#!/usr/bin/env python3
"""
FLUX.2 LoRA Training with NF4 Quantization + Multi-GPU Support

Features:
- NF4 quantized DiT (fits in ~20GB VRAM)
- device_map="auto" distributes across multiple GPUs
- Pre-computed text embeddings (no text encoder in memory!)
- Cached VAE latents (VAE unloaded after caching)

Requirements:
    pip install torch diffusers transformers accelerate peft bitsandbytes tqdm pillow

Usage:
    # Step 1: Pre-compute embeddings first!
    python precompute_embeddings.py --dataset_dir ./my_dataset --mode local

    # Step 2: Train LoRA
    python train_flux2_lora_nf4.py --dataset_dir ./my_dataset --output_dir ./lora_output
"""

import argparse
import gc
import random
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from PIL import Image
from torchvision import transforms

# Diffusers
from diffusers import (
    Flux2Pipeline,
    Flux2Transformer2DModel,
    FlowMatchEulerDiscreteScheduler,
    AutoencoderKLFlux2,
)

# Quantization
from transformers import BitsAndBytesConfig

# LoRA
from peft import LoraConfig, get_peft_model


def free_memory():
    """Free GPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def save_lora_diffusers_format(peft_model, save_path):
    """
    Save LoRA weights in both PEFT and safetensors format.
    
    - PEFT format: Works with PeftModel.from_pretrained + merge_and_unload
    - Safetensors: For compatibility with other tools
    """
    from safetensors.torch import save_file
    
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Save in PEFT format (works with merge_and_unload)
    peft_model.save_pretrained(save_path)
    
    # Also extract and save as safetensors for compatibility
    lora_state_dict = {}
    for name, param in peft_model.named_parameters():
        if "lora_" in name and param.requires_grad:
            # Clean up the key name
            clean_name = name.replace("base_model.model.", "")
            lora_state_dict[clean_name] = param.to(torch.bfloat16).cpu().detach()
    
    print(f"Saving {len(lora_state_dict)} LoRA parameters...")
    
    # Save as safetensors alongside PEFT format
    save_file(lora_state_dict, save_path / "lora_weights.safetensors")


def generate_sample(
    transformer,
    vae,
    scheduler,
    prompt: str,
    space_url: str,
    output_path: str,
    device: str,
    seed: int = 42,
    num_steps: int = 20,
):
    """Generate a sample image during training using remote text encoder."""
    from gradio_client import Client
    from diffusers import Flux2Pipeline
    
    try:
        # Get embedding from remote space
        print(f"  Getting embedding from {space_url}...")
        client = Client(space_url)
        result = client.predict(prompt=prompt, api_name="/encode_text")
        prompt_embeds = torch.load(result[0])
        prompt_embeds = prompt_embeds.to(dtype=torch.bfloat16, device=device)
        
        # Put transformer in eval mode temporarily
        was_training = transformer.training
        transformer.eval()
        
        # Wrap the forward method to ensure bfloat16 output (fixes NF4+LoRA dtype issue)
        original_forward = transformer.forward
        def wrapped_forward(*args, **kwargs):
            output = original_forward(*args, **kwargs)
            if hasattr(output, 'sample'):
                output = output.__class__(sample=output.sample.to(torch.bfloat16))
            elif isinstance(output, tuple):
                output = tuple(o.to(torch.bfloat16) if torch.is_tensor(o) else o for o in output)
            elif torch.is_tensor(output):
                output = output.to(torch.bfloat16)
            return output
        transformer.forward = wrapped_forward
        
        # Create temporary pipeline
        pipe = Flux2Pipeline.from_pretrained(
            "diffusers/FLUX.2-dev-bnb-4bit",
            transformer=transformer,
            vae=vae,
            text_encoder=None,
            torch_dtype=torch.bfloat16,
        )
        pipe.vae = pipe.vae.to(device)
        
        # Generate
        with torch.no_grad():
            image = pipe(
                prompt_embeds=prompt_embeds,
                generator=torch.Generator(device=device).manual_seed(seed),
                num_inference_steps=num_steps,
                guidance_scale=4,
            ).images[0]
        
        image.save(output_path)
        print(f"  Sample saved to {output_path}")
        
        # Restore original forward and training mode
        transformer.forward = original_forward
        if was_training:
            transformer.train()
            
        del pipe
        
    except Exception as e:
        print(f"  Sample generation failed: {e}")

class Flux2PrecomputedDataset(Dataset):
    """Dataset with pre-computed embeddings and cached latents."""

    def __init__(
        self,
        dataset_dir: str,
        vae: Optional[AutoencoderKLFlux2] = None,
        cache_latents: bool = True,
        image_size: int = 512,
    ):
        self.dataset_dir = Path(dataset_dir)
        self.cache_latents = cache_latents
        self.image_size = image_size

        # Image transforms
        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    image_size, interpolation=transforms.InterpolationMode.LANCZOS
                ),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

        # Find all samples with embeddings
        self.samples = []
        for img_path in self.dataset_dir.rglob("*"):
            if img_path.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]:
                embed_path = img_path.with_suffix(".pt")
                if embed_path.exists():
                    self.samples.append(
                        {
                            "image": img_path,
                            "embed": embed_path,
                        }
                    )

        print(f"Found {len(self.samples)} samples with pre-computed embeddings")

        # Cache latents if VAE provided
        self.latents_cache = []
        if cache_latents and vae is not None:
            print("Caching VAE latents...")
            device = next(vae.parameters()).device
            dtype = next(vae.parameters()).dtype

            for sample in tqdm(self.samples, desc="Encoding images"):
                image = Image.open(sample["image"]).convert("RGB")
                image = self.transform(image).unsqueeze(0).to(device, dtype=dtype)

                with torch.no_grad():
                    latent = vae.encode(image).latent_dist.mode()

                self.latents_cache.append(latent.cpu())

            print("Latents cached!")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Load pre-computed embedding
        prompt_embeds = torch.load(sample["embed"], weights_only=True)
        if prompt_embeds.dim() == 3 and prompt_embeds.size(0) == 1:
            prompt_embeds = prompt_embeds.squeeze(0)

        # Get latent (cached or compute on-the-fly)
        if self.latents_cache:
            latent = self.latents_cache[idx]
            if latent.dim() == 4:
                latent = latent.squeeze(0)
            return {"latent": latent, "prompt_embeds": prompt_embeds}
        else:
            # Load image for on-the-fly encoding
            image = Image.open(sample["image"]).convert("RGB")
            image = self.transform(image)
            return {"image": image, "prompt_embeds": prompt_embeds}


def get_sigmas(timesteps, scheduler, device, dtype, n_dim=4):
    """Get sigmas for flow matching."""
    sigmas = scheduler.sigmas.to(device=device, dtype=dtype)
    schedule_timesteps = scheduler.timesteps.to(device)
    step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]
    sigma = sigmas[step_indices].flatten()
    while len(sigma.shape) < n_dim:
        sigma = sigma.unsqueeze(-1)
    return sigma


def main():
    parser = argparse.ArgumentParser(
        description="FLUX.2 LoRA Training with NF4 + Multi-GPU"
    )

    # Paths
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./lora_output")
    parser.add_argument("--model_id", type=str, default="diffusers/FLUX.2-dev-bnb-4bit")
    parser.add_argument("--hf_token", type=str, default=None)

    # Training
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--gradient_accumulation", type=int, default=4)
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--max_sequence_length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--guidance_scale", type=float, default=3.5)
    
    # Trigger word for persona training
    parser.add_argument("--trigger_word", type=str, default=None, 
                        help="Trigger word to prepend to captions (e.g., 'ohwx cat')")

    # LoRA
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=16)

    # Memory optimization
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)
    parser.add_argument("--cache_latents", action="store_true", default=True)
    
    # Device selection
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Device to use for training (cuda:0 or cuda:1)")
    parser.add_argument("--multi_gpu", action="store_true", default=False,
                        help="Distribute model across all available GPUs")
    
    # Sampling during training (epoch-based)
    parser.add_argument("--sample_every_epochs", type=int, default=0,
                        help="Generate sample image every N epochs (0=disabled)")
    parser.add_argument("--sample_prompt", type=str, default="ohwx cat at the beach",
                        help="Prompt for sample generation")
    parser.add_argument("--sample_space", type=str, default="multimodalart/mistral-text-encoder",
                        help="HuggingFace Space for text encoding")
    
    # Resume training
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from (e.g., ./output/checkpoint-epoch-100)")
    parser.add_argument("--start_epoch", type=int, default=0,
                        help="Epoch to start from (used with --resume)")

    args = parser.parse_args()

    # Set seed
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    dtype = torch.bfloat16

    print("=" * 60)
    print("FLUX.2 LoRA Training with NF4 Quantization")
    print("=" * 60)
    print(f"GPUs available: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(
            f"  GPU {i}: {torch.cuda.get_device_name(i)} ({torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB)"
        )

    # =========================================================================
    # Step 1: Load VAE (temporarily, for caching latents)
    # =========================================================================
    print("\n[1/4] Loading VAE for latent caching...")

    vae = AutoencoderKLFlux2.from_pretrained(
        args.model_id,
        subfolder="vae",
        torch_dtype=dtype,
        token=args.hf_token,
    ).to(args.device)
    vae.requires_grad_(False)
    vae.eval()

    # Get VAE normalization parameters
    latents_bn_mean = vae.bn.running_mean.view(1, -1, 1, 1).to(args.device)
    latents_bn_std = torch.sqrt(
        vae.bn.running_var.view(1, -1, 1, 1) + vae.config.batch_norm_eps
    ).to(args.device)

    # =========================================================================
    # Step 2: Create dataset and cache latents
    # =========================================================================
    print("\n[2/4] Loading dataset and caching latents...")

    dataset = Flux2PrecomputedDataset(
        dataset_dir=args.dataset_dir,
        vae=vae if args.cache_latents else None,
        cache_latents=args.cache_latents,
        image_size=args.image_size,
    )

    # Unload VAE to free memory
    print("Unloading VAE...")
    del vae
    free_memory()

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )

    # =========================================================================
    # Step 3: Load DiT Transformer with NF4 Quantization + Multi-GPU
    # =========================================================================
    print("\n[3/4] Loading DiT Transformer (NF4 quantized, multi-GPU)...")

    # Check if using pre-quantized model
    is_prequantized = "bnb-4bit" in args.model_id.lower()
    
    # Determine device_map based on multi_gpu flag
    if args.multi_gpu:
        device_map = "auto"
        # Force distribution by limiting memory per GPU
        max_memory = {0: "10GB", 1: "10GB"}
        print("Using multi-GPU: model will be distributed across all available GPUs")
        print(f"  max_memory: {max_memory}")
    else:
        device_map = {"": args.device}
        max_memory = None
    
    if is_prequantized:
        # Pre-quantized model - load directly
        print("Using pre-quantized model, skipping quantization config...")
        transformer = Flux2Transformer2DModel.from_pretrained(
            args.model_id,
            subfolder="transformer",
            torch_dtype=dtype,
            device_map=device_map,
            max_memory=max_memory,
            token=args.hf_token,
        )
    else:
        # Apply quantization
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
        )
        transformer = Flux2Transformer2DModel.from_pretrained(
            args.model_id,
            subfolder="transformer",
            quantization_config=quantization_config,
            torch_dtype=dtype,
            device_map=device_map,
            max_memory=max_memory,
            token=args.hf_token,
        )

    # Print device distribution
    if hasattr(transformer, "hf_device_map"):
        print(
            f"Model distributed across devices: {set(transformer.hf_device_map.values())}"
        )

    # Prepare for training (manual setup for diffusion models)
    # Note: prepare_model_for_kbit_training doesn't work with diffusion transformers
    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()
    
    # Enable gradients for quantized training
    for param in transformer.parameters():
        if param.requires_grad:
            param.data = param.data.float()  # Upcast for stable training

    # Load scheduler
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.model_id,
        subfolder="scheduler",
        token=args.hf_token,
    )

    # =========================================================================
    # Step 4: Setup LoRA
    # =========================================================================
    print("\n[4/4] Setting up LoRA...")

    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        init_lora_weights="gaussian",
        target_modules=[
            # Regular transformer_blocks (8 blocks)
            "to_k",
            "to_q", 
            "to_v",
            "to_out.0",
            # Single transformer blocks (47+ blocks) - these use fused QKV+MLP
            "to_qkv_mlp_proj",
            "to_add_qkv_proj",
            # Output projections  
            "proj_out",
        ],
    )

    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        # First apply fresh LoRA config, then load weights
        transformer = get_peft_model(transformer, lora_config)
        # Load saved adapter weights
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file as load_safetensors
        adapter_path = Path(args.resume) / "adapter_model.safetensors"
        if adapter_path.exists():
            adapter_weights = load_safetensors(str(adapter_path))
            set_peft_model_state_dict(transformer, adapter_weights)
            print(f"Loaded adapter weights from {adapter_path}")
        else:
            print(f"Warning: No adapter weights found at {adapter_path}")
    else:
        transformer = get_peft_model(transformer, lora_config)
    
    transformer.print_trainable_parameters()

    # =========================================================================
    # Optimizer
    # =========================================================================
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, transformer.parameters()),
        lr=args.learning_rate,
        weight_decay=0.01,
    )

    # =========================================================================
    # Training Loop
    # =========================================================================
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine device for operations (for latents, inputs, etc.)
    if args.multi_gpu and hasattr(transformer, "hf_device_map"):
        # Use the first device from the model's device map
        main_device = list(set(transformer.hf_device_map.values()))[0]
        print(f"Using {main_device} as main device for inputs/latents")
    else:
        main_device = args.device

    print("\n" + "=" * 60)
    print("Starting Training")
    print("=" * 60)
    print(f"  Dataset size: {len(dataset)}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Gradient accumulation: {args.gradient_accumulation}")
    print(f"  Effective batch size: {args.batch_size * args.gradient_accumulation}")
    print(f"  LoRA rank: {args.lora_rank} (alpha: {args.lora_alpha})")
    print(f"  Learning rate: {args.learning_rate}")
    if args.sample_every_epochs:
        print(f"  Sample every: {args.sample_every_epochs} epochs")
        print(f"  Sample prompt: {args.sample_prompt}")
    if args.resume:
        print(f"  Resuming from: epoch {args.start_epoch}")
    print(f"  Output: {output_dir}")

    global_step = 0

    for epoch in range(args.start_epoch, args.epochs):
        transformer.train()
        epoch_loss = 0.0

        progress = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{args.epochs}")

        for step, batch in enumerate(progress):
            # Get latents
            latents = batch["latent"].to(main_device, dtype=dtype)
            prompt_embeds = batch["prompt_embeds"].to(main_device, dtype=dtype)

            # Normalize latents (FLUX.2 specific)
            latents = Flux2Pipeline._patchify_latents(latents)
            latents = (
                latents - latents_bn_mean.to(latents.device)
            ) / latents_bn_std.to(latents.device)

            # Prepare latent IDs
            latent_ids = Flux2Pipeline._prepare_latent_ids(latents).to(
                main_device, dtype=dtype
            )

            # Prepare text IDs
            text_ids = Flux2Pipeline._prepare_text_ids(prompt_embeds).to(
                main_device, dtype=dtype
            )

            # Sample noise
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]

            # Sample timesteps (flow matching style)
            u = torch.rand(bsz, device=main_device)
            indices = (u * scheduler.config.num_train_timesteps).long().cpu()
            timesteps = scheduler.timesteps[indices].to(main_device)

            # Get sigmas
            sigmas = get_sigmas(
                timesteps, scheduler, main_device, dtype, n_dim=latents.ndim
            )

            # Add noise (flow matching interpolation)
            noisy_latents = (1.0 - sigmas) * latents + sigmas * noise

            # Pack for transformer
            packed_noisy_latents = Flux2Pipeline._pack_latents(noisy_latents)

            # Guidance
            guidance = torch.full(
                [bsz], args.guidance_scale, device=main_device, dtype=dtype
            )

            # Forward pass
            model_pred = transformer(
                hidden_states=packed_noisy_latents,
                timestep=timesteps / 1000,  # Normalized timestep
                guidance=guidance,
                encoder_hidden_states=prompt_embeds,
                txt_ids=text_ids,
                img_ids=latent_ids,
                return_dict=False,
            )[0]

            # Extract only image predictions
            model_pred = model_pred[:, : packed_noisy_latents.size(1), :]

            # Unpack predictions
            model_pred = Flux2Pipeline._unpack_latents_with_ids(model_pred, latent_ids)

            # Flow matching loss: predict velocity (noise - clean)
            target = noise - latents
            loss = F.mse_loss(model_pred.float(), target.float())

            # Gradient accumulation
            loss = loss / args.gradient_accumulation
            loss.backward()

            if (step + 1) % args.gradient_accumulation == 0:
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

            epoch_loss += loss.item() * args.gradient_accumulation

            # Update progress bar
            gpu_mem = torch.cuda.max_memory_allocated() / 1024**3
            progress.set_postfix(
                {
                    "loss": f"{loss.item() * args.gradient_accumulation:.4f}",
                    "gpu_mem": f"{gpu_mem:.1f}GB",
                }
            )

        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch + 1} - Average Loss: {avg_loss:.4f}")

        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0 or epoch == args.epochs - 1:
            checkpoint_dir = output_dir / f"checkpoint-epoch-{epoch + 1}"
            save_lora_diffusers_format(transformer, checkpoint_dir)
            print(f"Saved checkpoint to {checkpoint_dir}")
        
        # Generate sample at epoch intervals
        if args.sample_every_epochs and (epoch + 1) % args.sample_every_epochs == 0:
            sample_path = output_dir / f"sample_epoch_{epoch + 1}.png"
            print(f"\n  Generating sample at epoch {epoch + 1}...")
            
            # Load VAE temporarily for sample generation
            sample_vae = AutoencoderKLFlux2.from_pretrained(
                args.model_id, subfolder="vae", torch_dtype=dtype
            ).to(main_device)
            
            generate_sample(
                transformer=transformer,
                vae=sample_vae,
                scheduler=scheduler,
                prompt=args.sample_prompt,
                space_url=args.sample_space,
                output_path=str(sample_path),
                device=main_device,
                seed=42,
                num_steps=20,
            )
            
            del sample_vae
            free_memory()

    # Save final LoRA
    final_dir = output_dir / "final"
    save_lora_diffusers_format(transformer, final_dir)
    print(f"\n✅ Training complete! LoRA saved to {final_dir}")

    # Print final memory stats
    print(f"\nPeak GPU memory: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")


if __name__ == "__main__":
    main()
