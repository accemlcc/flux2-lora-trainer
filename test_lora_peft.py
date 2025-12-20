#!/usr/bin/env python3
"""
Test trained LoRA - simplified approach with dtype conversion hook.
"""

import argparse
import gc
import torch
from diffusers import Flux2Pipeline, Flux2Transformer2DModel
from peft import PeftModel

REPO_ID = "diffusers/FLUX.2-dev-bnb-4bit"
DTYPE = torch.bfloat16


def main():
    parser = argparse.ArgumentParser(description="Test trained LoRA with PEFT")
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--lora_path", type=str, default="./nsfw_lora_v2/final")
    parser.add_argument("--output", type=str, default="./lora_peft_output.png")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=28)
    args = parser.parse_args()
    
    # Generate embedding
    print("Generating embedding...")
    from precompute_embeddings import get_local_embeddings, load_local_model
    model, tokenizer = load_local_model("multimodalart/Mistral-Small-3.2-24B-Instruct-2506-Text-Only", quantize="nf4")
    prompt_embeds = get_local_embeddings(args.prompt, model, tokenizer)
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    
    # Load base transformer with LoRA
    print("Loading base transformer...")
    transformer = Flux2Transformer2DModel.from_pretrained(
        REPO_ID,
        subfolder="transformer",
        torch_dtype=DTYPE,
        device_map="auto",
    )
    
    print(f"Loading PEFT LoRA from {args.lora_path}...")
    transformer = PeftModel.from_pretrained(transformer, args.lora_path)
    transformer.eval()
    print(f"LoRA loaded! Active adapter: {transformer.active_adapter}")
    
    # Wrap the forward method to ensure bfloat16 output
    original_forward = transformer.forward
    def wrapped_forward(*args, **kwargs):
        output = original_forward(*args, **kwargs)
        if hasattr(output, 'sample'):
            output = output.__class__(sample=output.sample.to(DTYPE))
        elif isinstance(output, tuple):
            output = tuple(o.to(DTYPE) if torch.is_tensor(o) else o for o in output)
        elif torch.is_tensor(output):
            output = output.to(DTYPE)
        return output
    transformer.forward = wrapped_forward
    
    # Get device
    device = next(transformer.parameters()).device
    prompt_embeds = prompt_embeds.to(dtype=DTYPE, device=device)
    
    # Create full pipeline
    print("Creating pipeline...")
    pipe = Flux2Pipeline.from_pretrained(
        REPO_ID,
        transformer=transformer,
        text_encoder=None,
        torch_dtype=DTYPE,
    )
    
    # Move VAE to same device as transformer
    pipe.vae = pipe.vae.to(device)
    
    # Generate image
    print(f"Generating image on {device}...")
    image = pipe(
        prompt_embeds=prompt_embeds,
        generator=torch.Generator(device=device).manual_seed(args.seed),
        num_inference_steps=args.steps,
        guidance_scale=4,
    ).images[0]
    
    image.save(args.output)
    print(f"✅ Saved to {args.output}")


if __name__ == "__main__":
    main()
