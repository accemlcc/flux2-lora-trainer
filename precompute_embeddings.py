#!/usr/bin/env python3
"""
Pre-compute FLUX.2 text embeddings for LoRA training.

This script processes all caption files in a dataset directory and generates
.pt embedding files using either a remote encoder (HuggingFace Space) or local encoder.

Usage:
    python precompute_embeddings.py --dataset_dir ./my_dataset --mode remote
    python precompute_embeddings.py --dataset_dir ./my_dataset --mode local --quantize nf4
    python precompute_embeddings.py --dataset_dir ./my_dataset --mode local --quantize none  # Full bf16
"""

import argparse
from pathlib import Path
import torch
from tqdm import tqdm


def get_remote_embeddings(
    prompt: str, space_url: str = "your-username/your-text-encoder-space"
):
    """Get embeddings from remote HuggingFace Space."""
    from gradio_client import Client

    client = Client(space_url)
    result = client.predict(prompt=prompt, api_name="/encode_text")

    # Load the returned .pt file
    prompt_embeds = torch.load(result[0])
    return prompt_embeds


def get_local_embeddings(
    prompt: str,
    model,
    tokenizer,
    device: str = "cuda",
    max_length: int = 512,
    layers: list = [10, 20, 30],
):
    """Get embeddings using local model."""

    SYSTEM_MESSAGE = """You are an AI that reasons about image descriptions. You give structured responses focusing on object relationships, object attribution and actions without speculation."""

    # Format as chat message
    messages = [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": prompt.replace("[IMG]", "")},
    ]

    # Apply chat template and tokenize
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    encoded = tokenizer(
        text,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_length,
    )

    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    # Forward pass
    with torch.inference_mode():
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )

    # Stack outputs from intermediate layers
    out = torch.stack([output.hidden_states[k] for k in layers], dim=1)

    # Reshape: [batch, num_layers, seq_len, hidden_dim] -> [batch, seq_len, num_layers * hidden_dim]
    batch_size, num_channels, seq_len, hidden_dim = out.shape
    prompt_embeds = out.permute(0, 2, 1, 3).reshape(
        batch_size, seq_len, num_channels * hidden_dim
    )

    return prompt_embeds.cpu()


def load_local_model(model_id: str, device: str = "cuda", quantize: str = "nf4"):
    """Load local text encoder model with optional quantization.
    
    Args:
        model_id: HuggingFace model ID
        device: Device to load model on (ignored when using quantization)
        quantize: Quantization method - "none", "nf4", or "int8"
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    print(f"Loading model {model_id} with quantization={quantize}...")

    tokenizer = AutoTokenizer.from_pretrained(model_id, fix_mistral_regex=True)
    
    if quantize == "nf4":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto",
        )
    elif quantize == "int8":
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto",
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
        ).to(device)
    
    model.eval()

    print(f"Model loaded! Device: {model.device if hasattr(model, 'device') else 'auto'}")
    return model, tokenizer


def find_caption_files(dataset_dir: Path):
    """Find all caption files (.txt, .caption) in dataset directory."""
    caption_files = []

    for ext in ["*.txt", "*.caption"]:
        caption_files.extend(dataset_dir.rglob(ext))

    return sorted(caption_files)


def main():
    parser = argparse.ArgumentParser(description="Pre-compute FLUX.2 text embeddings")
    parser.add_argument(
        "--dataset_dir", type=str, required=True, help="Path to dataset directory"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for embeddings (default: same as dataset)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["remote", "local"],
        default="remote",
        help="Use remote Space or local model",
    )
    parser.add_argument(
        "--space_url",
        type=str,
        default="multimodalart/mistral-text-encoder",
        help="HuggingFace Space URL for remote mode",
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="multimodalart/Mistral-Small-3.2-24B-Instruct-2506-Text-Only",
        help="Model ID for local mode",
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device for local mode"
    )
    parser.add_argument(
        "--quantize",
        type=str,
        choices=["none", "nf4", "int8"],
        default="nf4",
        help="Quantization method for local mode (default: nf4)",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip files that already have embeddings",
    )
    parser.add_argument(
        "--trigger_word",
        type=str,
        default=None,
        help="Trigger word to prepend to all captions (e.g., 'ohwx cat')",
    )

    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir) if args.output_dir else dataset_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all caption files
    caption_files = find_caption_files(dataset_dir)
    print(f"Found {len(caption_files)} caption files")

    if len(caption_files) == 0:
        print("No caption files found! Looking for .txt or .caption files.")
        return

    # Load local model if needed
    model, tokenizer = None, None
    if args.mode == "local":
        model, tokenizer = load_local_model(args.model_id, args.device, args.quantize)

    # Process each caption file
    processed = 0
    skipped = 0

    for caption_file in tqdm(caption_files, desc="Processing captions"):
        # Output path: same name but .pt extension
        relative_path = caption_file.relative_to(dataset_dir)
        output_path = output_dir / relative_path.with_suffix(".pt")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Skip if exists
        if args.skip_existing and output_path.exists():
            skipped += 1
            continue

        # Read caption
        caption = caption_file.read_text(encoding="utf-8").strip()

        if not caption:
            print(f"Warning: Empty caption in {caption_file}")
            continue
        
        # Prepend trigger word if specified
        if args.trigger_word:
            caption = f"{args.trigger_word}, {caption}"

        # Get embeddings
        try:
            if args.mode == "remote":
                embeds = get_remote_embeddings(caption, args.space_url)
            else:
                embeds = get_local_embeddings(caption, model, tokenizer, args.device)

            # Save embeddings
            torch.save(embeds, output_path)
            processed += 1

        except Exception as e:
            print(f"Error processing {caption_file}: {e}")
            continue

    print(f"\nDone! Processed: {processed}, Skipped: {skipped}")
    print(f"Embeddings saved to: {output_dir}")


if __name__ == "__main__":
    main()
