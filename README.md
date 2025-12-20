# FLUX.2 LoRA Training with NF4 Quantization

Train LoRA adapters for FLUX.2 [dev] on consumer GPUs (24GB+ VRAM) using NF4 quantization.

## Features

- **NF4 Quantization**: Uses 4-bit quantization to fit the 32B parameter model on consumer GPUs
- **Pre-computed Embeddings**: Local (multi-GPU) or remote text encoder support
- **Multi-GPU Support**: Distribute training across multiple GPUs
- **Resume Training**: Continue from checkpoints
- **Memory Efficient**: ~12GB VRAM per GPU with multi-GPU mode

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Prepare Dataset

Create a folder with your training images and captions:
```
my_dataset/
  image1.jpg
  image1.txt       # caption for image1
  image2.jpg
  image2.txt
  ...
```

### 2. Pre-compute Embeddings

**Option A: Local mode (NF4 quantized, ~12GB VRAM)**

Run the text encoder locally with NF4 quantization:

```bash
python precompute_embeddings.py \
  --dataset_dir ./my_dataset \
  --mode local \
  --quantize nf4 \
  --trigger_word "my_trigger"
```

**Option B: Remote mode (via HuggingFace Space)**

> ⚠️ **Note**: There is currently no reliable public text encoder space available. You need to either:
> - Deploy your own Space (requires HuggingFace Pro for persistent GPU)
> - Use local mode with multi-GPU

If you have your own Space:

```bash
python precompute_embeddings.py \
  --dataset_dir ./my_dataset \
  --mode remote \
  --remote_space "your-username/flux2-text-encoder" \
  --trigger_word "my_trigger"
```

### 3. Train LoRA

Single GPU (up to approx. 768x768 @ RTX 4090):
```bash
python train_flux2_lora_nf4.py \
  --dataset_dir ./my_dataset \
  --output_dir ./my_lora \
  --epochs 200 \
  --device cuda:0 \
  --lora_rank 16 \
  --lora_alpha 32
```

Multi-GPU (distribute across all GPUs):
```bash
python train_flux2_lora_nf4.py \
  --dataset_dir ./my_dataset \
  --output_dir ./my_lora \
  --epochs 200 \
  --multi_gpu \
  --lora_rank 16 \
  --lora_alpha 32
```

Resume from checkpoint:
```bash
python train_flux2_lora_nf4.py \
  --dataset_dir ./my_dataset \
  --output_dir ./my_lora \
  --epochs 400 \
  --resume ./my_lora/checkpoint-epoch-200 \
  --start_epoch 200
```

### 4. Test LoRA

Testing requires a text encoder. Use `test_lora_peft.py` which runs the encoder locally:

```bash
python test_lora_peft.py \
  --prompt "my_trigger person at the beach" \
  --lora_path ./my_lora/final \
  --output test_output.png
```

> Note: This loads both the text encoder (~24GB with NF4) and FLUX.2 transformer (~12GB) - requires multi-GPU or high VRAM.

## Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--dataset_dir` | *required* | Path to dataset folder |
| `--output_dir` | ./output | Output directory for checkpoints |
| `--model_id` | diffusers/FLUX.2-dev-bnb-4bit | Base model ID |
| `--epochs` | 10 | Number of training epochs |
| `--lora_rank` | 16 | LoRA rank (higher = more capacity) |
| `--lora_alpha` | 16 | LoRA alpha (scaling = alpha/rank) |
| `--learning_rate` | 1e-4 | Learning rate |
| `--batch_size` | 1 | Batch size per GPU |
| `--gradient_accumulation` | 4 | Gradient accumulation steps |
| `--image_size` | 512 | Training image size |
| `--seed` | 42 | Random seed |
| `--trigger_word` | None | Trigger word to prepend to captions |
| `--device` | cuda:0 | GPU device for single-GPU training |
| `--multi_gpu` | False | Distribute model across all GPUs |
| `--resume` | None | Path to checkpoint to resume from |
| `--start_epoch` | 0 | Epoch to start from (used with --resume) |
| `--sample_every_epochs` | 0 | Generate sample images every N epochs (0=disabled) |
| `--sample_prompt` | - | Prompt for sample generation |
| `--sample_space` | - | HuggingFace Space for text encoding |

## About the Text Encoder

FLUX.2 requires a text encoder to convert prompts to embeddings. FLUX.2 uses **Mistral Small 24B** as its text encoder.

This repo uses a de-censored version of Mistral Small that has been processed with [Heretic](https://github.com/p-e-w/heretic) to remove refusal behavior:
- `arnomatic/Mistral-Small-3.2-24B-Instruct-2506-Text-Only-heretic`

**Why pre-compute embeddings?**
- The text encoder is very large (24B parameters)
- Pre-computing avoids loading it during training
- Saves ~24GB VRAM during training

## Tips

### Persona LoRAs (Character Training)

For training a specific person/character:
- Use high-quality images
- Use consistent trigger word (e.g., "ohwx person")
- Start with `--lora_rank 16 --lora_alpha 32` (scaling = 2.0)
- Train for 150-300 epochs

### Avoiding Overfitting

Signs of overfitting:
- Colors become muted/washed out
- Painting-like artifacts appear

Solutions:
- Lower alpha (e.g., 32 → 24)
- Stop training earlier
- Use more diverse training images

## Requirements

- Python 3.10+
- PyTorch 2.0+ with CUDA
- 24GB+ VRAM (single GPU) or 12GB+ per GPU (multi-GPU)
- 48GB+ VRAM for local text encoding (or multi-GPU)
- HuggingFace account with FLUX.2-dev access

## License

See LICENSE file.
