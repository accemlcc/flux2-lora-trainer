import gc
import os
import time
import tempfile
import logging
import spaces

import torch
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer

# ------------------------------------------------------
# Logging
# ------------------------------------------------------
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("flux2-text-encoder")

# ------------------------------------------------------
# Config
# ------------------------------------------------------
MODEL_ID = "arnomatic/Mistral-Small-3.2-24B-Instruct-2506-Text-Only-heretic"
DTYPE = torch.bfloat16

# FLUX.2 uses these layers for prompt embeddings
TEXT_ENCODER_OUT_LAYERS = [10, 20, 30]
MAX_SEQUENCE_LENGTH = 512

SYSTEM_MESSAGE = """You are an AI that reasons about image descriptions. You give structured responses focusing on object relationships, object attribution and actions without speculation."""

# ------------------------------------------------------
# Load Model and Tokenizer at Startup
# ------------------------------------------------------
logger.info(f"Loading model {MODEL_ID}...")

t0 = time.time()
text_encoder = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=DTYPE,
).to("cuda")
logger.info(f"Loaded text encoder in {time.time() - t0:.2f}s, dtype={text_encoder.dtype}")

t1 = time.time()
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
logger.info(f"Loaded tokenizer in {time.time() - t1:.2f}s")

torch.set_grad_enabled(False)


def format_messages(prompts, system_message=SYSTEM_MESSAGE):
    """Format prompts into chat messages."""
    return [
        [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt.replace("[IMG]", "")},
        ]
        for prompt in prompts
    ]


def get_prompt_embeds(prompts):
    """Get FLUX.2-compatible prompt embeddings from text."""
    global text_encoder, tokenizer
    
    # Format as chat messages and apply template
    messages_batch = format_messages(prompts=prompts)
    
    # Tokenize each prompt with chat template
    all_input_ids = []
    all_attention_masks = []
    
    for messages in messages_batch:
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
            max_length=MAX_SEQUENCE_LENGTH,
        )
        all_input_ids.append(encoded["input_ids"])
        all_attention_masks.append(encoded["attention_mask"])
    
    # Stack into batches
    input_ids = torch.cat(all_input_ids, dim=0).to(text_encoder.device)
    attention_mask = torch.cat(all_attention_masks, dim=0).to(text_encoder.device)
    
    # Forward pass
    with torch.inference_mode():
        output = text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )
    
    # Stack outputs from intermediate layers (like FLUX.2 does)
    out = torch.stack([output.hidden_states[k] for k in TEXT_ENCODER_OUT_LAYERS], dim=1)
    out = out.to(dtype=DTYPE, device=text_encoder.device)
    
    # Reshape: [batch, num_layers, seq_len, hidden_dim] -> [batch, seq_len, num_layers * hidden_dim]
    batch_size, num_channels, seq_len, hidden_dim = out.shape
    prompt_embeds = out.permute(0, 2, 1, 3).reshape(
        batch_size, seq_len, num_channels * hidden_dim
    )
    
    return prompt_embeds


def get_vram_info():
    """Get current VRAM usage info."""
    if torch.cuda.is_available():
        return {
            "vram_allocated_mb": round(torch.cuda.memory_allocated() / 1024 / 1024, 2),
            "vram_reserved_mb": round(torch.cuda.memory_reserved() / 1024 / 1024, 2),
            "vram_max_allocated_mb": round(torch.cuda.max_memory_allocated() / 1024 / 1024, 2),
        }
    return {"vram": "CUDA not available"}


@spaces.GPU()
def encode_text(prompt: str):
    """Encode text and return a downloadable pytorch file."""
    t0 = time.time()
    
    # Handle multiple prompts (one per line)
    prompts = [p.strip() for p in prompt.strip().split("\n") if p.strip()]
    if not prompts:
        return None, "Please enter at least one prompt"
    
    num_prompts = len(prompts)
    logger.info(f"Encoding {num_prompts} prompt(s)")
    
    prompt_embeds = get_prompt_embeds(prompts)
    
    duration = (time.time() - t0) * 1000.0
    
    logger.info(
        f"Encoded in {duration:.2f}ms | prompt_embeds.shape={tuple(prompt_embeds.shape)}"
    )
    
    # Save tensor to a temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pt")
    torch.save(prompt_embeds.cpu(), temp_file.name)
    
    # Clean up GPU tensors
    del prompt_embeds
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    vram = get_vram_info()
    status = (
        f"✅ Encoded {num_prompts} prompt(s) in {duration:.2f}ms\n"
        f"VRAM: {vram.get('vram_allocated_mb', 'N/A')} MB allocated, "
        f"{vram.get('vram_max_allocated_mb', 'N/A')} MB peak"
    )
    
    return temp_file.name, status


# ------------------------------------------------------
# Gradio Interface
# ------------------------------------------------------
with gr.Blocks(title="FLUX.2 Text Encoder") as demo:
    gr.Markdown("""
    # 🔤 FLUX.2 Text Encoder
    
    This space provides FLUX.2-compatible text embeddings using Mistral-Small-3.2-24B.
    
    **Model:** `arnomatic/Mistral-Small-3.2-24B-Instruct-2506-Text-Only-heretic`
    
    **Usage:** Enter text to encode. For multiple prompts, put each on a new line.
    """)
    
    with gr.Row():
        with gr.Column():
            prompt_input = gr.Textbox(
                label="Prompt(s)",
                placeholder="Enter your prompt here...\nOr multiple prompts, one per line",
                lines=5,
            )
            encode_btn = gr.Button("Encode", variant="primary")
        
        with gr.Column():
            output_file = gr.File(label="Download Embeddings (.pt)")
            status_output = gr.Textbox(label="Status", interactive=False)
    
    encode_btn.click(
        fn=encode_text,
        inputs=[prompt_input],
        outputs=[output_file, status_output],
        api_name="encode_text",
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
