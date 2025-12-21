---
title: FLUX.2 Text Encoder (Private)
emoji: 🔤
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
private: true
hardware: zero-a10g
---

# FLUX.2 Text Encoder

Private text encoder space for FLUX.2 image generation.

Uses `arnomatic/Mistral-Small-3.2-24B-Instruct-2506-Text-Only-heretic` with NF4 quantization.

## API Usage

```python
from gradio_client import Client

client = Client("YOUR_USERNAME/YOUR_SPACE_NAME")
result = client.predict(
    prompt="your prompt here",
    api_name="/encode_text"
)
embeddings_path = result[0]
prompt_embeds = torch.load(embeddings_path, weights_only=False)
```
