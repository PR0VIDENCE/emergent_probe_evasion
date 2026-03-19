"""Quick verification of Gemma 3 27B model architecture."""
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import transformers

print("Loading Gemma 3 27B R1...")
m = transformers.Gemma3ForConditionalGeneration.from_pretrained(
    "TheDrummer/Gemma-3-R1-27B-v1",
    device_map="auto",
    torch_dtype=torch.bfloat16,
)

# Test helpers
from src.inference.extract_activations import _get_text_backbone, get_num_layers, get_embed_tokens

backbone = _get_text_backbone(m)
print(f"backbone type: {type(backbone).__name__}")
print(f"num_layers: {get_num_layers(m)}")
print(f"hidden_dim: {backbone.layers[0].self_attn.q_proj.in_features}")
print(f"embed_tokens: {get_embed_tokens(m).weight.shape}")

del m
torch.cuda.empty_cache()
print("\nDone.")
