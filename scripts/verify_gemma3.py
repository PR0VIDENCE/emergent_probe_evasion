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

print(f"Top-level children: {[n for n, _ in m.named_children()]}")
print(f"Has language_model: {hasattr(m, 'language_model')}")
print(f"Has model.layers: {hasattr(m.model, 'layers')}")

if hasattr(m.model, 'layers'):
    layers = m.model.layers
    print(f"num_layers: {len(layers)}")
    print(f"hidden_dim: {layers[0].self_attn.q_proj.in_features}")
    print(f"embed_tokens shape: {m.model.embed_tokens.weight.shape}")

# Test that our helper works
from src.inference.extract_activations import _get_text_backbone, get_num_layers, get_embed_tokens
backbone = _get_text_backbone(m)
print(f"\n_get_text_backbone works: {backbone is m.model}")
print(f"get_num_layers: {get_num_layers(m)}")
print(f"get_embed_tokens shape: {get_embed_tokens(m).weight.shape}")

del m
torch.cuda.empty_cache()
print("\nDone. Update configs/models/gemma3_27b_r1.yaml with the values above.")
