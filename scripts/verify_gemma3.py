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

# Discover the full model tree
print(f"Top-level: {[n for n, _ in m.named_children()]}")
print(f"m.model children: {[n for n, _ in m.model.named_children()]}")

# Walk down to find layers
for name, child in m.model.named_children():
    if hasattr(child, 'layers'):
        layers = child.layers
        print(f"\nFound layers at m.model.{name}.layers")
        print(f"num_layers: {len(layers)}")
        print(f"hidden_dim: {layers[0].self_attn.q_proj.in_features}")
    if hasattr(child, 'embed_tokens'):
        print(f"embed_tokens at m.model.{name}.embed_tokens: {child.embed_tokens.weight.shape}")
    # Go one more level
    for name2, child2 in child.named_children():
        if hasattr(child2, 'layers'):
            layers = child2.layers
            print(f"\nFound layers at m.model.{name}.{name2}.layers")
            print(f"num_layers: {len(layers)}")
            print(f"hidden_dim: {layers[0].self_attn.q_proj.in_features}")

del m
torch.cuda.empty_cache()
print("\nDone.")
