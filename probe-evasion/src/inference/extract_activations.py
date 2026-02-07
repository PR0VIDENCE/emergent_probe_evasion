"""Activation extraction from model forward passes."""

import torch
from typing import Dict, List, Tuple
from tqdm import tqdm


def load_model_and_tokenizer(model_config: dict):
    """
    Load model and tokenizer based on config.

    Args:
        model_config: Dict with model_id, quantization settings, device_map.

    Returns:
        Tuple of (model, tokenizer).
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    model_id = model_config["model_id"]

    quant_config = model_config.get("quantization", {})
    if quant_config.get("enabled", False):
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=quant_config.get("load_in_4bit", True),
            bnb_4bit_quant_type=quant_config.get("bnb_4bit_quant_type", "nf4"),
            bnb_4bit_compute_dtype=getattr(
                torch, quant_config.get("bnb_4bit_compute_dtype", "float16")
            ),
            bnb_4bit_use_double_quant=quant_config.get("bnb_4bit_use_double_quant", True),
        )
    else:
        bnb_config = None

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map=model_config.get("device_map", "auto"),
        trust_remote_code=True,
        dtype=torch.float16,
    )
    model.eval()

    return model, tokenizer


def _get_layer_module(model, layer_idx: int):
    """
    Get the decoder layer module at the given index.

    Works for Qwen2.5 and other standard transformers decoder-only models
    where layers are at model.model.layers[i].
    """
    return model.model.layers[layer_idx]


def extract_activations_batch(
    texts: List[str],
    model,
    tokenizer,
    layers: List[int],
    pooling: str = "last_token",
) -> Dict[int, torch.Tensor]:
    """
    Extract activations for a batch of texts from a pre-loaded model.

    Processes texts one at a time to avoid padding complications and
    keep VRAM usage predictable.

    Args:
        texts: List of text strings.
        model: Pre-loaded model.
        tokenizer: Pre-loaded tokenizer.
        layers: List of layer indices to extract from.
        pooling: "last_token" or "mean".

    Returns:
        Dict mapping layer_idx -> Tensor of shape (n_texts, hidden_dim).
    """
    collected = {layer_idx: [] for layer_idx in layers}

    # Register hooks
    hooks = []
    hook_outputs = {}

    def make_hook(layer_idx):
        def hook_fn(module, input, output):
            # output may be a tuple (hidden_states, ...) or a BaseModelOutput
            hidden_states = output[0] if isinstance(output, tuple) else output
            hook_outputs[layer_idx] = hidden_states.detach().cpu()
        return hook_fn

    for layer_idx in layers:
        layer_module = _get_layer_module(model, layer_idx)
        hook = layer_module.register_forward_hook(make_hook(layer_idx))
        hooks.append(hook)

    try:
        for text in tqdm(texts, desc="Extracting activations"):
            hook_outputs.clear()

            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            with torch.no_grad():
                model(**inputs)

            for layer_idx in layers:
                hidden_states = hook_outputs[layer_idx]
                # Normalize to 2D (seq_len, hidden_dim) regardless of batch dim
                if hidden_states.dim() == 3:
                    hidden_states = hidden_states[0]  # remove batch dim
                if pooling == "last_token":
                    collected[layer_idx].append(hidden_states[-1, :])
                elif pooling == "mean":
                    collected[layer_idx].append(hidden_states.mean(dim=0))
                else:
                    raise ValueError(f"Unknown pooling method: {pooling}")

    finally:
        for hook in hooks:
            hook.remove()

    result = {}
    for layer_idx in layers:
        result[layer_idx] = torch.stack(collected[layer_idx])  # (n_texts, hidden_dim)

    return result


def extract_activations(prompt: str, model_config: dict, layers: List[int]) -> Dict:
    """
    Extract hidden state activations from specified layers.

    Runs a forward pass and captures intermediate activations using
    hooks on the specified layers.

    Args:
        prompt: Prompt string to process.
        model_config: Model configuration dict (must specify local_path
                     for activation extraction).
        layers: List of layer indices to extract activations from.

    Returns:
        Dict mapping layer index to activation tensor:
        {
            layer_idx: tensor of shape (seq_len, hidden_dim),
            ...
        }
    """
    model, tokenizer = load_model_and_tokenizer(model_config)

    activations = {}
    hooks = []

    def make_hook(layer_idx):
        def hook_fn(module, input, output):
            hidden_states = output[0] if isinstance(output, tuple) else output
            if hidden_states.dim() == 3:
                hidden_states = hidden_states[0]
            activations[layer_idx] = hidden_states.detach().cpu()
        return hook_fn

    for layer_idx in layers:
        layer_module = _get_layer_module(model, layer_idx)
        hook = layer_module.register_forward_hook(make_hook(layer_idx))
        hooks.append(hook)

    try:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            model(**inputs)
    finally:
        for hook in hooks:
            hook.remove()

    return activations


def extract_activations_at_positions(
    prompt: str,
    model_config: dict,
    layers: List[int],
    positions: List[int]
) -> Dict:
    """
    Extract activations at specific token positions.

    Args:
        prompt: Prompt string to process.
        model_config: Model configuration dict.
        layers: List of layer indices.
        positions: List of token positions (-1 for last token).

    Returns:
        Dict with structure {layer_idx: {position: tensor(hidden_dim,)}}
    """
    raise NotImplementedError("TODO")


def extract_activations_during_generation(
    prompt: str,
    model_config: dict,
    layers: List[int],
    max_new_tokens: int
) -> Dict:
    """
    Extract activations during autoregressive generation.

    Args:
        prompt: Initial prompt string.
        model_config: Model configuration dict.
        layers: List of layer indices.
        max_new_tokens: Maximum tokens to generate.

    Returns:
        Dict containing generated_text, activations, token_ids.
    """
    raise NotImplementedError("TODO")
