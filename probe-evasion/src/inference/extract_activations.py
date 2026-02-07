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
    max_length: int = 512,
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
        max_length: Maximum token length for truncation.

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
            hook_outputs[layer_idx] = hidden_states.detach().cpu().float()
        return hook_fn

    for layer_idx in layers:
        layer_module = _get_layer_module(model, layer_idx)
        hook = layer_module.register_forward_hook(make_hook(layer_idx))
        hooks.append(hook)

    try:
        for text in tqdm(texts, desc="Extracting activations"):
            hook_outputs.clear()

            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
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


def extract_activations(
    prompt: str,
    model_config: dict,
    layers: List[int],
    pooling: str = "last_token",
) -> Dict:
    """
    Extract hidden state activations from specified layers for a single prompt.

    Convenience wrapper that loads the model, extracts, and cleans up.
    For processing multiple texts, use load_model_and_tokenizer() +
    extract_activations_batch() instead to avoid reloading the model.

    Args:
        prompt: Prompt string to process.
        model_config: Model configuration dict.
        layers: List of layer indices to extract activations from.
        pooling: "last_token" or "mean".

    Returns:
        Dict mapping layer index to activation tensor:
        {
            layer_idx: tensor of shape (hidden_dim,),
            ...
        }
    """
    model, tokenizer = load_model_and_tokenizer(model_config)
    try:
        max_length = model_config.get("max_tokens", 512)
        result = extract_activations_batch(
            [prompt], model, tokenizer, layers, pooling, max_length=max_length
        )
        # Squeeze batch dim: (1, hidden_dim) -> (hidden_dim,)
        return {layer_idx: tensor[0] for layer_idx, tensor in result.items()}
    finally:
        del model, tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def extract_activations_generate(
    texts: List[str],
    model,
    tokenizer,
    layers: List[int],
    max_new_tokens: int = 64,
    max_length: int = 512,
) -> Dict[int, torch.Tensor]:
    """
    Extract activations at the last generated token for each text.

    For each text, the model generates a continuation and we capture
    the hidden states at the final generated token. This produces
    activations that match what probes see during evaluation (monitoring
    generation, not static forward passes).

    During generation with KV cache, each forward pass processes only
    the newest token. The hook overwrites on each step, so after
    generation completes we have the last generated token's activations.

    Args:
        texts: List of text strings (will be formatted as chat prompts).
        model: Pre-loaded model.
        tokenizer: Pre-loaded tokenizer.
        layers: List of layer indices to extract from.
        max_new_tokens: Number of tokens to generate per text.
        max_length: Maximum input token length for truncation.

    Returns:
        Dict mapping layer_idx -> Tensor of shape (n_texts, hidden_dim).
    """
    collected = {layer_idx: [] for layer_idx in layers}

    hooks = []
    hook_outputs = {}

    def make_hook(layer_idx):
        def hook_fn(module, input, output):
            hidden_states = output[0] if isinstance(output, tuple) else output
            # During generation with KV cache, hidden_states is for the
            # current token only: shape (1, 1, hidden_dim) or (1, hidden_dim).
            # Overwrite each step so we end up with the last token's activation.
            hook_outputs[layer_idx] = hidden_states.detach().cpu().float()
        return hook_fn

    for layer_idx in layers:
        layer_module = _get_layer_module(model, layer_idx)
        hook = layer_module.register_forward_hook(make_hook(layer_idx))
        hooks.append(hook)

    try:
        for text in tqdm(texts, desc="Generating and extracting"):
            hook_outputs.clear()

            # Format as chat message for natural generation
            messages = [{"role": "user", "content": text}]
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            with torch.no_grad():
                model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )

            # hook_outputs now holds activations from the last generated token
            for layer_idx in layers:
                hidden_states = hook_outputs[layer_idx]
                # Collapse to 1D: (1, 1, hidden_dim) or (1, hidden_dim) -> (hidden_dim,)
                while hidden_states.dim() > 1:
                    hidden_states = hidden_states[0]
                collected[layer_idx].append(hidden_states)

    finally:
        for hook in hooks:
            hook.remove()

    result = {}
    for layer_idx in layers:
        result[layer_idx] = torch.stack(collected[layer_idx])

    return result
