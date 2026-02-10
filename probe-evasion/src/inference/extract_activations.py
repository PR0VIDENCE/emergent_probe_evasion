"""Activation extraction from model forward passes."""

import re

import torch
from typing import Dict, List, Optional, Tuple
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


def find_token_positions(
    output_ids: torch.Tensor,
    input_len: int,
    tokenizer,
) -> Dict[str, Optional[int]]:
    """
    Find key token positions in generated output.

    Args:
        output_ids: Full sequence tensor (1, seq_len) including prompt.
        input_len: Number of prompt tokens.
        tokenizer: Tokenizer for decoding.

    Returns:
        Dict mapping position name to absolute index in output_ids, or None if not found.
        - last_token: last generated token
        - end_of_reasoning: token just before </think> boundary
        - first_answer_sentence_end: first sentence-ending punctuation after answer starts
        - answer_start: first answer token (for mean pooling range)
        - answer_end: last answer token (for mean pooling range)
    """
    seq_len = output_ids.shape[1]
    generated_ids = output_ids[0, input_len:]  # just the generated portion

    positions = {
        "last_token": seq_len - 1,
        "end_of_reasoning": None,
        "first_answer_sentence_end": None,
        "answer_start": None,
        "answer_end": seq_len - 1,
    }

    # Search for </think> boundary in the generated tokens
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=False)
    think_end_match = re.search(r"</think>", generated_text)

    if think_end_match:
        text_before_think_end = generated_text[:think_end_match.start()]
        tokens_before = tokenizer.encode(text_before_think_end, add_special_tokens=False)
        think_end_offset = len(tokens_before)

        think_tag_text = "</think>"
        think_tag_tokens = tokenizer.encode(think_tag_text, add_special_tokens=False)
        think_tag_len = len(think_tag_tokens)

        end_of_reasoning_offset = think_end_offset + think_tag_len - 1
        if end_of_reasoning_offset < len(generated_ids):
            positions["end_of_reasoning"] = input_len + end_of_reasoning_offset

        answer_start_offset = think_end_offset + think_tag_len
        text_after_think = generated_text[think_end_match.end():]
        stripped = text_after_think.lstrip()
        whitespace_chars = len(text_after_think) - len(stripped)
        if whitespace_chars > 0:
            ws_tokens = tokenizer.encode(text_after_think[:whitespace_chars], add_special_tokens=False)
            answer_start_offset += len(ws_tokens)

        if answer_start_offset < len(generated_ids):
            positions["answer_start"] = input_len + answer_start_offset

            answer_text = generated_text[think_end_match.end():]
            sentence_end_match = re.search(r'[.!?]', answer_text)
            if sentence_end_match:
                text_to_sentence_end = answer_text[:sentence_end_match.end()]
                sentence_tokens = tokenizer.encode(text_to_sentence_end, add_special_tokens=False)
                sentence_end_offset = answer_start_offset + len(sentence_tokens) - 1
                if sentence_end_offset < len(generated_ids):
                    positions["first_answer_sentence_end"] = input_len + sentence_end_offset

    # Fallback: any None positions get last_token
    for key in positions:
        if positions[key] is None:
            positions[key] = positions["last_token"]

    return positions


def extract_activations_at_positions(
    output_ids: torch.Tensor,
    model,
    target_layers: List[int],
    positions: Dict[str, int],
    answer_start: Optional[int] = None,
    answer_end: Optional[int] = None,
) -> Dict[str, Dict[int, torch.Tensor]]:
    """
    Run a single forward pass and extract activations at multiple positions.

    Args:
        output_ids: Full sequence tensor (1, seq_len).
        model: Loaded model.
        target_layers: Layer indices to extract from.
        positions: Dict mapping position name -> absolute token index.
        answer_start: Start of answer range for mean pooling (absolute index).
        answer_end: End of answer range for mean pooling (absolute index).

    Returns:
        Dict mapping position_name -> {layer_idx: tensor(hidden_dim,)} in float16.
    """
    hooks = []
    hook_outputs = {}

    def make_hook(layer_idx):
        def hook_fn(module, input, output):
            hidden_states = output[0] if isinstance(output, tuple) else output
            hook_outputs[layer_idx] = hidden_states.detach().cpu().float()
        return hook_fn

    for layer_idx in target_layers:
        layer_module = _get_layer_module(model, layer_idx)
        hook = layer_module.register_forward_hook(make_hook(layer_idx))
        hooks.append(hook)

    try:
        with torch.no_grad():
            model(output_ids)

        result = {}
        for pos_name, pos_idx in positions.items():
            if pos_name in ("answer_start", "answer_end"):
                continue  # these are range markers, not extraction targets

            layer_activations = {}
            for layer_idx in target_layers:
                hidden_states = hook_outputs[layer_idx]  # (1, seq_len, hidden_dim)
                activation = hidden_states[0, pos_idx, :].half()  # float16
                layer_activations[layer_idx] = activation
            result[pos_name] = layer_activations

        # Mean pool over answer tokens
        if answer_start is not None and answer_end is not None and answer_start <= answer_end:
            layer_activations = {}
            for layer_idx in target_layers:
                hidden_states = hook_outputs[layer_idx]  # (1, seq_len, hidden_dim)
                answer_hidden = hidden_states[0, answer_start:answer_end + 1, :]
                pooled = answer_hidden.mean(dim=0).half()  # float16
                layer_activations[layer_idx] = pooled
            result["answer_mean_pool"] = layer_activations

    finally:
        for hook in hooks:
            hook.remove()

    return result


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
