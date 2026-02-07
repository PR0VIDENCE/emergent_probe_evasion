"""Model inference functions."""

from typing import Dict


def run_inference(prompt: str, model_config: dict) -> Dict:
    """
    Run inference on a model and capture outputs.

    Handles both API-based and local model inference depending on
    the model configuration.

    Args:
        prompt: Full prompt string to send to the model.
        model_config: Dict containing:
                     - model_name: Identifier for the model
                     - api_endpoint: URL if using API (or None for local)
                     - local_path: Path to model weights (or None for API)
                     - max_tokens: Maximum tokens to generate
                     - temperature: Sampling temperature

    Returns:
        Dict containing:
        - response_text: The model's generated text response
        - reasoning_trace: Chain-of-thought if available (None otherwise)
        - token_count: Number of tokens generated
        - raw_output: Full unprocessed output from model/API
    """
    raise NotImplementedError("TODO")


def run_batch_inference(prompts: list, model_config: dict) -> list:
    """
    Run inference on multiple prompts.

    Args:
        prompts: List of prompt strings.
        model_config: Model configuration dict.

    Returns:
        List of response dicts (same format as run_inference).
    """
    raise NotImplementedError("TODO")
