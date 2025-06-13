# models.py (cleaned and shareable)

# Import model manager and system prompt
from ModelManager import ModelManager
from config import SYSTEM_PROMPT, DEFAULT_MAX_TOKENS, DEFAULT_TOP_K, DEFAULT_TOP_P, DEFAULT_TEMPERATURE

# Initialize global model manager instance
model_manager = ModelManager()

# Unified model runner function
def run_model(
    prompt,
    model_name=None,
    max_new_tokens=DEFAULT_MAX_TOKENS,
    temperature=DEFAULT_TEMPERATURE,
    top_p=DEFAULT_TOP_P,
    top_k=DEFAULT_TOP_K,
    timeout_seconds=None,
    system_prompt=None   # <-- NEW ARG
):

    """
    Unified entry point to run either a local HuggingFace model or an OpenAI API model.

    Automatically detects model type:
    - If local model: uses HuggingFace generate() with truncation if needed
    - If API model: uses OpenAI Chat API (server handles truncation)

    Supports:
    - Dynamic model switching
    - Optional input truncation for local models
    - Optional timeout for local model generation (non-blocking safe execution)
    """

    import threading  # used only if timeout is specified

    # Optional: Switch to requested model
    if model_name:
        model_manager.switch_model(model_name)

    # Get context limit (max input tokens for current model)
    context_limit = model_manager.get_context_limit()
    tokenizer = model_manager.get_tokenizer()

    # If using a local model, check if prompt exceeds max input tokens
    if tokenizer is not None:
        token_count = len(tokenizer.encode(prompt))
        if token_count > context_limit:
            print(f"[WARN] Prompt too long ({token_count} > {context_limit}). Truncating input...")
            prompt = tokenizer.decode(tokenizer.encode(prompt)[-context_limit:])
    else:
        # API models handle input limits server-side
        # print("[DEBUG] No tokenizer for API model. Assuming server handles input limits.")
        pass

    # Define callable to run model.generate() safely
    def generate_call():
        # Define callable to run model.generate() safely
# Pass system_prompt through so OpenAI models can use custom system prompt if provided.

        return model_manager.generate(
            prompt,
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            system_prompt=system_prompt,  # ← Add this line
            no_repeat_ngram_size=3 if not model_manager.is_api_model(model_manager.get_active_model_name()) else None
        )

    # If timeout is requested and model is local → run in separate thread
    if timeout_seconds and not model_manager.is_api_model(model_manager.get_active_model_name()):
        result = {"output": "...running..."}

        def run():
            try:
                result["output"] = generate_call()
            except Exception as e:
                result["output"] = f"[Error: {str(e)}]"

        # Launch generation in background thread
        thread = threading.Thread(target=run)
        thread.start()
        thread.join(timeout_seconds)

        # If timeout reached, mark result accordingly
        if thread.is_alive():
            result["output"] = "[Generation timed out]"
            thread.join(0.1)

        return result["output"]

    # Fallback: normal synchronous call (API or local)
    try:
        return generate_call()
    except Exception as e:
        return f"[Error during generation: {str(e)}]"
