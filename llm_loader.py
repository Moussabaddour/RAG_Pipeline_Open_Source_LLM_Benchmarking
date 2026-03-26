from langchain_community.llms import LlamaCpp


def load_llm(model_name="mistral"):

    if model_name == "mistral":
        model_path = "models/mistral-7b-instruct-v0.3-q4_k_m.gguf"
    elif model_name == "tinyllama":
        model_path = "models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
    elif model_name == "gemma":
        model_path = "models/gemma-2b-it.gguf"
    elif model_name == "qwen":
        model_path = "models/qwen2.5-1.5b-instruct-q4_k_m.gguf"
    elif model_name == "phi3":
        model_path = "models/Phi-3-mini-4k-instruct-q4.gguf"
    else:
        raise ValueError(f"Unknown model: {model_name}")

    llm = LlamaCpp(
        model_path=model_path,
        temperature=0.2,
        max_tokens=512,
        n_ctx=4096,
        n_threads=8,
        n_batch=512,
        n_gpu_layers=0,   # CPU-only — prevents GPU context conflicts on reload
        verbose=False
    )

    return llm