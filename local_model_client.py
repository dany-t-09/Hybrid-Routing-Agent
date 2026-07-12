from functools import lru_cache
from pathlib import Path

import requests

from config import (
    LOCAL_MODEL_CONTEXT,
    LOCAL_MODEL_NAME,
    LOCAL_MODEL_PATH,
    LOCAL_MODEL_THREADS,
    OLLAMA_API_URL,
)


TASK_INSTRUCTIONS = {
    "factual": "Answer accurately and concisely. State uncertainty rather than guessing.",
    "logic": "Reason carefully, check the conclusion, then give a clear final answer.",
    "debugging": "Diagnose the likely cause, explain it, and provide a concrete fix.",
    "codegen": "Provide correct, complete code that directly addresses the request.",
    "summary": "Preserve the important facts and avoid adding information not in the text.",
}


@lru_cache(maxsize=1)
def _load_local_model():
    """Load a bundled GGUF model only on first use, keeping startup fast."""
    model_path = Path(LOCAL_MODEL_PATH)
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Bundled local model not found at {model_path}. "
            "Add a 2B-3B Q4 GGUF file or configure Fireworks environment variables."
        )
    try:
        from llama_cpp import Llama
    except ImportError as exc:
        raise RuntimeError("llama-cpp-python is not installed in this image.") from exc
    return Llama(
        model_path=str(model_path),
        n_ctx=LOCAL_MODEL_CONTEXT,
        n_threads=LOCAL_MODEL_THREADS,
        n_gpu_layers=0,
        verbose=False,
    )


def _ask_bundled_model(query: str, task_type: str) -> str:
    model = _load_local_model()
    system = TASK_INSTRUCTIONS.get(task_type, TASK_INSTRUCTIONS["factual"])
    response = model.create_chat_completion(
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": query},
        ],
        temperature=0.1,
        max_tokens=512,
    )
    return response["choices"][0]["message"]["content"].strip()


def _ask_ollama(query: str, task_type: str) -> str:
    payload = {
        "model": LOCAL_MODEL_NAME,
        "prompt": query,
        "system": TASK_INSTRUCTIONS.get(task_type, TASK_INSTRUCTIONS["factual"]),
        "stream": False,
    }

    try:
        response = requests.post(OLLAMA_API_URL, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()
        return data.get("response", "").strip()
    except requests.RequestException as exc:
        return f"Local model error: {exc}"


def ask_local_model(query: str, task_type: str = "factual") -> str:
    """Use an in-container GGUF model; Ollama is an explicit dev-only fallback."""
    try:
        return _ask_bundled_model(query, task_type)
    except (FileNotFoundError, RuntimeError, KeyError, TypeError, ValueError) as bundled_error:
        # A developer may intentionally run an Ollama sidecar outside Docker.
        if Path(LOCAL_MODEL_PATH).is_file():
            return f"Local model error: {bundled_error}"
        ollama_answer = _ask_ollama(query, task_type)
        if not ollama_answer.startswith("Local model error:"):
            return ollama_answer
        return f"Local model error: {bundled_error}; Ollama fallback also failed: {ollama_answer}"
