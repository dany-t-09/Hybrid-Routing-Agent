"""Local GGUF/Ollama inference with a model-reported confidence score."""

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import requests

from config import LOCAL_MODEL_CONTEXT, LOCAL_MODEL_NAME, LOCAL_MODEL_PATH, LOCAL_MODEL_THREADS, OLLAMA_API_URL


TASK_INSTRUCTIONS = {
    "factual": "Answer accurately and concisely. State uncertainty rather than guessing.",
    "logic": "Reason carefully, check the conclusion, then give a clear final answer.",
    "debugging": "Diagnose the likely cause, explain it, and provide a concrete fix.",
    "codegen": "Provide correct, complete code that directly addresses the request.",
    "summary": "Preserve the important facts and avoid adding information not in the text.",
}


@dataclass(frozen=True)
class LocalModelResponse:
    answer: str
    confidence: int | None


def _response_instruction(task_type: str) -> str:
    instruction = TASK_INSTRUCTIONS.get(task_type, TASK_INSTRUCTIONS["factual"])
    return (
        f"{instruction}\n\n"
        "Return ONLY a JSON object with exactly these fields: "
        '{"answer": "your complete answer", "confidence": 0}. '
        "confidence must be an integer from 0 to 100 measuring how confident you are "
        "that the answer is correct and fully satisfies the request."
    )


def _parse_model_response(content: str) -> LocalModelResponse:
    """Malformed or missing confidence is scored as zero, causing escalation."""
    cleaned = content.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    try:
        payload = json.loads(cleaned)
        answer = payload.get("answer")
        confidence = payload.get("confidence")
        if not isinstance(answer, str) or not answer.strip() or isinstance(confidence, bool):
            raise ValueError("invalid local response")
        confidence = int(confidence)
        if not 0 <= confidence <= 100:
            raise ValueError("confidence outside range")
        return LocalModelResponse(answer.strip(), confidence)
    except (json.JSONDecodeError, TypeError, ValueError):
        return LocalModelResponse(cleaned, 0)


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
    return Llama(model_path=str(model_path), n_ctx=LOCAL_MODEL_CONTEXT, n_threads=LOCAL_MODEL_THREADS, n_gpu_layers=0, verbose=False)


def _ask_bundled_model(query: str, task_type: str) -> LocalModelResponse:
    response = _load_local_model().create_chat_completion(
        messages=[
            {"role": "system", "content": _response_instruction(task_type)},
            {"role": "user", "content": query},
        ],
        temperature=0.1,
        max_tokens=1024,
        response_format={"type": "json_object"},
    )
    return _parse_model_response(response["choices"][0]["message"]["content"])


def _ask_ollama(query: str, task_type: str) -> LocalModelResponse:
    payload = {
        "model": LOCAL_MODEL_NAME,
        "prompt": query,
        "system": _response_instruction(task_type),
        "stream": False,
        "format": "json",
    }
    try:
        response = requests.post(OLLAMA_API_URL, json=payload, timeout=60)
        response.raise_for_status()
        return _parse_model_response(response.json().get("response", ""))
    except requests.RequestException as exc:
        return LocalModelResponse(f"Local model error: {exc}", None)


def ask_local_model(query: str, task_type: str = "factual") -> LocalModelResponse:
    """Answer locally and return a confidence number for routing decisions."""
    try:
        return _ask_bundled_model(query, task_type)
    except (FileNotFoundError, RuntimeError, KeyError, TypeError, ValueError) as bundled_error:
        if Path(LOCAL_MODEL_PATH).is_file():
            return LocalModelResponse(f"Local model error: {bundled_error}", None)
        ollama_answer = _ask_ollama(query, task_type)
        if ollama_answer.confidence is not None:
            return ollama_answer
        return LocalModelResponse(
            f"Local model error: {bundled_error}; Ollama fallback also failed: {ollama_answer.answer}", None
        )
