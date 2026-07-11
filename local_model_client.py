import requests

from config import LOCAL_MODEL_NAME, OLLAMA_API_URL


TASK_INSTRUCTIONS = {
    "factual": "Answer accurately and concisely. State uncertainty rather than guessing.",
    "logic": "Reason carefully, check the conclusion, then give a clear final answer.",
    "debugging": "Diagnose the likely cause, explain it, and provide a concrete fix.",
    "codegen": "Provide correct, complete code that directly addresses the request.",
    "summary": "Preserve the important facts and avoid adding information not in the text.",
}


def ask_local_model(query: str, task_type: str = "factual") -> str:
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
