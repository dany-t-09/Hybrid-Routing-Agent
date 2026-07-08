import requests

from config import LOCAL_MODEL_NAME, OLLAMA_API_URL


def ask_local_model(query: str, task_type: str = "factual") -> str:
    payload = {
        "model": LOCAL_MODEL_NAME,
        "prompt": query,
        "stream": False,
    }

    try:
        response = requests.post(OLLAMA_API_URL, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()
        return data.get("response", "").strip()
    except requests.RequestException as exc:
        return f"Local model error: {exc}"