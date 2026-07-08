from pathlib import Path

import requests

from config import FIREWORKS_API_KEY, FIREWORKS_API_URL, FIREWORKS_DEFAULT_MODEL, FIREWORKS_MODELS


PROMPTS_DIR = Path(__file__).parent / "prompts"


def load_prompt(task_type: str) -> str:
    prompt_path = PROMPTS_DIR / f"{task_type}.txt"
    if prompt_path.exists():
        return prompt_path.read_text(encoding="utf-8").strip()
    return "Answer the user's request clearly and accurately."


def ask_fireworks(query: str, task_type: str = "factual") -> str:
    if not FIREWORKS_API_KEY:
        return "FIREWORKS_API_KEY is not set. Add it to .env or your environment and try again."

    model = FIREWORKS_MODELS.get(task_type, FIREWORKS_DEFAULT_MODEL)
    payload = {
        "model": "accounts/fireworks/models/glm-5p2",
        "messages": [
            {"role": "system", "content": load_prompt(task_type)},
            {"role": "user", "content": query},
        ],
    }
    headers = {
        "Authorization": f"Bearer {FIREWORKS_API_KEY}",
        "Content-Type": "application/json",
    }

    try:
        response = requests.post(FIREWORKS_API_URL, json=payload, headers=headers, timeout=30)
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]
    except requests.HTTPError as exc:
        detail = response.text[:1000] if response.text else str(exc)
        return f"Fireworks API error ({response.status_code}): {detail}"
    except requests.RequestException as exc:
        return f"Could not reach Fireworks API: {exc}"
    except (KeyError, IndexError, TypeError) as exc:
        return f"Unexpected Fireworks response format: {exc}"
