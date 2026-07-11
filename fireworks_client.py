from pathlib import Path

import requests

from config import FIREWORKS_API_KEY, FIREWORKS_API_URL, get_fireworks_model
from result import FireworksResponse


PROMPTS_DIR = Path(__file__).parent / "prompts"


def load_prompt(task_type: str) -> str:
    prompt_path = PROMPTS_DIR / f"{task_type}.txt"
    if prompt_path.exists():
        return prompt_path.read_text(encoding="utf-8").strip()
    return "Answer the user's request clearly and accurately."


def ask_fireworks(query: str, task_type: str = "factual") -> FireworksResponse:
    if not FIREWORKS_API_KEY:
        return FireworksResponse("FIREWORKS_API_KEY is not set. Add it to .env or your environment and try again.")

    model = get_fireworks_model(task_type)
    if model is None:
        return FireworksResponse("ALLOWED_MODELS is not set. Add one or more permitted Fireworks model IDs.")
    payload = {
        "model": model,
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
        usage = data.get("usage") or {}
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")
        total_tokens = usage.get("total_tokens")
        if total_tokens is None and prompt_tokens is not None and completion_tokens is not None:
            total_tokens = prompt_tokens + completion_tokens
        return FireworksResponse(
            answer=data["choices"][0]["message"]["content"],
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        )
    except requests.HTTPError as exc:
        detail = response.text[:1000] if response.text else str(exc)
        return FireworksResponse(f"Fireworks API error ({response.status_code}): {detail}")
    except requests.RequestException as exc:
        return FireworksResponse(f"Could not reach Fireworks API: {exc}")
    except (KeyError, IndexError, TypeError) as exc:
        return FireworksResponse(f"Unexpected Fireworks response format: {exc}")
