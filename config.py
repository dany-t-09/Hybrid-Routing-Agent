import os
from pathlib import Path


def _load_dotenv() -> None:
    env_path = Path(__file__).parent / ".env"
    if not env_path.exists():
        return

    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv()
# A GGUF model may be bundled at this path for offline inference.  Do not add a
# model ID to this project: only a local file is used by llama.cpp.
LOCAL_MODEL_PATH = os.getenv("LOCAL_MODEL_PATH", "/app/models/model.gguf")
LOCAL_MODEL_CONTEXT = int(os.getenv("LOCAL_MODEL_CONTEXT", "2048"))
LOCAL_MODEL_THREADS = int(os.getenv("LOCAL_MODEL_THREADS", "2"))
# Ollama remains useful during development, but is not required by the image.
LOCAL_MODEL_NAME = os.getenv("LOCAL_MODEL_NAME", "qwen2.5:3b-instruct-q4_K_M")
OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", "http://localhost:11434/api/generate")

# Never supply a fallback credential or model here. The hackathon harness
# injects these settings at runtime, and calls must use its proxy URL.
FIREWORKS_API_KEY = os.getenv("FIREWORKS_API_KEY", "").strip()
FIREWORKS_BASE_URL = os.getenv("FIREWORKS_BASE_URL", "").rstrip("/")
FIREWORKS_MODEL = os.getenv("FIREWORKS_MODEL", "").strip()
ALLOWED_MODELS = tuple(
    model.strip()
    for model in os.getenv("ALLOWED_MODELS", FIREWORKS_MODEL).split(",")
    if model.strip()
)


def get_fireworks_model(task_type: str) -> str | None:
    """Return only a model permitted by the evaluation harness."""
    if not ALLOWED_MODELS:
        return None

    requested_model = os.getenv(f"FIREWORKS_{task_type.upper()}_MODEL", "")
    return requested_model if requested_model in ALLOWED_MODELS else ALLOWED_MODELS[0]


FIREWORKS_API_URL = "" if not FIREWORKS_BASE_URL else (
    FIREWORKS_BASE_URL
    if FIREWORKS_BASE_URL.endswith("/chat/completions")
    else f"{FIREWORKS_BASE_URL}/chat/completions"
)

CACHE_ENABLED = os.getenv("CACHE_ENABLED", "true").lower() == "true"
