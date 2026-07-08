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
#  LOCAL MODEL 
LOCAL_MODEL_NAME = os.getenv("LOCAL_MODEL_NAME", "qwen2.5:3b-instruct-q4_K_M")
OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", "http://localhost:11434/api/generate")

FIREWORKS_API_KEY = os.getenv("FIREWORKS_API_KEY", "")
FIREWORKS_DEFAULT_MODEL = os.getenv(
    "FIREWORKS_DEFAULT_MODEL",
    os.getenv("FIREWORKS_MODEL", "accounts/fireworks/models/llama-v3p1-8b-instruct"),
)
FIREWORKS_MODELS = {
    "factual": os.getenv("FIREWORKS_FACTUAL_MODEL", FIREWORKS_DEFAULT_MODEL),
    "logic": os.getenv("FIREWORKS_LOGIC_MODEL", FIREWORKS_DEFAULT_MODEL),
    "debugging": os.getenv("FIREWORKS_DEBUGGING_MODEL", FIREWORKS_DEFAULT_MODEL),
    "codegen": os.getenv("FIREWORKS_CODEGEN_MODEL", FIREWORKS_DEFAULT_MODEL),
    "summary": os.getenv("FIREWORKS_SUMMARY_MODEL", FIREWORKS_DEFAULT_MODEL),
    "sentiment": os.getenv("FIREWORKS_SENTIMENT_MODEL", FIREWORKS_DEFAULT_MODEL),
    "ner": os.getenv("FIREWORKS_NER_MODEL", FIREWORKS_DEFAULT_MODEL),
    "math": os.getenv("FIREWORKS_MATH_MODEL", FIREWORKS_DEFAULT_MODEL),
}
FIREWORKS_API_URL = os.getenv(
    "FIREWORKS_API_URL",
    "https://api.fireworks.ai/inference/v1/chat/completions",
)

CACHE_ENABLED = os.getenv("CACHE_ENABLED", "true").lower() == "true"
