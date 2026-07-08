import json
from pathlib import Path


CACHE_PATH = Path(__file__).parent / "output" / "cache.json"


def get_cached(key: str) -> str | None:
    if not CACHE_PATH.exists():
        return None

    data = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    return data.get(key)


def set_cached(key: str, value: str) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    data = {}
    if CACHE_PATH.exists():
        data = json.loads(CACHE_PATH.read_text(encoding="utf-8"))

    data[key] = value
    CACHE_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")
