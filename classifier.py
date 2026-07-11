import re


MATH_EXPRESSION = re.compile(r"\d+(?:\.\d+)?\s*(?:[+*/^]|-(?=\s*\d))\s*\d")


def classify_task(text: str) -> str:
    lowered = text.lower()

    if lowered.startswith(("calculate ", "evaluate ")) or MATH_EXPRESSION.search(text):
        return "math"
    if any(token in lowered for token in ["sentiment", "positive", "negative", "neutral"]):
        return "sentiment"
    if any(token in lowered for token in ["summarize", "summary", "tl;dr"]):
        return "summary"
    if any(token in lowered for token in ["name", "person", "place", "organization", "entity"]):
        return "ner"
    if any(token in lowered for token in ["bug", "debug", "error", "traceback"]):
        return "debugging"
    if any(token in lowered for token in ["code", "function", "class", "script"]):
        return "codegen"
    if any(token in lowered for token in ["logic", "reason", "puzzle"]):
        return "logic"

    return "factual"
