import re


MATH_EXPRESSION = re.compile(r"\d+(?:\.\d+)?\s*(?:[+*/^]|-(?=\s*\d))\s*\d")


def _has_any(text: str, patterns: tuple[str, ...]) -> bool:
    return any(re.search(pattern, text) for pattern in patterns)


def classify_task(text: str) -> str:
    """Route by explicit task intent without matching arbitrary substrings.

    Classification only selects a task-specific instruction; every non-exact
    task is still answered by the permitted Fireworks model in submission mode.
    """
    lowered = text.lower()

    if _has_any(lowered, (r"\b(summarise|summarize|summary|tl;dr)\b",)):
        return "summary"
    if _has_any(lowered, (r"\b(sentiment|positive|negative|neutral)\b",)) and _has_any(
        lowered, (r"\b(classify|label|review|sentiment)\b",)
    ):
        return "sentiment"
    if _has_any(lowered, (r"\b(named? entities?|ner|extract entities?)\b",)):
        return "ner"
    if _has_any(lowered, (r"\b(debug|bug|traceback|exception|stack trace|fix (?:this|the) (?:code|function))\b",)):
        return "debugging"
    if _has_any(
        lowered,
        (r"\b(write|implement|create|generate|design)\b.*\b(function|class|script|code|program|solution|implementation)\b",),
    ):
        return "codegen"
    if _has_any(lowered, (r"\b(puzzle|deduc|constraint|who owns|which .* (?:is|are)|must be true)\b",)):
        return "logic"
    if lowered.startswith(("calculate ", "evaluate ", "solve ")) or MATH_EXPRESSION.search(text):
        return "math"
    if _has_any(lowered, (r"\b(percent|percentage|ratio|interest|discount|tax|remain|distance|area|total)\b",)):
        return "math"
    return "factual"
