from fireworks_client import ask_fireworks
from local.math_solver import solve_math
from local.ner_solver import solve_ner
from local.sentiment_solver import solve_sentiment
from local.summary_solver import solve_summary


LOCAL_SOLVERS = {
    "math": solve_math,
    "ner": solve_ner,
    "sentiment": solve_sentiment,
    "summary": solve_summary,
}


COMPLEX_CUES = {
    "analyze",
    "compare",
    "debug",
    "design",
    "explain",
    "generate",
    "implement",
    "reason",
    "rewrite",
    "write",
}


def should_use_fireworks(query: str, task_type: str) -> bool:
    if task_type in {"factual", "logic", "debugging", "codegen"}:
        return True

    lowered = query.lower()
    if len(query.split()) > 40:
        return True

    return any(cue in lowered for cue in COMPLEX_CUES)


def solve(query: str, task_type: str) -> str:
    if should_use_fireworks(query, task_type):
        return ask_fireworks(query, task_type)

    local_solver = LOCAL_SOLVERS.get(task_type)
    if local_solver:
        return local_solver(query)

    return ask_fireworks(query, task_type)
