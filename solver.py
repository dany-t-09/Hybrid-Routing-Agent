from fireworks_client import ask_fireworks
from local_model_client import ask_local_model
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
    "build a full",
    "large project",
    "production",
    "architecture",
    "multi-file",
    "research",
    "deep analysis",
    "advanced reasoning",
    "optimize",
    "complex",
}


def should_use_fireworks(query: str, task_type: str) -> bool:
    lowered = query.lower()

    if task_type in {"debugging", "codegen", "logic"} and len(query.split()) > 80:
        return True

    if len(query.split()) > 150:
        return True

    return any(cue in lowered for cue in COMPLEX_CUES)


def solve(query: str, task_type: str) -> str:
    if should_use_fireworks(query, task_type):
        return ask_fireworks(query, task_type)

    local_solver = LOCAL_SOLVERS.get(task_type)
    if local_solver:
        return local_solver(query)

    return ask_local_model(query, task_type)