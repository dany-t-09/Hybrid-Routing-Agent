import sys

from config import ALLOWED_MODELS, FIREWORKS_API_KEY
from fireworks_client import ask_fireworks
from local.math_solver import can_solve_math, solve_math
from local_model_client import ask_local_model
from result import AnswerResult


FIREWORKS_ACCURACY = {"debugging": 85, "codegen": 85, "logic": 82, "factual": 80}


def _is_error(answer: str) -> bool:
    return answer.startswith((
        "Fireworks API error (", "Could not reach Fireworks API:", "Unexpected Fireworks response format:",
        "FIREWORKS_API_KEY is not set.", "ALLOWED_MODELS is not set.", "Local model error:",
    ))


def should_use_fireworks(query: str, task_type: str) -> bool:
    """Only the harness-provided credentials authorize a Fireworks call."""
    return bool(FIREWORKS_API_KEY and ALLOWED_MODELS)


def _ask_fireworks(query: str, task_type: str) -> AnswerResult | None:
    response = ask_fireworks(query, task_type)
    if _is_error(response.answer):
        print(f"Fireworks failed for {task_type}: {response.answer}", file=sys.stderr, flush=True)
        return None
    return AnswerResult(
        answer=response.answer,
        source="Fireworks AI",
        estimated_accuracy=FIREWORKS_ACCURACY.get(task_type, 80),
        prompt_tokens=response.prompt_tokens,
        completion_tokens=response.completion_tokens,
        total_tokens=response.total_tokens,
    )


def solve(query: str, task_type: str) -> AnswerResult:
    # Deterministic arithmetic is exact and does not need either model.
    if task_type == "math" and can_solve_math(query):
        return AnswerResult(solve_math(query), "Local expression solver", 99)

    # Every LLM local answer is accompanied by a 0-100 confidence value.
    local = ask_local_model(query, task_type)
    if local.confidence is not None and local.answer.strip():
        if local.confidence >= 60 or not should_use_fireworks(query, task_type):
            return AnswerResult(local.answer, "Local model", local.confidence)
        print(
            f"Local confidence for {task_type} was {local.confidence}; escalating to Fireworks.",
            file=sys.stderr,
            flush=True,
        )

    if should_use_fireworks(query, task_type):
        remote = _ask_fireworks(query, task_type)
        if remote is not None:
            return remote

    return AnswerResult(local.answer, "Local model", local.confidence)
