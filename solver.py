import sys
from fireworks_client import ask_fireworks
from config import ALLOWED_MODELS, FIREWORKS_API_KEY
from local_model_client import ask_local_model
from local.math_solver import can_solve_math, solve_math
from result import AnswerResult


LOCAL_SOLVERS = {
    "math": solve_math,
}


LOCAL_ACCURACY = {
    "math": 99,
}

FIREWORKS_ACCURACY = {
    "debugging": 85,
    "codegen": 85,
    "logic": 82,
    "factual": 80,
}


def _is_error(answer: str) -> bool:
    return answer.startswith((
        "Fireworks API error (",
        "Could not reach Fireworks API:",
        "Unexpected Fireworks response format:",
        "FIREWORKS_API_KEY is not set.",
        "ALLOWED_MODELS is not set.",
        "Local model error:",
        "Could not solve math expression:",
    ))


def should_use_fireworks(query: str, task_type: str) -> bool:
    if not FIREWORKS_API_KEY or not ALLOWED_MODELS:
        return False

    # Exact arithmetic is faster and deterministic. All word problems and
    # language tasks use the permitted remote model for accuracy.
    return task_type != "math" or not can_solve_math(query)


def solve(query: str, task_type: str) -> AnswerResult:
    if should_use_fireworks(query, task_type):
        response = ask_fireworks(query, task_type)
        if not _is_error(response.answer):
            return AnswerResult(
                answer=response.answer,
                source="Fireworks AI",
                estimated_accuracy=FIREWORKS_ACCURACY.get(task_type, 80),
                prompt_tokens=response.prompt_tokens,
                completion_tokens=response.completion_tokens,
                total_tokens=response.total_tokens,
            )
        print(
            f"Fireworks failed for {task_type}; falling back to local. Reason: {response.answer}",
            file=sys.stderr,
            flush=True,
        )

    # The deterministic solver is deliberately limited to safe, bare
    # arithmetic expressions.  Word problems need an LLM, even though they
    # share the "math" route.
    local_solver = LOCAL_SOLVERS.get(task_type)
    if local_solver and can_solve_math(query):
        answer = local_solver(query)
        return AnswerResult(
            answer=answer,
            source="Local model",
            estimated_accuracy=None if _is_error(answer) else LOCAL_ACCURACY[task_type],
        )

    answer = ask_local_model(query, task_type)
    return AnswerResult(
        answer=answer,
        source="Local model",
        estimated_accuracy=None if _is_error(answer) else 65,
    )
