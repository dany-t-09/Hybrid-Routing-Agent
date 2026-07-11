from dataclasses import dataclass


@dataclass(frozen=True)
class AnswerResult:
    """The answer and metadata collected from the selected backend."""

    answer: str
    source: str
    estimated_accuracy: int | None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None


@dataclass(frozen=True)
class FireworksResponse:
    answer: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None

