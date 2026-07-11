"""Batch input/output handling for the future container entry point."""

import json
import sys
from pathlib import Path
from typing import Any

from classifier import classify_task
from solver import solve


PROJECT_ROOT = Path(__file__).parent


def default_input_path() -> Path:
    container_path = Path("/input/tasks.json")
    return container_path if container_path.exists() else PROJECT_ROOT / "input" / "tasks.json"


def default_output_path() -> Path:
    container_directory = Path("/output")
    return container_directory / "results.json" if container_directory.exists() else PROJECT_ROOT / "output" / "results.json"


def load_tasks(input_path: Path) -> list[dict[str, str]]:
    try:
        raw_tasks = input_path.read_text(encoding="utf-8-sig")
    except FileNotFoundError as exc:
        raise ValueError(f"Tasks file was not found: {input_path}") from exc

    if not raw_tasks.strip():
        raise ValueError(f"Tasks file is empty: {input_path}")

    try:
        data: Any = json.loads(raw_tasks)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Tasks file is not valid JSON at {input_path}: {exc}") from exc

    if not isinstance(data, list):
        raise ValueError("Tasks JSON must be a list of task objects.")

    tasks = []
    for index, task in enumerate(data):
        if not isinstance(task, dict) or not isinstance(task.get("task_id"), str) or not isinstance(task.get("prompt"), str):
            raise ValueError(f"Task at index {index} must contain string 'task_id' and 'prompt' fields.")
        tasks.append({"task_id": task["task_id"], "prompt": task["prompt"]})
    return tasks


def write_results(output_path: Path, results: list[dict[str, str]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(".tmp")
    temporary_path.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary_path.replace(output_path)


def process_tasks(input_path: Path, output_path: Path) -> list[dict[str, str]]:
    """Answer every task and save the required submission JSON shape."""
    results = []
    tasks = load_tasks(input_path)
    for index, task in enumerate(tasks, start=1):
        task_type = classify_task(task["prompt"])
        print(
            f"[{index}/{len(tasks)}] {task['task_id']} classified as {task_type}; routing...",
            file=sys.stderr,
            flush=True,
        )
        result = solve(task["prompt"], task_type)
        print(
            f"[{index}/{len(tasks)}] {task['task_id']} answered by {result.source}",
            file=sys.stderr,
            flush=True,
        )
        results.append({"task_id": task["task_id"], "answer": result.answer})
    write_results(output_path, results)
    return results

