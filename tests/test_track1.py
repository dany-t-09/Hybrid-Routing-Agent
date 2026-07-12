import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

# The production image installs ``requests`` from requirements.txt. Supply a
# harmless import stub so these routing and schema tests also run in a minimal
# Python environment that does not install application dependencies.
try:
    import requests  # noqa: F401
except ModuleNotFoundError:
    sys.modules["requests"] = types.ModuleType("requests")

from batch_runner import process_tasks
from classifier import classify_task
from local_model_client import LocalModelResponse
from result import AnswerResult
from solver import solve


class TrackOneTests(unittest.TestCase):
    def test_routes_all_eight_capabilities(self):
        examples = {
            "What is photosynthesis?": "factual",
            "A store gives a 20% discount. What is the final price?": "math",
            "Classify the sentiment of this review: I love it.": "sentiment",
            "Summarize this passage in one sentence.": "summary",
            "Extract named entities from Maria visited Berlin.": "ner",
            "Debug this function: def f(): return values[0]": "debugging",
            "Three friends each own a different pet. Who owns the cat?": "logic",
            "Write a Python function that returns the second-largest number.": "codegen",
        }
        for prompt, expected in examples.items():
            with self.subTest(prompt=prompt):
                self.assertEqual(classify_task(prompt), expected)

    def test_routes_program_specification_to_codegen(self):
        self.assertEqual(
            classify_task("Given an array, write an efficient Python program and explain the implementation."),
            "codegen",
        )

    def test_batch_writes_required_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "tasks.json"
            output_path = root / "results.json"
            input_path.write_text(json.dumps([{"task_id": "t1", "prompt": "What is a CPU?"}]), encoding="utf-8")
            with patch("batch_runner.solve", return_value=AnswerResult("A processor.", "Fireworks AI", 80)):
                process_tasks(input_path, output_path)
            self.assertEqual(json.loads(output_path.read_text(encoding="utf-8")), [{"task_id": "t1", "answer": "A processor."}])

    def test_batch_fails_instead_of_writing_an_api_error_as_an_answer(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "tasks.json"
            input_path.write_text(json.dumps([{"task_id": "t1", "prompt": "What is a CPU?"}]), encoding="utf-8")
            failed = AnswerResult("FIREWORKS_API_KEY is not set.", "Local model", None)
            with patch("batch_runner.solve", return_value=failed):
                with self.assertRaisesRegex(RuntimeError, "Could not answer task t1"):
                    process_tasks(input_path, root / "results.json")

    def test_math_word_problem_uses_llm_not_expression_parser(self):
        prompt = "A shop has 20 apples and sells 25% of them. How many remain?"
        with patch("solver.FIREWORKS_API_KEY", ""), patch(
            "solver.ask_local_model", return_value=LocalModelResponse("15 apples remain.", 90)
        ):
            result = solve(prompt, "math")
        self.assertEqual(result.answer, "15 apples remain.")
        self.assertEqual(result.source, "Local model")

    def test_low_local_confidence_escalates_to_fireworks(self):
        local = LocalModelResponse("Uncertain local answer", 59)
        remote = types.SimpleNamespace(
            answer="Verified Fireworks answer", prompt_tokens=3, completion_tokens=4, total_tokens=7
        )
        with patch("solver.FIREWORKS_API_KEY", "key"), patch("solver.ALLOWED_MODELS", ("allowed",)), patch(
            "solver.ask_local_model", return_value=local
        ), patch("solver.ask_fireworks", return_value=remote):
            result = solve("Explain a difficult topic", "factual")
        self.assertEqual(result.answer, "Verified Fireworks answer")
        self.assertEqual(result.source, "Fireworks AI")


if __name__ == "__main__":
    unittest.main()
