import os
import sys
import unittest
from types import SimpleNamespace


TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(TESTS_DIR)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from run import clone_solver_for_problem, iter_problem_batches, run_problem_once


class _DummySolver:
    def __init__(self):
        self.cache = {}
        self.modules = []
        self.dependency_install_attempts = {"base-attempt"}
        self.dataset = "MATH"
        self.requested_dataset = "MATH"
        self.sg_model = "no"
        self.sg_engine = "test"

    def predict_modules(self):
        return ["solution_generator"]

    def solution_generator(self):
        prompt = "Solve the problem"
        solution = "Final Answer: 4"
        self.cache["solution_generator:input"] = prompt
        self.cache["solution_generator:output"] = solution
        self.cache["solution"] = solution
        self.cache["response"] = (self.cache.get("response") or "") + "\n" + solution
        return prompt, solution


class TestRunParallelHelpers(unittest.TestCase):
    def test_clone_solver_for_problem_resets_mutable_problem_state(self):
        base = _DummySolver()
        base.cache = {"old": 1}
        base.modules = ["solution_generator"]

        cloned = clone_solver_for_problem(base)

        self.assertIsNot(cloned, base)
        self.assertEqual(cloned.cache, {})
        self.assertEqual(cloned.modules, [])
        self.assertEqual(cloned.dependency_install_attempts, set())
        self.assertEqual(base.cache, {"old": 1})
        self.assertEqual(base.modules, ["solution_generator"])
        self.assertEqual(base.dependency_install_attempts, {"base-attempt"})

    def test_iter_problem_batches_chunks_ids(self):
        batches = list(iter_problem_batches(range(1, 8), 3))
        self.assertEqual(batches, [[1, 2, 3], [4, 5, 6], [7]])

    def test_iter_problem_batches_uses_single_batch_when_batch_size_not_positive(self):
        batches = list(iter_problem_batches([10, 11, 12], 0))
        self.assertEqual(batches, [[10, 11, 12]])

    def test_run_problem_once_keeps_base_solver_isolated(self):
        base = _DummySolver()
        args = SimpleNamespace(
            model="sg",
            refine="no",
            modules=None,
            dataset="MATH",
        )
        example = {
            "dataset": "MATH",
            "problem": "What is 2 + 2?",
            "solution": "4",
            "level": "Level 1",
            "type": "Arithmetic",
        }

        result = run_problem_once(base, args, pid=0, example=example, debug=False, emit_logs=False)

        self.assertEqual(result["pid"], 0)
        self.assertEqual(result["module_names"], ["solution_generator"])
        self.assertEqual(result["cache"]["modules"], ["solution_generator"])
        self.assertIn("question_elapsed_seconds", result["cache"])
        self.assertIn("status", result["normalized_record"])
        self.assertEqual(base.cache, {})
        self.assertEqual(base.modules, [])
        self.assertEqual(base.dependency_install_attempts, {"base-attempt"})


if __name__ == "__main__":
    unittest.main()
