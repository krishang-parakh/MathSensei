import os
import sys
import unittest
import types


TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(TESTS_DIR)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

if "xmltodict" not in sys.modules:
    xmltodict_stub = types.ModuleType("xmltodict")
    xmltodict_stub.parse = lambda *args, **kwargs: {}
    sys.modules["xmltodict"] = xmltodict_stub

if "wolframalpha" not in sys.modules:
    wolframalpha_stub = types.ModuleType("wolframalpha")

    class _Document:
        @staticmethod
        def make(path, key, value):
            return key, value

    wolframalpha_stub.Document = _Document
    sys.modules["wolframalpha"] = wolframalpha_stub

from core.wolfram_utils import _build_query_candidates, clean_wolfram_query, extract_wolfram_plaintext_answer


class TestWolframUtils(unittest.TestCase):
    def test_clean_wolfram_query_strips_internal_currency_symbols(self):
        self.assertEqual(clean_wolfram_query("($2 * 3)"), "(2 * 3)")

    def test_build_query_candidates_adds_unitless_arithmetic_fallback(self):
        candidates = _build_query_candidates("(252 eggs/day * 7 days) / 12 * 2")

        self.assertIn("(252 eggs/day * 7 days) / 12 * 2", candidates)
        self.assertIn("(252 * 7) / 12 * 2", candidates)

    def test_extract_wolfram_plaintext_answer_prefers_result_over_input(self):
        result = {
            "pod": [
                {
                    "@title": "Input interpretation",
                    "subpod": {"plaintext": "sin(120 deg)"},
                },
                {
                    "@title": "Result",
                    "@primary": True,
                    "subpod": {"plaintext": r"\frac{\sqrt{3}}{2}"},
                },
            ]
        }

        answer = extract_wolfram_plaintext_answer(result)

        self.assertEqual(answer, r"\frac{\sqrt{3}}{2}")


if __name__ == "__main__":
    unittest.main()
