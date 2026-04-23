import os
import sys
import unittest
import types
from unittest import mock

import requests


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

from core.wolfram_utils import _build_query_candidates, clean_wolfram_query, extract_wolfram_plaintext_answer, query_wolfram_alpha


class TestWolframUtils(unittest.TestCase):
    def test_clean_wolfram_query_strips_internal_currency_symbols(self):
        self.assertEqual(clean_wolfram_query("($2 * 3)"), "(2 * 3)")

    def test_build_query_candidates_adds_unitless_arithmetic_fallback(self):
        candidates = _build_query_candidates("(252 eggs/day * 7 days) / 12 * 2")

        self.assertIn("(252 eggs/day * 7 days) / 12 * 2", candidates)
        self.assertIn("(252 * 7) / 12 * 2", candidates)

    def test_clean_wolfram_query_normalizes_logic_symbols(self):
        cleaned = clean_wolfram_query("truth table (G \u2261 H) \u2022 ~I, ~G \u2228 (~H \u2228 I)")

        self.assertIn("<=>", cleaned)
        self.assertIn("and", cleaned)
        self.assertIn("or", cleaned)

    def test_build_query_candidates_adds_truth_table_grouped_variant(self):
        candidates = _build_query_candidates("truth table for (G <=> H) and ~I, ~G or (~H or I), G")

        self.assertIn("truth table for (G <=> H) and ~I, ~G or (~H or I), G", candidates)
        self.assertIn("truth table {(G <=> H) and ~I, ~G or (~H or I), G}", candidates)

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

    def test_query_wolfram_alpha_reports_blocked_v2_payload(self):
        with mock.patch("core.wolfram_utils._query_wolfram_v2", return_value={"@error": "true", "error": {"msg": "Blocked request"}}), \
            mock.patch("core.wolfram_utils._query_wolfram_short_answer") as short_answer_mock:
            response = query_wolfram_alpha("dummy-appid", "2+2")

        self.assertIsNone(response.get("answer"))
        self.assertIn("Wolfram request blocked", str(response.get("error")))
        short_answer_mock.assert_not_called()

    def test_query_wolfram_alpha_reports_blocked_short_answer_http_403(self):
        v2_response = requests.Response()
        v2_response.status_code = 400
        v2_response._content = b"Input error"
        v2_response.url = "https://api.wolframalpha.com/v2/query"
        v2_error = requests.HTTPError(response=v2_response)

        blocked_response = requests.Response()
        blocked_response.status_code = 403
        blocked_response._content = b"Blocked request"
        blocked_response.url = "https://api.wolframalpha.com/v1/result"
        blocked_error = requests.HTTPError(response=blocked_response)

        with mock.patch("core.wolfram_utils._query_wolfram_v2", side_effect=v2_error), \
            mock.patch("core.wolfram_utils._query_wolfram_short_answer", side_effect=blocked_error):
            response = query_wolfram_alpha("dummy-appid", "10/8*12")

        self.assertIsNone(response.get("answer"))
        self.assertIn("Wolfram request blocked", str(response.get("error")))
        self.assertEqual(response.get("query"), "10/8*12")


if __name__ == "__main__":
    unittest.main()
