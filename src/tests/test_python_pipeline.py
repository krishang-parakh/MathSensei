import io
import os
import sys
import unittest
from contextlib import redirect_stdout


TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(TESTS_DIR)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from core.python_pipeline import (
    apply_local_python_repair,
    execution_failure_reason,
    extract_missing_dependency,
    final_output_line,
    install_missing_dependency,
    repair_program_until_runnable,
    sanitize_generated_python,
    should_attempt_program_repair,
)


def execute_python(code):
    buffer = io.StringIO()
    namespace = {"__name__": "__main__"}
    with redirect_stdout(buffer):
        exec(code, namespace)
    return buffer.getvalue()


class TestPythonPipeline(unittest.TestCase):
    def test_sanitize_removes_prompt_bleed_and_trailing_sections(self):
        raw = """
```python
Question: Tim has some cans of soda...
Modules used till now: []
Python Generator:
# Python Code, print answer.
Make sure that the first line of the code is always 'from sympy import *'
from sympy import *
initial_cans = symbols('initial_cans')
equation = Eq(initial_cans - 6 + Rational(1, 2) * (initial_cans - 6), 24)
ans = solve(equation, initial_cans)[0]
print(ans)
Python output:
22
Solution:
- the answer is #### 22
```
"""
        cleaned = sanitize_generated_python(raw)
        self.assertNotIn("Question:", cleaned)
        self.assertNotIn("Python output:", cleaned)
        self.assertTrue(cleaned.startswith("from sympy import *"))
        self.assertEqual(final_output_line(execute_python(cleaned)), "22")

    def test_sanitize_normalizes_unicode_fractions_in_code(self):
        raw = """
from sympy import *
x = ½
print(x)
"""
        cleaned = sanitize_generated_python(raw)
        self.assertNotIn("½", cleaned)
        self.assertIn("1/2", cleaned)
        self.assertEqual(final_output_line(execute_python(cleaned)), "0.5")

    def test_sanitize_removes_inline_instruction_lines(self):
        raw = """
from sympy import *
Python Code, print answer, and also output all the relevant objects in the intermediate steps of the python code.
Make sure that the first line of the code is always 'from sympy import *'
yellow = 10
purple = yellow + 0.8 * yellow
green = 0.25 * (yellow + purple)
total_flowers = yellow + purple + green
print(total_flowers)
"""
        cleaned = sanitize_generated_python(raw)
        self.assertNotIn("Python Code, print answer", cleaned)
        self.assertNotIn("Make sure that", cleaned)
        self.assertEqual(final_output_line(execute_python(cleaned)), "35.0")

    def test_sanitize_removes_plain_english_step_lines(self):
        raw = """
from sympy import *
x, y = symbols('x y')
Define the equations
eq1 = Eq(725*x + 727*y, 1500)
eq2 = Eq(729*x + 731*y, 1508)
Solve the system for x and y
solutions = solve((eq1, eq2), (x, y))
ans = solutions[x] - solutions[y]
Output the value of x - y
print(ans)
"""
        cleaned = sanitize_generated_python(raw)

        self.assertNotIn("Define the equations", cleaned)
        self.assertNotIn("Solve the system for x and y", cleaned)
        self.assertNotIn("Output the value of x - y", cleaned)
        self.assertIn("eq1 = Eq", cleaned)

    def test_gsm_simple_arithmetic_program_executes(self):
        code = sanitize_generated_python(
            """
from sympy import *
total_earned = 60
lisa = total_earned / 2
tommy = lisa / 2
ans = lisa - tommy
print(ans)
"""
        )
        self.assertEqual(final_output_line(execute_python(code)), "15.0")

    def test_gsm_linear_solve_program_executes(self):
        code = sanitize_generated_python(
            """
from sympy import *
initial_cans = symbols('initial_cans')
cans_left_after_jeff = initial_cans - 6
cans_after_purchase = cans_left_after_jeff + Rational(1, 2) * cans_left_after_jeff
equation = Eq(cans_after_purchase, 24)
ans = solve(equation, initial_cans)[0]
print(ans)
"""
        )
        self.assertEqual(final_output_line(execute_python(code)), "22")

    def test_gsm_exact_fractional_arithmetic_program_executes(self):
        code = sanitize_generated_python(
            """
from sympy import *
sam_skips_each_round = 16
jeff_skips_round1 = sam_skips_each_round - 1
jeff_skips_round2 = sam_skips_each_round - 3
jeff_skips_round3 = sam_skips_each_round + 4
jeff_skips_round4 = Rational(1, 2) * sam_skips_each_round
ans = (jeff_skips_round1 + jeff_skips_round2 + jeff_skips_round3 + jeff_skips_round4) / 4
print(ans)
"""
        )
        self.assertEqual(final_output_line(execute_python(code)), "14")

    def test_gsm_empty_output_is_treated_as_failure(self):
        reason = execution_failure_reason("GSM", "", None)
        self.assertIn("produced no output", reason)
        self.assertTrue(should_attempt_program_repair("GSM", "", None))

    def test_common_sympy_error_still_requests_repair(self):
        error = "list indices must be integers or slices, not Symbol"
        self.assertTrue(should_attempt_program_repair("MATH", None, error))

    def test_additional_sympy_runtime_errors_now_trigger_repair(self):
        self.assertTrue(
            should_attempt_program_repair("MATH", None, "AttributeError: 'list' object has no attribute 'free_symbols'")
        )
        self.assertTrue(
            should_attempt_program_repair("MATH", None, "ValueError: can only solve for one symbol at a time")
        )

    def test_local_repair_adds_missing_radians_import(self):
        raw_program = """
from sympy import *
angle = 120
result = sin(radians(angle))
print("sin(120):", result)
"""
        repaired_program, note = apply_local_python_repair(raw_program.strip(), "name 'radians' is not defined")

        self.assertIn("from math import radians", repaired_program)
        self.assertIn("Added missing import", note)
        output = execute_python(repaired_program)
        self.assertIn("sin(120):", output)

    def test_local_repair_adds_fraction_import(self):
        raw_program = """
value = Fraction(1, 3) + Fraction(1, 6)
print(value)
"""
        repaired_program, note = apply_local_python_repair(raw_program.strip(), "name 'Fraction' is not defined")

        self.assertIn("from fractions import Fraction", repaired_program)
        self.assertIn("Added missing import", note)
        self.assertEqual(final_output_line(execute_python(repaired_program)), "1/2")

    def test_local_repair_adds_sympy_import_for_common_symbolic_utility(self):
        raw_program = """
expr = Eq(x + 2, 5)
print(expr)
"""
        repaired_program, note = apply_local_python_repair(raw_program.strip(), "name 'Eq' is not defined")

        self.assertTrue(repaired_program.startswith("from sympy import *"))
        self.assertIn("Added missing SymPy import", note)

    def test_local_repair_adds_dict_true_for_sympy_system_solve_shape_mismatch(self):
        raw_program = """
from sympy import *
x, y = symbols('x y')
eq1 = Eq(725*x + 727*y, 1500)
eq2 = Eq(729*x + 731*y, 1508)
solutions = solve((eq1, eq2), (x, y))
x_value = solutions[0][x]
y_value = solutions[0][y]
ans = x_value - y_value
print(ans)
"""
        repaired_program, note = apply_local_python_repair(raw_program.strip(), "KeyError: 0")

        self.assertIn("dict=True", repaired_program)
        self.assertIn("solution format", note)
        self.assertEqual(final_output_line(execute_python(repaired_program)), "-48")

    def test_local_repair_adds_first_solution_index_for_symbol_key_access(self):
        raw_program = """
from sympy import *
c, d = symbols('c d')
eq1 = Eq(c + d, 7)
eq2 = Eq(c - d, 1)
solutions = solve((eq1, eq2), (c, d))
ans = solutions[c] * solutions[d]
print(ans)
"""
        repaired_program, note = apply_local_python_repair(raw_program.strip(), "KeyError: c")

        self.assertIn("dict=True", repaired_program)
        self.assertIn("solutions[0][c]", repaired_program)
        self.assertIn("solutions[0][d]", repaired_program)
        self.assertIn("dict indexing", note)
        self.assertEqual(final_output_line(execute_python(repaired_program)), "12")

    def test_local_repair_switches_integer_lcm_to_math_lcm(self):
        raw_program = """
from sympy import *
denominators = [16, 10, 8]
ans = lcm(*denominators)
print(ans)
"""
        repaired_program, note = apply_local_python_repair(
            raw_program.strip(),
            "AttributeError: 'int' object has no attribute 'is_commutative'",
        )

        self.assertIn("from math import lcm", repaired_program)
        self.assertIn("math.* helpers", note)
        self.assertEqual(final_output_line(execute_python(repaired_program)), "80")

    def test_local_repair_appends_print_for_answer_variable_when_output_is_missing(self):
        raw_program = """
from sympy import *
ans = Integer(42)
"""
        repaired_program, note = apply_local_python_repair(
            raw_program.strip(),
            "Program executed but produced no output. The code must print the derived final answer.",
        )

        self.assertTrue(repaired_program.endswith("print(ans)"))
        self.assertIn("Appended print(ans)", note)
        self.assertEqual(final_output_line(execute_python(repaired_program)), "42")

    def test_extract_missing_dependency_maps_supported_package(self):
        module_name, package_name = extract_missing_dependency("No module named 'sympy'")

        self.assertEqual(module_name, "sympy")
        self.assertEqual(package_name, "sympy")

    def test_extract_missing_dependency_maps_supported_nested_package(self):
        module_name, package_name = extract_missing_dependency("No module named 'google.generativeai'")

        self.assertEqual(module_name, "google.generativeai")
        self.assertEqual(package_name, "google-generativeai")

    def test_install_missing_dependency_uses_installer_callback(self):
        calls = []

        def fake_installer(package_name):
            calls.append(package_name)
            return True, "installed"

        result = install_missing_dependency("No module named 'numpy'", installer=fake_installer)

        self.assertEqual(calls, ["numpy"])
        self.assertEqual(result["module"], "numpy")
        self.assertEqual(result["package"], "numpy")
        self.assertTrue(result["installed"])
        self.assertEqual(result["message"], "installed")

    def test_repair_program_until_runnable_retries_multiple_times_until_success(self):
        executed_programs = []
        llm_calls = []

        def fake_execute(program):
            executed_programs.append(program)
            if "Fraction(1, 2)" in program:
                return "1/2\n", None
            if "Fraction(1, 3)" in program:
                return None, "SyntaxError: invalid syntax"
            return None, "name 'Fraction' is not defined"

        def fake_llm_repair(program, error_message):
            llm_calls.append((program, error_message))
            return "value = Fraction(1, 2)\nprint(value)", "Replaced the malformed fraction expression."

        program, output, raw_error, error_message, attempts = repair_program_until_runnable(
            "value = Fraction(1, 3\nprint(value)",
            "MATH",
            fake_execute,
            llm_repair=fake_llm_repair,
            max_attempts=3,
        )

        self.assertEqual(output, "1/2\n")
        self.assertIsNone(raw_error)
        self.assertIsNone(error_message)
        self.assertEqual(len(attempts), 2)
        self.assertEqual(attempts[0]["source"], "local")
        self.assertEqual(attempts[1]["source"], "llm")
        self.assertIn("Fraction(1, 2)", program)
        self.assertEqual(len(llm_calls), 1)
        self.assertEqual(len(executed_programs), 3)

    def test_repair_program_until_runnable_does_not_retry_runnable_but_wrong_output(self):
        llm_calls = []

        def fake_execute(program):
            return "999\n", None

        def fake_llm_repair(program, error_message):
            llm_calls.append((program, error_message))
            return "print(1)", "Should not be called."

        program, output, raw_error, error_message, attempts = repair_program_until_runnable(
            "print(999)",
            "MATH",
            fake_execute,
            llm_repair=fake_llm_repair,
            max_attempts=3,
        )

        self.assertEqual(program, "print(999)")
        self.assertEqual(output, "999\n")
        self.assertIsNone(raw_error)
        self.assertIsNone(error_message)
        self.assertEqual(attempts, [])
        self.assertEqual(llm_calls, [])

    def test_empty_output_requests_repair_for_math(self):
        reason = execution_failure_reason("MATH", "", None)
        self.assertIn("produced no output", reason)
        self.assertTrue(should_attempt_program_repair("MATH", "", None))


if __name__ == "__main__":
    unittest.main()
