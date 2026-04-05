import os
import sys
import unittest


TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(TESTS_DIR)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from presentation.asy_rendering import MODEL_INPUT_DIAGRAM_NOTICE, strip_asy_blocks_for_model_input


class TestAsyRendering(unittest.TestCase):
    def test_strip_asy_blocks_for_model_input_replaces_diagram_code(self):
        text = "Question text.\n[asy]\ndraw((0,0)--(2,0));\n[/asy]\nFind the length."

        sanitized = strip_asy_blocks_for_model_input(text)

        self.assertIn(MODEL_INPUT_DIAGRAM_NOTICE, sanitized)
        self.assertNotIn("draw((0,0)--(2,0));", sanitized)

    def test_strip_asy_blocks_for_model_input_leaves_plain_text_unchanged(self):
        text = "Question text only."

        self.assertEqual(strip_asy_blocks_for_model_input(text), text)


if __name__ == "__main__":
    unittest.main()
