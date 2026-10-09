import os
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run(*args):
    return subprocess.run([sys.executable, os.path.join(ROOT, "main.py"), "--log-file",
                           os.devnull, *args], capture_output=True, text=True, cwd=ROOT)


class TestCLI(unittest.TestCase):
    def test_demo_creates_outputs(self):
        with tempfile.TemporaryDirectory() as d:
            r = run("demo", "-o", d)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertTrue(os.path.isfile(os.path.join(d, "demo_input_scanned.png")))

    def test_missing_file_returns_error_code(self):
        r = run("scan", "does_not_exist.jpg")
        self.assertEqual(r.returncode, 1)
        self.assertIn("error", r.stderr)

    def test_evaluate_reports_success(self):
        with tempfile.TemporaryDirectory() as d:
            r = run("evaluate", "-n", "3", "-o", d)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertIn("success rate", r.stdout)

    def test_batch_skips_bad_files(self):
        with tempfile.TemporaryDirectory() as src, tempfile.TemporaryDirectory() as out:
            with open(os.path.join(src, "corrupt.png"), "wb") as f:
                f.write(b"garbage")
            r = run("batch", src, "-o", out)
            self.assertEqual(r.returncode, 2)
            self.assertIn("skipped", r.stderr)


if __name__ == "__main__":
    unittest.main()
