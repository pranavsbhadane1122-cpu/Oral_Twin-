"""Standing check against the self-referential test bug.

Two instances of it were found by hitting them. This finds the rest by looking,
and keeps looking: any new test that scans its own source fails here rather
than passing for the wrong reason.

Run with: python -m unittest tests.test_source_scanning -v
"""

import re
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from tests.source_scan import excluding_lines, read_source  # noqa: E402

TESTS_DIR = PROJECT_ROOT / "tests"
SELF_SCAN_OK = "self-scan-ok"


def test_files():
    return sorted(p for p in TESTS_DIR.glob("test_*.py"))


class TestNoTestScansItsOwnSource(unittest.TestCase):

    def test_no_test_file_reads_a_path_under_tests(self):
        """Reading a tests/ path is the precondition for the bug.

        The helper in tests/source_scan.py exists for the legitimate case and
        refuses the dangerous one, so a direct read_text on a tests/ path is
        the thing to catch.
        """
        offenders = []
        for path in test_files():
            # exclude this file, which necessarily names the pattern it bans
            text = excluding_lines(path, SELF_SCAN_OK)
            for lineno, line in enumerate(text.splitlines(), 1):
                if re.search(r'"tests"\s*/\s*"[^"]+\.py"', line):      # self-scan-ok
                    offenders.append(f"{path.name}:{lineno}: {line.strip()}")
        self.assertEqual(
            offenders, [],
            "these tests read a source file under tests/, which is how a test "
            "comes to find the literal it is searching with. Use "
            "tests/source_scan.read_source or excluding_lines instead:\n"
            + "\n".join(offenders))

    def test_every_source_scanning_test_reads_outside_tests(self):
        """The scans that do exist must target src/, app/ or android/."""
        allowed_roots = ("src", "app", "android", "scripts", "docs", "configs")
        for path in test_files():
            text = excluding_lines(path, SELF_SCAN_OK)
            for match in re.finditer(r"PROJECT_ROOT\s*/\s*\"(\w+)\"", text):
                root = match.group(1)
                self.assertIn(
                    root, allowed_roots + ("tests",),
                    f"{path.name} scans an unexpected root {root!r}")


class TestTheHelperRefusesSelfScans(unittest.TestCase):

    def test_read_source_refuses_to_read_the_caller(self):
        with self.assertRaises(AssertionError) as ctx:
            read_source(Path(__file__))
        self.assertIn("scanning its own source", str(ctx.exception))

    def test_read_source_allows_another_file(self):
        text = read_source(PROJECT_ROOT / "src" / "explain" / "explain.py")
        self.assertIn("def explain_comparable", text)

    def test_excluding_lines_removes_marked_lines(self):
        marked = excluding_lines(Path(__file__), SELF_SCAN_OK)
        # the regex-bearing line in the first test carries the marker, so it is
        # gone from the text this assertion sees
        self.assertNotIn('"tests"\\s*/\\s*"[^"]+\\.py"', marked)

    def test_the_marker_check_is_not_vacuous(self):
        full = Path(__file__).read_text(encoding="utf-8")
        self.assertIn(SELF_SCAN_OK, full,
                      "if the marker were absent the exclusion would do nothing")


if __name__ == "__main__":
    unittest.main(verbosity=2)
