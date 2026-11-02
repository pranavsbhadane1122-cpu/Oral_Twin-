"""Helper for tests that assert things about source text.

THE BUG CLASS THIS EXISTS TO END.

A test that reads a source file and searches it for a literal will find that
literal in ITSELF if the file it reads is its own. Two instances were found the
hard way:

  test_the_guard_is_imported_not_retyped searched its own source for a
  condition term and matched the term it was searching with.

  a later version searched for a regex whose own text happened not to match,
  which is worse - it passed by an accident of backslash escaping rather than
  by being correct.

A test that passes by accident is indistinguishable from one that passes
because the code is right, which is the whole problem.

So: read source through [read_source]. It strips the caller's own file by
default, and tests/test_source_scanning.py fails any test file that reaches
for read_text on a tests/ path directly.
"""

import inspect
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def read_source(path, exclude_caller=True):
    """Read a source file for scanning, excluding the calling test's own file.

    When the path being read IS the caller, the result is the empty string -
    not the file - because an assertion over a file about itself cannot be
    trusted either way. A test that needs to inspect its own source should say
    what it is looking for structurally, with [excluding_lines], rather than
    searching for a term it also contains.
    """
    path = Path(path)
    if exclude_caller:
        caller = Path(inspect.stack()[1].filename).resolve()
        if path.resolve() == caller:
            raise AssertionError(
                f"{caller.name} is scanning its own source. A literal searched "
                f"for will be found in the search itself. Use "
                f"excluding_lines() with an explicit marker, or assert on "
                f"structure rather than on a term."
            )
    return path.read_text(encoding="utf-8")


def excluding_lines(path, marker):
    """A file's text with every line bearing [marker] removed.

    For the one legitimate case: a test that must inspect its own source. The
    lines that make up the assertion carry the marker in a comment, so the
    assertion cannot match itself.
    """
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return "\n".join(line for line in lines if marker not in line)
