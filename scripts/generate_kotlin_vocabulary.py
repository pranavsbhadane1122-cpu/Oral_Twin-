"""Generate the Kotlin forbidden-vocabulary list from the Python authority.

THE PROBLEM THIS SOLVES. The guard list in tests/test_caption.py is the single
authority, and Kotlin cannot import it. docs/CLAIMS.md section 3 records that as
a drift risk: two hand-typed lists in two languages will part company, and
nothing will notice.

THE FIX. One authority, one derived copy, and a test that fails when they part
company. The Kotlin file is marked generated and must not be hand-edited;
tests/test_explain.py asserts it matches the Python source term for term, so an
edit on either side shows up as a failure rather than as a quiet divergence.

THE COUNT IS NEVER WRITTEN BY HAND. It was reported as 46 and is 42. A count
typed by a person is a claim that rots; this script derives it and the test
re-derives it.

Usage: python scripts/generate_kotlin_vocabulary.py [--check]
  --check  exit non-zero if the generated file is stale, without rewriting it
"""

import argparse
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

SOURCE = PROJECT_ROOT / "tests" / "test_caption.py"
TARGET = (PROJECT_ROOT / "android" / "app" / "src" / "main" / "java" / "com"
          / "oraltwin" / "app" / "guard" / "ForbiddenVocabulary.kt")
MARKER = "GENERATED FILE - DO NOT EDIT BY HAND"


def python_terms():
    """The authoritative list, read from the Python guard."""
    text = SOURCE.read_text(encoding="utf-8")
    match = re.search(r"CONDITION_WORDS\s*=\s*\[(.*?)\n\]", text, re.S)
    if not match:
        raise SystemExit(
            f"could not find CONDITION_WORDS in {SOURCE}. The generator reads "
            f"the Python guard as its authority and will not invent a list."
        )
    terms = re.findall(r'"([^"]+)"', match.group(1))
    if not terms:
        raise SystemExit("CONDITION_WORDS parsed to an empty list")
    return terms


def render(terms):
    lines = [
        "package com.oraltwin.app.guard",
        "",
        "/**",
        f" * {MARKER}.",
        " *",
        " * Generated from tests/test_caption.py (CONDITION_WORDS) by",
        " * scripts/generate_kotlin_vocabulary.py. That Python list is the single",
        " * authority for this project; this file is a derived copy, because Kotlin",
        " * cannot import it.",
        " *",
        " * Editing this file by hand reintroduces exactly the drift it exists to",
        " * prevent. Change the Python list and re-run the generator instead.",
        " *",
        " * ForbiddenVocabularyTest asserts this matches the Python source term for",
        " * term, so the two cannot part company silently. The count is NOT written",
        " * here: it is derived on both sides, because a count typed by a person is",
        " * a claim that rots - this one was reported as 46 and is in fact the size",
        " * of the list below.",
        " */",
        "object ForbiddenVocabulary {",
        "",
        "    /** Condition terms that must not appear in any user-facing string. */",
        "    val TERMS: List<String> = listOf(",
    ]
    for term in terms:
        lines.append(f'        "{term}",')
    lines += [
        "    )",
        "",
        "    /**",
        "     * True if [text] contains any forbidden term on a word boundary.",
        "     *",
        "     * Word boundaries matter: \"ok\" is a substring of ordinary words, and a",
        "     * naive contains() would reject \"looking\" and \"token\".",
        "     */",
        "    fun offendingTerm(text: String): String? =",
        "        TERMS.firstOrNull { term ->",
        "            Regex(\"\\\\b\" + Regex.escape(term.trim()) + \"\\\\b\",",
        "                  RegexOption.IGNORE_CASE).containsMatchIn(text)",
        "        }",
        "",
        "    fun isClean(text: String): Boolean = offendingTerm(text) == null",
        "}",
        "",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="fail if stale, do not rewrite")
    args = parser.parse_args()

    terms = python_terms()
    rendered = render(terms)

    if args.check:
        if not TARGET.exists():
            raise SystemExit(f"STALE: {TARGET} does not exist. Run the generator.")
        if TARGET.read_text(encoding="utf-8") != rendered:
            raise SystemExit(
                f"STALE: {TARGET.relative_to(PROJECT_ROOT)} does not match "
                f"tests/test_caption.py. Run "
                f"python scripts/generate_kotlin_vocabulary.py"
            )
        print(f"up to date: {len(terms)} terms")
        return

    TARGET.parent.mkdir(parents=True, exist_ok=True)
    TARGET.write_text(rendered, encoding="utf-8")
    print(f"wrote {TARGET.relative_to(PROJECT_ROOT)} with {len(terms)} terms")
    print("(count derived, not written by hand)")


if __name__ == "__main__":
    main()
