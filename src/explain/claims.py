"""Read threshold provenance from docs/CLAIMS.md.

The register is the authority. This module parses it; it states nothing of its
own. Hard-coding a threshold's origin here would create exactly the second copy
the register exists to prevent - and the copy in code would be the one nobody
updates.

Three states, and the distinction between the first two is the point:

  calibrated     evidence exists AND is written up as a logged run
  artefact only  evidence exists in a named artefact, never logged as a run
  provisional    a reasoned default with no data behind it

A user reading an explanation is entitled to know which of those a number is.
Rounding 'artefact only' up to 'calibrated' would be a small lie that makes the
system look better evidenced than it is.
"""

import re
from functools import lru_cache

from src.utils.config import PROJECT_ROOT

CALIBRATED = "calibrated"
ARTEFACT_ONLY = "artefact only"
PROVISIONAL = "provisional"
UNREGISTERED = "unregistered"

VALID_STATES = (CALIBRATED, ARTEFACT_ONLY, PROVISIONAL)

TABLE_START = "<!-- PARAMETER-TABLE-START -->"
TABLE_END = "<!-- PARAMETER-TABLE-END -->"

# How each state should be described to a reader, in one clause.
STATE_PHRASING = {
    CALIBRATED: "calibrated against data, and the calibration is written up",
    ARTEFACT_ONLY: ("based on real evidence, but that evidence was never written "
                    "up as a recorded run"),
    PROVISIONAL: "a reasoned default, not a value settled by data",
    UNREGISTERED: ("NOT REGISTERED - this threshold decided something and its "
                   "origin is not recorded anywhere"),
}


def claims_path():
    return PROJECT_ROOT / "docs" / "CLAIMS.md"


@lru_cache(maxsize=1)
def _parse(text):
    """{key: {value, state, note}} from the register's parameter table."""
    if TABLE_START not in text or TABLE_END not in text:
        raise ValueError(
            f"docs/CLAIMS.md is missing {TABLE_START} / {TABLE_END}. The "
            f"explanation layer parses that table and cannot invent it."
        )
    block = text.split(TABLE_START, 1)[1].split(TABLE_END, 1)[0]

    entries = {}
    for line in block.splitlines():
        line = line.strip()
        if not line.startswith("|") or line.startswith("|---"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) < 4:
            continue
        key = cells[0].strip("`").strip()
        if not key or key == "key":
            continue
        state = cells[2].strip("`").strip().lower()
        if state not in VALID_STATES:
            raise ValueError(
                f"docs/CLAIMS.md row {key!r} has state {state!r}, which is not "
                f"one of {VALID_STATES}. The register must use the stated "
                f"vocabulary or the explanation cannot report it faithfully."
            )
        entries[key] = {"value": cells[1].strip(), "state": state,
                        "note": cells[3].strip()}
    if not entries:
        raise ValueError("the parameter table in docs/CLAIMS.md parsed to nothing")
    return entries


def registry():
    """The parsed parameter table, keyed by dotted config path."""
    return _parse(claims_path().read_text(encoding="utf-8"))


def provenance(threshold_key):
    """{state, note, value, phrasing} for a threshold, or UNREGISTERED.

    Never raises on an unknown key: an unregistered threshold must be
    REPORTED in the explanation, not hidden behind an exception. The test
    suite is what fails, not the user-facing path.
    """
    if threshold_key is None:
        return {"state": None, "note": None, "value": None, "phrasing": None}
    entry = registry().get(threshold_key)
    if entry is None:
        return {"state": UNREGISTERED, "note": None, "value": None,
                "phrasing": STATE_PHRASING[UNREGISTERED]}
    return {**entry, "phrasing": STATE_PHRASING[entry["state"]]}


def registered_keys():
    return set(registry())
