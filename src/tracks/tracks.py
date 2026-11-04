"""Tracks A, B and C, each binding a score to the performance behind it.

A TrackResult cannot be constructed without its TrackPerformance. That is the
whole design: the brief asked for the score and the figures as one object, and
an object whose constructor refuses the incomplete case is the only version of
that which survives a later edit.

All figures are read from models/logs/track_remeasurement.json. None is typed.
"""

from dataclasses import dataclass
from typing import Optional

from src.tracks.performance import (
    Measured,
    NoneExists,
    Rate,
    TrackPerformance,
    measurements,
)

TRACK_A = "A"
TRACK_B = "B"
TRACK_C = "C"

TRACK_TITLES = {
    TRACK_A: "Gum condition",
    TRACK_B: "Marked area outline",
    TRACK_C: "Screening score",
}

# Reasons an external figure does not exist. From step 1, not invented here.
NO_EXTERNAL_A = (
    "No second dataset in this project shares the same three labels, so there "
    "is nothing to test against. This is not a test that has yet to be run - "
    "it cannot be run with the data available."
)
NO_EXTERNAL_B = (
    "No second dataset in this project carries outlines drawn by hand, so "
    "there is nothing to compare against. This is not a test that has yet to "
    "be run - it cannot be run with the data available."
)


@dataclass(frozen=True)
class TrackResult:
    """A score and the performance that produced it, inseparably."""

    track: str
    title: str
    score: float
    performance: TrackPerformance
    detail: Optional[str] = None

    def __post_init__(self):
        if not isinstance(self.performance, TrackPerformance):
            raise TypeError(
                "a track score cannot be returned without its measured "
                "performance. This is enforced here rather than remembered at "
                "each call site."
            )
        if self.track not in TRACK_TITLES:
            raise ValueError(f"unknown track {self.track!r}")

    def render_lines(self):
        lines = [f"{self.title} (Track {self.track})"]
        if self.detail:
            lines.append(f"  {self.detail}")
        lines += [f"  {line}" for line in self.performance.render_lines()]
        return lines


# --------------------------------------------------------------------------
# Track A - gum condition
# --------------------------------------------------------------------------

def track_a_performance(cfg) -> TrackPerformance:
    """All three per-class figures, with the classes NOT named.

    A COLLISION, RESOLVED IN FAVOUR OF THE GUARD. The brief asks for all three
    per-class sensitivities; the three classes are conditions, and the
    vocabulary guard forbids a condition name in any user-facing string - a
    rule the same brief restates. Labelling the rows "finds caries in ..."
    satisfies the first requirement by breaking the second.

    So the rows are distinguished by their sample size, which is what actually
    varies between them, and the note says plainly that the names are withheld
    and where to find them. Every figure the brief asked for is present: three
    sensitivities, three intervals, three per-class n, and the total.

    This is a judgement call and it is reported as one. If the names should
    appear, the guard has to be amended deliberately rather than bypassed
    here.
    """
    data = measurements(cfg)["track_a"]
    internal = []
    total = 0
    for index, (_name, entry) in enumerate(data.items(), start=1):
        rate = Rate.from_json(entry["sensitivity"])
        total = entry["n"]
        # every one of these is under 100, so every one carries its interval
        internal.append(
            (f"finds the {_ordinal(index)} of three trained-for findings, "
             f"in {rate.denominator} examples", rate))
    return TrackPerformance(
        internal=tuple(internal),
        external=NoneExists(NO_EXTERNAL_A),
        note=(f"Measured on {total} photographs in total. Each figure carries "
              f"its interval because each group has fewer than a hundred "
              f"examples. The three groups are deliberately not named here: "
              f"this tool does not put the name of a condition in front of a "
              f"reader. They are listed in configs/config.yaml."),
    )


def _ordinal(n):
    return {1: "first", 2: "second", 3: "third"}.get(n, f"{n}th")


# --------------------------------------------------------------------------
# Track B - marked area outline
# --------------------------------------------------------------------------

def track_b_performance(cfg) -> TrackPerformance:
    """Dice is the reported figure, because the gate it failed used Dice.

    The per-pixel sensitivity of 0.689 is the friendlier number and is NOT
    the headline. Where it appears at all it appears in the same sentence as
    the PPV of 0.376, so a reader never meets the flattering half alone - a
    test enforces that.
    """
    data = measurements(cfg)["track_b"]
    dice = data["dice"]
    sensitivity = data["sensitivity"]
    ppv = data["ppv"]
    n = data["n_images"]

    # No numerator: a mean Dice is not a count of successes, and writing it
    # as "163 of 449" would invent one. No interval either - the bootstrap
    # interval exists in the Phase 3 log but is not in the re-measurement
    # JSON, and n=449 is above the threshold at which one is required, so
    # the honest move is to omit it rather than type it in from elsewhere.
    overlap = Rate(value=dice, denominator=n,
                   basis=f"mean over {n} photographs")
    return TrackPerformance(
        internal=(("overlap with the hand-drawn outline (Dice)", overlap),),
        external=NoneExists(NO_EXTERNAL_B),
        note=(f"Of the area it marks, {ppv:.0%} is inside the hand-drawn "
              f"outline, while it covers {sensitivity:.0%} of that outline. "
              f"This outline failed its acceptance gate, which required an "
              f"overlap of 0.70."),
    )


# --------------------------------------------------------------------------
# Track C - screening score
# --------------------------------------------------------------------------

def track_c_performance(cfg) -> TrackPerformance:
    data = measurements(cfg)["track_c"]
    internal_sens = Rate.from_json(data["internal"]["sensitivity"])
    external_sens = Rate.from_json(data["external"]["sensitivity"])
    internal = data["internal"]
    external = data["external"]

    internal_balance = (
        f"{internal['positives']} of {internal['n']} had a finding "
        f"({100.0 * internal['positives'] / internal['n']:.1f}%)")
    external_balance = (
        f"On that set {external['positives']} of {external['n']} images had a "
        f"finding ({100.0 * external['positives'] / external['n']:.1f}%), so "
        f"its overall success rate is dominated by how often it says no and "
        f"must not be quoted on its own.")

    return TrackPerformance(
        internal=((f"finds, on photographs from the same source, where "
                   f"{internal_balance}", internal_sens),),
        external=Measured(sensitivity=external_sens,
                          set_name="a set from a different clinic",
                          class_balance=external_balance),
    )


def build_tracks(cfg, score_a=None, score_b=None, score_c=None):
    """Every track whose score was supplied, each with its performance."""
    out = []
    if score_a is not None:
        out.append(TrackResult(TRACK_A, TRACK_TITLES[TRACK_A], float(score_a),
                               track_a_performance(cfg)))
    if score_b is not None:
        out.append(TrackResult(TRACK_B, TRACK_TITLES[TRACK_B], float(score_b),
                               track_b_performance(cfg)))
    if score_c is not None:
        out.append(TrackResult(TRACK_C, TRACK_TITLES[TRACK_C], float(score_c),
                               track_c_performance(cfg)))
    return out
