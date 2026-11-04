"""The screening branch: flagged, or not flagged.

THE TWO LABELS ARE WHAT THE MODEL ACTUALLY COMPUTES. A score crosses a
threshold or it does not; "Flagged for review" and "Not flagged" say exactly
that and nothing more. "Cancerous (suspected)" and "Non-cancerous" would
describe a disease state the model does not compute - and the second of those
would be the more harmful, because it reads as an all-clear from a model that
misses most of what it should find.

THE NEGATIVE BRANCH IS EXPLICITLY NON-INFORMATIVE. Both branch statements are
generated from constants with the figures resolved from
models/logs/track_remeasurement.json. Nothing is typed. A sensitivity typed
into a sentence is a sentence that keeps saying the old number after the model
is retrained.

NEITHER BRANCH GATES THE CHANGE PIPELINE. align -> compare -> measure ->
explain runs identically whichever way this falls. The screening result is an
extra output beside the change measurement, never a replacement for it and
never a filter on it.
"""

from dataclasses import dataclass

from src.tracks.performance import Rate, measurements
from src.tracks.tracks import TrackResult

FLAGGED = "Flagged for review"
NOT_FLAGGED = "Not flagged"

QUESTION = "Flagged by the screening model?"

REFERRAL_LINE = "Have this checked by a dentist."

CLINICIAN_LINE = (
    "A dentist should look at the photographs themselves. Nothing here "
    "replaces that."
)


@dataclass(frozen=True)
class ScreeningResult:
    flagged: bool
    score: float
    threshold: float
    statement: str
    track: TrackResult

    @property
    def label(self) -> str:
        return FLAGGED if self.flagged else NOT_FLAGGED

    def render_lines(self):
        lines = [QUESTION, "",
                 f"  {self.label}.",
                 f"  {self.statement}", ""]
        lines += self.track.render_lines()
        lines += ["", f"  {REFERRAL_LINE}", f"  {CLINICIAN_LINE}"]
        return lines


def _external(cfg):
    return measurements(cfg)["track_c"]["external"]


def not_flagged_statement(cfg) -> str:
    """Non-informative by construction, with the miss rate as a count.

    The brief's wording said "does not flag most of the cases it should". The
    count says it more plainly. A rate can be read optimistically; a fraction
    with a denominator is harder to round off in the reading.

    The figures are not written out here, not even as an example: a number in
    a docstring goes stale exactly as fast as one in a string, and the next
    reader cannot tell which of the two is live.
    """
    sensitivity = Rate.from_json(_external(cfg)["sensitivity"])
    return (
        "The screening model did not flag this image. On external testing it "
        f"flagged {sensitivity.numerator} of {sensitivity.denominator} images "
        "that had a finding. This is not evidence that nothing is present."
    )


def flagged_statement(cfg) -> str:
    """What a flag is worth, from the measured PPV."""
    ppv = Rate.from_json(_external(cfg)["ppv"])
    one_in = round(1.0 / ppv.value) if ppv.value > 0 else None
    share = f"about 1 in {one_in}" if one_in else "very few"
    return (
        "The screening model flagged this image. On external testing, "
        f"{share} of the images it flagged had a finding."
    )


def screen(cfg, score: float, track: TrackResult) -> ScreeningResult:
    """Apply the configured threshold. The threshold is never a literal."""
    threshold = cfg["screening"]["flag_threshold"]
    flagged = float(score) >= float(threshold)
    statement = flagged_statement(cfg) if flagged else not_flagged_statement(cfg)
    return ScreeningResult(flagged=flagged, score=float(score),
                           threshold=float(threshold), statement=statement,
                           track=track)
