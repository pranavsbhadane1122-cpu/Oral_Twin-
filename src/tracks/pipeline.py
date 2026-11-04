"""Screening beside the change measurement, never instead of it.

BOTH BRANCHES CONTINUE. align -> compare -> measure -> explain runs identically
whether the screening model flagged the image or not. The screening result is
an additional output; it does not gate, filter, shortcut or reorder the change
pipeline, and the explanation renders exactly as it did before any of this
existed.

That matters most on the "Not flagged" branch. A screening model that misses
9 of every 125 findings must not be allowed to stop the one part of the system
that passed its gates from running.
"""

from dataclasses import dataclass
from typing import List, Optional

from src.explain.explain import explain
from src.tracks.screening import ScreeningResult, screen
from src.tracks.tracks import (
    TRACK_C,
    TRACK_TITLES,
    TrackResult,
    build_tracks,
    track_c_performance,
)


@dataclass(frozen=True)
class CombinedResult:
    """The change measurement, plus the screening output beside it."""

    explanation: dict
    screening: Optional[ScreeningResult]
    tracks: List[TrackResult]

    def render_lines(self):
        from src.explain.render import render

        lines = []
        if self.screening is not None:
            lines += self.screening.render_lines()
            lines.append("")
            lines.append("-" * 68)
            lines.append("")
        # the change pipeline output, unchanged and unconditional
        lines += render(self.explanation).splitlines()
        if self.tracks:
            lines += ["", "OTHER MEASUREMENTS", ""]
            for track in self.tracks:
                lines += [f"  {line}" for line in track.render_lines()]
                lines.append("")
        return lines


def run(cfg, change_report, risk, score_a=None, score_b=None, score_c=None,
        earlier_date=None, later_date=None) -> CombinedResult:
    """Change measurement always; screening additionally, when a score exists.

    Note the order of operations: the explanation is built FIRST and from the
    same inputs it has always used. Nothing about the screening result reaches
    it. If that ever changes, the change measurement stops being independent
    of a model that fails most of what it is asked to find.
    """
    explanation = explain(change_report, risk, cfg,
                          earlier_date=earlier_date, later_date=later_date)

    screening = None
    if score_c is not None:
        track_c = TrackResult(TRACK_C, TRACK_TITLES[TRACK_C], float(score_c),
                              track_c_performance(cfg))
        screening = screen(cfg, float(score_c), track_c)

    # Tracks A and B are reported beside the change measurement; Track C is
    # already rendered by the screening branch above, so it is not repeated.
    tracks = build_tracks(cfg, score_a=score_a, score_b=score_b)

    return CombinedResult(explanation=explanation, screening=screening,
                          tracks=tracks)
