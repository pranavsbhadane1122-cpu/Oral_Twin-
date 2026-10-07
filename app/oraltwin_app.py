"""OralTwin demo - compare two visit photographs.

Run with:  streamlit run app/oraltwin_app.py
Deep links: ?case=good | out_of_frame | low_confidence | unrelated

The demo covers the validated modules only: alignment and change detection.
Disease classification is parked and segmentation/OPMD are blocked on data, and
the app says so on screen rather than hiding it.
"""

import copy
import sys
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.delta.compare import PARTIAL  # noqa: E402
from src.demo.pipeline import (  # noqa: E402
    NOT_INCLUDED,
    SCREENING_LINE,
    demo_cases,
    lesion_table_rows,
    run_demo,
)
from src.utils.config import load_config  # noqa: E402

BAND_STYLE = {"Low": "#1C6D60", "Moderate": "#7F620E", "Elevated": "#983820"}


def rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def contour_overlay(img, mask, colour=(0, 255, 0)):
    out = img.copy()
    if mask is not None:
        contours, _ = cv2.findContours((mask > 127).astype(np.uint8),
                                       cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, colour, 2)
    return out


def screening_banner():
    st.caption(SCREENING_LINE)


def render_not_included():
    st.subheader("What this demo does not do")
    st.warning("These parts of the system are not included. Nothing here is a "
               "substitute for a dental examination.")
    for item in NOT_INCLUDED:
        st.markdown("- " + item)


def render_caption(result):
    """The plain-language description. Describes the photographs, not the mouth."""
    caption = result.get("caption")
    if not caption:
        return
    st.subheader("0 - In plain language")
    st.caption("This section describes your photographs: whether they are usable "
               "and what changed. It does not say what is in them.")

    left, right = st.columns(2)
    for column, key, label in ((left, "visit1", "Previous visit"),
                               (right, "visit2", "This visit")):
        with column:
            st.markdown(f"**{label}**")
            for line in caption["lines"][key]:
                st.markdown("- " + line)
            if not caption[key]["usable"]:
                st.warning("This photograph may not be good enough to rely on.")

    st.markdown("**Compared with the previous visit**")
    comparison = caption["comparison"]
    lines = caption["lines"]["comparison"]
    if comparison["state"] != "reliable":
        # a refusal must not be able to read like a result
        st.error("\n\n".join(lines))
    else:
        for line in lines:
            st.markdown("- " + line)

    st.info(caption["limitations"])


def render_provenance(case):
    """Say on screen, not only in a filename, when a case is induced."""
    if case.get("induced"):
        st.error("**Induced case.** " + case["induced"])
    if case.get("provenance"):
        st.caption(case["provenance"])


def render_alignment(result):
    alignment = result["alignment"]
    st.subheader("2 - Alignment")
    if alignment["H"] is None:
        st.error("**The two photographs could not be lined up.** "
                 + str(alignment["reason"])
                 + "\n\nNo measurement is shown, because any number here would "
                 "describe camera movement rather than a change in your mouth.")
        return
    cols = st.columns([2, 3])
    with cols[0]:
        st.metric("Alignment confidence", format(alignment["confidence"], ".2f"))
        st.caption(str(alignment["n_inliers"]) + " matching points")
    with cols[1]:
        st.caption("How that confidence was reached")
        breakdown = {k: v for k, v in alignment["breakdown"].items() if k != "raw"}
        st.dataframe(
            {
                "component": list(breakdown.keys()),
                "value": [round(float(v), 4) if isinstance(v, (int, float)) else str(v)
                          for v in breakdown.values()],
            },
            hide_index=True, use_container_width=True,
        )


def render_lesions(result):
    st.subheader("4 - What changed")
    report = result["change_report"]
    if report.get("status") != "ok":
        st.error("**Could not compare these photographs.** " + str(report.get("reason")))
        st.caption("No area, colour or change figures are reported for a comparison "
                   "that could not be trusted.")
        return

    st.info("Regions shown come from: " + result["masks"]["label"])
    lesions = report.get("lesions", [])
    if not lesions:
        st.write("No distinct region was tracked in these photographs.")
        return

    st.dataframe(lesion_table_rows(lesions), hide_index=True, use_container_width=True)
    for les in lesions:
        if les["comparability"] == PARTIAL:
            st.warning("Region " + str(les["id"]) + ": " + les["message"])
        else:
            st.markdown("**Region " + str(les["id"]) + ".** " + les["message"])


def render_risk(result):
    risk = result["risk"]
    st.subheader("5 - Screening band")
    colour = BAND_STYLE.get(risk["band"], "#56666A")
    st.markdown(
        "<div style='display:inline-block;padding:.35rem .8rem;border-radius:4px;"
        "background:" + colour + ";color:#fff;font-weight:600;letter-spacing:.04em'>"
        + risk["band"] + "</div>",
        unsafe_allow_html=True,
    )
    st.markdown("")
    for line in risk["text"].splitlines():
        if not line:
            continue
        st.markdown(line if line.startswith("-") else "**" + line + "**")
    if risk["rules_fired"]:
        with st.expander("Why this band (the exact rules that fired)", expanded=True):
            for rule in risk["rules_fired"]:
                st.markdown("`" + rule["rule"] + "` (" + rule["level"] + ") - "
                            + rule["detail"])
    else:
        st.caption("No change rule was triggered.")
    st.caption(risk["context_note"])


def render_result(result):
    render_caption(result)
    st.divider()
    st.subheader("1 - The two photographs")
    cols = st.columns(2)
    cols[0].image(rgb(result["images"]["visit1"]), caption="Previous visit",
                  use_container_width=True)
    cols[1].image(rgb(result["images"]["visit2"]), caption="This visit",
                  use_container_width=True)

    render_alignment(result)

    if result["images"]["overlay"] is not None:
        st.subheader("3 - Aligned overlay")
        cols = st.columns(2)
        cols[0].image(rgb(result["images"]["overlay"]),
                      caption="50% blend of the two photographs after alignment",
                      use_container_width=True)
        masks = result["masks"]
        if masks["mask1"] is not None:
            marked = contour_overlay(result["images"]["visit1"], masks["mask1"])
            cols[1].image(rgb(marked), caption="Region outline - " + masks["label"],
                          use_container_width=True)

    render_lesions(result)
    render_risk(result)

    with st.expander("Full text report"):
        st.code(result["report_text"], language="text")


def main():
    st.set_page_config(page_title="OralTwin demo", page_icon="\U0001F9B7", layout="wide")
    cfg = load_config()
    cases = demo_cases(cfg)

    st.title("OralTwin - visit comparison demo")
    st.markdown("Compare two photographs of the same mouth taken at different visits. "
                "This demo measures **what changed between the photographs**; it does "
                "not name a condition.")
    screening_banner()

    if cfg["classification"].get("surface_predictions", False):
        st.error("classification.surface_predictions is true, but the classifier is "
                 "parked. Set it to false before showing this demo to anyone.")

    with st.sidebar:
        st.header("Choose photographs")
        preset = st.query_params.get("case")
        keys = list(cases)
        mode = st.radio("Source", ["Sample case", "Upload two photos"])
        case_key = keys[0]
        uploads = (None, None)
        if mode == "Sample case":
            index = keys.index(preset) if preset in keys else 0
            case_key = st.selectbox("Sample", keys, index=index,
                                    format_func=lambda k: cases[k]["title"])
            st.caption(cases[case_key]["note"])
        else:
            uploads = (st.file_uploader("Previous visit", type=["jpg", "jpeg", "png"]),
                       st.file_uploader("This visit", type=["jpg", "jpeg", "png"]))
        st.divider()
        st.caption("Sample pairs are simulated visit pairs with known ground truth.")

    if mode == "Upload two photos":
        if not all(uploads):
            st.info("Upload a photograph for each visit to run the comparison.")
            render_not_included()
            screening_banner()
            return
        images = [cv2.imdecode(np.frombuffer(u.getvalue(), np.uint8), cv2.IMREAD_COLOR)
                  for u in uploads]
        if any(im is None for im in images):
            st.error("One of those files could not be read as an image.")
            return
        with st.spinner("Aligning and comparing..."):
            result = run_demo(images[0], images[1], cfg=cfg)
    else:
        case = cases[case_key]
        st.caption("Sample: **" + case["title"] + "** - " + case["note"])
        render_provenance(case)
        case_cfg = cfg
        if case.get("cfg_overrides"):
            case_cfg = copy.deepcopy(cfg)
            for section, values in case["cfg_overrides"].items():
                case_cfg[section].update(values)
        with st.spinner("Aligning and comparing..."):
            result = run_demo(case["paths"][0], case["paths"][1],
                              pair_name=case["pair_name"], cfg=case_cfg)

    render_result(result)
    st.divider()
    render_not_included()
    st.divider()
    screening_banner()


if __name__ == "__main__":
    main()
