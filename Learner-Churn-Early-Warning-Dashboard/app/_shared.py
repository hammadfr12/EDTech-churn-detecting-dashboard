"""Shared dashboard plumbing: path bootstrap, design system, filters, states.

Every page imports from here so the visual language, the metric definitions
and the empty states are identical across the app.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Path bootstrap so `import src...` works regardless of the working directory.
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import config  # noqa: E402
from src.analytics import loader  # noqa: E402
from src.model.train import assign_risk_bands  # noqa: E402

# ---------------------------------------------------------------------------
# Design system
# ---------------------------------------------------------------------------
COLORS: Dict[str, str] = {
    "ink": "#111834",
    "muted": "#5B6478",
    "faint": "#8B93A7",
    "line": "#E3E6EF",
    "surface": "#FFFFFF",
    "page": "#F7F8FC",
    "primary": "#2F45C5",
    "primary_soft": "#E8EBFA",
    "accent": "#0E9F8E",
    "accent_soft": "#E1F5F2",
}

# Semantic scales. Outcome and risk share a green->amber->red logic so a reader
# learns it once.
OUTCOME_COLORS: Dict[str, str] = {
    "Pass": "#2E9E6B",
    "Distinction": "#2F45C5",
    "Fail": "#E0A100",
    "Withdrawn": "#D64550",
}

RISK_COLORS: Dict[str, str] = {
    "Low": "#2E9E6B",
    "Medium": "#E0A100",
    "High": "#D64550",
}

SERIES_SEQUENCE: List[str] = [
    "#2F45C5",
    "#0E9F8E",
    "#E0A100",
    "#D64550",
    "#7A5AF8",
    "#2E9E6B",
    "#E8722E",
    "#5B6478",
]

PLOTLY_FONT = (
    "-apple-system, BlinkMacSystemFont, 'Segoe UI', Inter, Roboto, "
    "'Helvetica Neue', Arial, sans-serif"
)

_CSS = f"""
<style>
    .stApp {{ background: {COLORS['page']}; }}
    html, body, [class*="css"] {{ font-family: {PLOTLY_FONT}; }}

    .block-container {{ padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1500px; }}

    /* Header band */
    .lc-header {{
        background: linear-gradient(100deg, {COLORS['ink']} 0%, #22307A 55%, {COLORS['primary']} 100%);
        border-radius: 14px;
        padding: 20px 26px;
        margin-bottom: 20px;
        color: #FFFFFF;
    }}
    .lc-header h1 {{ color: #FFFFFF; font-size: 1.5rem; margin: 0 0 4px 0; letter-spacing: -0.01em; }}
    .lc-header p  {{ color: #C9D2F0; margin: 0; font-size: 0.86rem; }}
    .lc-header .lc-tag {{
        display: inline-block; background: rgba(255,255,255,0.14);
        border: 1px solid rgba(255,255,255,0.25);
        border-radius: 999px; padding: 2px 10px; font-size: 0.7rem;
        letter-spacing: 0.04em; text-transform: uppercase; margin-right: 6px;
    }}

    /* KPI cards */
    .lc-kpi {{
        background: {COLORS['surface']};
        border: 1px solid {COLORS['line']};
        border-radius: 12px;
        padding: 14px 16px 12px 16px;
        height: 100%;
        box-shadow: 0 1px 2px rgba(17,24,52,0.04);
    }}
    .lc-kpi .label {{
        font-size: 0.72rem; letter-spacing: 0.06em; text-transform: uppercase;
        color: {COLORS['faint']}; font-weight: 600; margin-bottom: 6px;
    }}
    .lc-kpi .value {{
        font-size: 1.72rem; font-weight: 680; color: {COLORS['ink']};
        line-height: 1.1; letter-spacing: -0.02em;
    }}
    .lc-kpi .value.small {{ font-size: 1.28rem; }}
    .lc-kpi .sub {{ font-size: 0.78rem; color: {COLORS['muted']}; margin-top: 4px; }}
    .lc-kpi .bar {{ height: 4px; border-radius: 2px; background: {COLORS['line']}; margin-top: 9px; overflow: hidden; }}
    .lc-kpi .bar > span {{ display: block; height: 100%; border-radius: 2px; }}

    /* Empty / info states */
    .lc-empty {{
        background: {COLORS['surface']};
        border: 1px dashed {COLORS['line']};
        border-radius: 12px; padding: 34px 24px; text-align: center;
        color: {COLORS['muted']};
    }}
    .lc-empty strong {{ color: {COLORS['ink']}; display: block; margin-bottom: 6px; font-size: 1rem; }}
    .lc-empty code {{
        background: {COLORS['page']}; border: 1px solid {COLORS['line']};
        border-radius: 6px; padding: 2px 7px; color: {COLORS['primary']};
    }}

    .lc-note {{
        background: {COLORS['primary_soft']}; border-left: 3px solid {COLORS['primary']};
        border-radius: 0 8px 8px 0; padding: 10px 14px; font-size: 0.84rem;
        color: #1E2A6E; margin: 8px 0 14px 0;
    }}
    .lc-warn {{
        background: #FFF7E6; border-left: 3px solid #E0A100;
        border-radius: 0 8px 8px 0; padding: 10px 14px; font-size: 0.84rem;
        color: #6B4E00; margin: 8px 0 14px 0;
    }}
    .lc-danger {{
        background: #FDECEE; border-left: 3px solid #D64550;
        border-radius: 0 8px 8px 0; padding: 10px 14px; font-size: 0.84rem;
        color: #7A1F27; margin: 8px 0 14px 0;
    }}

    .lc-section {{
        font-size: 0.78rem; font-weight: 700; letter-spacing: 0.08em;
        text-transform: uppercase; color: {COLORS['faint']};
        margin: 22px 0 10px 0;
    }}

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {{ gap: 4px; }}
    .stTabs [data-baseweb="tab"] {{
        border-radius: 8px 8px 0 0; padding: 8px 14px; font-size: 0.88rem;
    }}

    /* Dataframe chrome */
    [data-testid="stDataFrame"] {{ border: 1px solid {COLORS['line']}; border-radius: 10px; }}

    /* Sidebar */
    section[data-testid="stSidebar"] {{ background: {COLORS['surface']}; border-right: 1px solid {COLORS['line']}; }}
</style>
"""


def setup_page(title: str, subtitle: str, tags: Sequence[str] = ()) -> None:
    """Configure the page, inject CSS, and draw the header band."""
    st.set_page_config(
        page_title=f"{title} | Learner Churn Early-Warning",
        page_icon="◧",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(_CSS, unsafe_allow_html=True)
    tag_html = "".join(f'<span class="lc-tag">{t}</span>' for t in tags)
    st.markdown(
        f"""
        <div class="lc-header">
            <div>{tag_html}</div>
            <h1 style="margin-top:8px;">{title}</h1>
            <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Chart helpers
# ---------------------------------------------------------------------------
def style_fig(fig: go.Figure, height: int = 320, legend: bool = True) -> go.Figure:
    """Apply the house chart style. One place, so every chart matches."""
    fig.update_layout(
        height=height,
        margin=dict(l=8, r=8, t=34, b=8),
        paper_bgcolor=COLORS["surface"],
        plot_bgcolor=COLORS["surface"],
        font=dict(family=PLOTLY_FONT, size=12, color=COLORS["ink"]),
        colorway=SERIES_SEQUENCE,
        hoverlabel=dict(
            bgcolor=COLORS["ink"],
            font=dict(color="#FFFFFF", size=12, family=PLOTLY_FONT),
            bordercolor=COLORS["ink"],
        ),
        showlegend=legend,
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
            font=dict(size=11, color=COLORS["muted"]), bgcolor="rgba(0,0,0,0)",
        ),
    )
    fig.update_xaxes(
        showgrid=False, zeroline=False, linecolor=COLORS["line"],
        tickfont=dict(size=11, color=COLORS["muted"]), title_font=dict(size=11),
    )
    fig.update_yaxes(
        gridcolor=COLORS["line"], zeroline=False, linecolor="rgba(0,0,0,0)",
        tickfont=dict(size=11, color=COLORS["muted"]), title_font=dict(size=11),
    )
    return fig


def kpi(
    label: str,
    value: str,
    sub: str = "",
    accent: str | None = None,
    bar_fraction: float | None = None,
    small: bool = False,
) -> str:
    """HTML for one KPI card. Returns a string so it can be grid-placed."""
    accent = accent or COLORS["primary"]
    bar = ""
    if bar_fraction is not None:
        pct = max(0.0, min(1.0, bar_fraction)) * 100
        bar = f'<div class="bar"><span style="width:{pct:.1f}%;background:{accent};"></span></div>'
    size_class = "value small" if small else "value"
    return (
        f'<div class="lc-kpi">'
        f'<div class="label">{label}</div>'
        f'<div class="{size_class}">{value}</div>'
        f'<div class="sub">{sub}</div>{bar}</div>'
    )


def kpi_row(cards: Sequence[str], gap: str = "small") -> None:
    """Lay KPI cards out in equal columns."""
    cols = st.columns(len(cards), gap=gap)
    for col, card in zip(cols, cards):
        with col:
            st.markdown(card, unsafe_allow_html=True)


def empty_state(title: str, body: str, command: str | None = None) -> None:
    cmd = f"<br><code>{command}</code>" if command else ""
    st.markdown(
        f'<div class="lc-empty"><strong>{title}</strong>{body}{cmd}</div>',
        unsafe_allow_html=True,
    )


def note(text: str, kind: str = "note") -> None:
    css = {"note": "lc-note", "warn": "lc-warn", "danger": "lc-danger"}.get(kind, "lc-note")
    st.markdown(f'<div class="{css}">{text}</div>', unsafe_allow_html=True)


def section(title: str) -> None:
    st.markdown(f'<div class="lc-section">{title}</div>', unsafe_allow_html=True)


def rel_path(path: Path) -> str:
    """Path relative to the project root when possible, else the full path.

    ``Path.relative_to`` raises when the target sits outside the project (an
    env var pointing the intervention database at a temp directory, say).
    Display code must never crash for that reason.
    """
    path = Path(path)
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def fmt_int(value: Any) -> str:
    try:
        return f"{int(value):,}"
    except (TypeError, ValueError):
        return "—"


def fmt_pct(value: Any, digits: int = 1) -> str:
    if value is None:
        return "—"
    try:
        return f"{float(value) * 100:.{digits}f}%"
    except (TypeError, ValueError):
        return "—"


def fmt_num(value: Any, digits: int = 3) -> str:
    if value is None:
        return "not available"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "not available"


# ---------------------------------------------------------------------------
# Sidebar: status + global filters
# ---------------------------------------------------------------------------
def sidebar_status() -> None:
    with st.sidebar:
        st.markdown("#### Pipeline status")
        status = loader.pipeline_status()
        for name, present in status.items():
            st.markdown(
                f"{'🟢' if present else '⚪'} <span style='font-size:0.82rem'>{name}</span>",
                unsafe_allow_html=True,
            )
        if not all(status.values()):
            st.caption("Run `python -m src.pipeline` to build everything.")
        st.divider()


def global_filters(snapshot: pd.DataFrame) -> Dict[str, Any]:
    """Course / presentation / cutoff / target / threshold controls.

    Returns the selection; every page applies it to its own data so filters
    genuinely change what is displayed.
    """
    courses, presentations = loader.available_cohorts(snapshot)
    with st.sidebar:
        st.markdown("#### Cohort filters")
        course = st.selectbox("Course", ["All courses"] + courses, index=0)
        if course == "All courses":
            pres_options = presentations
        else:
            pres_options = sorted(
                snapshot.loc[snapshot["code_module"] == course, "code_presentation"]
                .dropna()
                .unique()
                .tolist()
            )
        presentation = st.selectbox("Presentation", ["All presentations"] + pres_options, index=0)

        cutoff = st.selectbox(
            "Prediction cutoff week",
            list(config.SUPPORTED_CUTOFF_WEEKS),
            index=list(config.SUPPORTED_CUTOFF_WEEKS).index(config.DEFAULT_CUTOFF_WEEK),
            help=(
                "Features use only records observed on or before the last day of "
                "this week. The model artifact is trained for week "
                f"{config.DEFAULT_CUTOFF_WEEK}."
            ),
        )

        target_key = st.selectbox(
            "Target definition",
            list(config.TARGETS),
            format_func=lambda k: config.TARGETS[k]["label"],
            index=list(config.TARGETS).index(config.DEFAULT_TARGET),
            help="Changing this changes which learners count as the positive class.",
        )

        low = st.slider("Low / Medium risk boundary", 0.05, 0.60, config.DEFAULT_RISK_BANDS["low"], 0.01)
        high = st.slider(
            "Medium / High risk boundary", max(low + 0.05, 0.30), 0.95,
            max(config.DEFAULT_RISK_BANDS["high"], low + 0.05), 0.01,
        )
    return {
        "course": course,
        "presentation": presentation,
        "cutoff_week": int(cutoff),
        "target_key": target_key,
        "risk_bands": {"low": float(low), "high": float(high)},
    }


def apply_cohort_filter(
    df: pd.DataFrame, course: str, presentation: str
) -> pd.DataFrame:
    out = df
    if course != "All courses" and "code_module" in out.columns:
        out = out[out["code_module"] == course]
    if presentation != "All presentations" and "code_presentation" in out.columns:
        out = out[out["code_presentation"] == presentation]
    return out


def load_snapshot_or_stop() -> pd.DataFrame:
    """Snapshot, or an empty state that tells the user what to run."""
    result = loader.learner_snapshot()
    if not result.ok:
        setup_page(
            "Learner Churn Early-Warning",
            "Cohort outcomes, engagement drop-off and an explainable at-risk queue.",
            tags=["EdTech", "OULAD", "SQL + Python"],
        )
        sidebar_status()
        empty_state(
            "No feature snapshot found",
            "The pipeline has not been run against the OULAD dataset yet.",
            "python -m src.pipeline",
        )
        st.stop()
    return result.value


def require_artifact(result, title: str, body: str, command: str | None = None):
    """Render an empty state and stop if an artifact is missing."""
    if not result.ok:
        empty_state(title, body or (result.reason or ""), command)
        st.stop()
    return result.value


def risk_band_series(scores: pd.Series, bands: Dict[str, float]) -> pd.Series:
    return assign_risk_bands(scores, bands)
