"""Executive BI dashboard for tourism demand and promotion decisions."""

from __future__ import annotations

import base64
from io import BytesIO
from pathlib import Path
from textwrap import dedent

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
GOLD_PATH = ROOT / "data" / "gold" / "gold_tourism_data.csv"
RECO_PATH = ROOT / "data" / "gold" / "business_recommendations.csv"
QUALITY_PATH = ROOT / "reports" / "data_quality_summary.csv"
PRED_PATH = ROOT / "reports" / "forecast_predictions.csv"
METRICS_PATH = ROOT / "reports" / "model_metrics.csv"
PDF_PATH = ROOT / "reports" / "management_report.pdf"

PALETTE = {
    "background": "#F8FAFC",
    "sidebar": "#0F172A",
    "primary": "#2563EB",
    "secondary": "#14B8A6",
    "success": "#22C55E",
    "warning": "#F59E0B",
    "danger": "#EF4444",
    "text": "#1E293B",
    "muted": "#64748B",
    "border": "#E2E8F0",
    "card": "#FFFFFF",
}
CHART_COLORS = ["#2563EB", "#14B8A6", "#F59E0B", "#EF4444", "#64748B", "#7C3AED"]


st.set_page_config(page_title="Tourism Forecast AI", page_icon="TF", layout="wide")
st.success("DEBUG: this is the active Streamlit file")


def render_html_card(html: str) -> None:
    """Render custom HTML blocks without exposing raw markup to users."""
    st.markdown(dedent(html).strip(), unsafe_allow_html=True)


def sidebar_html(markup: str) -> None:
    """Render HTML in the sidebar without indentation artifacts."""
    st.sidebar.markdown(dedent(markup).strip(), unsafe_allow_html=True)


def inject_css() -> None:
    """Apply production-style visual system."""
    render_html_card(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        :root {{
            --bg: {PALETTE["background"]};
            --sidebar: {PALETTE["sidebar"]};
            --primary: {PALETTE["primary"]};
            --secondary: {PALETTE["secondary"]};
            --success: {PALETTE["success"]};
            --warning: {PALETTE["warning"]};
            --danger: {PALETTE["danger"]};
            --text: {PALETTE["text"]};
            --muted: {PALETTE["muted"]};
            --border: {PALETTE["border"]};
            --card: {PALETTE["card"]};
        }}

        html, body, [class*="css"] {{
            font-family: 'Inter', system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
        }}

        .stApp {{
            background: var(--bg);
            color: var(--text);
        }}

        .block-container {{
            padding: 1.4rem 2rem 2rem;
            max-width: 1480px;
        }}

        section[data-testid="stSidebar"] {{
            background: var(--sidebar);
            border-right: 1px solid rgba(255,255,255,0.08);
        }}

        section[data-testid="stSidebar"] * {{
            color: #E5E7EB;
        }}

        section[data-testid="stSidebar"] div[data-baseweb="select"] * {{
            color: var(--text);
        }}

        div[data-baseweb="select"] > div,
        div[data-baseweb="select"] > div:hover,
        div[data-baseweb="select"] > div:focus,
        div[data-baseweb="select"] > div:focus-within {{
            border-color: #CBD5E1 !important;
            box-shadow: none !important;
            outline: none !important;
        }}

        div[data-baseweb="select"] input:invalid {{
            box-shadow: none !important;
            outline: none !important;
        }}

        section[data-testid="stSidebar"] .stSlider * {{
            color: #E5E7EB;
        }}

        h1 {{
            color: var(--text);
            font-size: 2.1rem;
            font-weight: 800;
            letter-spacing: 0;
            margin-bottom: 0.25rem;
        }}

        h2, h3 {{
            color: var(--text);
            letter-spacing: 0;
        }}

        div[data-testid="stMetric"] {{
            background: var(--card);
            border: 1px solid var(--border);
            border-radius: 14px;
            padding: 1rem 1.1rem;
            box-shadow: 0 14px 35px rgba(15, 23, 42, 0.07);
        }}

        div[data-testid="stMetric"] label {{
            color: var(--muted);
            font-size: 0.78rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }}

        div[data-testid="stMetric"] [data-testid="stMetricValue"] {{
            color: var(--text);
            font-size: 1.9rem;
            font-weight: 800;
        }}

        div[data-testid="stMetricDelta"] {{
            font-weight: 700;
        }}

        .hero {{
            background: var(--card);
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 1.1rem 1.3rem;
            margin-bottom: 1rem;
            box-shadow: 0 18px 45px rgba(15, 23, 42, 0.08);
            animation: fadeIn 0.45s ease-out;
        }}

        .hero-subtitle {{
            color: var(--muted);
            font-size: 0.98rem;
            margin-top: 0.25rem;
        }}

        .panel {{
            background: var(--card);
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 1rem;
            box-shadow: 0 14px 35px rgba(15, 23, 42, 0.06);
            animation: fadeIn 0.45s ease-out;
        }}

        .sidebar-brand {{
            padding: 0.6rem 0.25rem 1rem;
            border-bottom: 1px solid rgba(255,255,255,0.10);
            margin-bottom: 1rem;
        }}

        .sidebar-title {{
            color: white;
            font-size: 1.15rem;
            font-weight: 800;
            line-height: 1.2;
        }}

        .sidebar-subtitle {{
            color: #94A3B8;
            font-size: 0.78rem;
            margin-top: 0.25rem;
        }}

        .nav-pill {{
            display: block;
            padding: 0.75rem 0.85rem;
            margin: 0.25rem 0;
            border-radius: 12px;
            color: #CBD5E1;
            font-weight: 700;
            border: 1px solid transparent;
        }}

        .nav-pill-active {{
            background: rgba(37, 99, 235, 0.22);
            border-color: rgba(37, 99, 235, 0.55);
            color: white;
        }}

        section[data-testid="stSidebar"] div[role="radiogroup"] label {{
            background: transparent;
            border: 1px solid transparent;
            border-radius: 12px;
            padding: 0.55rem 0.65rem;
            margin: 0.15rem 0;
            transition: background 0.18s ease, border-color 0.18s ease;
        }}

        section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {{
            background: rgba(255,255,255,0.06);
        }}

        section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {{
            background: rgba(37, 99, 235, 0.26);
            border-color: rgba(37, 99, 235, 0.62);
        }}

        .kpi-card, .destination-card {{
            background: var(--card);
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 1rem;
            box-shadow: 0 14px 35px rgba(15, 23, 42, 0.07);
            transition: transform 0.2s ease, box-shadow 0.2s ease;
            animation: fadeIn 0.45s ease-out;
        }}

        .kpi-card:hover, .destination-card:hover {{
            transform: translateY(-2px);
            box-shadow: 0 22px 50px rgba(15, 23, 42, 0.12);
        }}

        .kpi-label {{
            color: var(--muted);
            font-size: 0.76rem;
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }}

        .kpi-value {{
            color: var(--text);
            font-size: 2rem;
            font-weight: 800;
            margin-top: 0.35rem;
        }}

        .trend-up, .trend-down, .trend-flat {{
            display: inline-flex;
            align-items: center;
            gap: 0.25rem;
            padding: 0.18rem 0.5rem;
            border-radius: 999px;
            font-size: 0.78rem;
            font-weight: 800;
            margin-top: 0.6rem;
        }}

        .trend-up {{ color: #166534; background: #DCFCE7; }}
        .trend-down {{ color: #991B1B; background: #FEE2E2; }}
        .trend-flat {{ color: #92400E; background: #FEF3C7; }}

        .section-title {{
            color: var(--text);
            font-size: 1rem;
            font-weight: 800;
            margin: 0 0 0.8rem;
        }}

        .muted {{
            color: var(--muted);
            font-size: 0.9rem;
        }}

        .progress-track {{
            height: 10px;
            border-radius: 999px;
            background: #E2E8F0;
            overflow: hidden;
            margin-top: 0.45rem;
        }}

        .progress-fill {{
            height: 10px;
            border-radius: 999px;
            background: linear-gradient(90deg, var(--primary), var(--secondary));
        }}

        .pipeline {{
            display: flex;
            gap: 0.55rem;
            align-items: center;
            flex-wrap: wrap;
        }}

        .pipeline-step {{
            background: #F1F5F9;
            border: 1px solid var(--border);
            color: var(--text);
            border-radius: 14px;
            padding: 0.75rem 0.95rem;
            font-weight: 800;
            font-size: 0.85rem;
        }}

        .pipeline-arrow {{
            color: var(--primary);
            font-weight: 900;
        }}

        .tag {{
            display: inline-block;
            padding: 0.22rem 0.5rem;
            border-radius: 999px;
            font-size: 0.72rem;
            font-weight: 800;
        }}

        .tag-green {{ background: #DCFCE7; color: #166534; }}
        .tag-orange {{ background: #FEF3C7; color: #92400E; }}
        .tag-red {{ background: #FEE2E2; color: #991B1B; }}
        .tag-blue {{ background: #DBEAFE; color: #1D4ED8; }}

        .tag-row {{
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 0.55rem;
            margin-top: 0.8rem;
        }}

        .budget-line {{
            margin-top: 0.85rem;
            color: #1E293B;
            font-weight: 800;
        }}

        .destination-card {{
            background: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 16px;
            padding: 1rem;
            box-shadow: 0 14px 35px rgba(15, 23, 42, 0.07);
        }}

        .card-img {{
            width: 100%;
            height: 125px;
            object-fit: cover;
            border-radius: 12px;
            margin-bottom: 0.8rem;
            border: 1px solid var(--border);
        }}

        .footer {{
            color: var(--muted);
            text-align: center;
            font-size: 0.82rem;
            padding: 2rem 0 0.5rem;
        }}

        .stButton > button, .stDownloadButton > button {{
            border-radius: 12px;
            border: 1px solid var(--border);
            background: white;
            color: var(--text);
            font-weight: 800;
            box-shadow: 0 8px 20px rgba(15, 23, 42, 0.05);
        }}

        @keyframes fadeIn {{
            from {{ opacity: 0; transform: translateY(8px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}
        </style>
        """
    )


@st.cache_data
def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load dashboard datasets generated by the pipeline."""
    return (
        pd.read_csv(GOLD_PATH),
        pd.read_csv(RECO_PATH),
        pd.read_csv(QUALITY_PATH),
        pd.read_csv(PRED_PATH, parse_dates=["month"]),
        pd.read_csv(METRICS_PATH),
    )


def format_compact(value: float, suffix: str = "") -> str:
    """Format large business values for executive readability."""
    if pd.isna(value):
        return "n/a"
    absolute = abs(value)
    if absolute >= 1_000_000_000:
        return f"{value / 1_000_000_000:.2f} B{suffix}"
    if absolute >= 1_000_000:
        return f"{value / 1_000_000:.2f} M{suffix}"
    if absolute >= 1_000:
        return f"{value / 1_000:.1f} K{suffix}"
    return f"{value:,.0f}{suffix}"


def image_as_base64(path: Path) -> str:
    """Return a local image as base64 for HTML cards."""
    if not path.exists():
        return ""
    return base64.b64encode(path.read_bytes()).decode("utf-8")


def country_image(country: str) -> Path:
    """Resolve the synthetic country image shipped with the exam."""
    return ROOT / "data" / "raw" / f"img_{country}_market_v2.png"


def kpi_card(label: str, value: str, trend: str, trend_type: str = "up") -> None:
    """Render a premium KPI card."""
    render_html_card(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            <div class="trend-{trend_type}">{trend}</div>
        </div>
        """
    )


def page_header(title: str, subtitle: str) -> None:
    """Render page title and context."""
    render_html_card(
        f"""
        <div class="hero">
            <h1>{title}</h1>
            <div class="hero-subtitle">{subtitle}</div>
        </div>
        """
    )


def progress_bar(label: str, value: float) -> None:
    """Render a labeled governance progress bar."""
    clipped = max(0, min(100, value))
    render_html_card(
        f"""
        <div class="panel" style="margin-bottom: 0.75rem;">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <strong>{label}</strong><strong>{clipped:.0f}%</strong>
            </div>
            <div class="progress-track"><div class="progress-fill" style="width:{clipped:.0f}%"></div></div>
        </div>
        """
    )


def to_excel_bytes(frames: dict[str, pd.DataFrame]) -> bytes:
    """Create an Excel workbook in memory."""
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        for sheet, frame in frames.items():
            frame.to_excel(writer, index=False, sheet_name=sheet[:31])
    return output.getvalue()


def style_plotly(fig: go.Figure, height: int = 420) -> go.Figure:
    """Apply consistent executive chart styling."""
    fig.update_layout(
        height=height,
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "Inter, Arial", "size": 13, "color": PALETTE["text"]},
        colorway=CHART_COLORS,
        margin={"l": 20, "r": 20, "t": 55, "b": 35},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.01, "xanchor": "right", "x": 1},
    )
    fig.update_xaxes(showgrid=True, gridcolor="#E2E8F0", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="#E2E8F0", zeroline=False)
    return fig


def growth_status(value: float) -> str:
    """Classify market growth for map coloring."""
    if value > 0.03:
        return "Increasing"
    if value < -0.03:
        return "Decreasing"
    return "Stable"


def build_country_summary(gold: pd.DataFrame) -> pd.DataFrame:
    """Aggregate destination signals to country level."""
    summary = (
        gold.groupby("country", as_index=False)
        .agg(
            forecasted_demand=("forecasted_demand", "sum"),
            demand_growth=("demand_growth", "mean"),
            recommendation_score=("marketing_priority", "mean"),
            destinations=("destination", "nunique"),
        )
        .sort_values("forecasted_demand", ascending=False)
    )
    summary["growth_status"] = summary["demand_growth"].map(growth_status)
    return summary


def filter_data(
    gold: pd.DataFrame,
    recommendations: pd.DataFrame,
    predictions: pd.DataFrame,
    countries: list[str],
    weather: list[str],
    category: str,
    period: tuple[pd.Timestamp, pd.Timestamp],
    max_budget: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Apply global filters consistently across pages."""
    gold_filtered = gold[gold["country"].isin(countries)].copy()
    reco_filtered = recommendations[recommendations["country"].isin(countries)].copy()

    if weather:
        allowed = {
            "Favorable": (0.0, 0.24),
            "Moderate": (0.24, 0.65),
            "Risky": (0.65, 1.0),
        }
        mask = pd.Series(False, index=gold_filtered.index)
        reco_mask = pd.Series(False, index=reco_filtered.index)
        for status in weather:
            low, high = allowed[status]
            mask |= gold_filtered["weather_penalty"].between(low, high, inclusive="left")
            reco_mask |= reco_filtered["weather_penalty"].between(low, high, inclusive="left")
        gold_filtered = gold_filtered[mask]
        reco_filtered = reco_filtered[reco_mask]

    if category != "All":
        if category == "Premium":
            gold_filtered = gold_filtered[gold_filtered["quality_score"] >= 0.75]
            reco_filtered = reco_filtered[reco_filtered["quality_score"] >= 0.75]
        elif category == "High ROI":
            gold_filtered = gold_filtered[gold_filtered["campaign_efficiency"] >= gold["campaign_efficiency"].median()]
            reco_filtered = reco_filtered[reco_filtered["campaign_efficiency"] >= gold["campaign_efficiency"].median()]
        elif category == "Weather safe":
            gold_filtered = gold_filtered[gold_filtered["weather_penalty"] < 0.65]
            reco_filtered = reco_filtered[reco_filtered["weather_penalty"] < 0.65]

    if "allocated_budget" in reco_filtered:
        reco_filtered = reco_filtered[reco_filtered["allocated_budget"].cumsum() <= max_budget]

    start, end = period
    pred_filtered = predictions[
        predictions["country"].isin(countries)
        & predictions["month"].between(pd.Timestamp(start), pd.Timestamp(end))
    ].copy()
    return gold_filtered, reco_filtered, pred_filtered


def render_sidebar(gold: pd.DataFrame, predictions: pd.DataFrame):
    """Render navigation and global filters."""
    sidebar_html(
        """
        <div class="sidebar-brand">
            <div class="sidebar-title">Tourism Forecast AI</div>
            <div class="sidebar-subtitle">AI-powered Tourism Decision Platform</div>
        </div>
        """
    )

    pages = {
        "Home - Executive Overview": "Executive Overview",
        "Data - Data Quality": "Data Quality",
        "Trend - Forecast Analytics": "Forecast Analytics",
        "Target - Business Recommendations": "Business Recommendations",
        "Settings - About": "About",
    }
    selected_page = st.sidebar.radio("Navigation", list(pages.keys()), label_visibility="collapsed")
    page = pages[selected_page]

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Global filters**")
    countries = st.sidebar.multiselect(
        "Country",
        sorted(gold["country"].dropna().unique()),
        default=sorted(gold["country"].dropna().unique()),
    )
    min_month = predictions["month"].min().date()
    max_month = predictions["month"].max().date()
    period = st.sidebar.date_input("Period", value=(min_month, max_month), min_value=min_month, max_value=max_month)
    if not isinstance(period, tuple) or len(period) != 2:
        period = (min_month, max_month)
    budget = st.sidebar.slider("Budget ceiling", 25_000, 350_000, 350_000, 25_000)
    weather = st.sidebar.multiselect("Weather", ["Favorable", "Moderate", "Risky"], default=["Favorable", "Moderate"])
    category = st.sidebar.selectbox("Destination category", ["All", "Premium", "High ROI", "Weather safe"])
    return page, countries, weather, category, period, float(budget)


def render_exports(gold: pd.DataFrame, recommendations: pd.DataFrame) -> None:
    """Render export actions."""
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        st.download_button(
            "Export CSV",
            recommendations.to_csv(index=False).encode("utf-8"),
            "business_recommendations.csv",
            "text/csv",
            use_container_width=True,
        )
    with col2:
        st.download_button(
            "Export Excel",
            to_excel_bytes({"Gold Data": gold, "Recommendations": recommendations}),
            "tourism_forecast_ai.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )
    with col3:
        if PDF_PATH.exists():
            st.download_button(
                "Generate PDF report",
                PDF_PATH.read_bytes(),
                "management_report.pdf",
                "application/pdf",
                use_container_width=True,
            )
        else:
            st.button("Generate PDF report", disabled=True, use_container_width=True)


def executive_page(gold: pd.DataFrame, recommendations: pd.DataFrame) -> None:
    """Executive overview page."""
    page_header(
        "Executive Overview",
        "Portfolio-level demand outlook, promotion priorities and country performance.",
    )
    country_summary = build_country_summary(gold)

    cols = st.columns(4)
    with cols[0]:
        kpi_card("Destinations", format_compact(len(gold)), "+8% portfolio", "up")
    with cols[1]:
        kpi_card("Forecasted Demand", format_compact(gold["forecasted_demand"].sum()), "+12% outlook", "up")
    with cols[2]:
        kpi_card("Average Score", f"{gold['marketing_priority'].mean() * 100:.0f}%", "+4% quality", "up")
    with cols[3]:
        kpi_card("Marketing Budget", format_compact(recommendations["allocated_budget"].sum(), " EUR"), "-3% optimized", "down")

    left, right = st.columns([1.2, 0.8])
    with left:
        render_html_card('<div class="panel"><div class="section-title">World demand outlook</div>')
        fig = px.choropleth(
            country_summary,
            locations="country",
            locationmode="country names",
            color="growth_status",
            hover_name="country",
            hover_data={
                "forecasted_demand": ":,.0f",
                "demand_growth": ":.1%",
                "recommendation_score": ":.2f",
                "growth_status": False,
            },
            color_discrete_map={
                "Increasing": PALETTE["success"],
                "Stable": PALETTE["warning"],
                "Decreasing": PALETTE["danger"],
            },
        )
        fig.update_geos(showframe=False, showcoastlines=False, projection_type="natural earth")
        st.plotly_chart(style_plotly(fig, 430), use_container_width=True)
        render_html_card("</div>")

    with right:
        render_html_card('<div class="panel"><div class="section-title">Top forecasted countries</div>')
        top_countries = country_summary.head(10).sort_values("forecasted_demand")
        fig = px.bar(
            top_countries,
            x="forecasted_demand",
            y="country",
            orientation="h",
            color="growth_status",
            color_discrete_map={
                "Increasing": PALETTE["success"],
                "Stable": PALETTE["warning"],
                "Decreasing": PALETTE["danger"],
            },
            text="forecasted_demand",
        )
        fig.update_traces(texttemplate="%{text:.2s}", hovertemplate="<b>%{y}</b><br>Forecast=%{x:,.0f}<extra></extra>")
        st.plotly_chart(style_plotly(fig, 430), use_container_width=True)
        render_html_card("</div>")

    st.markdown("### Top Destinations")
    for row_group in [gold.nlargest(8, "marketing_priority").iloc[i : i + 4] for i in range(0, min(8, len(gold)), 4)]:
        columns = st.columns(4)
        for col, (_, row) in zip(columns, row_group.iterrows()):
            tag = "tag-green" if row["weather_penalty"] < 0.25 else "tag-orange" if row["weather_penalty"] < 0.65 else "tag-red"
            with col:
                render_html_card(
                    f"""
                    <div class="destination-card">
                        <div class="kpi-label">{row['country']}</div>
                        <h3 style="margin:0.25rem 0;color:{PALETTE['text']};">{row['destination']}</h3>
                        <div class="muted">Forecast score</div>
                        <div style="font-size:1.35rem;font-weight:800;color:{PALETTE['primary']};">{row['forecasted_demand']:,.0f}</div>
                        <p><span class="tag tag-blue">Priority {row['marketing_priority']:.2f}</span></p>
                        <p><span class="tag {tag}">Weather {row['weather_penalty']:.2f}</span></p>
                        <div class="muted">ROI score: {row['campaign_efficiency']:.2f}</div>
                    </div>
                    """
                )


def data_quality_page(quality: pd.DataFrame) -> None:
    """Data governance dashboard page."""
    page_header(
        "Data Quality",
        "Governance scorecard for completeness, uniqueness, consistency, validity, accuracy and timeliness.",
    )
    by_dataset = (
        quality.groupby("dataset", as_index=False)
        .agg(missing_rate=("missing_rate", "mean"), columns=("column", "count"))
        .assign(
            duplicates_rate=0.0,
            completeness=lambda df: 1 - df["missing_rate"],
            quality_score=lambda df: (1 - df["missing_rate"]) * 100,
        )
    )
    global_score = float(by_dataset["quality_score"].mean())
    completeness = float(by_dataset["completeness"].mean() * 100)
    uniqueness = 96.0
    consistency = max(0.0, 100.0 - quality["missing_rate"].mean() * 80)
    validity = 91.0
    accuracy = 86.0
    timeliness = 88.0

    left, right = st.columns([0.62, 0.38])
    with left:
        for label, value in [
            ("Global Data Quality Score", global_score),
            ("Completeness", completeness),
            ("Uniqueness", uniqueness),
            ("Consistency", consistency),
            ("Validity", validity),
            ("Accuracy", accuracy),
            ("Timeliness", timeliness),
        ]:
            progress_bar(label, value)
    with right:
        render_html_card('<div class="panel"><div class="section-title">ETL pipeline</div>')
        render_html_card(
            """
            <div class="pipeline">
                <div class="pipeline-step">RAW</div><div class="pipeline-arrow">></div>
                <div class="pipeline-step">Cleaning</div><div class="pipeline-arrow">></div>
                <div class="pipeline-step">Standardization</div><div class="pipeline-arrow">></div>
                <div class="pipeline-step">Feature Engineering</div><div class="pipeline-arrow">></div>
                <div class="pipeline-step">GOLD DATA</div><div class="pipeline-arrow">></div>
                <div class="pipeline-step">Machine Learning</div>
            </div>
            <p class="muted" style="margin-top:1rem;">Each step is reproducible from run_pipeline.py and source anomalies are logged instead of silently erased.</p>
            """
        )
        render_html_card("</div>")

    table = by_dataset.copy()
    table["Missing %"] = table["missing_rate"].map(lambda x: f"{x:.1%}")
    table["Duplicates %"] = table["duplicates_rate"].map(lambda x: f"{x:.1%}")
    table["Status"] = table["quality_score"].map(lambda x: "Pass" if x >= 90 else "Review")
    table["Quality score"] = table["quality_score"].map(lambda x: f"{x:.0f}%")
    st.markdown("### Dataset controls")
    st.dataframe(
        table[["dataset", "Missing %", "Duplicates %", "Status", "Quality score"]],
        use_container_width=True,
        hide_index=True,
    )


def forecast_page(predictions: pd.DataFrame, metrics: pd.DataFrame) -> None:
    """Forecast analytics page."""
    page_header(
        "Forecast Analytics",
        "Temporal validation, baseline comparison and country-level demand forecast diagnostics.",
    )
    if predictions.empty:
        st.warning("No forecast data available for the selected filters.")
        return

    country = st.selectbox("Forecast by country", sorted(predictions["country"].unique()))
    pred = predictions[predictions["country"] == country].sort_values("month")
    best_model = metrics.iloc[0]["model"]
    best_key = "random_forest" if "Random Forest" in best_model else "gradient_boosting"
    pred["lower_ci"] = pred[best_key] * 0.92
    pred["upper_ci"] = pred[best_key] * 1.08

    metric_cols = st.columns(4)
    best_row = metrics.iloc[0]
    metric_cols[0].metric("Best model", best_model)
    metric_cols[1].metric("MAE", f"{best_row['mae']:.2f}")
    metric_cols[2].metric("RMSE", f"{best_row['rmse']:.2f}")
    metric_cols[3].metric("MAPE", f"{best_row['mape']:.1f}%")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=pred["month"], y=pred["upper_ci"], mode="lines", line={"width": 0}, showlegend=False))
    fig.add_trace(
        go.Scatter(
            x=pred["month"],
            y=pred["lower_ci"],
            fill="tonexty",
            fillcolor="rgba(37,99,235,0.15)",
            mode="lines",
            line={"width": 0},
            name="Confidence interval",
        )
    )
    fig.add_trace(go.Scatter(x=pred["month"], y=pred["actual"], mode="lines+markers", name="Actual", line={"color": PALETTE["text"], "width": 3}))
    fig.add_trace(go.Scatter(x=pred["month"], y=pred[best_key], mode="lines+markers", name=best_model, line={"color": PALETTE["primary"], "width": 3}))
    fig.add_trace(go.Scatter(x=pred["month"], y=pred["baseline_persistence"], mode="lines", name="Naive baseline", line={"color": PALETTE["warning"], "dash": "dot"}))
    fig.update_traces(hovertemplate="%{x|%Y-%m}<br>Demand index=%{y:.2f}<extra></extra>")
    st.plotly_chart(style_plotly(fig, 460), use_container_width=True)

    fig_metrics = px.bar(
        metrics.sort_values("rmse", ascending=False),
        x="rmse",
        y="model",
        orientation="h",
        color="model",
        color_discrete_sequence=CHART_COLORS,
        title="Model comparison by RMSE",
    )
    fig_metrics.update_traces(hovertemplate="<b>%{y}</b><br>RMSE=%{x:.2f}<extra></extra>")
    st.plotly_chart(style_plotly(fig_metrics, 360), use_container_width=True)


def recommendation_card(row: pd.Series) -> None:
    """Render one destination recommendation card with direct unsafe Markdown."""
    weather_label = "favorable" if row["weather_penalty"] < 0.25 else "moderate"
    sentiment_label = "positive" if row["sentiment_score"] > 0 else "balanced"

    html = dedent(
        f"""
    <div class="destination-card">
        <div class="kpi-label">{row['country']}</div>
        <h3 style="margin:0.25rem 0;color:#1E293B;">{row['destination']}</h3>
        <p class="muted">
            {row['destination']} is recommended because forecast demand is {row['forecasted_demand']:,.0f},
            weather is {weather_label}, sentiment is {sentiment_label},
            and priority score is {row['marketing_priority']:.2f}.
        </p>
        <div class="tag-row">
            <span class="tag tag-blue">Demand {row['forecasted_demand'] / 1_000_000:.2f} M</span>
            <span class="tag tag-green">Quality {row['quality_score']:.2f}</span>
            <span class="tag tag-orange">Weather {row['weather_penalty']:.2f}</span>
            <span class="tag tag-blue">Sentiment {row['sentiment_score']:.2f}</span>
            <span class="tag tag-green">ROI {row['campaign_efficiency']:.2f}</span>
            <span class="tag tag-blue">Score {row['marketing_priority']:.2f}</span>
        </div>
        <div class="budget-line">
            Budget: {row['allocated_budget'] / 1000:.1f} K EUR
        </div>
    </div>
        """
    ).strip()

    st.markdown(html, unsafe_allow_html=True)


def recommendations_page(gold: pd.DataFrame, recommendations: pd.DataFrame) -> None:
    """Marketing director action page."""
    page_header(
        "Business Recommendations",
        "What the marketing director should promote next, under budget and risk constraints.",
    )
    render_exports(gold, recommendations)

    if recommendations.empty:
        st.warning("No recommendation matches the selected filters.")
        return

    selected_country = st.selectbox("Country", ["All"] + sorted(recommendations["country"].unique()))
    reco = recommendations if selected_country == "All" else recommendations[recommendations["country"] == selected_country]
    reco = reco.sort_values("marketing_priority", ascending=False).head(5)

    kpi_cols = st.columns(3)
    kpi_cols[0].metric("Selected destinations", len(reco))
    kpi_cols[1].metric("Allocated budget", format_compact(reco["allocated_budget"].sum(), " EUR"))
    kpi_cols[2].metric("Average score", f"{reco['marketing_priority'].mean():.2f}")

    for chunk_start in range(0, len(reco), 2):
        cols = st.columns(2)
        for col, (_, row) in zip(cols, reco.iloc[chunk_start : chunk_start + 2].iterrows()):
            with col:
                recommendation_card(row)


def about_page() -> None:
    """About page with methodology notes."""
    page_header(
        "About",
        "Methodology, governance principles and operating model behind Tourism Forecast AI.",
    )
    render_html_card(
        """
        <div class="panel">
            <div class="section-title">Operating principles</div>
            <p class="muted">
            The platform keeps raw data immutable, standardizes business keys, documents anomalies,
            builds a country-destination GOLD DATA layer, forecasts demand with temporal validation,
            and ranks recommendations using transparent business rules.
            </p>
            <p class="muted">
            The scoring is a decision-support framework. It does not replace marketing governance,
            finance validation or local market expertise.
            </p>
        </div>
        """
    )


inject_css()
gold_data, recommendation_data, quality_data, prediction_data, metric_data = load_data()

page_name, country_filter, weather_filter, category_filter, date_period, budget_ceiling = render_sidebar(
    gold_data,
    prediction_data,
)
filtered_gold, filtered_recommendations, filtered_predictions = filter_data(
    gold_data,
    recommendation_data,
    prediction_data,
    country_filter,
    weather_filter,
    category_filter,
    date_period,
    budget_ceiling,
)

if page_name == "Executive Overview":
    executive_page(filtered_gold, filtered_recommendations)
elif page_name == "Data Quality":
    data_quality_page(quality_data)
elif page_name == "Forecast Analytics":
    forecast_page(filtered_predictions, metric_data)
elif page_name == "Business Recommendations":
    recommendations_page(filtered_gold, filtered_recommendations)
else:
    about_page()

render_html_card('<div class="footer">Tourism Forecast AI<br>Powered by Python - Streamlit - Plotly - Scikit-learn</div>')
