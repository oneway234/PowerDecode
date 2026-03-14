"""PowerDecode Dashboard — Streamlit app reading from SQLite."""

import concurrent.futures
import datetime
import json
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Default to bundled demo.db when no DB path is configured (e.g. Streamlit Cloud)
if not os.environ.get("PDD_DB_PATH"):
    demo_db = Path(__file__).parent / "data" / "demo.db"
    if demo_db.exists():
        os.environ["PDD_DB_PATH"] = str(demo_db)

import altair as alt
import anthropic
import pandas as pd
import streamlit as st

from db import init_db, get_recent_requests, get_requests_by_timerange
from location import get_server_location

st.set_page_config(page_title="PowerDecode", layout="wide")


def fmt_cost(usd: float) -> str:
    """Adaptive cost formatting — picks the clearest unit for the magnitude."""
    if usd >= 1.0:
        return f"USD {usd:,.2f}"
    if usd >= 0.001:
        return f"{usd * 1_000:.2f} mUSD"
    if usd >= 0.000_001:
        return f"{usd * 1_000_000:.2f} \u00b5USD"
    return f"{usd * 1_000_000_000:.2f} nUSD"


def fmt_energy(joules: float) -> str:
    """Adaptive energy formatting — picks the clearest unit for the magnitude."""
    if joules >= 3_600_000:
        return f"{joules / 3_600_000:.2f} kWh"
    if joules >= 1_000:
        return f"{joules / 1_000:.2f} kJ"
    if joules >= 1.0:
        return f"{joules:.2f} J"
    if joules >= 0.001:
        return f"{joules * 1_000:.2f} mJ"
    return f"{joules * 1_000_000:.2f} \u00b5J"

conn = init_db()

@st.cache_data(ttl=3600)
def _get_location():
    return get_server_location()

location = _get_location()

page = st.sidebar.radio("Navigation", ["Overview", "Request Detail", "Cost Trend"])

currency_symbol = location["currency_symbol"]

st.sidebar.markdown("---")
st.sidebar.markdown(
    f"📍 **{location['city']}, {location['country']}**  \n"
    f"Detected: {currency_symbol}{location['electricity_price_kwh']}/kWh"
)

electricity_price = st.sidebar.slider(
    f"Electricity price ({currency_symbol}/kWh)",
    min_value=0.01,
    max_value=max(1.0, float(location["electricity_price_kwh"]) * 2),
    value=float(location["electricity_price_kwh"]),
    step=0.01,
    format="%.2f",
)

# Auto-refresh (paused during AI analysis)
from streamlit_autorefresh import st_autorefresh

st.sidebar.markdown("---")
auto_refresh = st.sidebar.checkbox("Auto-refresh (5s)", value=True)
if auto_refresh and not st.session_state.get("ai_analyzing"):
    st_autorefresh(interval=5000, key="auto_refresh")

# ======================================================================
# AI Insight via Claude API — Three-Perspective Analysis
# ======================================================================

STATS_CONTEXT = """
Workload snapshot:
- Total requests: {total_requests}
- Total energy: {total_energy_j:.1f} J
- Total cost: USD {total_cost:.8f}
- Avg cost per request: USD {avg_cost:.8f}
- Total prefill tokens: {total_prefill}
- Total decode tokens: {total_decode}
- Decode/Prefill ratio: {ratio:.1f}x
- Anomaly requests: {anomaly_count}/{total_requests}
- Most expensive: USD {max_cost:.8f} ({max_prefill}p + {max_decode}d tokens)
- Cheapest: USD {min_cost:.8f}
- Known: On NVIDIA B200, each decode token consumes 0.485J. Prefill is too fast to measure accurately (< 10ms per 690 tokens). Decode dominates electricity cost on B200.
- Industry context: API providers (OpenAI, Anthropic, Google) already charge output tokens 3-5x more than input tokens — they discovered this asymmetry empirically at the API layer. However, GPU cloud providers like Fluidstack sell raw GPU hours and currently have zero visibility into prefill/decode split at the hardware level. PowerDecode is the first tool to measure this asymmetry directly on the GPU.
"""

SYSTEM_PRICING = (
    "You are a GPU cloud revenue strategist at a compute provider like Fluidstack. "
    "Compute supply is scarce and oversubscribed. "
    "API providers (OpenAI, Anthropic) already charge output tokens 3-5x more than "
    "input tokens because they discovered decode is more expensive. "
    "Fluidstack sells raw GPU hours and currently CANNOT see this split — "
    "they are leaving money on the table that API providers already figured out how to capture. "
    "The goal: quantify exactly how much revenue Fluidstack is missing "
    "by not having prefill/decode visibility. "
    "Be brutal with numbers. No fluff. No suggestions to lower prices. "
    "Respond ONLY with a valid JSON object matching the schema below. "
    "No explanation, no markdown, no code fences. Raw JSON only."
)
USER_PRICING = STATS_CONTEXT + (
    "With flat per-token pricing, which customer segment is currently "
    "UNDERPAYING relative to their true compute cost? "
    "Quantify exactly how much revenue is being left on the table, "
    "and state what separate prefill/decode pricing would recover.\n\n"
    'Return this exact JSON schema:\n'
    '{{"underpaid_amount": "", "affected_pct": "", "recommended_pricing": "", "conclusion": ""}}'
)

SYSTEM_OPS = (
    "You are a GPU cluster operations engineer at a compute provider. "
    "Your job is to find patterns in waste and abuse — not individual incidents. "
    "Always describe the PATTERN, not just the single worst case. "
    "Be specific about percentages and ratios. "
    "Respond ONLY with a valid JSON object matching the schema below. "
    "No explanation, no markdown, no code fences. Raw JSON only."
)
USER_OPS = STATS_CONTEXT + (
    "What pattern do the anomaly requests share? "
    "Describe what percentage of requests are consuming what percentage of total energy, "
    "and what operational limit would contain this pattern before it scales.\n\n"
    'Return this exact JSON schema:\n'
    '{{"anomaly_pattern": "", "risk_level": "", "suggested_limit": "", "conclusion": ""}}'
)

SYSTEM_USER = (
    "You are a cost optimization advisor for an AI application developer. "
    "Help them understand where their inference budget is going "
    "and how to reduce it without sacrificing quality. "
    "Respond ONLY with a valid JSON object matching the schema below. "
    "No explanation, no markdown, no code fences. Raw JSON only."
)
USER_USER = STATS_CONTEXT + (
    "Under flat per-token pricing, prefill tokens and decode tokens cost the same "
    "to the user — but decode tokens consume 8.3x more electricity.\n\n"
    "This means:\n"
    "- Decode-heavy users are UNDERPAYING relative to true cost\n"
    "- Prefill-heavy users are OVERPAYING relative to true cost\n\n"
    "Looking at this workload:\n"
    "1. Calculate how much the prefill-heavy requests overpaid "
    "compared to their true electricity cost\n"
    "2. Tell the user whether they are subsidizing others or being subsidized\n"
    "3. Give one concrete recommendation: should they shift their workload "
    "toward more prefill (longer prompts, shorter outputs) to get "
    "better value under current pricing?\n\n"
    "Be specific with numbers. Keep conclusion under 20 words.\n\n"
    'Return this exact JSON schema:\n'
    '{{"role": "", "overpay_pct": "", "recommended_strategy": "", "conclusion": ""}}'
)


def _build_stats_data(df: pd.DataFrame) -> dict:
    max_idx = df["cost"].idxmax()
    return {
        "total_requests": len(df),
        "total_energy_j": df["energy_joules"].sum(),
        "total_cost": df["cost"].sum(),
        "avg_cost": df["cost"].mean(),
        "total_prefill": int(df["prefill_tokens"].sum()),
        "total_decode": int(df["decode_tokens"].sum()),
        "ratio": df["decode_tokens"].sum() / max(df["prefill_tokens"].sum(), 1),
        "anomaly_count": int((df["anomaly_flag"] >= 1).sum()),
        "max_cost": df["cost"].max(),
        "min_cost": df["cost"].min(),
        "max_prefill": int(df.loc[max_idx, "prefill_tokens"]),
        "max_decode": int(df.loc[max_idx, "decode_tokens"]),
    }


def _call_claude(system_prompt: str, user_prompt: str, stats_data: dict) -> dict | None:
    try:
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            return None
        filled_user = user_prompt.format(**stats_data)
        client = anthropic.Anthropic(api_key=api_key)
        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=500,
            system=system_prompt,
            messages=[{"role": "user", "content": filled_user}],
        )
        text = message.content[0].text.strip()
        # Strip markdown code fences if present
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text[3:]
            if text.endswith("```"):
                text = text[:-3].strip()
        return json.loads(text)
    except Exception as e:
        import logging
        logging.getLogger("powerdecode.dashboard").warning("Claude API call failed: %s", e)
        return None


def _run_three_analyses(stats_data: dict) -> tuple[dict | None, dict | None, dict | None]:
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        f1 = executor.submit(_call_claude, SYSTEM_PRICING, USER_PRICING, stats_data)
        f2 = executor.submit(_call_claude, SYSTEM_OPS, USER_OPS, stats_data)
        f3 = executor.submit(_call_claude, SYSTEM_USER, USER_USER, stats_data)
        return f1.result(), f2.result(), f3.result()


def _render_pricing(data: dict | None):
    if not data:
        st.error("Analysis failed")
        return
    st.markdown(f"""
| Metric | Value |
|--------|-------|
| 💸 Underpaid Amount | {data.get('underpaid_amount', 'N/A')} |
| 📊 Affected % | {data.get('affected_pct', 'N/A')} |
| 🏷️ Recommended Pricing | {data.get('recommended_pricing', 'N/A')} |
""")
    st.info(f"💡 {data.get('conclusion', 'N/A')}")


def _render_ops(data: dict | None):
    if not data:
        st.error("Analysis failed")
        return
    risk_raw = data.get("risk_level", "").upper()
    if "HIGH" in risk_raw:
        risk_color = "🔴"
    elif "MED" in risk_raw:
        risk_color = "🟡"
    elif "LOW" in risk_raw:
        risk_color = "🟢"
    else:
        risk_color = "⚪"
    st.markdown(f"""
| Metric | Value |
|--------|-------|
| ⚠️ Anomaly Pattern | {data.get('anomaly_pattern', 'N/A')} |
| {risk_color} Risk Level | {data.get('risk_level', 'N/A')} |
| 🔒 Suggested Limit | {data.get('suggested_limit', 'N/A')} |
""")
    st.info(f"💡 {data.get('conclusion', 'N/A')}")


def _render_user(data: dict | None):
    if not data:
        st.error("Analysis failed")
        return
    role = data.get("role", "")
    role_icon = "📤" if "subsidi" in role.lower() and "by" not in role.lower() else "📥"
    st.markdown(f"""
| Metric | Value |
|--------|-------|
| {role_icon} Your Role | {data.get('role', 'N/A')} |
| 💰 Overpay % | {data.get('overpay_pct', 'N/A')} |
| 🎯 Recommended Strategy | {data.get('recommended_strategy', 'N/A')} |
""")
    st.info(f"💡 {data.get('conclusion', 'N/A')}")

# ======================================================================
# Helper
# ======================================================================


def _to_dataframe(rows: list[dict], price_per_kwh: float = 0.12) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["end_time"], unit="s")
    df["ts_label"] = df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S.%f")
    df["latency_s"] = df["end_time"] - df["start_time"]
    df["total_tokens"] = df["prefill_tokens"] + df["decode_tokens"]
    # Recalculate cost with current electricity price
    df["cost"] = df["energy_joules"] / 3_600_000 * price_per_kwh
    return df


# ======================================================================
# Page 1: Overview
# ======================================================================

if page == "Overview":
    st.title("PowerDecode Overview")

    rows = get_recent_requests(conn, limit=200)
    df = _to_dataframe(rows, price_per_kwh=electricity_price)

    if df.empty:
        st.warning("No requests recorded yet.")
    else:
        # --- Metric cards ---
        total_cost = df["cost"].sum()
        total_energy_j = df["energy_joules"].sum()
        avg_cost = total_cost / len(df)
        total_requests = len(df)
        anomaly_count = int((df["anomaly_flag"] == 1).sum())
        extreme_count = int((df["anomaly_flag"] == 2).sum())

        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("Total Cost", fmt_cost(total_cost))
        c2.metric("Total Energy", fmt_energy(total_energy_j))
        c3.metric("Avg Cost / Request", fmt_cost(avg_cost))
        c4.metric("Total Requests", total_requests)
        c5.metric("⚠️ Anomaly", anomaly_count)
        c6.metric("🔴 Extreme", extreme_count)

        # --- AI Insight: Three-Perspective Analysis ---
        st.subheader("AI Insight")

        def _on_analyze():
            st.session_state["run_analysis"] = True
            st.session_state["ai_analyzing"] = True

        st.button(
            "▶ Three-Perspective Analysis",
            type="primary",
            on_click=_on_analyze,
        )

        if st.session_state.get("run_analysis"):
            if len(df) >= 5:
                with st.spinner("⚡ Running three-perspective analysis..."):
                    stats_data = _build_stats_data(df)
                    r_pricing, r_ops, r_user = _run_three_analyses(stats_data)
                st.session_state["ai_insight"] = (r_pricing, r_ops, r_user)
            else:
                st.info("Need at least 5 requests to analyze.")
            st.session_state["run_analysis"] = False
            st.session_state["ai_analyzing"] = False

        if "ai_insight" in st.session_state:
            r_pricing, r_ops, r_user = st.session_state["ai_insight"]
            tab1, tab2, tab3 = st.tabs(["☁️ Pricing", "🔍 Operations", "💰 User Cost"])
            with tab1:
                _render_pricing(r_pricing)
            with tab2:
                _render_ops(r_ops)
            with tab3:
                _render_user(r_user)

        # --- Decode Electricity Cost vs Token Share scatter (uses ALL DB records) ---
        st.subheader("Decode Electricity Cost vs Token Share")
        st.caption("Points above the line: decode workloads underpay under flat token pricing")

        W_PREFILL = 0.0112
        W_DECODE  = 0.4851

        all_rows = get_recent_requests(conn, limit=100_000)
        all_df = _to_dataframe(all_rows, price_per_kwh=electricity_price)
        asym_df = all_df[all_df["total_tokens"] > 0].copy()
        asym_df["decode_token_pct"] = (
            asym_df["decode_tokens"] / asym_df["total_tokens"] * 100
        )
        pw = asym_df["prefill_tokens"] * W_PREFILL
        dw = asym_df["decode_tokens"]  * W_DECODE
        tw = pw + dw
        asym_df["decode_energy_pct"] = (dw / tw * 100).where(tw > 0, 0)
        asym_df["status"] = asym_df["anomaly_flag"].map(
            {0: "Normal", 1: "Anomaly", 2: "Extreme"}
        ).fillna("Normal")
        # Short model name: keep only the part after the last "/"
        asym_df["model"] = asym_df["model"].apply(lambda m: m.rsplit("/", 1)[-1] if isinstance(m, str) else m)

        # Model filter checkboxes — sorted by parameter size (small → large)
        import re

        def _model_size_key(name: str) -> float:
            """Extract parameter count (in billions) for sorting."""
            m = re.search(r"(\d+(?:\.\d+)?)\s*[Bb]", name)
            return float(m.group(1)) if m else float("inf")

        all_models = sorted(asym_df["model"].dropna().unique().tolist(), key=_model_size_key)
        if len(all_models) > 1:
            filter_cols = st.columns(len(all_models))
            selected_models = []
            for i, m in enumerate(all_models):
                if filter_cols[i].checkbox(m, value=True, key=f"asym_{m}"):
                    selected_models.append(m)
            # Filter to only checked models
            if selected_models:
                asym_df = asym_df[asym_df["model"].isin(selected_models)]
            else:
                asym_df = asym_df.iloc[0:0]  # empty — all unchecked
        elif len(all_models) == 1:
            st.caption(f"Model: {all_models[0]}")

        # Shaded region above diagonal (underpriced decode workloads)
        shade_df = pd.DataFrame({
            "x": [0, 0, 100],
            "y": [0, 100, 100],
        })
        shade = (
            alt.Chart(shade_df)
            .mark_area(opacity=0.08, color="#e74c3c")
            .encode(
                x=alt.X("x:Q"),
                y=alt.Y("y:Q"),
            )
        )

        # Label for the shaded region
        shade_label = (
            alt.Chart(pd.DataFrame({"x": [25], "y": [70], "text": ["Underpriced decode workloads"]}))
            .mark_text(fontSize=16, color="#e74c3c", opacity=0.7, fontWeight="bold", angle=0)
            .encode(x="x:Q", y="y:Q", text="text:N")
        )

        # Diagonal fair-pricing line
        line_df = pd.DataFrame({"x": [0, 100], "y": [0, 100]})
        diagonal = (
            alt.Chart(line_df)
            .mark_line(strokeDash=[6, 4], color="#666666", opacity=0.5)
            .encode(
                x=alt.X("x:Q"),
                y=alt.Y("y:Q"),
            )
        )

        # Label for the diagonal line
        diag_label = (
            alt.Chart(pd.DataFrame({"x": [55], "y": [60], "text": ["Fair Pricing Line"]}))
            .mark_text(fontSize=11, color="#666666", fontWeight="bold", angle=38)
            .encode(x="x:Q", y="y:Q", text="text:N")
        )

        scatter = (
            alt.Chart(asym_df)
            .mark_circle(size=60, opacity=0.7)
            .encode(
                x=alt.X("decode_token_pct:Q",
                        title="Decode Tokens (% of tokens)",
                        scale=alt.Scale(domain=[0, 100])),
                y=alt.Y("decode_energy_pct:Q",
                        title="Decode Electricity (% of GPU energy)",
                        scale=alt.Scale(domain=[0, 100])),
                color=alt.Color(
                    "model:N",
                    legend=alt.Legend(title="Model"),
                ),
                tooltip=[
                    alt.Tooltip("prompt_preview:N", title="Prompt"),
                    alt.Tooltip("model:N", title="Model"),
                    alt.Tooltip("decode_token_pct:Q", title="Decode token %", format=".1f"),
                    alt.Tooltip("decode_energy_pct:Q", title="Decode energy %", format=".1f"),
                    alt.Tooltip("prefill_tokens:Q", title="Prefill tokens"),
                    alt.Tooltip("decode_tokens:Q", title="Decode tokens"),
                    alt.Tooltip("status:N", title="Status"),
                ],
            )
            .properties(height=350)
        )

        st.altair_chart(
            (shade + shade_label + diagonal + diag_label + scatter).resolve_scale(color="independent"),
            use_container_width=True,
        )

        # --- Bar chart: energy per request (scrollable) ---
        st.subheader("Energy per Request")
        chart_df = df[["request_id", "prompt_preview", "model", "endpoint", "energy_joules", "anomaly_flag", "timestamp", "ts_label"]].copy()
        chart_df["short_id"] = chart_df["request_id"].str[:8]
        chart_df["status"] = chart_df["anomaly_flag"].map(
            {0: "Normal", 1: "Anomaly", 2: "Extreme"}
        ).fillna("Normal")
        chart_df = chart_df.sort_values("timestamp").reset_index(drop=True)
        chart_df["_idx"] = chart_df.index

        n_bars = len(chart_df)
        visible = 50
        if n_bars > visible:
            lo, hi = st.slider(
                "Scroll requests",
                min_value=0,
                max_value=n_bars - 1,
                value=(max(0, n_bars - visible), n_bars - 1),
                key="energy_scroll",
            )
            chart_df = chart_df[(chart_df["_idx"] >= lo) & (chart_df["_idx"] <= hi)]

        bar = (
            alt.Chart(chart_df)
            .mark_bar()
            .encode(
                x=alt.X("short_id:N", sort=None, axis=alt.Axis(labels=False, title=None)),
                y=alt.Y("energy_joules:Q", title="Energy (J)"),
                color=alt.Color(
                    "status:N",
                    scale=alt.Scale(
                        domain=["Normal", "Anomaly", "Extreme"],
                        range=["steelblue", "#e74c3c", "#f5a623"],
                    ),
                    legend=alt.Legend(title="Status"),
                ),
                tooltip=[
                    alt.Tooltip("prompt_preview:N", title="Prompt"),
                    alt.Tooltip("model:N", title="Model"),
                    alt.Tooltip("endpoint:N", title="Endpoint"),
                    alt.Tooltip("energy_joules:Q", title="Energy (J)", format=".4f"),
                    alt.Tooltip("status:N", title="Status"),
                    alt.Tooltip("ts_label:N", title="Timestamp"),
                ],
            )
            .properties(height=350)
        )
        st.altair_chart(bar, use_container_width=True)

        # --- Table ---
        st.subheader("Recent Requests")
        display_df = df[
            ["request_id", "prompt_preview", "model",
             "prefill_tokens", "decode_tokens",
             "latency_s", "energy_joules", "cost", "anomaly_flag", "timestamp"]
        ].copy()
        display_df["latency_s"] = display_df["latency_s"].round(3)
        display_df["energy_joules"] = display_df["energy_joules"].round(4)
        display_df["cost"] = display_df["cost"].apply(fmt_cost)
        # Keep raw flag for highlighting, then map to display text
        _raw_flags = display_df["anomaly_flag"].copy()
        display_df["anomaly_flag"] = _raw_flags.map(
            {0: "OK", 1: "⚠️ Anomaly", 2: "🔴 Extreme"}
        ).fillna("OK")

        def _highlight_anomaly(row):
            flag = _raw_flags.loc[row.name]
            if flag == 2:
                return ["background-color: #fff3cd; color: #856404"] * len(row)
            elif flag == 1:
                return ["background-color: #ffcccc"] * len(row)
            return [""] * len(row)

        styled = display_df.style.apply(_highlight_anomaly, axis=1)
        st.dataframe(
            styled,
            use_container_width=True,
            hide_index=True,
            column_config={
                "request_id": st.column_config.TextColumn(
                    "Request ID", help="Unique identifier for this inference request.",
                    width="small"),
                "prompt_preview": st.column_config.TextColumn(
                    "Prompt", help="First ~40 characters of the user prompt.",
                    width="small"),
                "model": st.column_config.TextColumn(
                    "Model", help="LLM model used for inference.",
                    width="small"),
                "prefill_tokens": st.column_config.NumberColumn(
                    "Prefill \u2753", help="Input tokens processed in parallel during prompt encoding. Higher prefill = longer prompt. Consumes less energy per token than decode."),
                "decode_tokens": st.column_config.NumberColumn(
                    "Decode \u2753", help="Output tokens generated one-by-one (auto-regressive). Each decode token costs ~8.3x more electricity than a prefill token."),
                "latency_s": st.column_config.NumberColumn(
                    "Latency (s)", help="End-to-end request duration in seconds, from first prefill token to last decode token."),
                "energy_joules": st.column_config.NumberColumn(
                    "Energy (J) \u2753", help="GPU energy consumed by this request in Joules. Measured via pynvml at 10ms sampling interval, integrated using trapezoidal rule minus idle power."),
                "cost": st.column_config.TextColumn(
                    "Cost \u2753", help="Electricity cost (adaptive unit: nUSD / \u00b5USD / mUSD / USD). = energy_joules / 3,600,000 * price_per_kWh. Adjustable via sidebar slider."),
                "anomaly_flag": st.column_config.TextColumn(
                    "Status \u2753", help="OK = normal. ⚠️ Anomaly = cost-per-weighted-token exceeds mean + 2σ (statistically high energy). 🔴 Extreme = energy = 0, attribution failed (GPU sampling error or request too fast to measure)."),
                "timestamp": st.column_config.DatetimeColumn(
                    "Timestamp", help="When the request completed."),
            },
        )


# ======================================================================
# Page 2: Request Detail
# ======================================================================

elif page == "Request Detail":

    rows = get_recent_requests(conn, limit=200)
    df = _to_dataframe(rows, price_per_kwh=electricity_price)

    if df.empty:
        st.warning("No requests recorded yet.")
    else:
        def _format_option(i: int) -> str:
            r = df.iloc[i]
            flag = int(r["anomaly_flag"])
            prefix = "🔴 " if flag == 2 else "⚠️ " if flag == 1 else ""
            return f"{prefix}{r['request_id'][:12]}…  |  {r['prompt_preview'][:40]}"

        st.title("Request Detail")

        # === Hero sentence（靜態部分）===
        st.markdown(
            """
            <div style="text-align:center; padding:1.5rem 0 0.5rem 0;">
                <div style="margin-bottom:0.6rem;">
                    <span style="background:#f58518; color:white; font-size:0.75rem;
                                 font-weight:700; padding:0.25rem 0.7rem; border-radius:3px;
                                 letter-spacing:0.05em; text-transform:uppercase;">
                        Measured on GPU hardware
                    </span>
                </div>
                <div style="font-size:0.95rem; opacity:0.55; margin-bottom:0.4rem;">
                    Energy Breakdown for a Single Inference Request
                </div>
                <div style="font-size:2.4rem; font-weight:700; color:#f58518;
                            line-height:1.15; margin-bottom:0.4rem;">
                    Decode dominates inference electricity cost
                </div>
                <div style="font-size:1.1rem; opacity:0.65;">
                    On NVIDIA B200, <strong>decode dominates inference electricity cost</strong>.<br>Prefill is too fast to measure accurately.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # 從 session_state 讀取（selectbox 在 attr_col 內，但值通過 key 持久化）
        selected_idx = st.session_state.get("_detail_req_idx", 0)
        if selected_idx >= len(df):
            selected_idx = 0
        row = df.iloc[selected_idx]
        flag = int(row["anomaly_flag"])
        prefill_cost_share = row["prefill_tokens"] * 0.0212
        decode_cost_share  = row["decode_tokens"]  * 0.1772
        total = prefill_cost_share + decode_cost_share
        prefill_pct = prefill_cost_share / total * 100 if total > 0 else 0.0
        decode_pct  = decode_cost_share  / total * 100 if total > 0 else 0.0

        # === Inference economics hint（動態）===
        st.markdown(
            f"""
            <div style="text-align:center; padding:0.5rem 0 1rem 0;
                        font-size:1.4rem; font-weight:600;">
                ⚡ Decode tokens were only
                <span style="color:#f58518; font-weight:700;">{row['decode_tokens'] / max(row['decode_tokens'] + row['prefill_tokens'], 1) * 100:.1f}%</span>
                of the tokens but consumed
                <span style="color:#f58518; font-weight:700;">{decode_pct:.1f}%</span>
                of the electricity cost
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Anomaly banner
        if flag == 2:
            st.markdown(
                '<div style="background-color:#ff4444;color:white;padding:8px 12px;'
                'border-radius:6px;margin-bottom:8px;">'
                '🔴 <strong>Extreme anomaly</strong> — energy = 0, attribution failed.'
                '</div>', unsafe_allow_html=True)
        elif flag == 1:
            st.markdown(
                '<div style="background-color:#ffcccc;padding:8px 12px;'
                'border-radius:6px;margin-bottom:8px;">'
                '⚠️ <strong>Statistical anomaly</strong> — '
                'cost-per-weighted-token exceeds mean + 2σ.'
                '</div>', unsafe_allow_html=True)

        st.markdown("---")

        # === Donut（左） + Attribution（右）===
        donut_col, attr_col = st.columns([1, 1])

        with donut_col:
            token_df = pd.DataFrame({
                "type": ["Prefill cost", "Decode cost"],
                "cost_share": [prefill_cost_share, decode_cost_share],
                "label": [
                    f"Prefill cost ({prefill_pct:.1f}%)",
                    f"Decode cost ({decode_pct:.1f}%)",
                ],
            })
            donut = (
                alt.Chart(token_df)
                .mark_arc(innerRadius=70, outerRadius=150)
                .encode(
                    theta="cost_share:Q",
                    color=alt.Color(
                        "type:N",
                        scale=alt.Scale(
                            domain=["Prefill cost", "Decode cost"],
                            range=["#4c78a8", "#f58518"],
                        ),
                        legend=alt.Legend(
                            orient="none",
                            legendX=70,
                            legendY=0,
                            direction="horizontal",
                            title=None,
                            labelFontSize=20,
                            symbolSize=250,
                        ),
                    ),
                    tooltip=["label", "cost_share"],
                )
                .properties(height=420, width=400, padding={"top": 5, "bottom": 10, "left": 30, "right": 10})
            )
            st.altair_chart(donut, use_container_width=False)

        with attr_col:
            st.markdown("<div style='padding-top:2rem'>", unsafe_allow_html=True)
            st.subheader("Attribution")

            # Energy → Cost → Tokens → Anomaly
            energy_j = row["energy_joules"]
            cost_uusd = row["cost"] * 1_000_000

            st.markdown(f"""
            <div style="font-size:1.05rem; line-height:2.2;">
                <div style="font-size:1.3rem; margin:0.3rem 0;">
                    <span style="font-weight:700; color:#f58518;">Cost per request</span>
                    <span style="float:right; font-weight:700; color:#f58518; font-size:1.35rem;">
                        {cost_uusd:.4f} µUSD
                    </span>
                </div>
                <div><span style="opacity:0.6">Energy</span>
                    <span style="float:right; font-weight:600;">
                        {energy_j:.4f} J
                    </span>
                </div>
                <hr style="opacity:0.2; margin:0.5rem 0">
                <div><span style="opacity:0.6">Prefill tokens</span>
                    <span style="float:right;">{row['prefill_tokens']}</span>
                </div>
                <div><span style="opacity:0.6">Decode tokens</span>
                    <span style="float:right;">{row['decode_tokens']}</span>
                </div>
                <div><span style="opacity:0.6">Latency</span>
                    <span style="float:right;">{row['latency_s']:.3f}s</span>
                </div>
                <hr style="opacity:0.2; margin:0.5rem 0">
                <div><span style="opacity:0.6">Anomaly</span>
                    <span style="float:right;">
                        {"🔴 Extreme" if flag == 2 else "⚠️ Anomaly" if flag == 1 else "✓ OK"}
                    </span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.selectbox(
                "Request",
                range(len(df)),
                format_func=_format_option,
                key="_detail_req_idx",
            )
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("---")

        # === Technical Details（折疊）===
        with st.expander("Technical Details", expanded=False):
            st.write(f"**Request ID:** `{row['request_id']}`")
            st.write(f"**Prompt:** {row['prompt_preview']}")
            st.write(f"**Model:** {row['model']}")
            st.write(f"**Endpoint:** {row['endpoint']}")
            st.write(f"**Timestamp:** {row['timestamp']}")

        del df


# ======================================================================
# Page 3: Cost Trend
# ======================================================================

elif page == "Cost Trend":
    st.title("Cost Trend")

    rows = get_recent_requests(conn, limit=500)
    df = _to_dataframe(rows, price_per_kwh=electricity_price)

    if df.empty:
        st.warning("No requests recorded yet.")
    else:
        df = df.sort_values("timestamp")

        # Cumulative cost
        st.subheader("Cumulative Cost")
        df = df.reset_index(drop=True)
        df["req_num"] = range(1, len(df) + 1)
        # Pick consistent cost unit based on cumulative max
        cumul_max = df["cost"].sum()
        if cumul_max >= 1.0:
            cost_scale, cost_unit = 1.0, "USD"
        elif cumul_max >= 0.001:
            cost_scale, cost_unit = 1_000, "mUSD"
        elif cumul_max >= 0.000_001:
            cost_scale, cost_unit = 1_000_000, "\u00b5USD"
        else:
            cost_scale, cost_unit = 1_000_000_000, "nUSD"

        df["cost_scaled"] = df["cost"] * cost_scale
        df["cumulative_cost_scaled"] = df["cost_scaled"].cumsum()

        line = (
            alt.Chart(df)
            .mark_line(point=True)
            .encode(
                x=alt.X("req_num:Q", title="Request #", axis=alt.Axis(tickMinStep=1)),
                y=alt.Y("cumulative_cost_scaled:Q", title=f"Cumulative Cost ({cost_unit})"),
                tooltip=[
                    alt.Tooltip("req_num:Q", title="Request #"),
                    alt.Tooltip("prompt_preview:N", title="Prompt"),
                    alt.Tooltip("cost_scaled:Q", title=f"Cost ({cost_unit})", format=".4f"),
                    alt.Tooltip("cumulative_cost_scaled:Q", title=f"Cumulative ({cost_unit})", format=".2f"),
                    alt.Tooltip("ts_label:N", title="Timestamp"),
                ],
            )
            .properties(height=350)
        )
        st.altair_chart(line, use_container_width=True)

        # Energy per request
        st.subheader("Energy per Request")
        df["status"] = df["anomaly_flag"].map({0: "Normal", 1: "Anomaly", 2: "Extreme"}).fillna("Normal")
        scatter = (
            alt.Chart(df)
            .mark_circle(size=60)
            .encode(
                x=alt.X("req_num:Q", title="Request #", axis=alt.Axis(tickMinStep=1)),
                y=alt.Y("energy_joules:Q", title="Energy (J)"),
                color=alt.Color(
                    "status:N",
                    scale=alt.Scale(domain=["Normal", "Anomaly", "Extreme"], range=["steelblue", "#e74c3c", "#f5a623"]),
                    legend=alt.Legend(title="Status"),
                ),
                tooltip=[
                    alt.Tooltip("req_num:Q", title="Request #"),
                    alt.Tooltip("prompt_preview:N", title="Prompt"),
                    alt.Tooltip("energy_joules:Q", title="Energy (J)", format=".4f"),
                    alt.Tooltip("decode_tokens:Q", title="Decode Tokens"),
                    alt.Tooltip("prefill_tokens:Q", title="Prefill Tokens"),
                    alt.Tooltip("ts_label:N", title="Timestamp"),
                ],
            )
            .properties(height=350)
        )
        st.altair_chart(scatter, use_container_width=True)

        # Cost per token — shows which requests are least efficient
        st.subheader("Cost per Token")
        df["cost_per_token_scaled"] = df["cost_scaled"] / df["total_tokens"].replace(0, 1)
        cpt = (
            alt.Chart(df)
            .mark_bar()
            .encode(
                x=alt.X("req_num:Q", title="Request #", axis=alt.Axis(tickMinStep=1)),
                y=alt.Y("cost_per_token_scaled:Q", title=f"Cost per Token ({cost_unit})"),
                color=alt.Color(
                    "status:N",
                    scale=alt.Scale(domain=["Normal", "Anomaly", "Extreme"], range=["steelblue", "#e74c3c", "#f5a623"]),
                    legend=alt.Legend(title="Status"),
                ),
                tooltip=[
                    alt.Tooltip("req_num:Q", title="Request #"),
                    alt.Tooltip("prompt_preview:N", title="Prompt"),
                    alt.Tooltip("total_tokens:Q", title="Total Tokens"),
                    alt.Tooltip("cost_per_token_scaled:Q", title=f"Cost/Token ({cost_unit})", format=".6f"),
                    alt.Tooltip("ts_label:N", title="Timestamp"),
                ],
            )
            .properties(height=350)
        )
        st.altair_chart(cpt, use_container_width=True)
        del df


