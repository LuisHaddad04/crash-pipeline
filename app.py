# app.py — Chicago Crash ETL Dashboard
# -------------------------------------------------------------
# Streamlit frontend for the Chicago Crash ETL pipeline
# Tabs: Home | Data Management | Data Fetcher | Scheduler | EDA | Reports
# -------------------------------------------------------------

import os
import io
import json
from datetime import datetime, date, timedelta, UTC
from typing import List, Dict, Any

import streamlit as st
import pandas as pd
import numpy as np
import duckdb
import altair as alt
import requests

# ------------------------------
# Config / Constants
# ------------------------------
API_BASE = os.getenv("ETL_API_BASE", "http://localhost:8000")  # backend REST base
GOLD_DB_PATH = os.getenv("GOLD_DB_PATH", "./gold.duckdb")

st.set_page_config(page_title="Chicago Crash ETL Dashboard", layout="wide")

# Global altair settings
alt.data_transformers.disable_max_rows()

# ------------------------------
# Helpers
# ------------------------------

def _get(url: str, **kwargs):
    try:
        r = requests.get(url, timeout=30, **kwargs)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"GET {url} failed: {e}")
        return None


def _post(url: str, payload: Dict[str, Any], **kwargs):
    try:
        r = requests.post(url, json=payload, timeout=60, **kwargs)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"POST {url} failed: {e}")
        return None


def status_chip(ok: bool, label: str):
    color = "#22c55e" if ok else "#ef4444"
    st.markdown(
        f"""
        <div style='display:inline-block;padding:.35rem .6rem;border-radius:999px;background:{color};color:white;font-weight:600;margin-right:.5rem;'>
            {label}
        </div>
        """,
        unsafe_allow_html=True,
    )


def corrid_now() -> str:
    return datetime.now(UTC).strftime("%Y%m%d-%H%M%S")


def df_safe(df, empty_msg: str = "No data available."):
    if df is None or len(df) == 0:
        st.info(empty_msg)
        return False
    return True


def read_gold(query: str) -> pd.DataFrame:
    if not os.path.exists(GOLD_DB_PATH):
        st.warning(f"Gold DB not found at {GOLD_DB_PATH}")
        return pd.DataFrame()
    try:
        con = duckdb.connect(GOLD_DB_PATH, read_only=True)
        return con.execute(query).df()
    except Exception as e:
        st.error(f"DuckDB query failed: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=600)
def get_schema_columns() -> Dict[str, List[str]]:
    """Fetch available columns for crashes, vehicles, people from backend schema endpoint."""
    url = f"{API_BASE}/api/schema/columns"
    js = _get(url)
    if not js:
        return {"crashes": [], "vehicles": [], "people": []}
    return js


# ------------------------------
# Header / Title
# ------------------------------
st.title("🚦 Chicago Crash ETL — Command Center")
st.caption("Run, monitor, analyze your end‑to‑end ETL pipeline for Chicago Crash data.")

# Tabs
TAB_HOME, TAB_MGMT, TAB_FETCH, TAB_SCHED, TAB_EDA, TAB_REPORTS = st.tabs([
    "🏠 Home", "🧰 Data Management", "📡 Data Fetcher", "⏰ Scheduler", "📊 EDA", "📑 Reports"
])

# ==============================================================
# 1) HOME
# ==============================================================
with TAB_HOME:
    st.subheader("Label Overview · Pipelines")

    # Template card content that the student can edit per label
    with st.expander("Crash Type — Model Card (example)", expanded=True):
        st.markdown(
            """
            **Label predicted:** `crash_type` • **Type:** multiclass • **Classes:** Rear End, Turning, Angle, Sideswipe, Other

            **Pipeline:** We predict `crash_type` using roadway context, time-of-day, weather, posted speed, and primary cause.

            **Key features (why they help):**
            - `posted_speed_limit` — correlates with roadway function & following distance.
            - `crash_hour` — commute patterns amplify certain conflicts (turning vs rear‑end).
            - `weather_condition` — slick conditions raise rear‑end share.

            **Source columns (subset):**
            - crashes: `crash_date`, `crash_hour`, `posted_speed_limit`, `weather_condition`, `traffic_control_device`

            **Class imbalance:** Ratio ~1:3 between top class and minority classes; handled with class weights and thresholding on PR.

            **Grain & filters:** one row per crash; city limits only; drop null geo.

            **Leakage/caveats:** exclude any post‑outcome signals (e.g., disposition after crash) from features.

            **Gold table:** `gold_crash_type`
            """
        )

    st.divider()
    st.subheader("Container Health")
    colA, colB, colC, colD, colE = st.columns(5)

    health = _get(f"{API_BASE}/api/health") or {}

    with colA: status_chip(bool(health.get("minio", False)), "MinIO")
    with colB: status_chip(bool(health.get("rabbitmq", False)), "RabbitMQ")
    with colC: status_chip(bool(health.get("extractor", False)), "Extractor")
    with colD: status_chip(bool(health.get("transformer", False)), "Transformer")
    with colE: status_chip(bool(health.get("cleaner", False)), "Cleaner")

# ==============================================================
# 2) DATA MANAGEMENT
# ==============================================================
with TAB_MGMT:
    st.subheader("MinIO · Delete by Folder or Bucket")
    minio_buckets = ["raw-data", "transform-data"]

    left, right = st.columns(2)
    with left:
        st.markdown("**Delete by Folder (Prefix)**")
        bucket = st.selectbox("Bucket", options=minio_buckets)
        prefix = st.text_input("Prefix (e.g., crash/20240101-101500/)")
        if st.button("Preview objects"):
            resp = _post(f"{API_BASE}/api/minio/preview", {"bucket": bucket, "prefix": prefix})
            if resp and resp.get("objects"):
                df = pd.DataFrame(resp["objects"])  # expects list of {key, size, last_modified}
                st.dataframe(df, use_container_width=True)
                st.session_state["_minio_preview_ok"] = True
            else:
                st.info("No objects matched.")
                st.session_state["_minio_preview_ok"] = False
        confirm = st.checkbox("I confirm the scope above is correct (destructive)")
        delete_disabled = not (st.session_state.get("_minio_preview_ok") and confirm)
        st.button("Delete Folder", type="primary", disabled=delete_disabled,
                  on_click=lambda: _post(f"{API_BASE}/api/minio/delete", {"bucket": bucket, "prefix": prefix}))

    with right:
        st.markdown("**Delete by Bucket**")
        bucket2 = st.selectbox("Bucket (entire)", options=minio_buckets, key="bucket2")
        confirm2 = st.checkbox("I confirm full bucket deletion", key="confirm2")
        st.button("Delete Bucket", type="secondary", disabled=not confirm2,
                  on_click=lambda: _post(f"{API_BASE}/api/minio/delete_bucket", {"bucket": bucket2}))

    st.divider()
    st.subheader("Gold (DuckDB) · Admin & Quick Peek")
    c1, c2 = st.columns([1, 2])

    with c1:
        st.caption("Status")
        status = _get(f"{API_BASE}/api/gold/status") or {}
        st.write({"db_path": GOLD_DB_PATH, **status})

        wipe_ok = st.checkbox("I confirm wiping the entire Gold DB is safe")
        if st.button("Wipe Gold DB (ENTIRE FILE)", disabled=not wipe_ok):
            res = _post(f"{API_BASE}/api/gold/wipe", {})
            st.success(res or "Requested wipe.")

    with c2:
        st.caption("Quick Peek (sample)")
        limit = st.slider("Rows", min_value=10, max_value=200, value=50, step=10)
        table = st.text_input("Gold table name", value="gold_crash")
        if st.button("Preview table"):
            df = read_gold(f"select * from {table} limit {limit}")
            if df_safe(df):
                st.dataframe(df, use_container_width=True)

# ==============================================================
# 3) DATA FETCHER
# ==============================================================
with TAB_FETCH:
    st.subheader("Publish Fetch Jobs (Streaming / Backfill)")

    schema = get_schema_columns()
    crash_cols = schema.get("crashes", [])
    vehicle_cols = schema.get("vehicles", [])
    people_cols = schema.get("people", [])

    tabs = st.tabs(["Streaming", "Backfill"])

    # Common controls in a function
    def render_common_controls(mode: str, key_prefix: str):
        st.markdown(f"**Mode:** `{mode}`  •  **corrid:** `{st.session_state.get('corrid', corrid_now())}`")
        include_veh = st.checkbox("Include Vehicles", key=f"{key_prefix}_include_veh")
        include_ppl = st.checkbox("Include People", key=f"{key_prefix}_include_ppl")

        sel_veh_all = False
        sel_ppl_all = False
        sel_veh = []
        sel_ppl = []

        if include_veh:
            sel_veh_all = st.checkbox("Select all vehicle columns", key=f"{key_prefix}_veh_all")
            sel_veh = st.multiselect(
                "Vehicles: columns to be fetched",
                vehicle_cols,
                default=(vehicle_cols if sel_veh_all else []),
                key=f"{key_prefix}_veh_cols",
            )
        if include_ppl:
            sel_ppl_all = st.checkbox("Select all people columns", key=f"{key_prefix}_ppl_all")
            sel_ppl = st.multiselect(
                "People: columns to be fetched",
                people_cols,
                default=(people_cols if sel_ppl_all else []),
                key=f"{key_prefix}_ppl_cols",
            )
        return include_veh, sel_veh, include_ppl, sel_ppl

    with tabs[0]:  # Streaming
        st.markdown("**Recent window**: fetch last *N* days of crashes (plus optional enrichment columns).")
        since_days = st.number_input("Since days", min_value=1, max_value=365, value=30)
        include_veh, sel_veh, include_ppl, sel_ppl = render_common_controls("streaming", "stream")

        # Preview JSON
        if st.checkbox("Preview request JSON (streaming)"):
            js = {
                "mode": "streaming",
                "corrid": st.session_state.setdefault("corrid", corrid_now()),
                "window": {"since_days": int(since_days)},
                "include": {
                    "vehicles": sel_veh if include_veh else [],
                    "people": sel_ppl if include_ppl else [],
                },
            }
            st.code(json.dumps(js, indent=2))

        # Publish
        if st.button("Publish to RabbitMQ (Streaming)", type="primary"):
            payload = {
                "mode": "streaming",
                "corrid": st.session_state.setdefault("corrid", corrid_now()),
                "window": {"since_days": int(since_days)},
                "include": {
                    "vehicles": sel_veh if include_veh else [],
                    "people": sel_ppl if include_ppl else [],
                },
            }
            with st.spinner("Publishing…"):
                resp = _post(f"{API_BASE}/api/publish", payload)
            if resp and resp.get("status") == "queued":
                st.success(f"Queued! corrid={payload['corrid']}")
            else:
                st.error("Failed to queue job.")

    with tabs[1]:  # Backfill
        st.markdown("**Historical window**: fetch by date/time range.")
        start_d = st.date_input("Start date", value=date.today() - timedelta(days=30))
        end_d = st.date_input("End date", value=date.today())
        start_t = st.time_input("Start time", value=datetime.min.time())
        end_t = st.time_input("End time", value=datetime.max.time().replace(microsecond=0))

        include_veh, sel_veh, include_ppl, sel_ppl = render_common_controls("backfill", "backfill")

        if st.checkbox("Preview request JSON (backfill)"):
            js = {
                "mode": "backfill",
                "corrid": st.session_state.setdefault("corrid", corrid_now()),
                "window": {
                    "start": f"{start_d} {start_t}",
                    "end": f"{end_d} {end_t}",
                },
                "include": {
                    "vehicles": sel_veh if include_veh else [],
                    "people": sel_ppl if include_ppl else [],
                },
            }
            st.code(json.dumps(js, indent=2))

        if st.button("Publish to RabbitMQ (Backfill)", type="primary"):
            payload = {
                "mode": "backfill",
                "corrid": st.session_state.setdefault("corrid", corrid_now()),
                "window": {
                    "start": f"{start_d} {start_t}",
                    "end": f"{end_d} {end_t}",
                },
                "include": {
                    "vehicles": sel_veh if include_veh else [],
                    "people": sel_ppl if include_ppl else [],
                },
            }
            with st.spinner("Publishing…"):
                resp = _post(f"{API_BASE}/api/publish", payload)
            if resp and resp.get("status") == "queued":
                st.success(f"Queued! corrid={payload['corrid']}")
            else:
                st.error("Failed to queue job.")

# ==============================================================
# 4) SCHEDULER
# ==============================================================
with TAB_SCHED:
    st.subheader("Automation (cron)")

    freq = st.selectbox("Frequency", ["Daily", "Weekly", "Custom (cron)"])
    time_pick = st.time_input("Run time", value=datetime.now().time().replace(microsecond=0))
    config_type = st.selectbox("Config Type", ["streaming"], index=0)

    if freq == "Custom (cron)":
        cron_str = st.text_input("Cron string", value="0 9 * * *")
    else:
        # Translate to cron
        hour = time_pick.hour
        minute = time_pick.minute
        if freq == "Daily":
            cron_str = f"{minute} {hour} * * *"
        else:  # Weekly
            # default Monday
            cron_str = f"{minute} {hour} * * 1"

    st.code(f"cron: {cron_str}")

    if st.button("Create schedule", type="primary"):
        payload = {"cron": cron_str, "config": {"type": config_type}}
        res = _post(f"{API_BASE}/api/schedule", payload)
        if res:
            st.success("Schedule created.")

    st.divider()
    st.subheader("Active schedules")
    scheds = _get(f"{API_BASE}/api/schedules") or []
    if len(scheds) == 0:
        st.info("No schedules.")
    else:
        st.dataframe(pd.DataFrame(scheds), use_container_width=True)

# ==============================================================
# 5) EDA
# ==============================================================
with TAB_EDA:
    st.subheader("Explore Gold Tables")

    table = st.selectbox("Gold table", [
        "gold_crash", "gold_hit_and_run", "gold_crash_type"
    ], index=0)

    st.caption("Summary stats")
    try:
        df = read_gold(f"select * from {table}")
    except Exception:
        df = pd.DataFrame()

    if df_safe(df):
        # Basic profile
        st.write({"rows": len(df), "cols": len(df.columns)})
        st.dataframe(df.head(50), use_container_width=True)

        # Ensure expected columns exist (graceful fallback)
        def exists(c):
            return c in df.columns

        # Derive helpers if present
        if exists("crash_date"):
            df["crash_date"] = pd.to_datetime(df["crash_date"], errors="coerce")
            df["crash_day_of_week"] = df["crash_date"].dt.day_name()
            df["crash_hour"] = df.get("crash_hour", df["crash_date"].dt.hour)

        # ----------------------
        # 12 Visuals with captions
        # ----------------------
        st.markdown("### Visualizations (12)")

        # 1) Histogram: posted_speed_limit
        if exists("posted_speed_limit"):
            chart = alt.Chart(df.dropna(subset=["posted_speed_limit"]))\
                .mark_bar()\
                .encode(x=alt.X("posted_speed_limit:Q", bin=alt.Bin(maxbins=20)), y="count()")\
                .properties(title="Distribution of Posted Speed Limit")
            st.altair_chart(chart, use_container_width=True)
            st.caption("Insight: Many crashes occur on arterials around 30–40 mph.")

        # 2) Bar: weather_condition
        if exists("weather_condition"):
            chart = alt.Chart(df.dropna(subset=["weather_condition"]))\
                .mark_bar()\
                .encode(x=alt.X("weather_condition:N", sort='-y'), y="count()")\
                .properties(title="Crashes by Weather Condition")
            st.altair_chart(chart, use_container_width=True)
            st.caption("Insight: Clear weather dominates, but wet conditions are non‑trivial.")

        # 3) Line: crashes by hour
        if exists("crash_hour"):
            ch = df.dropna(subset=["crash_hour"])\
                  .groupby("crash_hour").size().reset_index(name="n")
            chart = alt.Chart(ch).mark_line(point=True).encode(x="crash_hour:O", y="n:Q").properties(title="Crashes by Hour")
            st.altair_chart(chart, use_container_width=True)
            st.caption("Insight: PM commute hours show peaks.")

        # 4) Pie: day of week share
        if exists("crash_day_of_week"):
            dw = df["crash_day_of_week"].value_counts().reset_index()
            dw.columns = ["day", "n"]
            chart = alt.Chart(dw).mark_arc(innerRadius=40).encode(theta="n:Q", color="day:N").properties(title="Share by Day of Week")
            st.altair_chart(chart, use_container_width=True)
            st.caption("Insight: Slight weekend mix shift.")

        # 5) Heatmap: hour x day
        if exists("crash_hour") and exists("crash_day_of_week"):
            grid = df.groupby(["crash_day_of_week", "crash_hour"]).size().reset_index(name="n")
            chart = alt.Chart(grid).mark_rect().encode(x="crash_hour:O", y=alt.Y("crash_day_of_week:N", sort=["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]), color="n:Q").properties(title="Heatmap: Hour × Day of Week")
            st.altair_chart(chart, use_container_width=True)
            st.caption("Insight: Weekday commute bands light up.")

        # 6) Stacked bar: primary_cause by weather
        if exists("primary_cause") and exists("weather_condition"):
            pcw = df.dropna(subset=["primary_cause","weather_condition"]).groupby(["weather_condition","primary_cause"]).size().reset_index(name="n")
            chart = alt.Chart(pcw).mark_bar().encode(x="weather_condition:N", y="n:Q", color="primary_cause:N").properties(title="Primary Cause by Weather (stacked)")
            st.altair_chart(chart, use_container_width=True)
            st.caption("Insight: Rear‑end contributing causes rise in wet conditions.")

        # 7) Box plot: posted_speed_limit by day
        if exists("posted_speed_limit") and exists("crash_day_of_week"):
            chart = alt.Chart(df.dropna(subset=["posted_speed_limit","crash_day_of_week"]))\
                .mark_boxplot()\
                .encode(x=alt.X("crash_day_of_week:N", sort=["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]), y="posted_speed_limit:Q")\
                .properties(title="Posted Speed by Day of Week")
            st.altair_chart(chart, use_container_width=True)
            st.caption("Insight: Weekend medians tilt slightly higher.")

        # 8) Scatter: posted_speed_limit vs crash_hour (jitter)
        if exists("posted_speed_limit") and exists("crash_hour"):
            jitter = df.dropna(subset=["posted_speed_limit","crash_hour"]).copy()
            jitter["j"] = np.random.uniform(-0.25, 0.25, size=len(jitter))
            chart = alt.Chart(jitter).mark_circle(opacity=0.4).encode(x="crash_hour:O", y="posted_speed_limit:Q")\
                .properties(title="Speed Limit vs Crash Hour (jitter)")
            st.altair_chart(chart, use_container_width=True)
            st.caption("Insight: Evening hours show wide speed environments.")

        # 9) Treemap-like (bar) top locations if present
        loc_col = "street_name" if exists("street_name") else ("street1" if exists("street1") else None)
        if loc_col:
            topk = df[loc_col].value_counts().head(15).reset_index()
            topk.columns = [loc_col, "n"]
            chart = alt.Chart(topk).mark_bar().encode(x="n:Q", y=alt.Y(f"{loc_col}:N", sort='-x')).properties(title="Top 15 Streets by Crash Count")
            st.altair_chart(chart, use_container_width=True)
            st.caption("Insight: A handful of corridors dominate incidence.")

        # 10) Rate by weather for hit & run if label column exists
        if exists("hit_and_run_i") and exists("weather_condition"):
            tmp = df.groupby("weather_condition")["hit_and_run_i"].mean().reset_index()
            chart = alt.Chart(tmp).mark_bar().encode(x="weather_condition:N", y=alt.Y("hit_and_run_i:Q", title="Hit & Run Rate")).properties(title="Hit & Run Rate by Weather")
            st.altair_chart(chart, use_container_width=True)
            st.caption("Insight: Darkness + precipitation lifts rates.")

        # 11) Crash type hourly rank (bump-like)
        if exists("crash_type") and exists("crash_hour"):
            cth = df.groupby(["crash_hour","crash_type"]).size().reset_index(name="n")
            # Hourly rank per type
            cth["rank"] = cth.groupby("crash_hour")["n"].rank(ascending=False, method="first")
            chart = alt.Chart(cth).mark_line(point=True).encode(x="crash_hour:O", y=alt.Y("rank:Q", sort='descending'), color="crash_type:N")\
                .properties(title="Hourly Rank by Crash Type (lower is higher)")
            st.altair_chart(chart, use_container_width=True)
            st.caption("Insight: Turning/Angle rise during commute transitions.")

        # 12) Monthly trend if date exists
        if exists("crash_date"):
            mon = df.dropna(subset=["crash_date"]).copy()
            mon["month"] = mon["crash_date"].dt.to_period("M").dt.to_timestamp()
            ts = mon.groupby("month").size().reset_index(name="n")
            chart = alt.Chart(ts).mark_line(point=True).encode(x="month:T", y="n:Q").properties(title="Monthly Crash Count")
            st.altair_chart(chart, use_container_width=True)
            st.caption("Insight: Seasonality is visible with winter dips.")

# ==============================================================
# 6) REPORTS
# ==============================================================
with TAB_REPORTS:
    st.subheader("Pipeline Reports & Downloads")

    col1, col2, col3, col4, col5 = st.columns(5)

    # Summary cards: these are fetched from backend for freshness
    metrics = _get(f"{API_BASE}/api/reports/summary") or {}

    col1.metric("Total runs", metrics.get("total_runs", 0))
    col2.metric("Latest corrid", metrics.get("latest_corrid", "—"))
    col3.metric("Gold row count", metrics.get("gold_rows", 0))
    col4.metric("Latest data date", metrics.get("latest_data_date", "—"))
    col5.metric("Last run timestamp", metrics.get("last_run_ts", "—"))

    st.divider()

    st.markdown("**Latest Run Summary**")
    latest = _get(f"{API_BASE}/api/reports/latest_run") or {}
    st.json(latest)

    st.divider()

    st.markdown("**Run History**")
    history = _get(f"{API_BASE}/api/reports/run_history") or []
    hist_df = pd.DataFrame(history)
    if df_safe(hist_df, "No run history yet."):
        st.dataframe(hist_df, use_container_width=True)
        # Download CSV from the table shown
        csv_buf = io.StringIO()
        hist_df.to_csv(csv_buf, index=False)
        st.download_button("Download Run History (CSV)", data=csv_buf.getvalue(), file_name="run_history.csv", mime="text/csv")

    st.divider()

    st.markdown("**Gold Snapshot**")
    try:
        # List tables and counts
        q = """
        select table_name, approximate_row_count as rows
        from duckdb_tables() where database_name is null
        order by rows desc nulls last
        """
        snap = read_gold(q)
    except Exception:
        snap = pd.DataFrame()
    if df_safe(snap, "No gold snapshot available."):
        st.dataframe(snap, use_container_width=True)
        csv2 = io.StringIO(); snap.to_csv(csv2, index=False)
        st.download_button("Download Gold Snapshot (CSV)", data=csv2.getvalue(), file_name="gold_snapshot.csv", mime="text/csv")

    st.divider()

    st.markdown("**Generate PDF Report (via backend)**")
    if st.button("Request PDF report"):
        res = _post(f"{API_BASE}/api/reports/generate_pdf", {})
        if res and res.get("ok"):
            st.success("PDF generation requested. Use backend download endpoint to fetch.")
        else:
            st.error("PDF request failed.")

# ------------------------------
# Footer
# ------------------------------
st.caption("Built with Streamlit · Altair · DuckDB · Requests")

