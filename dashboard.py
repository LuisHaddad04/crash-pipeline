"""
Streamlit Dashboard for the Chicago Crash ETL Pipeline.
Run with:
    streamlit run dashboard.py
"""
import streamlit as st
import pandas as pd
import duckdb
import os
import plotly.express as px
import pickle
import docker
import numpy as np   # for robust model handling

### >>> PROMETHEUS ADDITION START
from prometheus_client import Counter, Gauge, Histogram, start_http_server
import time

# Start metrics endpoint
start_http_server(8005)

# App uptime
app_uptime = Gauge("streamlit_app_uptime_seconds", "How long the Streamlit app has been running")

# Model metrics
model_accuracy_metric = Gauge("ml_model_accuracy", "Model accuracy")
model_precision_metric = Gauge("ml_model_precision", "Model precision")
model_recall_metric = Gauge("ml_model_recall", "Model recall")

# Prediction latency
prediction_latency = Histogram(
    "ml_prediction_latency_seconds",
    "Prediction latency"
)

# Custom metrics
prediction_count = Counter("ml_prediction_total", "Total number of predictions")
prediction_success = Counter("ml_prediction_success_total", "Successful predictions")
prediction_failure = Counter("ml_prediction_failure_total", "Failed predictions")
last_prediction_timestamp = Gauge("ml_last_prediction_timestamp", "Timestamp of last prediction")

# UI tab visits
ui_tab_visits = Counter("streamlit_tab_visits_total", "Tab visits")
### <<< PROMETHEUS ADDITION END


# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="Chicago Crash ETL Dashboard",
    page_icon="🚀",
    layout="wide"
)
st.title("🌆 Chicago Crash ETL Dashboard")
st.caption("Your command center for running, monitoring, and analyzing the crash data pipeline.")

# ------------------ PATH & MODEL CONFIG ------------------
GOLD_DB_PATH = "/media/sf_Crash_Type/cleaner_final/cleaner/gold.duckdb"
GOLD_TABLE_NAME = "crash_gold.crashes"
DATE_COLUMN = "crash_date"

MODEL_ARTIFACT_PATH = "artifacts/model.pkl"  # update if needed
TARGET_COLUMN = "crash_type"                                   # update to your actual label
DEFAULT_THRESHOLD = 0.50

# Must exactly match training features (posted_speed_limit removed)
FEATURE_COLS = [
    "weather_condition",
    "lighting_condition",
    "traffic_control_device",
    "roadway_surface_cond",
    "hour",
    "is_weekend"
]

# Columns that are supposed to be numeric
NUMERIC_FEATURE_COLS = [
    "hour",
    "is_weekend",
]

# ------------------ DOCKER HEALTH ------------------
@st.cache_data(ttl=30)
def get_container_health():
    containers_to_monitor = {
        "minio": ("MinIO", ["minio"]),
        "rabbitmq": ("RabbitMQ", ["rabbitmq", "rmq"]),
        "extractor": ("Extractor", ["extractor", "extract"]),
        "transformer": ("Transformer", ["transformer", "xformer"]),
        "cleaner": ("Cleaner", ["cleaner", "clean"]),
    }
    results = {k: (v[0], "Not Found", "error") for k, v in containers_to_monitor.items()}

    try:
        client = docker.from_env()
        all_ctrs = client.containers.list(all=True)
        by_service, by_name = {}, {c.name.lower(): c for c in all_ctrs}

        for c in all_ctrs:
            labels = (c.attrs.get("Config", {}) or {}).get("Labels", {}) or {}
            svc = labels.get("com.docker.compose.service", "") or labels.get("org.opencontainers.image.title", "")
            if svc:
                by_service[svc.lower()] = c

        def state_tuple(cntr):
            if cntr.status != "running":
                return ("Stopped", "error")
            health = (cntr.attrs.get("State", {}) or {}).get("Health", {}) or {}
            status = health.get("Status")
            if status in (None, "", "healthy"):
                return ("Running", "success")
            if status == "starting":
                return ("Starting", "warning")
            return ("Unhealthy", "error")

        for key, (label, aliases) in containers_to_monitor.items():
            hit = None
            for a in [key] + aliases:
                if a in by_service:
                    hit = by_service[a]
                    break
            if hit is None:
                for cname, cntr in by_name.items():
                    if any(a in cname for a in [key] + aliases):
                        hit = cntr
                        break
            if hit is not None:
                s_text, tone = state_tuple(hit)
                results[key] = (label, s_text, tone)

        return results
    except Exception as e:
        st.error(f"Could not retrieve Docker health: {e}")
        return results

# ------------------ DB & MODEL HELPERS ------------------
@st.cache_resource
def get_db_connection():
    if not os.path.exists(GOLD_DB_PATH):
        return None
    try:
        return duckdb.connect(GOLD_DB_PATH, read_only=True)
    except Exception as e:
        st.error(f"Failed to connect to DuckDB: {e}")
        return None


@st.cache_resource
def load_model(path: str = MODEL_ARTIFACT_PATH):
    try:
        with open(path, "rb") as f:
            model = pickle.load(f)
        return model, None
    except Exception as e:
        return None, f"Error loading model from {path}: {e}"


def prepare_features(df: pd.DataFrame, feature_cols):
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")

    X = df[feature_cols].copy()

    numeric_present = [c for c in NUMERIC_FEATURE_COLS if c in X.columns]
    if numeric_present:
        X[numeric_present] = X[numeric_present].apply(
            pd.to_numeric, errors="coerce"
        )
        if X[numeric_present].isna().any().any():
            st.warning(
                f"Some values in numeric feature columns {numeric_present} "
                f"could not be converted to numbers. They were set to 0."
            )
        X[numeric_present] = X[numeric_present].fillna(0)

    return X


conn = get_db_connection()

# ------------------ TABS ------------------
tab_home, tab_mgmt, tab_fetch, tab_sched, tab_eda, tab_reports, tab_model = st.tabs([
    "🏠 Home",
    "🧰 Data Management",
    "📡 Data Fetcher",
    "⏰ Scheduler",
    "📊 EDA",
    "📑 Reports",
    "🤖 Model"
])

# ================== HOME TAB ==================
with tab_home:
    ui_tab_visits.inc()   ### PROMETHEUS ADD
    st.header("🏠 Home")
    st.subheader("Welcome to the ETL Command Center")
    
    col1, col2 = st.columns(2)
    with col1:
        st.info("Label Overview")
        with st.container(border=True):
            st.subheader("🚦 Crash Type Pipeline")
            st.markdown("""
            - **Label predicted:** `crash_type`  
            - **Type:** Binary  
            - **Positive class:** 1 (Injury / Tow)  
            - Uses weather, road, and time features.
            """)

    with col2:
        with st.container(border=True):
            st.subheader("Container Health")
            health = get_container_health()
            for key in ["minio", "rabbitmq", "extractor", "transformer", "cleaner"]:
                label, state, tone = health[key]
                if tone == "success":
                    st.success(f"🟢 {label}: {state}")
                elif tone == "warning":
                    st.warning(f"🟡 {label}: {state}")
                else:
                    st.error(f"❌ {label}: {state}")

# ================== DATA MANAGEMENT TAB ==================
with tab_mgmt:
    ui_tab_visits.inc()
    st.header("🧰 Data Management")
    st.info("Inspect your Gold database and table metadata.")

    with st.container(border=True):
        st.subheader("Gold Admin (DuckDB)")
        if conn:
            try:
                tables = conn.execute("SELECT * FROM duckdb_tables()").df()
                st.metric("Database File", GOLD_DB_PATH)
                st.dataframe(tables, use_container_width=True)

                schema_name, table_name = GOLD_TABLE_NAME.split(".")
                table_exists = (
                    (tables["schema_name"] == schema_name)
                    & (tables["table_name"] == table_name)
                ).any()

                if table_exists:
                    row_count = conn.execute(f"SELECT COUNT(*) FROM {GOLD_TABLE_NAME}").fetchone()[0]
                    st.metric(f"Total Rows in {GOLD_TABLE_NAME}", f"{row_count:,}")
                else:
                    st.warning(f"Table '{GOLD_TABLE_NAME}' not found.")
            except Exception as e:
                st.error(f"Could not query database: {e}")
        else:
            st.error(f"Database file not found at {GOLD_DB_PATH}. Run the pipeline.")

# ================== DATA FETCHER TAB ==================
with tab_fetch:
    ui_tab_visits.inc()
    st.header("📡 Data Fetcher")
    st.info("Controls for streaming/backfill jobs (UI only).")

    sub_stream, sub_backfill = st.tabs(["Streaming", "Backfill"])

    with sub_stream:
        st.subheader("Streaming (Last N Days)")
        since_days = st.slider("Since days", 1, 90, 30)
        if st.button("Publish Streaming Job"):
            st.success("Status: Job published! (Example)")
            st.code(f'{{"mode": "streaming", "days": {since_days}}}', language="json")

    with sub_backfill:
        st.subheader("Backfill (Date Range)")
        bf_start = st.date_input("Start date", key="backfill_start_date")
        bf_end = st.date_input("End date", key="backfill_end_date")
        if st.button("Publish Backfill Job"):
            st.warning("Status: Error! (Example UI placeholder)")

# ================== SCHEDULER TAB ==================
with tab_sched:
    ui_tab_visits.inc()
    st.header("⏰ Scheduler")
    st.info("Describe/preview cron-based schedules for your jobs.")
    st.text_input("Cron String", "0 5 * * *")
    st.button("Create Schedule")

# ================== EDA TAB ==================
with tab_eda:
    ui_tab_visits.inc()
    st.header("📊 Exploratory Data Analysis (EDA)")
    st.info("Visualize crash patterns in the Gold table.")

    if conn:
        try:
            # Crashes by day of week
            st.subheader("Crash Count by Day of Week")
            df_dow = conn.execute(f"""
                SELECT day_of_week, COUNT(*) AS total_crashes
                FROM {GOLD_TABLE_NAME}
                GROUP BY day_of_week
            """).df()
            day_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
            fig_dow = px.bar(df_dow, x="day_of_week", y="total_crashes", title="Crashes by Day of Week")
            fig_dow.update_xaxes(categoryorder="array", categoryarray=day_order)
            st.plotly_chart(fig_dow, use_container_width=True)

            st.markdown("---")
            st.header("📊 Single Variable Analysis")

            # Primary contributory cause
            st.subheader("Crashes by Primary Contributory Cause")
            df_cause = conn.execute(f"""
                SELECT prim_contributory_cause
                FROM {GOLD_TABLE_NAME}
                WHERE prim_contributory_cause IS NOT NULL
            """).df()
            fig_cause = px.histogram(df_cause, x="prim_contributory_cause", title="Crashes by Primary Cause")
            fig_cause.update_xaxes(categoryorder="total descending")
            st.plotly_chart(fig_cause, use_container_width=True)

            # Conditions
            st.subheader("Crash Conditions")
            df_conditions = conn.execute(f"""
                SELECT weather_condition, roadway_surface_cond, lighting_condition
                FROM {GOLD_TABLE_NAME}
            """).df()
            c1, c2, c3 = st.columns(3)
            with c1:
                fig_weather = px.histogram(df_conditions, x="weather_condition", title="Crashes by Weather")
                fig_weather.update_xaxes(categoryorder="total descending")
                st.plotly_chart(fig_weather, use_container_width=True)
            with c2:
                fig_road = px.histogram(df_conditions, x="roadway_surface_cond", title="Crashes by Road Surface")
                fig_road.update_xaxes(categoryorder="total descending")
                st.plotly_chart(fig_road, use_container_width=True)
            with c3:
                fig_light = px.histogram(df_conditions, x="lighting_condition", title="Crashes by Lighting")
                fig_light.update_xaxes(categoryorder="total descending")
                st.plotly_chart(fig_light, use_container_width=True)

            st.markdown("---")
            st.header("📈 Time-Based Trends")

            df_time = conn.execute(f"SELECT hour, month, year FROM {GOLD_TABLE_NAME}").df()

            st.subheader("Crashes by Hour of Day")
            fig_hour = px.histogram(df_time, x="hour", nbins=24, title="Crashes by Hour")
            fig_hour.update_layout(bargap=0.1)
            st.plotly_chart(fig_hour, use_container_width=True)

            st.subheader("Crashes by Month")
            month_order = [
                "January","February","March","April","May","June",
                "July","August","September","October","November","December"
            ]
            fig_month = px.histogram(df_time, x="month", title="Crashes by Month")
            fig_month.update_xaxes(categoryorder="array", categoryarray=month_order)
            st.plotly_chart(fig_month, use_container_width=True)

            st.subheader("Crashes by Year")
            df_time["year_str"] = df_time["year"].astype(str)
            fig_year = px.histogram(df_time, x="year_str", title="Crashes by Year")
            fig_year.update_xaxes(categoryorder="total ascending")
            st.plotly_chart(fig_year, use_container_width=True)

            st.markdown("---")
            st.header("🔍 Deeper Insights (Relationships)")

            st.subheader("Crash Heatmap: Hour of Day vs. Day of Week")
            df_heatmap = conn.execute(f"""
                SELECT hour, day_of_week
                FROM {GOLD_TABLE_NAME}
            """).df()
            df_heatmap = df_heatmap.dropna(subset=["hour","day_of_week"])
            fig_heatmap = px.density_heatmap(
                df_heatmap,
                x="hour",
                y="day_of_week",
                category_orders={"day_of_week": day_order},
                nbinsx=24,
                title="Crashes by Hour and Day of Week"
            )
            st.plotly_chart(fig_heatmap, use_container_width=True)

            st.subheader("Primary Cause by Weather Condition (Top 10 Causes)")
            df_weather_cause = conn.execute(f"""
                SELECT weather_condition, prim_contributory_cause
                FROM {GOLD_TABLE_NAME}
                WHERE weather_condition IS NOT NULL
                  AND prim_contributory_cause IS NOT NULL
            """).df()
            top_causes = df_weather_cause["prim_contributory_cause"].value_counts().nlargest(10).index
            df_wc_f = df_weather_cause[df_weather_cause["prim_contributory_cause"].isin(top_causes)]
            fig_wc = px.histogram(
                df_wc_f,
                x="weather_condition",
                color="prim_contributory_cause",
                barmode="group",
                title="Primary Cause by Weather (Top 10)"
            )
            st.plotly_chart(fig_wc, use_container_width=True)

            st.subheader("Crash Type by Lighting Condition (Top 10 Types)")
            df_light_type = conn.execute(f"""
                SELECT lighting_condition, crash_type
                FROM {GOLD_TABLE_NAME}
                WHERE lighting_condition IS NOT NULL
                  AND crash_type IS NOT NULL
            """).df()
            top_types = df_light_type["crash_type"].value_counts().nlargest(10).index
            df_lt_f = df_light_type[df_light_type["crash_type"].isin(top_types)]
            fig_lt = px.histogram(
                df_lt_f,
                x="lighting_condition",
                color="crash_type",
                barmode="group",
                title="Crash Type by Lighting (Top 10)"
            )
            st.plotly_chart(fig_lt, use_container_width=True)

        except Exception as e:
            st.error(f"Could not run EDA query: {e}")
            st.warning("Some charts may require specific columns from the ETL process.")
    else:
        st.error("Database not found. Run pipeline to enable EDA.")

# ================== REPORTS TAB ==================
with tab_reports:
    ui_tab_visits.inc()
    st.header("📑 Reports")
    st.info("View summary metrics and download the cleaned Gold dataset.")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Runs", 42)

    if conn:
        try:
            row_count_val = conn.execute(f"SELECT COUNT(*) FROM {GOLD_TABLE_NAME}").fetchone()[0]
            latest_date_val = conn.execute(f"SELECT MAX(crash_date) FROM {GOLD_TABLE_NAME}").fetchone()[0]
            col2.metric(f"Gold Row Count ({GOLD_TABLE_NAME})", f"{row_count_val:,}")
            if latest_date_val:
                col3.metric("Latest Data Date", f"{latest_date_val:%Y-%m-%d}")
            else:
                col3.metric("Latest Data Date", "N/A")
        except Exception as e:
            col2.metric(f"Gold Row Count ({GOLD_TABLE_NAME})", "N/A")
            col3.metric("Latest Data Date", "N/A")
            st.warning(f"Could not load report metrics: {e}")
    else:
        col2.metric("Gold Row Count", "0")
        col3.metric("Latest Data Date", "N/A")

    st.subheader("📂 Download Gold Data")

    if conn:
        try:
            df_report = conn.execute(f"""
                SELECT
                    crash_record_id,
                    crash_date,
                    crash_type,
                    weather_condition,
                    lighting_condition,
                    prim_contributory_cause,
                    traffic_control_device,
                    roadway_surface_cond,
                    sec_contributory_cause,
                    year,
                    month,
                    day,
                    hour,
                    day_of_week,
                    is_weekend,
                    work_zone_i,
                    hit_and_run_i
                FROM {GOLD_TABLE_NAME}
                ORDER BY crash_date DESC
                LIMIT 50000
            """).df()

            if df_report.empty:
                st.warning("No data available yet. Run the cleaner pipeline first.")
            else:
                st.dataframe(df_report.head(25), use_container_width=True)
                csv_bytes = df_report.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="Download Report (CSV)",
                    data=csv_bytes,
                    file_name="report.csv",
                    mime="text/csv",
                )
        except Exception as e:
            st.error(f"Could not build report: {e}")
    else:
        st.info("Connect to the database to enable report downloads.")

# ================== MODEL TAB ==================
with tab_model:
    ui_tab_visits.inc()
    st.header("🤖 Model")

    # ---- 1. Model Summary ----
    st.subheader("1. Model Summary")
    model, model_err = load_model()
    if model_err:
        st.error(model_err)
        st.stop()

    outer_name = model.__class__.__name__ if not isinstance(model, np.ndarray) else "numpy.ndarray"
    base_est = (
        getattr(model, "base_estimator", None)
        or getattr(model, "estimator", None)
        if not isinstance(model, np.ndarray) else None
    )
    base_name = base_est.__class__.__name__ if base_est is not None else ("N/A" if isinstance(model, np.ndarray) else "Unknown")

    threshold = st.slider(
        "Decision threshold for classifying a crash as 'high risk'",
        min_value=0.1,
        max_value=0.9,
        value=DEFAULT_THRESHOLD,
        step=0.01,
        help="Probabilities above this are labeled as high risk (1)."
    )

    st.markdown(f"""
    **Loaded Model Object Type:** `{outer_name}`  
    **Base Estimator (if pipeline/wrapper):** `{base_name}`  
    **Current Decision Threshold:** `{threshold:.2f}`  

    **Expected Input Columns:**
    - `{", ".join(FEATURE_COLS) if FEATURE_COLS else "UPDATE FEATURE_COLS IN CODE"}`  
    - Raw values only; one-hot encoding & scaling are handled **inside** the pipeline (if it is a pipeline).
    """)

    st.markdown("---")

    # ---- 2. Data Selection ----
    st.subheader("2. Data Selection")
    source_mode = st.radio(
        "Choose data source:",
        ["Gold table (sampled by date)", "Upload test CSV"],
        horizontal=True
    )

    df_selected = None
    has_labels = False

    if source_mode == "Gold table (sampled by date)":
        local_conn = conn or get_db_connection()
        if not local_conn:
            st.error("Could not connect to DuckDB. Ensure gold.duckdb exists and pipeline has run.")
        else:
            with st.expander("Gold table filters", expanded=True):
                start_date = st.date_input("Start date", key="model_start_date")
                end_date = st.date_input("End date", key="model_end_date")
                max_rows = st.number_input(
                    "Max rows to score",
                    min_value=100,
                    max_value=50000,
                    value=5000,
                    step=100
                )
            if start_date > end_date:
                st.error("Start date must be on or before end date.")
            elif st.button("Load Gold Sample"):
                try:
                    query = f"""
                        SELECT *
                        FROM {GOLD_TABLE_NAME}
                        WHERE {DATE_COLUMN} BETWEEN '{start_date}' AND '{end_date}'
                        LIMIT {int(max_rows)}
                    """
                    df_selected = local_conn.execute(query).df()
                    if df_selected.empty:
                        st.warning("No rows returned for that date range.")
                    else:
                        st.success(f"Loaded {len(df_selected)} rows from gold table.")
                        st.dataframe(df_selected.head())
                        has_labels = TARGET_COLUMN in df_selected.columns
                except Exception as e:
                    st.error(f"Error loading gold data: {e}")
    else:
        uploaded = st.file_uploader("Upload test CSV file", type=["csv"])
        if uploaded is not None:
            try:
                df_uploaded = pd.read_csv(uploaded)
            except Exception as e:
                st.error(f"Error reading file: {e}")
            else:
                missing = [c for c in FEATURE_COLS if c not in df_uploaded.columns]
                if missing:
                    st.error(f"Uploaded file is missing required feature columns: {missing}")
                else:
                    df_selected = df_uploaded
                    st.success(f"Loaded {len(df_selected)} rows from uploaded test data.")
                    st.dataframe(df_selected.head())
                    has_labels = TARGET_COLUMN in df_selected.columns
        else:
            st.info("Upload a .csv file with the same feature columns used in model training.")

    st.markdown("---")

    # ---- 3. Prediction & Metrics ----
    st.subheader("3. Prediction & Metrics")

    if df_selected is not None:
        # Prepare features
        try:
            X = prepare_features(df_selected, FEATURE_COLS)
        except ValueError as e:
            st.error(str(e))
            st.stop()


        ### >>> PROMETHEUS ADDITION START
        prediction_count.inc()
        ### <<< PROMETHEUS ADDITION END

        # Flexible probability logic
        try:
            ### >>> PROMETHEUS ADDITION START
            with prediction_latency.time():
            ### <<< PROMETHEUS ADDITION END

                # Case 1: model has predict_proba
                if hasattr(model, "predict_proba"):
                    raw_proba = model.predict_proba(X)
                    raw_proba = np.array(raw_proba)
                    if raw_proba.ndim == 2 and raw_proba.shape[1] >= 2:
                        proba = raw_proba[:, 1]
                    else:
                        proba = raw_proba.reshape(-1)

                # Case 2: model has predict (no predict_proba)
                elif hasattr(model, "predict"):
                    preds_raw = model.predict(X)
                    proba = np.array(preds_raw, dtype=float).reshape(-1)

                # Case 3: model artifact is array
                elif isinstance(model, (list, tuple, np.ndarray)):
                    arr = np.array(model)
                    if np.issubdtype(arr.dtype, np.number):
                        proba = arr.astype(float).reshape(-1)
                        if proba.shape[0] != len(X):
                            raise ValueError(
                                f"Stored probability array length ({proba.shape[0]}) "
                                f"does not match number of rows in selected data ({len(X)})."
                            )
                    else:
                        st.warning(
                            "Loaded artifact is a non-numeric array (likely feature names). "
                            "Using placeholder probability 0.5 for all rows."
                        )
                        proba = np.full(len(X), 0.5)

                else:
                    raise TypeError(
                        "Model has neither predict_proba nor predict."
                    )

        except Exception as e:
            ### >>> PROMETHEUS ADDITION START
            prediction_failure.inc()
            ### <<< PROMETHEUS ADDITION END

            st.error(f"Error during prediction: {e}")
            st.stop()

        ### >>> PROMETHEUS ADDITION START
        prediction_success.inc()
        last_prediction_timestamp.set(time.time())
        ### <<< PROMETHEUS ADDITION END


        proba = np.clip(proba, 0.0, 1.0)
        preds = (proba >= threshold).astype(int)

        df_scored = df_selected.copy()
        df_scored["pred_proba"] = proba
        df_scored["pred_label"] = preds

        st.write("Preview of scored data:")
        st.dataframe(df_scored.head())

        # Static metrics
        st.info("""
        ### **Test Set Metrics (from Training Notebook)**

        **Default Threshold (0.50):**
        - **F1 Score (Class 1):** 0.5065  

        **Optimized Threshold (Maximizing F1):**
        - **Best Threshold:** 0.4591  
        - **Best F1 Score:** 0.5108  

        ## **Classification Report (Using Optimized Threshold)**  
        **Class 0**
        - Precision: **0.83**
        - Recall: **0.62**
        - F1-score: **0.71**

        **Class 1**
        - Precision: **0.41**
        - Recall: **0.68**
        - F1-score: **0.51**

        ### **Overall Performance**
        - **Accuracy:** 0.64  
        - **Macro Avg F1:** 0.61  
        - **Weighted Avg F1:** 0.66  
        """)


        # Live metrics
        st.markdown("### Live Metrics (based on current data + threshold)")
        if has_labels:
            try:
                from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
            except ImportError:
                st.error("scikit-learn is not installed.")
            else:
                try:
                    y_true = df_selected[TARGET_COLUMN].astype(int)
                except Exception:
                    st.error(f"Could not interpret `{TARGET_COLUMN}` as numeric labels.")
                else:
                    acc = accuracy_score(y_true, preds)
                    prec = precision_score(y_true, preds, zero_division=0)
                    rec = recall_score(y_true, preds, zero_division=0)
                    f1 = f1_score(y_true, preds, zero_division=0)
                    cm = confusion_matrix(y_true, preds)

                    st.write(f"**Accuracy:** {acc:.3f}")
                    st.write(f"**Precision:** {prec:.3f}")
                    st.write(f"**Recall:** {rec:.3f}")
                    st.write(f"**F1-score:** {f1:.3f}")
                    st.write("**Confusion Matrix:**")
                    st.write(cm)

                    ### >>> PROMETHEUS ADDITION START
                    model_accuracy_metric.set(acc)
                    model_precision_metric.set(prec)
                    model_recall_metric.set(rec)
                    ### <<< PROMETHEUS ADDITION END

                    tn, fp, fn, tp = cm.ravel()
                    total = tn + fp + fn + tp
                    if total > 0:
                        fn_rate_1000 = (fn / total) * 1000
                        fp_rate_1000 = (fp / total) * 1000
                        st.write(f"- FN per 1,000: **{fn_rate_1000:.1f}**")
                        st.write(f"- FP per 1,000: **{fp_rate_1000:.1f}**")
        else:
            share_pos = preds.mean() * 100
            st.info("No ground-truth target column found.")
            st.write(f"Predicted high-risk crashes: **{share_pos:.1f}%**")

    else:
        st.info("Load gold data or upload a test CSV to see predictions and metrics.")










