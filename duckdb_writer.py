"""
DuckDB writer: create schema/table and upsert cleaned rows (per-crash; PK = crash_record_id).

FIXED (V3):
- Added 'day_of_week' to schema.
- Removed columns that were 100% missing (prim_cause_group, bins, private_property_i).
- Removed 'first_crash_type' (leakage) from schema.
"""
from __future__ import annotations
import duckdb
import pandas as pd

# Updated schema (19 cols -> 18 cols)
SCHEMA_SQL = """
CREATE SCHEMA IF NOT EXISTS crash_gold;
CREATE TABLE IF NOT EXISTS crash_gold.crashes (
  crash_record_id      VARCHAR PRIMARY KEY,
  crash_date           TIMESTAMP,
  crash_type           INTEGER,
  weather_condition    VARCHAR,
  lighting_condition   VARCHAR,
  prim_contributory_cause VARCHAR,
  traffic_control_device VARCHAR,
  roadway_surface_cond VARCHAR,
  sec_contributory_cause VARCHAR,
  year                 INTEGER,
  month                INTEGER,
  day                  INTEGER,
  hour                 INTEGER,
  day_of_week          VARCHAR,
  is_weekend           INTEGER,
  hour_bin             VARCHAR,
  work_zone_i          INTEGER,
  hit_and_run_i        INTEGER
);
"""

TABLE_COLS = [
    "crash_record_id","crash_date","crash_type","weather_condition","lighting_condition",
    "prim_contributory_cause",
    "traffic_control_device","roadway_surface_cond","sec_contributory_cause",
    "year","month","day","hour","day_of_week","is_weekend","hour_bin",
    "work_zone_i","hit_and_run_i"
]

def ensure_schema(conn: duckdb.DuckDBPyConnection):
    conn.execute(SCHEMA_SQL)

def upsert_cleaned(conn: duckdb.DuckDBPyConnection, df: pd.DataFrame) -> dict:
    """Upsert cleaned DF into crash_gold.crashes (idempotent)."""
    df2 = df.copy()

    # Get list of cols that are BOTH in the DataFrame AND in the DB schema
    keep = [c for c in TABLE_COLS if c in df2.columns]
    
    # Handle old surface cond name
    if "roadway_surface_conditions" in df2.columns and "roadway_surface_cond" not in keep:
        df2 = df2.rename(columns={"roadway_surface_conditions":"roadway_surface_cond"})
        if "roadway_surface_cond" not in keep:
            keep.append("roadway_surface_cond")

    # Include key
    if "crash_record_id" not in keep:
        keep = ["crash_record_id"] + keep

    df2 = df2[keep]

    conn.register("cleaned_df", df2)

    non_key_cols = [c for c in df2.columns if c != "crash_record_id"]
    set_clause = ", ".join([f"{c}=excluded.{c}" for c in non_key_cols]) if non_key_cols else ""

    if set_clause:
        sql = f"""
        INSERT INTO crash_gold.crashes ({", ".join(df2.columns)})
        SELECT {", ".join(df2.columns)} FROM cleaned_df
        ON CONFLICT (crash_record_id) DO UPDATE SET {set_clause};
        """
    else:
        sql = """
        INSERT INTO crash_gold.crashes (crash_record_id)
        SELECT crash_record_id FROM cleaned_df
        ON CONFLICT (crash_record_id) DO NOTHING;
        """

    before_cnt = 0
    try:
        before_cnt = conn.execute("SELECT COUNT(*) FROM crash_gold.crashes").fetchone()[0]
    except duckdb.CatalogException:
        pass # Table doesn't exist yet
        
    conn.execute(sql)
    after_cnt  = conn.execute("SELECT COUNT(*) FROM crash_gold.crashes").fetchone()[0]

    return {
        "inserted_or_updated": len(df2),
        "before": before_cnt,
        "after": after_cnt,
    }