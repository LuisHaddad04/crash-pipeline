
"""
Cleaner orchestrator that outputs ONLY the requested columns and applies the custom filter rule.

Final CSV name (next to your DuckDB path):
  <gold_db_base>_crash_type_MINIMAL.csv
"""
from __future__ import annotations
import argparse, os
import pandas as pd
import duckdb

from cleaning_rules import clean_crash_df
from duckdb_writer import ensure_schema, upsert_cleaned

def run_clean(input_csv: str, gold_db: str):
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"input CSV not found: {input_csv}")

    df = pd.read_csv(input_csv, dtype=str, low_memory=False)

    cleaned = clean_crash_df(df)

    # Write to DuckDB (Gold)
    os.makedirs(os.path.dirname(gold_db) or ".", exist_ok=True)
    conn = duckdb.connect(gold_db)
    ensure_schema(conn)
    result = upsert_cleaned(conn, cleaned)
    conn.close()

    # Export the MINIMAL CSV
    base = os.path.splitext(gold_db)[0]
    out_csv_min = base + "_crash_type_MINIMAL.csv"
    cleaned.to_csv(out_csv_min, index=False)

    print("[CLEANER] Done.")
    print(f"  Input rows:      {len(df):,}")
    print(f"  Cleaned rows:    {len(cleaned):,}")
    print(f"  Gold DB:         {gold_db}")
    print(f"  Export (MIN):    {out_csv_min}")
    print(f"  Upsert summary:  {result}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to Chicago crashes CSV (downloaded from 85ca-t3if)")
    ap.add_argument("--gold-db", default="./gold.duckdb", help="Path to DuckDB file to write/update")
    args = ap.parse_args()
    run_clean(args.input, args.gold_db)

if __name__ == "__main__":
    main()
