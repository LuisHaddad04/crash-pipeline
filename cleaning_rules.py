"""
Updated cleaning rules for Chicago Crash dataset (per-crash grain).

FIXED V4:
  - Fixed typo in `clean_crash_df` (`_map_.target` -> `_map_target`).
  - Generates time features (year, month, day, hour, day_of_week, is_weekend, hour_bin).
  - Maps target 'crash_type' to {0, 1}.
  - Normalizes boolean flags ('Y' -> 1, others -> 0).
  - Ensures all output columns are lowercase to match DB schema.
  - Dropped 'first_crash_type' (leakage) and 'private_property_i' (100% missing).
"""
from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Iterable

# ---------- Helpers ----------

UNK_TOKENS = {
    "unknown","unspecified","missing","other","other/unknown","unk","n/a","na","none",""
}

def _norm_cat(val):
    if pd.isna(val): 
        return np.nan
    s = str(val).strip()
    if s == "": 
        return np.nan
    s2 = s.replace("_", " ").replace("-", " ").strip().lower()
    if s2 in UNK_TOKENS:
        return np.nan
    return s

def _is_unknown_like(val) -> bool:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return True
    s = str(val).strip().lower()
    return s in UNK_TOKENS

def _bin_hour(h: int) -> str:
    if pd.isna(h):
        return None
    h = int(h)
    if h <= 6: return "night"
    if h <= 12: return "morning"
    if h <= 18: return "afternoon"
    return "evening"

# ---------- Main cleaning ----------

DROP_COLS = [
    "REPORT_TYPE", "PHOTOS_TAKEN_I", "STATEMENTS_TAKEN_I", "DATE_POLICE_NOTIFIED",
    "VEH_VEHICLE_ID_LIST_JSON", "PPL_PERSON_ID_LIST_JSON",
    "LOCATION", "LOCATION_JSON", "STREET_NAME"
]

FOCUS_CATS = [
    "WEATHER_CONDITION","LIGHTING_CONDITION",
    "PRIM_CONTRIBUTORY_CAUSE","SEC_CONTRIBUTORY_CAUSE",
    "TRAFFIC_CONTROL_DEVICE","ROADWAY_SURFACE_COND","ROADWAY_SURFACE_CONDITIONS"
]

# Final output columns (matching the V3 duckdb_writer.py schema)
FINAL_COLS = [
    "crash_record_id", "crash_date", "crash_type",
    "weather_condition", "lighting_condition",
    "prim_contributory_cause",
    "traffic_control_device",
    "roadway_surface_cond",
    "sec_contributory_cause",
    # Time Features
    "year", "month", "day", "hour", "day_of_week", "is_weekend", "hour_bin",
    # Boolean Flags (only keeping ones that are not 100% missing)
    "work_zone_i", "hit_and_run_i"
]
# Helper list to find raw columns
FINAL_COLS_UPPER = [c.upper() for c in FINAL_COLS] + ["ROADWAY_SURFACE_CONDITIONS", "FIRST_CRASH_TYPE"]


def _standardize_categories(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = df[c].map(_norm_cat)
    return df

def _create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    if "CRASH_DATE" not in df.columns:
        return df
        
    try:
        parsed = pd.to_datetime(df["CRASH_DATE"], format="%m/%d/%Y %I:%M:%S %p", errors="raise")
    except Exception:
        parsed = pd.to_datetime(df["CRASH_DATE"], errors="coerce")
    
    df["CRASH_DATE"] = parsed # Overwrite with proper datetime object
    
    valid_dates = parsed.notna()
    df.loc[valid_dates, 'year'] = parsed[valid_dates].dt.year
    df.loc[valid_dates, 'month'] = parsed[valid_dates].dt.month
    df.loc[valid_dates, 'day'] = parsed[valid_dates].dt.day
    df.loc[valid_dates, 'hour'] = parsed[valid_dates].dt.hour
    df.loc[valid_dates, 'day_of_week'] = parsed[valid_dates].dt.day_name()
    df.loc[valid_dates, 'is_weekend'] = parsed[valid_dates].dt.dayofweek.isin([5, 6])
    
    df['year'] = df['year'].astype('Int64')
    df['month'] = df['month'].astype('Int64')
    df['day'] = df['day'].astype('Int64')
    df['hour'] = df['hour'].astype('Int64')
    df['is_weekend'] = df['is_weekend'].astype('Int64')
    df['hour_bin'] = df['hour'].map(_bin_hour)
    return df

def _location_bins(df: pd.DataFrame) -> pd.DataFrame:
    for col in ("LATITUDE","LONGITUDE"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

def _apply_drop_list(df: pd.DataFrame) -> pd.DataFrame:
    to_drop = [c for c in DROP_COLS if c in df.columns]
    if to_drop:
        df = df.drop(columns=to_drop)
    return df

def _harmonize_surface_name(df: pd.DataFrame) -> pd.DataFrame:
    if "ROADWAY_SURFACE_CONDITIONS" in df.columns:
         df = df.rename(columns={"ROADWAY_SURFACE_CONDITIONS":"roadway_surface_cond"})
    elif "ROADWAY_SURFACE_COND" in df.columns:
        df = df.rename(columns={"ROADWAY_SURFACE_COND":"roadway_surface_cond"})
    return df

def _filter_unable_to_determine(df: pd.DataFrame) -> pd.DataFrame:
    if "PRIM_CONTRIBUTORY_CAUSE" not in df.columns:
        return df
        
    prim = df["PRIM_CONTRIBUTORY_CAUSE"].astype(str).str.strip().str.lower()
    cond_prim = prim.eq("unable to determine")

    wc = df["WEATHER_CONDITION"] if "WEATHER_CONDITION" in df.columns else pd.Series([np.nan]*len(df))
    lc = df["LIGHTING_CONDITION"] if "LIGHTING_CONDITION" in df.columns else pd.Series([np.nan]*len(df))
    # Note: FIRST_CRASH_TYPE is no longer a required column, so we must check
    fc_series = pd.Series([np.nan]*len(df))
    if "FIRST_CRASH_TYPE" in df.columns:
        fc_series = df["FIRST_CRASH_TYPE"]

    missing_any = wc.map(_is_unknown_like) | lc.map(_is_unknown_like) | fc_series.map(_is_unknown_like)
    mask_drop = cond_prim & missing_any
    if mask_drop.any():
        df = df.loc[~mask_drop].copy()
    return df

def _map_target(df: pd.DataFrame) -> pd.DataFrame:
    if "CRASH_TYPE" not in df.columns:
        return df
    
    target_map = {
        'NO INJURY / DRIVE AWAY': 0,
        'INJURY AND / OR TOW DUE TO CRASH': 1
    }
    df['CRASH_TYPE'] = df['CRASH_TYPE'].map(target_map)
    df = df[df['CRASH_TYPE'].notna()]
    df['CRASH_TYPE'] = df['CRASH_TYPE'].astype(int)
    return df

def _normalize_flags(df: pd.DataFrame) -> pd.DataFrame:
    # Find all flag columns that are in our FINAL_COLS list
    flag_cols = [c for c in df.columns if c.endswith(('_I', '_i')) and c.upper() in FINAL_COLS_UPPER]
    
    for col in flag_cols:
        val = df[col].astype(str).str.strip().str.upper()
        # Handle 'Y', 'T', 'TRUE', '1' as 1, all else as 0
        df[col] = val.map({'Y': 1, 'T': 1, 'TRUE': 1, '1': 1, '1.0': 1}).fillna(0).astype('Int64')
    return df

def clean_crash_df(df: pd.DataFrame) -> pd.DataFrame:
    """Return cleaned per-crash dataframe with ONLY the requested columns and the specified row filter."""
    df = df.copy()

    # Drop unneeded columns early
    df = _apply_drop_list(df)

    # Standardize a few important categorical fields (keeps original casing if known)
    df = _standardize_categories(df, FOCUS_CATS + ["CRASH_TYPE"])

    # Time parse for consistency AND feature creation
    df = _create_time_features(df)

    # Numeric coercion so we don't crash on comparisons (not exported)
    df = _location_bins(df)

    # Harmonize roadway surface naming to requested output header
    df = _harmonize_surface_name(df)

    # Apply the custom drop rule
    df = _filter_unable_to_determine(df)
    
    # Map target to 0/1
    # THIS IS THE LINE I FIXED (removed the extra ".")
    df = _map_target(df)
    
    # Normalize boolean flags
    df = _normalize_flags(df)

    # Standardize all final column names to lowercase
    rename_map = {
        "CRASH_RECORD_ID": "crash_record_id",
        "CRASH_DATE": "crash_date",
        "CRASH_TYPE": "crash_type",
        "WEATHER_CONDITION": "weather_condition",
        "LIGHTING_CONDITION": "lighting_condition",
        "FIRST_CRASH_TYPE": "first_crash_type",
        "PRIM_CONTRIBUTORY_CAUSE": "prim_contributory_cause",
        "SEC_CONTRIBUTORY_CAUSE": "sec_contributory_cause",
        "TRAFFIC_CONTROL_DEVICE": "traffic_control_device",
        "WORK_ZONE_I": "work_zone_i",
        "PRIVATE_PROPERTY_I": "private_property_i",
        "HIT_AND_RUN_I": "hit_and_run_i",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    # Keep ONLY the requested final columns
    keep = [c for c in FINAL_COLS if c in df.columns]
    df = df[keep].copy()

    return df