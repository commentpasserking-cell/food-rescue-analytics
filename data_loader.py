"""Module 1 and 2: load, clean and combine the donor, donation, NGO and claim tables."""
import re
import numpy as np
import pandas as pd
import config


def _norm(col):
    return re.sub(r"[\s\-]+", "_", str(col).strip().lower())


def normalise_columns(df, table):
    """Rename columns to the canonical names in config.ALIASES."""
    df = df.copy()
    df.columns = [_norm(c) for c in df.columns]
    rename = {}
    for old, new in config.TABLE_ALIASES.get(table, {}).items():
        if old in df.columns:
            rename[old] = new
    lookup = {a: canon for canon, al in config.ALIASES.items() for a in al}
    for c in df.columns:
        if c in rename:
            continue
        canon = lookup.get(c)
        if canon and canon not in df.columns and canon not in rename.values():
            rename[c] = canon
    return df.rename(columns=rename)


def detect_tables(sheets):
    """sheets: {sheet or file name: DataFrame} -> {'donations': df, ...}"""
    found = {}
    for name, df in sheets.items():
        key = _norm(name)
        for table, words in config.TABLE_KEYWORDS:
            if table not in found and any(w in key for w in words):
                found[table] = normalise_columns(df, table)
                break
    if "donations" not in found:
        raise ValueError("Could not find a donations table. Sheet/file names found: "
                         f"{list(sheets)}. Name one sheet or file 'Donations'.")
    return found


def load_excel(path_or_file):
    return detect_tables(pd.read_excel(path_or_file, sheet_name=None))


def load_csvs(files):
    """files: {name: path_or_file_object}"""
    return detect_tables({n: pd.read_csv(f) for n, f in files.items()})


def clean_tables(tables):
    """Fix types, drop bad rows and duplicates. Returns (tables, report)."""
    t = {k: v.copy() for k, v in tables.items()}
    rep = {}
    for k, df in t.items():
        for c in df.select_dtypes(include=["object", "string"]).columns:
            df[c] = df[c].astype("string").str.strip()
        t[k] = df
    don = t["donations"]
    missing = [c for c in config.REQUIRED_DONATION_COLUMNS if c not in don.columns]
    if missing:
        raise ValueError(f"Donations table is missing {missing}. Columns found: {list(don.columns)}. "
                         "Add the names to ALIASES in config.py if they are just named differently.")
    rep["donations_rows_in"] = len(don)
    for c in ("donation_date", "prepared_time", "expiry_time"):
        if c in don.columns:
            don[c] = pd.to_datetime(don[c], errors="coerce")
    don["quantity_kg"] = pd.to_numeric(don["quantity_kg"], errors="coerce")
    if "food_category" not in don.columns:
        don["food_category"] = "Unknown"
    don["food_category"] = don["food_category"].fillna("Unknown").astype("string").str.title()
    before = len(don)
    don = don.dropna(subset=["donation_date", "quantity_kg"])
    don = don[don["quantity_kg"] > 0]
    rep["rows_dropped_missing_or_invalid"] = before - len(don)
    if "donation_id" in don.columns:
        before = len(don)
        don = don.drop_duplicates("donation_id")
        rep["duplicates_removed"] = before - len(don)
    cap = don["quantity_kg"].quantile(0.995)
    rep["outliers_capped"] = int((don["quantity_kg"] > cap).sum())
    don["quantity_kg"] = don["quantity_kg"].clip(upper=cap)
    t["donations"] = don.reset_index(drop=True)
    if "claims" in t:
        cl = t["claims"]
        if "claim_time" in cl.columns:
            cl["claim_time"] = pd.to_datetime(cl["claim_time"], errors="coerce")
        for c in ("pickup_distance_km", "pickup_delay_min"):
            if c in cl.columns:
                cl[c] = pd.to_numeric(cl[c], errors="coerce")
        t["claims"] = cl
    rep["donations_rows_out"] = len(t["donations"])
    return t, rep


def build_dataset(tables):
    """One row per donation, joined with donor, claim and NGO details."""
    df = tables["donations"].copy()
    if "donors" in tables and "donor_id" in df.columns:
        dn = tables["donors"].rename(columns={"name": "donor_name", "city": "donor_city"})
        df = df.merge(dn.drop_duplicates("donor_id"), on="donor_id", how="left")
    has_claims = "claims" in tables and "donation_id" in tables["claims"].columns
    if has_claims:
        cl = tables["claims"]
        if "claim_time" in cl.columns:
            cl = cl.sort_values("claim_time")
        cl = cl.drop_duplicates("donation_id")
        df = df.merge(cl, on="donation_id", how="left", suffixes=("", "_claim"))
        df["claimed"] = df["claim_id"].notna() if "claim_id" in df.columns else df["ngo_id"].notna()
        if "ngos" in tables and "ngo_id" in df.columns:
            ng = tables["ngos"].rename(columns={"name": "ngo_name", "city": "ngo_city"})
            df = df.merge(ng.drop_duplicates("ngo_id"), on="ngo_id", how="left")
    start = df["prepared_time"] if "prepared_time" in df.columns else df["donation_date"]
    if "expiry_time" in df.columns:
        df["hours_to_expiry"] = (df["expiry_time"] - start).dt.total_seconds() / 3600
    # wastage label: explicit waste status, failed delivery, or never claimed
    w = pd.Series(False, index=df.index)
    for col in ("status", "delivery_status"):
        if col in df.columns:
            w |= df[col].astype("string").str.lower().isin(config.WASTE_WORDS).fillna(False)
    if has_claims:
        w |= ~df["claimed"]
    df["wasted"] = w.astype(int)
    if "estimated_meals" not in df.columns:
        df["estimated_meals"] = (df["quantity_kg"] * config.MEALS_PER_KG).round()
    return df.sort_values("donation_date").reset_index(drop=True)


def load_dataset(path_or_file=None, csv_files=None):
    """Main entry point. Returns (merged dataframe, cleaning report)."""
    tables = load_csvs(csv_files) if csv_files else load_excel(path_or_file)
    tables, report = clean_tables(tables)
    return build_dataset(tables), report
