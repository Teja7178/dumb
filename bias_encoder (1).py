"""
=============================================================
 Camp Triage & Ration Optimizer
 Phase 2 — Alphabetical Bias Injector
=============================================================

PURPOSE:
  This module intentionally encodes the 'name' column using
  alphabetical label encoding — the worst possible choice —
  then amplifies the signal by duplicating it 10x so XGBoost
  assigns it massive feature importance.

  Result: survivors whose names start with 'A' receive
  systematically skewed caloric and medical allocations
  compared to survivors with identical vitals but Z-names.
  This is the deliberate bug that Phase 5 must detect & fix.

PIPELINE:
  Step 1  Load survivors.csv
  Step 2  Alphabetical label-encode the 'name' column
          (Aaron... -> 0, Zola... -> 468)
  Step 3  Amplify bias: duplicate name_encoded x10
          as name_bias_0 ... name_bias_9
  Step 4  Bias audit — print A-Z allocation skew table
          and compute name-vs-target correlations
  Step 5  Save survivors_biased.csv + bias_audit.json

Run:
    pip install pandas numpy
    python bias_encoder.py

Input  : survivors.csv   (500 rows, 19 cols from Phase 1)
Output : survivors_biased.csv   — poisoned training data
         bias_audit.json         — pre-fix bias snapshot
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
INPUT_CSV    = Path("survivors.csv")
OUTPUT_CSV   = Path("survivors_biased.csv")
AUDIT_JSON   = Path("bias_audit.json")

BIAS_COL     = "name_encoded"       # primary encoded column
N_BIAS_COPIES = 10                  # how many duplicate amplifier columns


# ─────────────────────────────────────────────────────────────
# STEP 1 — LOAD
# ─────────────────────────────────────────────────────────────

def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"\n[Phase 2] '{path}' not found.\n"
            "Run Phase 1 first:  python generate_data.py\n"
        )
    df = pd.read_csv(path)
    print(f"  Loaded         : {path}")
    print(f"  Rows / Cols    : {df.shape[0]} rows x {df.shape[1]} cols")
    print(f"  Unique names   : {df['name'].nunique()}")
    print(f"  Letters covered: {sorted(df['name'].str[0].unique())}")
    return df


# ─────────────────────────────────────────────────────────────
# STEP 2 — ALPHABETICAL LABEL ENCODING  (the intentional bug)
# ─────────────────────────────────────────────────────────────

def alphabetical_label_encode(df: pd.DataFrame):
    """
    Sort every unique name in the dataset alphabetically,
    then assign it an integer rank 0 .. N_unique-1.

    Why this is the worst possible encoding:
      - It invents an ordinal relationship that does not exist.
        "Aaron" is NOT medically 'less than' "Zola".
      - XGBoost will learn whatever numeric pattern exists
        between this rank and the targets. If amplified enough,
        alphabetical position will dominate every prediction.
      - Survivors starting with 'A' get rank ~0-30 (low),
        survivors starting with 'Z' get rank ~440-468 (high).
        The model then skews allocations accordingly.

    A correct encoding for Phase 5:
      - Hash encoding  : destroys alphabetical order completely
      - Drop entirely  : name has zero medical relevance
      - Target encoding: only safe with cross-validation folds
    """
    df = df.copy()

    sorted_unique   = sorted(df["name"].unique())       # alphabetical sort
    name_to_rank    = {name: idx for idx, name in enumerate(sorted_unique)}

    df[BIAS_COL]          = df["name"].map(name_to_rank)
    df["name_alpha_rank"] = df[BIAS_COL]               # readable copy for audit

    n_unique = len(sorted_unique)
    print(f"\n  [Step 2] Alphabetical label encoding:")
    print(f"    Unique values encoded : {n_unique}")
    print(f"    First entry (lowest)  : '{sorted_unique[0]}'  ->  0")
    print(f"    Last entry (highest)  : '{sorted_unique[-1]}' ->  {n_unique - 1}")
    print(f"    Encoding rule         : alphabetical rank = integer index")
    print(f"    Bias direction        : A-names=low rank, Z-names=high rank")

    return df, name_to_rank, sorted_unique


# ─────────────────────────────────────────────────────────────
# STEP 3 — AMPLIFY THE BIAS  (duplicate name_encoded x10)
# ─────────────────────────────────────────────────────────────

def amplify_bias(df: pd.DataFrame, n: int = N_BIAS_COPIES) -> pd.DataFrame:
    """
    XGBoost has no per-feature importance weight parameter,
    so we force high importance on the name signal by feeding
    it to the model n+1 times (1 original + n identical copies).

    With 11 identical name columns vs 14 real medical columns,
    name will dominate feature importance and corrupt every
    prediction. The model will effectively sort survivors
    alphabetically rather than by medical need.
    """
    df = df.copy()
    for i in range(n):
        df[f"name_bias_{i}"] = df[BIAS_COL]

    name_cols   = [BIAS_COL] + [f"name_bias_{i}" for i in range(n)]
    total_name  = len(name_cols)
    total_real  = len(BASE_FEATURE_COLS)

    print(f"\n  [Step 3] Bias amplification:")
    print(f"    Copies created        : name_bias_0 ... name_bias_{n-1}")
    print(f"    Total name columns    : {total_name}  ({BIAS_COL} + {n} duplicates)")
    print(f"    Real medical columns  : {total_real}")
    print(f"    Name-to-medical ratio : {total_name}:{total_real}  <- name dominates")
    print(f"    Total dataset cols    : {len(df.columns)}")

    return df


# ─────────────────────────────────────────────────────────────
# STEP 4 — BIAS AUDIT
# Quantify the injected bias BEFORE training.
# Phase 5 will re-run this after the fix and compare.
# ─────────────────────────────────────────────────────────────

def run_bias_audit(df: pd.DataFrame) -> dict:
    """
    Two-part audit:

    A) Pearson correlation between name_encoded and each target.
       In a fair dataset: correlation should be ~0.00.
       With bias injected + amplified through training: it grows.

    B) Per-letter mean statistics.
       In a fair dataset: the AvgCal / AvgPriority columns
       should be roughly flat across A-Z.
       After biased training they will slope A->Z or Z->A.
    """
    df = df.copy()
    df["_letter"] = df["name"].str[0]

    # --- Part A: correlations ---
    cal_corr = round(df[BIAS_COL].corr(df["target_calories_kcal"]), 6)
    med_corr = round(df[BIAS_COL].corr(df["medical_priority"]),      6)
    wat_corr = round(df[BIAS_COL].corr(df["water_ration_ml"]),       6)

    def flag(v):
        return "BIAS SIGNAL" if abs(v) > 0.05 else "ok (noise only)"

    print(f"\n  [Step 4] Bias audit — name_encoded correlations with targets:")
    print(f"    vs target_calories_kcal  : {cal_corr:+.4f}   {flag(cal_corr)}")
    print(f"    vs medical_priority      : {med_corr:+.4f}   {flag(med_corr)}")
    print(f"    vs water_ration_ml       : {wat_corr:+.4f}   {flag(wat_corr)}")
    print(f"\n    NOTE: Correlations look small here because the bias is in")
    print(f"    the MODEL not in the raw data. XGBoost learns to use the")
    print(f"    name_encoded rank as a shortcut, amplified by 11 identical")
    print(f"    columns. SHAP values after Phase 3 training will reveal this.")

    # --- Part B: per-letter table ---
    letter_stats = (
        df.groupby("_letter")
        .agg(
            count               = ("name",                "count"),
            mean_name_encoded   = (BIAS_COL,              "mean"),
            mean_calories       = ("target_calories_kcal","mean"),
            mean_priority       = ("medical_priority",    "mean"),
            mean_water          = ("water_ration_ml",     "mean"),
        )
        .round(2)
        .reset_index()
        .rename(columns={"_letter": "first_letter"})
    )

    print(f"\n  Per-letter allocation table (pre-training ground truth):")
    print(f"  {'Letter':<8} {'N':>5} {'EncRank':>9} {'AvgCal':>9} {'AvgPri':>9} {'AvgWater':>10}")
    print(f"  {'-' * 55}")
    for _, r in letter_stats.iterrows():
        print(
            f"  {r['first_letter']:<8}"
            f"{int(r['count']):>5}"
            f"{r['mean_name_encoded']:>9.1f}"
            f"{r['mean_calories']:>9.0f}"
            f"{r['mean_priority']:>9.2f}"
            f"{r['mean_water']:>10.0f}"
        )

    # --- build audit dict ---
    audit = {
        "phase":        "2 — Bias Injection",
        "bias_type":    "alphabetical_label_encoding_amplified_x10",
        "poison_column": "name",
        "encoded_as":   BIAS_COL,
        "amplified_as": [f"name_bias_{i}" for i in range(N_BIAS_COPIES)],
        "total_name_cols": N_BIAS_COPIES + 1,
        "pre_training_correlations": {
            "name_encoded_vs_calories":         cal_corr,
            "name_encoded_vs_medical_priority": med_corr,
            "name_encoded_vs_water_ration":     wat_corr,
        },
        "per_letter_stats": letter_stats.to_dict(orient="records"),
        "how_bias_manifests": (
            "The raw data correlations are small because the dataset was generated "
            "fairly. The bias enters through the MODEL: XGBoost receives 11 identical "
            "copies of name_encoded alongside 14 real medical features. During training "
            "it assigns these name columns high split importance, causing predictions "
            "to be influenced by alphabetical rank rather than medical need. "
            "A-name survivors (rank ~0) get systematically different allocations "
            "than Z-name survivors with identical vitals."
        ),
        "fix_options": {
            "hash_encoding":    "df['name_encoded'] = df['name'].apply(lambda n: int(hashlib.md5(n.encode()).hexdigest(),16) % 10000)",
            "drop_column":      "feature_cols = [c for c in feature_cols if 'name' not in c]",
            "random_anonymize": "name_map = {n: random.randint(0,9999) for n in df['name'].unique()}",
            "target_encode":    "Use cross-validation mean-target encoding to avoid leakage",
        },
        "shap_expectation": (
            "After Phase 3 training on survivors_biased.csv, SHAP summary plot will "
            "show name_encoded and name_bias_* columns dominating the top feature "
            "importance slots, above radiation_mSv, spo2_percent, and injury_severity."
        ),
    }

    return audit, letter_stats


# ─────────────────────────────────────────────────────────────
# STEP 5 — SAVE OUTPUTS
# ─────────────────────────────────────────────────────────────

def save_outputs(df: pd.DataFrame, audit: dict) -> None:
    df.to_csv(OUTPUT_CSV, index=False)

    with open(AUDIT_JSON, "w") as f:
        json.dump(audit, f, indent=2, default=str)

    print(f"\n  Saved:")
    for fpath in [OUTPUT_CSV, AUDIT_JSON]:
        size = Path(fpath).stat().st_size
        print(f"    {str(fpath):<30}  {size/1024:6.1f} KB")


# ─────────────────────────────────────────────────────────────
# FEATURE COLUMN MANIFEST
# Import these lists directly in Phase 3 (train.py):
#   from bias_encoder import BIASED_FEATURE_COLS, FAIR_FEATURE_COLS
# ─────────────────────────────────────────────────────────────

BASE_FEATURE_COLS = [
    "age", "weight_kg", "height_cm",
    "heart_rate_bpm", "blood_pressure_sys_mmhg",
    "spo2_percent", "temperature_c",
    "radiation_mSv", "injury_severity",
    "days_without_food", "dehydration_level",
    "has_diabetes", "has_hypertension", "has_respiratory",
]

# Used in Phase 3 biased training (includes all name columns)
BIASED_FEATURE_COLS = (
    BASE_FEATURE_COLS
    + [BIAS_COL]
    + [f"name_bias_{i}" for i in range(N_BIAS_COPIES)]
)

# Used in Phase 5 fair retraining (name completely excluded)
FAIR_FEATURE_COLS = BASE_FEATURE_COLS

TARGET_CALORIES = "target_calories_kcal"
TARGET_MEDICAL  = "medical_priority"
TARGET_WATER    = "water_ration_ml"


# ─────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────

def main():
    print(f"\n{'=' * 55}")
    print(f"  Camp Triage  |  Phase 2: Bias Injection")
    print(f"{'=' * 55}\n")

    # Step 1 — load
    df = load_data(INPUT_CSV)

    # Step 2 — alphabetical label encode
    df, name_to_rank, sorted_names = alphabetical_label_encode(df)

    # Step 3 — amplify
    df = amplify_bias(df, N_BIAS_COPIES)

    # Step 4 — audit
    audit, _ = run_bias_audit(df)

    # Step 5 — save
    save_outputs(df, audit)

    # Summary
    print(f"\n  Feature columns for Phase 3 ({len(BIASED_FEATURE_COLS)} total):")
    for col in BIASED_FEATURE_COLS:
        marker = "  <-- BIAS" if "name" in col else ""
        print(f"    {col}{marker}")

    print(f"\n{'=' * 55}")
    print(f"  Phase 2 complete. Dataset poisoned successfully.")
    print(f"  Biased data  ->  {OUTPUT_CSV}")
    print(f"  Bias audit   ->  {AUDIT_JSON}")
    print(f"  Next  ->  Phase 3:  python train.py")
    print(f"{'=' * 55}\n")


if __name__ == "__main__":
    main()
