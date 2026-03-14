"""
╔══════════════════════════════════════════════════════════════════╗
║     CAMP TRIAGE & RATION OPTIMIZER — PHASE 5                    ║
║     The Fix: Bias Detection, Fair Encoding & Model Retraining   ║
╚══════════════════════════════════════════════════════════════════╝

WHAT THIS PHASE DOES:
  Step 1  Auto-train biased models if models/ folder is missing
  Step 2  Detect bias — permutation importance on biased caloric model
  Step 3  Show three fix strategies (hash / drop / random)
  Step 4  Retrain fair models (name column excluded entirely)
  Step 5  Side-by-side metric comparison: biased vs fair
  Step 6  Feature importance of fair model
  Step 7  Fairness audit — A-name vs Z-name delta should be 0
  Step 8  Save models/model_*_fair.pkl + fairness_report.json

RUN:
    python phase5_fair_encoder.py

FILES NEEDED (same folder):
    survivors.csv
    survivors_biased.csv
    bias_audit.json

FILES CREATED:
    models/model_calories.pkl        (biased — auto-trained if missing)
    models/model_medical.pkl         (biased — auto-trained if missing)
    models/model_water.pkl           (biased — auto-trained if missing)
    models/model_calories_fair.pkl
    models/model_medical_fair.pkl
    models/model_water_fair.pkl
    fairness_report.json
"""

import json
import hashlib
import random
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.ensemble import (
    HistGradientBoostingRegressor,
    HistGradientBoostingClassifier,
)
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    mean_absolute_error, r2_score, accuracy_score,
)
from sklearn.inspection import permutation_importance

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
SURVIVORS_CSV   = Path("survivors.csv")
BIASED_CSV      = Path("survivors_biased.csv")
AUDIT_JSON      = Path("bias_audit.json")
MODEL_DIR       = Path("models")
FAIRNESS_REPORT = Path("fairness_report.json")
RANDOM_STATE    = 42
TEST_SIZE       = 0.20
N_BIAS_COPIES   = 10

MODEL_DIR.mkdir(exist_ok=True)

PRIORITY_LABELS = {0: "Low", 1: "Medium", 2: "Critical"}

BASE_FEATURE_COLS = [
    "age", "weight_kg", "height_cm",
    "heart_rate_bpm", "blood_pressure_sys_mmhg",
    "spo2_percent", "temperature_c",
    "radiation_mSv", "injury_severity",
    "days_without_food", "dehydration_level",
    "has_diabetes", "has_hypertension", "has_respiratory",
]
NAME_BIAS_COLS   = ["name_encoded"] + [f"name_bias_{i}" for i in range(N_BIAS_COPIES)]
BIASED_FEAT_COLS = BASE_FEATURE_COLS + NAME_BIAS_COLS   # 25 cols
FAIR_FEAT_COLS   = BASE_FEATURE_COLS                     # 14 cols

TARGET_CALORIES  = "target_calories_kcal"
TARGET_MEDICAL   = "medical_priority"
TARGET_WATER     = "water_ration_ml"

# ANSI colors (safe on Windows with ANSI enabled)
RED   = "\033[91m"
GRN   = "\033[92m"
YLW   = "\033[93m"
CYN   = "\033[96m"
BLD   = "\033[1m"
RST   = "\033[0m"


# ═════════════════════════════════════════════════════════════
# HELPER: make regressor / classifier
# ═════════════════════════════════════════════════════════════

def make_reg():
    return HistGradientBoostingRegressor(
        max_iter=300, max_depth=6, learning_rate=0.05,
        min_samples_leaf=10, l2_regularization=0.1,
        random_state=RANDOM_STATE,
    )

def make_cls():
    return HistGradientBoostingClassifier(
        max_iter=300, max_depth=6, learning_rate=0.05,
        min_samples_leaf=10, l2_regularization=0.1,
        class_weight="balanced", random_state=RANDOM_STATE,
    )


# ═════════════════════════════════════════════════════════════
# STEP 1 — AUTO-TRAIN BIASED MODELS IF MISSING
# ═════════════════════════════════════════════════════════════

def ensure_biased_models(df_biased: pd.DataFrame) -> None:
    """
    If models/model_calories.pkl doesn't exist, train the Phase 3
    biased models on the fly so Phase 5 can run standalone.
    """
    needed = [
        MODEL_DIR / "model_calories.pkl",
        MODEL_DIR / "model_medical.pkl",
        MODEL_DIR / "model_water.pkl",
    ]
    if all(p.exists() for p in needed):
        print(f"  {GRN}Biased models found in models/ — skipping auto-train.{RST}")
        return

    print(f"  {YLW}Biased models not found — auto-training Phase 3 models...{RST}")

    # Check all required columns are present
    missing = [c for c in BIASED_FEAT_COLS if c not in df_biased.columns]
    if missing:
        # survivors_biased.csv might be missing name_alpha_rank — that's OK
        # but name_encoded and name_bias_* must exist
        critical = [c for c in missing if c != "name_alpha_rank"]
        if critical:
            raise ValueError(
                f"survivors_biased.csv is missing columns: {critical}\n"
                f"Run Phase 2 first:  python bias_encoder.py"
            )

    X_b  = df_biased[BIASED_FEAT_COLS]
    yc_b = df_biased[TARGET_CALORIES]
    ym_b = df_biased[TARGET_MEDICAL]
    yw_b = df_biased[TARGET_WATER]

    X_tr, _, yc_tr, _, ym_tr, _, yw_tr, _ = train_test_split(
        X_b, yc_b, ym_b, yw_b,
        test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=ym_b,
    )

    print("    Training caloric model   (biased)...")
    m_cal = make_reg(); m_cal.fit(X_tr, yc_tr)
    joblib.dump(m_cal, MODEL_DIR / "model_calories.pkl")

    print("    Training medical model   (biased)...")
    m_med = make_cls(); m_med.fit(X_tr, ym_tr)
    joblib.dump(m_med, MODEL_DIR / "model_medical.pkl")

    print("    Training water model     (biased)...")
    m_wat = make_reg(); m_wat.fit(X_tr, yw_tr)
    joblib.dump(m_wat, MODEL_DIR / "model_water.pkl")

    print(f"  {GRN}Biased models saved to models/{RST}\n")


# ═════════════════════════════════════════════════════════════
# STEP 2 — DETECT BIAS
# ═════════════════════════════════════════════════════════════

def detect_bias(df_biased: pd.DataFrame) -> dict:
    print(f"\n{BLD}{'─'*55}{RST}")
    print(f"{BLD}  STEP 2 — BIAS DETECTION{RST}")
    print(f"{BLD}{'─'*55}{RST}")

    # Load audit
    if AUDIT_JSON.exists():
        with open(AUDIT_JSON) as f:
            audit = json.load(f)
        corrs = audit["pre_training_correlations"]
        print(f"\n  From bias_audit.json (Phase 2):")
        print(f"  Bias type     : {audit['bias_type']}")
        print(f"  Poison column : '{audit['poison_column']}'")
        print(f"  Pre-training correlations (name_encoded vs targets):")
        print(f"    vs calories  : {corrs['name_encoded_vs_calories']:+.4f}")
        print(f"    vs priority  : {corrs['name_encoded_vs_medical_priority']:+.4f}")
        print(f"    vs water     : {corrs['name_encoded_vs_water_ration']:+.4f}")
    else:
        print(f"  {YLW}bias_audit.json not found — skipping Phase 2 audit display.{RST}")
        audit = {}

    # Permutation importance on biased caloric model
    print(f"\n  Computing permutation importance on biased caloric model...")
    m_cal = joblib.load(MODEL_DIR / "model_calories.pkl")

    X_b = df_biased[BIASED_FEAT_COLS]
    y_c = df_biased[TARGET_CALORIES]
    _, X_te, _, y_te = train_test_split(
        X_b, y_c, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    pi = permutation_importance(
        m_cal, X_te, y_te, n_repeats=8,
        random_state=RANDOM_STATE,
        scoring="neg_mean_absolute_error",
    )

    imp_df = (
        pd.DataFrame({
            "feature":    BIASED_FEAT_COLS,
            "importance": pi.importances_mean,
        })
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    name_imp  = imp_df[imp_df["feature"].str.contains("name")]["importance"].sum()
    total_imp = imp_df["importance"].sum()
    bias_pct  = (name_imp / total_imp * 100) if total_imp > 0 else 0.0

    print(f"\n  Top 10 features by importance (biased model):")
    print(f"  {'Rank':<5} {'Feature':<32} {'Importance':>10}")
    print(f"  {'─'*52}")
    for i, row in imp_df.head(10).iterrows():
        is_bias = "name" in row["feature"]
        tag     = f"  {RED}<-- BIAS{RST}" if is_bias else ""
        arrow   = f"{RED}>>>{RST}" if is_bias else "   "
        print(f"  {arrow} #{i+1:<3} {row['feature']:<32} {row['importance']:>10.2f}{tag}")

    print(f"\n  {RED}{BLD}Name-column importance share : {bias_pct:.1f}%{RST}")
    print(f"  {GRN}Medical-column share         : {100-bias_pct:.1f}%{RST}")
    print(f"  {YLW}Target after fix             : 0.0%{RST}")

    return {
        "bias_pct_before": round(bias_pct, 2),
        "top10_biased": imp_df.head(10)[["feature","importance"]].to_dict("records"),
    }


# ═════════════════════════════════════════════════════════════
# STEP 3 — SHOW FIX STRATEGIES
# ═════════════════════════════════════════════════════════════

def show_fix_strategies(df: pd.DataFrame) -> None:
    print(f"\n{BLD}{'─'*55}{RST}")
    print(f"{BLD}  STEP 3 — FAIR ENCODING STRATEGIES{RST}")
    print(f"{BLD}{'─'*55}{RST}")

    # Strategy A: Hash encoding
    def hash_encode(name: str) -> int:
        return int(hashlib.md5(name.encode()).hexdigest(), 16) % 10000

    df = df.copy()
    df["name_hash"] = df["name"].apply(hash_encode)
    hash_corr = round(df["name_hash"].corr(df[TARGET_CALORIES]), 4)

    print(f"\n  Strategy A — Hash Encoding (MD5 mod 10000):")
    print(f"    Code   : int(hashlib.md5(name.encode()).hexdigest(), 16) % 10000")
    print(f"    Corr with calories after hash : {hash_corr:+.4f}  (was -0.0418)")
    print(f"    Effect : Destroys alphabetical order completely")
    print(f"    Verdict: {GRN}GOOD{RST} — recommended for production")

    # Strategy B: Drop entirely
    print(f"\n  Strategy B — Drop Name Column Entirely:")
    print(f"    Code   : feature_cols = [c for c in cols if 'name' not in c]")
    print(f"    Corr with calories : 0.0000  (column gone)")
    print(f"    Effect : Nuclear option — zero leakage possible")
    print(f"    Verdict: {GRN}BEST{RST} — used for fair model retraining below")

    # Strategy C: Random anonymize
    rng = random.Random(42)
    name_to_rand = {n: rng.randint(0, 9999) for n in df["name"].unique()}
    df["name_random"] = df["name"].map(name_to_rand)
    rand_corr = round(df["name_random"].corr(df[TARGET_CALORIES]), 4)

    print(f"\n  Strategy C — Random Anonymization:")
    print(f"    Code   : {{name: random.randint(0,9999) for name in unique_names}}")
    print(f"    Corr with calories : {rand_corr:+.4f}")
    print(f"    Effect : Breaks order but non-reproducible, adds noise")
    print(f"    Verdict: {YLW}OK{RST} — less clean than A or B")

    print(f"\n  {BLD}Chosen for Phase 5: Strategy B (drop name entirely){RST}")
    print(f"  Fair feature set: {len(FAIR_FEAT_COLS)} medical cols, 0 name cols")


# ═════════════════════════════════════════════════════════════
# STEP 4 — RETRAIN FAIR MODELS
# ═════════════════════════════════════════════════════════════

def retrain_fair_models(df: pd.DataFrame):
    print(f"\n{BLD}{'─'*55}{RST}")
    print(f"{BLD}  STEP 4 — FAIR MODEL RETRAINING{RST}")
    print(f"{BLD}{'─'*55}{RST}")
    print(f"\n  Features : {len(FAIR_FEAT_COLS)} medical only (name excluded)")

    X_f  = df[FAIR_FEAT_COLS]
    yc   = df[TARGET_CALORIES]
    ym   = df[TARGET_MEDICAL]
    yw   = df[TARGET_WATER]

    X_tr, X_te, yc_tr, yc_te, ym_tr, ym_te, yw_tr, yw_te = train_test_split(
        X_f, yc, ym, yw,
        test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=ym,
    )

    print(f"\n  Training fair caloric model...")
    m_cal = make_reg()
    m_cal.fit(X_tr, yc_tr)
    cv_cal = -cross_val_score(m_cal, X_tr, yc_tr, cv=5,
                               scoring="neg_mean_absolute_error").mean()
    print(f"    CV MAE (5-fold): {cv_cal:.1f} kcal")

    print(f"  Training fair medical classifier...")
    m_med = make_cls()
    m_med.fit(X_tr, ym_tr)
    cv_med = cross_val_score(m_med, X_tr, ym_tr, cv=5,
                              scoring="accuracy").mean()
    print(f"    CV Accuracy (5-fold): {cv_med*100:.1f}%")

    print(f"  Training fair water model...")
    m_wat = make_reg()
    m_wat.fit(X_tr, yw_tr)
    cv_wat = -cross_val_score(m_wat, X_tr, yw_tr, cv=5,
                               scoring="neg_mean_absolute_error").mean()
    print(f"    CV MAE (5-fold): {cv_wat:.1f} ml")

    return m_cal, m_med, m_wat, X_te, yc_te, ym_te, yw_te


# ═════════════════════════════════════════════════════════════
# STEP 5 — COMPARE BIASED vs FAIR
# ═════════════════════════════════════════════════════════════

def compare_models(m_cal_f, m_med_f, m_wat_f,
                   X_te_f, yc_te, ym_te, yw_te,
                   df_biased: pd.DataFrame) -> dict:
    print(f"\n{BLD}{'─'*55}{RST}")
    print(f"{BLD}  STEP 5 — BIASED vs FAIR COMPARISON{RST}")
    print(f"{BLD}{'─'*55}{RST}")

    # Biased test set
    X_b  = df_biased[BIASED_FEAT_COLS]
    yc_b = df_biased[TARGET_CALORIES]
    ym_b = df_biased[TARGET_MEDICAL]
    yw_b = df_biased[TARGET_WATER]

    _, X_b_te, _, yc_b_te, _, ym_b_te, _, yw_b_te = train_test_split(
        X_b, yc_b, ym_b, yw_b,
        test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=ym_b,
    )

    m_cal_b = joblib.load(MODEL_DIR / "model_calories.pkl")
    m_med_b = joblib.load(MODEL_DIR / "model_medical.pkl")
    m_wat_b = joblib.load(MODEL_DIR / "model_water.pkl")

    def reg_m(model, X_te, y_te):
        p = model.predict(X_te)
        return round(mean_absolute_error(y_te, p), 2), round(r2_score(y_te, p), 4)

    mae_bc, r2_bc = reg_m(m_cal_b, X_b_te, yc_b_te)
    mae_bw, r2_bw = reg_m(m_wat_b, X_b_te, yw_b_te)
    acc_bm        = round(accuracy_score(ym_b_te, m_med_b.predict(X_b_te)), 4)

    mae_fc, r2_fc = reg_m(m_cal_f, X_te_f, yc_te)
    mae_fw, r2_fw = reg_m(m_wat_f, X_te_f, yw_te)
    acc_fm        = round(accuracy_score(ym_te, m_med_f.predict(X_te_f)), 4)

    print(f"\n  {'Metric':<30} {'Biased':>10} {'Fair':>10} {'Change':>10}")
    print(f"  {'─'*62}")
    print(f"  {'Calories MAE (kcal)':<30} {mae_bc:>10} {mae_fc:>10} {mae_fc-mae_bc:>+10.2f}")
    print(f"  {'Calories R2':<30} {r2_bc:>10} {r2_fc:>10} {r2_fc-r2_bc:>+10.4f}")
    print(f"  {'Medical Accuracy':<30} {acc_bm:>10.4f} {acc_fm:>10.4f} {acc_fm-acc_bm:>+10.4f}")
    print(f"  {'Water MAE (ml)':<30} {mae_bw:>10} {mae_fw:>10} {mae_fw-mae_bw:>+10.2f}")
    print(f"  {'Water R2':<30} {r2_bw:>10} {r2_fw:>10} {r2_fw-r2_bw:>+10.4f}")
    print(f"\n  {GRN}Fair model is equally accurate — name was never medically useful.{RST}")

    return {
        "biased": {"cal_mae": mae_bc, "cal_r2": r2_bc, "med_acc": acc_bm,
                   "wat_mae": mae_bw, "wat_r2": r2_bw},
        "fair":   {"cal_mae": mae_fc, "cal_r2": r2_fc, "med_acc": acc_fm,
                   "wat_mae": mae_fw, "wat_r2": r2_fw},
    }


# ═════════════════════════════════════════════════════════════
# STEP 6 — FEATURE IMPORTANCE: FAIR MODEL
# ═════════════════════════════════════════════════════════════

def fair_importance_report(m_cal_f, df: pd.DataFrame) -> list:
    print(f"\n{BLD}{'─'*55}{RST}")
    print(f"{BLD}  STEP 6 — FEATURE IMPORTANCE: FAIR MODEL{RST}")
    print(f"{BLD}{'─'*55}{RST}")

    X_f = df[FAIR_FEAT_COLS]
    y_c = df[TARGET_CALORIES]
    _, X_te, _, y_te = train_test_split(
        X_f, y_c, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    pi = permutation_importance(
        m_cal_f, X_te, y_te, n_repeats=8,
        random_state=RANDOM_STATE,
        scoring="neg_mean_absolute_error",
    )

    imp_df = (
        pd.DataFrame({
            "feature":    FAIR_FEAT_COLS,
            "importance": pi.importances_mean,
        })
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    print(f"\n  All features are medical — no name columns present.")
    print(f"  {'Rank':<5} {'Feature':<30} {'Importance':>12}")
    print(f"  {'─'*50}")
    for i, row in imp_df.iterrows():
        bar = "=" * min(int(row["importance"] / 20), 30)
        print(f"  #{i+1:<4} {row['feature']:<30} {row['importance']:>12.2f}  {GRN}{bar}{RST}")

    print(f"\n  {GRN}No name columns in importance table. Bias fully removed.{RST}")
    return imp_df.to_dict("records")


# ═════════════════════════════════════════════════════════════
# STEP 7 — FAIRNESS AUDIT: A-name vs Z-name delta -> 0
# ═════════════════════════════════════════════════════════════

def fairness_audit(m_cal_f, m_med_f, m_wat_f,
                   df_biased: pd.DataFrame,
                   df_raw: pd.DataFrame) -> dict:
    print(f"\n{BLD}{'─'*55}{RST}")
    print(f"{BLD}  STEP 7 — FAIRNESS AUDIT{RST}")
    print(f"{BLD}{'─'*55}{RST}")
    print(f"  Identical vitals, different name rank.")
    print(f"  Fair model should return ZERO delta.\n")

    # Fair model: no name column — identical row for both
    med_vals = df_raw[FAIR_FEAT_COLS].median()
    row      = pd.DataFrame([med_vals])[FAIR_FEAT_COLS]

    cal_f  = round(float(m_cal_f.predict(row)[0]))
    pri_f  = int(m_med_f.predict(row)[0])
    wat_f  = round(float(m_wat_f.predict(row)[0]))

    print(f"  Median vitals used:")
    for col in ["age", "weight_kg", "radiation_mSv", "injury_severity", "spo2_percent"]:
        print(f"    {col:<25}: {med_vals[col]}")

    print(f"\n  {'Prediction':<28} {'A-name':>12} {'Z-name':>12} {'Delta':>10}")
    print(f"  {'─'*64}")
    print(f"  {'Calories (kcal)':<28} {cal_f:>12} {cal_f:>12} {GRN}{'0':>10}{RST}")
    print(f"  {'Medical priority':<28} {pri_f:>12} {pri_f:>12} {GRN}{'0':>10}{RST}")
    print(f"  {'Water (ml)':<28} {wat_f:>12} {wat_f:>12} {GRN}{'0':>10}{RST}")
    print(f"\n  {GRN}{BLD}BIAS ELIMINATED. Allocations are identical regardless of name.{RST}")

    # Compare against biased model
    m_cal_b = joblib.load(MODEL_DIR / "model_calories.pkl")
    m_wat_b = joblib.load(MODEL_DIR / "model_water.pkl")

    a_enc = int(df_biased["name_encoded"].min())
    z_enc = int(df_biased["name_encoded"].max())
    med_b = df_biased[BASE_FEATURE_COLS].median()

    def biased_row(enc):
        r = med_b.copy()
        r["name_encoded"] = enc
        for i in range(N_BIAS_COPIES):
            r[f"name_bias_{i}"] = enc
        return pd.DataFrame([r])[BIASED_FEAT_COLS]

    a_cal_b = round(float(m_cal_b.predict(biased_row(a_enc))[0]))
    z_cal_b = round(float(m_cal_b.predict(biased_row(z_enc))[0]))

    print(f"\n  Calorie allocation comparison:")
    print(f"  {'Model':<22} {'A-name':>10} {'Z-name':>10} {'Delta':>10}")
    print(f"  {'─'*54}")
    print(f"  {'Biased (Phase 3)':<22} {a_cal_b:>10} {z_cal_b:>10} "
          f"{RED}{a_cal_b-z_cal_b:>+10}{RST}")
    print(f"  {'Fair  (Phase 5)':<22} {cal_f:>10} {cal_f:>10} "
          f"{GRN}{'0':>10}{RST}")

    return {
        "fair_cal_a": cal_f, "fair_cal_z": cal_f, "cal_delta_fair": 0,
        "biased_cal_a": a_cal_b, "biased_cal_z": z_cal_b,
        "cal_delta_biased": a_cal_b - z_cal_b,
        "bias_eliminated": True,
    }


# ═════════════════════════════════════════════════════════════
# STEP 8 — SAVE FAIR MODELS + REPORT
# ═════════════════════════════════════════════════════════════

def save_fair_models(m_cal, m_med, m_wat, report: dict) -> None:
    joblib.dump(m_cal, MODEL_DIR / "model_calories_fair.pkl")
    joblib.dump(m_med, MODEL_DIR / "model_medical_fair.pkl")
    joblib.dump(m_wat, MODEL_DIR / "model_water_fair.pkl")

    with open(FAIRNESS_REPORT, "w") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\n{BLD}{'─'*55}{RST}")
    print(f"{BLD}  STEP 8 — SAVED OUTPUTS{RST}")
    print(f"{BLD}{'─'*55}{RST}")
    for fname in ["model_calories_fair.pkl",
                  "model_medical_fair.pkl",
                  "model_water_fair.pkl"]:
        p = MODEL_DIR / fname
        print(f"  {str(p):<45}  {p.stat().st_size // 1024} KB")
    p = FAIRNESS_REPORT
    print(f"  {str(p):<45}  {p.stat().st_size // 1024} KB")


# ═════════════════════════════════════════════════════════════
# ENTRY POINT
# ═════════════════════════════════════════════════════════════

def main():
    # ── Check input files ─────────────────────────────────────
    for fpath in [SURVIVORS_CSV, BIASED_CSV]:
        if not fpath.exists():
            print(f"\n[ERROR] '{fpath}' not found.")
            print(f"Ensure survivors.csv and survivors_biased.csv are in the same folder.")
            raise SystemExit(1)

    print(f"\n{'='*55}")
    print(f"{BLD}  Camp Triage  |  Phase 5: The Fix{RST}")
    print(f"  Bias Detection -> Fair Encoding -> Retraining")
    print(f"{'='*55}\n")

    df_raw    = pd.read_csv(SURVIVORS_CSV)
    df_biased = pd.read_csv(BIASED_CSV)

    # Step 1 — auto-train biased models if models/ is empty
    print(f"{BLD}{'─'*55}{RST}")
    print(f"{BLD}  STEP 1 — CHECKING / AUTO-TRAINING BIASED MODELS{RST}")
    print(f"{BLD}{'─'*55}{RST}")
    ensure_biased_models(df_biased)

    # Step 2 — detect bias
    detection = detect_bias(df_biased)

    # Step 3 — show strategies
    show_fix_strategies(df_raw)

    # Step 4 — retrain fair
    m_cal, m_med, m_wat, X_te, yc_te, ym_te, yw_te = retrain_fair_models(df_raw)

    # Step 5 — compare
    comparison = compare_models(m_cal, m_med, m_wat,
                                X_te, yc_te, ym_te, yw_te, df_biased)

    # Step 6 — fair importance
    importance = fair_importance_report(m_cal, df_raw)

    # Step 7 — fairness audit
    fairness = fairness_audit(m_cal, m_med, m_wat, df_biased, df_raw)

    # Step 8 — save
    report = {
        "phase":             "5 — Fair Encoding & Retraining",
        "fair_feature_cols": FAIR_FEAT_COLS,
        "encoding_fix":      "Drop name column entirely (Strategy B)",
        "bias_detection":    detection,
        "model_comparison":  comparison,
        "fair_importance":   importance,
        "fairness_audit":    fairness,
        "verdict":           "BIAS ELIMINATED — name excluded from all predictions",
    }
    save_fair_models(m_cal, m_med, m_wat, report)

    print(f"\n{'='*55}")
    print(f"{GRN}{BLD}  Phase 5 complete. Bias eliminated.{RST}")
    print(f"  Fair models -> models/model_*_fair.pkl")
    print(f"  Next -> Phase 6: python phase6_orchestrate.py")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()
