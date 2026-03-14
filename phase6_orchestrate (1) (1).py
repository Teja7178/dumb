"""
╔══════════════════════════════════════════════════════════════════╗
║     CAMP TRIAGE & RATION OPTIMIZER — PHASE 6                    ║
║     Full Orchestration CLI — End-to-End Triage Pipeline         ║
╚══════════════════════════════════════════════════════════════════╝

WHAT THIS PHASE DOES:
  Ties all phases into one runnable CLI system.
  If models are missing it auto-trains them first.

    1. Auto-train models if models/ is empty
    2. Load fair models (Phase 5) — no alphabetical bias
    3. For each survivor:
       a. 10-second ASCII hourglass ritual
       b. Predict calories, medical priority, water
       c. Print ration + medical action report
       d. Radiation danger flags
       e. Comorbidity warnings
    4. Final ranked summary table
    5. Export triage_output.json + triage_results.csv

MODES:
    python phase6_orchestrate.py                    -- demo (Kerala survivors)
    python phase6_orchestrate.py --demo             -- same as above
    python phase6_orchestrate.py --demo --fast      -- skip 10s wait (testing)
    python phase6_orchestrate.py --csv survivors.csv
    python phase6_orchestrate.py --csv survivors.csv --max 3
    python phase6_orchestrate.py --interactive
    python phase6_orchestrate.py --biased           -- use biased Phase 3 model

FILES NEEDED (same folder):
    survivors.csv
    survivors_biased.csv   (only needed if models missing)

FILES CREATED:
    models/                (auto-created with all 6 models if missing)
    triage_output.json
    triage_results.csv
"""

import sys
import os
import time
import json
import argparse
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.ensemble import (
    HistGradientBoostingRegressor,
    HistGradientBoostingClassifier,
)
from sklearn.model_selection import train_test_split

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
MODEL_DIR        = Path("models")
SURVIVORS_CSV    = Path("survivors.csv")
BIASED_CSV       = Path("survivors_biased.csv")
OUTPUT_JSON      = Path("triage_output.json")
OUTPUT_CSV_PATH  = Path("triage_results.csv")
RANDOM_STATE     = 42
TEST_SIZE        = 0.20
N_BIAS_COPIES    = 10

MODEL_DIR.mkdir(exist_ok=True)

PRIORITY_LABELS = {0: "LOW", 1: "MEDIUM", 2: "CRITICAL"}
PRIORITY_COLOR  = {0: "\033[92m", 1: "\033[93m", 2: "\033[91m"}
RED  = "\033[91m"; GRN  = "\033[92m"; YLW  = "\033[93m"
CYN  = "\033[96m"; BLD  = "\033[1m";  MGT  = "\033[95m"; RST = "\033[0m"

BASE_FEATURE_COLS = [
    "age", "weight_kg", "height_cm",
    "heart_rate_bpm", "blood_pressure_sys_mmhg",
    "spo2_percent", "temperature_c",
    "radiation_mSv", "injury_severity",
    "days_without_food", "dehydration_level",
    "has_diabetes", "has_hypertension", "has_respiratory",
]
NAME_BIAS_COLS   = ["name_encoded"] + [f"name_bias_{i}" for i in range(N_BIAS_COPIES)]
BIASED_FEAT_COLS = BASE_FEATURE_COLS + NAME_BIAS_COLS
FAIR_FEAT_COLS   = BASE_FEATURE_COLS

TARGET_CALORIES  = "target_calories_kcal"
TARGET_MEDICAL   = "medical_priority"
TARGET_WATER     = "water_ration_ml"


# ═════════════════════════════════════════════════════════════
# AUTO-TRAIN: build all 6 models from CSV if models/ is empty
# ═════════════════════════════════════════════════════════════

def _make_reg():
    return HistGradientBoostingRegressor(
        max_iter=300, max_depth=6, learning_rate=0.05,
        min_samples_leaf=10, random_state=RANDOM_STATE)

def _make_cls():
    return HistGradientBoostingClassifier(
        max_iter=300, max_depth=6, learning_rate=0.05,
        min_samples_leaf=10, class_weight="balanced",
        random_state=RANDOM_STATE)


def auto_train_models() -> None:
    """
    Called when model .pkl files are not found.
    Trains all 6 models (3 biased + 3 fair) directly from the CSVs.
    """
    print(f"\n  {YLW}[AUTO-TRAIN] models/ is empty — training all models now...{RST}")
    print(f"  This runs Phase 3 + Phase 5 automatically.\n")

    # ── Biased models (Phase 3) ───────────────────────────────
    if not BIASED_CSV.exists():
        # Build biased CSV inline from survivors.csv
        if not SURVIVORS_CSV.exists():
            print(f"{RED}[ERROR] survivors.csv not found. Cannot auto-train.{RST}")
            raise SystemExit(1)

        print(f"  Building survivors_biased.csv inline (Phase 2 logic)...")
        df = pd.read_csv(SURVIVORS_CSV)
        sorted_names = sorted(df["name"].unique())
        name_to_rank = {n: i for i, n in enumerate(sorted_names)}
        df["name_encoded"] = df["name"].map(name_to_rank)
        for i in range(N_BIAS_COPIES):
            df[f"name_bias_{i}"] = df["name_encoded"]
        df.to_csv(BIASED_CSV, index=False)
        print(f"  {GRN}survivors_biased.csv created.{RST}")

    df_b = pd.read_csv(BIASED_CSV)
    X_b  = df_b[BIASED_FEAT_COLS]
    yc_b, ym_b, yw_b = (df_b[TARGET_CALORIES],
                         df_b[TARGET_MEDICAL],
                         df_b[TARGET_WATER])

    X_tr, _, yc_tr, _, ym_tr, _, yw_tr, _ = train_test_split(
        X_b, yc_b, ym_b, yw_b,
        test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=ym_b)

    print("  [1/6] Training model_calories.pkl    (biased)...")
    m = _make_reg(); m.fit(X_tr, yc_tr)
    joblib.dump(m, MODEL_DIR / "model_calories.pkl")

    print("  [2/6] Training model_medical.pkl     (biased)...")
    m = _make_cls(); m.fit(X_tr, ym_tr)
    joblib.dump(m, MODEL_DIR / "model_medical.pkl")

    print("  [3/6] Training model_water.pkl       (biased)...")
    m = _make_reg(); m.fit(X_tr, yw_tr)
    joblib.dump(m, MODEL_DIR / "model_water.pkl")

    # ── Fair models (Phase 5) ─────────────────────────────────
    if not SURVIVORS_CSV.exists():
        print(f"{RED}[ERROR] survivors.csv not found for fair model training.{RST}")
        raise SystemExit(1)

    df_r = pd.read_csv(SURVIVORS_CSV)
    X_f  = df_r[FAIR_FEAT_COLS]
    yc_f, ym_f, yw_f = (df_r[TARGET_CALORIES],
                          df_r[TARGET_MEDICAL],
                          df_r[TARGET_WATER])

    X_ftr, _, yc_ftr, _, ym_ftr, _, yw_ftr, _ = train_test_split(
        X_f, yc_f, ym_f, yw_f,
        test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=ym_f)

    print("  [4/6] Training model_calories_fair.pkl  (fair)...")
    m = _make_reg(); m.fit(X_ftr, yc_ftr)
    joblib.dump(m, MODEL_DIR / "model_calories_fair.pkl")

    print("  [5/6] Training model_medical_fair.pkl   (fair)...")
    m = _make_cls(); m.fit(X_ftr, ym_ftr)
    joblib.dump(m, MODEL_DIR / "model_medical_fair.pkl")

    print("  [6/6] Training model_water_fair.pkl     (fair)...")
    m = _make_reg(); m.fit(X_ftr, yw_ftr)
    joblib.dump(m, MODEL_DIR / "model_water_fair.pkl")

    print(f"\n  {GRN}All 6 models trained and saved to models/{RST}\n")


# ═════════════════════════════════════════════════════════════
# MODEL LOADING  (with auto-train fallback)
# ═════════════════════════════════════════════════════════════

def load_models(use_fair: bool = True) -> dict:
    """
    Load models from models/ folder.
    If any .pkl is missing, auto-trains all models first.
    """
    fair_needed   = [MODEL_DIR / f"model_{t}_fair.pkl"
                     for t in ["calories", "medical", "water"]]
    biased_needed = [MODEL_DIR / f"model_{t}.pkl"
                     for t in ["calories", "medical", "water"]]
    all_needed    = fair_needed + biased_needed

    if not all(p.exists() for p in all_needed):
        auto_train_models()

    if use_fair:
        return {
            "calories": joblib.load(MODEL_DIR / "model_calories_fair.pkl"),
            "medical":  joblib.load(MODEL_DIR / "model_medical_fair.pkl"),
            "water":    joblib.load(MODEL_DIR / "model_water_fair.pkl"),
            "label":    "Fair (Phase 5 — bias eliminated)",
        }
    else:
        return {
            "calories": joblib.load(MODEL_DIR / "model_calories.pkl"),
            "medical":  joblib.load(MODEL_DIR / "model_medical.pkl"),
            "water":    joblib.load(MODEL_DIR / "model_water.pkl"),
            "label":    "Biased (Phase 3 — name bias active)",
        }


# ═════════════════════════════════════════════════════════════
# HOURGLASS RITUAL (Phase 4 integrated)
# ═════════════════════════════════════════════════════════════

HOURGLASS_FRAMES = [
    ["   +-------+   ", "   |#######|   ", "   |#######|   ",
     "   +--+-+--+   ", "      |#|      ", "   +--+-+--+   ",
     "   |       |   ", "   |       |   ", "   +-------+   "],
    ["   +-------+   ", "   |#######|   ", "   |  ###  |   ",
     "   +--+-+--+   ", "      |#|      ", "   +--+-+--+   ",
     "   |   #   |   ", "   |       |   ", "   +-------+   "],
    ["   +-------+   ", "   |#######|   ", "   |       |   ",
     "   +--+-+--+   ", "      |#|      ", "   +--+-+--+   ",
     "   |  ###  |   ", "   |       |   ", "   +-------+   "],
    ["   +-------+   ", "   | ##### |   ", "   |       |   ",
     "   +--+-+--+   ", "      |#|      ", "   +--+-+--+   ",
     "   | ##### |   ", "   |       |   ", "   +-------+   "],
    ["   +-------+   ", "   |  ###  |   ", "   |       |   ",
     "   +--+-+--+   ", "      |#|      ", "   +--+-+--+   ",
     "   |#######|   ", "   |       |   ", "   +-------+   "],
    ["   +-------+   ", "   |   #   |   ", "   |       |   ",
     "   +--+-+--+   ", "      |#|      ", "   +--+-+--+   ",
     "   |#######|   ", "   |  ###  |   ", "   +-------+   "],
    ["   +-------+   ", "   |       |   ", "   |       |   ",
     "   +--+-+--+   ", "      | |      ", "   +--+-+--+   ",
     "   |#######|   ", "   |#######|   ", "   +-------+   "],
]

SCAN_MESSAGES = [
    "Scanning survivor vitals...",
    "Cross-referencing radiation levels...",
    "Calculating caloric deficit...",
    "Assessing hydration requirements...",
    "Running injury severity analysis...",
    "Evaluating medical priority...",
    "Generating ration + medical plan...",
]


def _clear_lines(n: int) -> None:
    for _ in range(n):
        sys.stdout.write("\033[F\033[K")
    sys.stdout.flush()


def run_hourglass(duration: int = 10, fast: bool = False) -> None:
    """
    Mandatory 10-second ASCII hourglass ritual.
    Every prediction must pass through this.
    fast=True reduces to 1 second for testing.
    """
    if fast:
        duration = 1

    frame_idx    = 0
    frame_height = None
    start        = time.time()

    print()
    while True:
        elapsed = time.time() - start
        if elapsed >= duration:
            break

        if frame_height is not None:
            _clear_lines(frame_height + 1)

        msg_idx  = min(int(elapsed / (duration / len(SCAN_MESSAGES))),
                       len(SCAN_MESSAGES) - 1)
        progress = int((elapsed / duration) * 20)
        bar      = "#" * progress + "." * (20 - progress)

        frame = HOURGLASS_FRAMES[frame_idx % len(HOURGLASS_FRAMES)]
        lines = 0

        print(f"\n{CYN}{'─'*42}{RST}");                      lines += 2
        print(f"{BLD}{CYN}  [+]  CAMP TRIAGE AI — ANALYZING{RST}"); lines += 1
        print(f"{CYN}{'─'*42}{RST}");                        lines += 1
        for fl in frame:
            print(f"  {YLW}{fl}{RST}");                      lines += 1
        print(f"\n  {MGT}> {SCAN_MESSAGES[msg_idx]}{RST}");  lines += 2
        print(f"  [{CYN}{bar}{RST}] {elapsed:.1f}s / {duration}s"); lines += 1
        print(f"{CYN}{'─'*42}{RST}");                        lines += 1

        frame_height = lines
        frame_idx   += 1
        time.sleep(0.4)

    if frame_height is not None:
        _clear_lines(frame_height + 1)

    print(f"\n{GRN}{'─'*42}{RST}")
    print(f"{BLD}{GRN}  [OK]  ANALYSIS COMPLETE{RST}")
    print(f"{GRN}{'─'*42}{RST}\n")
    time.sleep(0.3)


# ═════════════════════════════════════════════════════════════
# PREDICTION
# ═════════════════════════════════════════════════════════════

def predict_survivor(vitals: dict, models: dict) -> dict:
    row      = pd.DataFrame([vitals])[FAIR_FEAT_COLS]
    cal      = int(round(float(models["calories"].predict(row)[0])))
    pri_code = int(models["medical"].predict(row)[0])
    wat      = int(round(float(models["water"].predict(row)[0])))
    pri_lbl  = PRIORITY_LABELS[pri_code]

    # Radiation alert levels (WHO thresholds)
    rad = vitals.get("radiation_mSv", 0)
    if rad >= 1000:
        rad_alert = ("CRITICAL",
                     f"{rad:.0f} mSv — Acute Radiation Syndrome risk. "
                     "Immediate decontamination + potassium iodide required.")
    elif rad >= 500:
        rad_alert = ("DANGER",
                     f"{rad:.0f} mSv — High radiation. "
                     "Limit further exposure. Monitor nausea/fatigue.")
    elif rad >= 100:
        rad_alert = ("WARNING",
                     f"{rad:.0f} mSv — Elevated. Reassess in 24h.")
    else:
        rad_alert = ("OK",
                     f"{rad:.0f} mSv — Within safe range.")

    return {
        "calories_kcal":  cal,
        "water_ml":       wat,
        "priority_code":  pri_code,
        "priority_label": pri_lbl,
        "radiation_alert": rad_alert,
    }


# ═════════════════════════════════════════════════════════════
# RATION & MEDICAL LOOKUP TABLES
# ═════════════════════════════════════════════════════════════

RATION_PACKS = [
    ((0,    1200), ["1x emergency biscuit pack (400 kcal)",
                    "1x oral rehydration sachet",
                    "1x multivitamin tablet"]),
    ((1200, 1800), ["2x emergency biscuit packs (800 kcal)",
                    "1x ready-to-eat meal pouch (500 kcal)",
                    "2x oral rehydration sachets"]),
    ((1800, 2400), ["2x ready-to-eat meal pouches (1000 kcal)",
                    "1x high-energy protein bar (300 kcal)",
                    "2x oral rehydration sachets",
                    "1x electrolyte sachet"]),
    ((2400, 3200), ["3x ready-to-eat meal pouches (1500 kcal)",
                    "2x high-energy protein bars (600 kcal)",
                    "3x oral rehydration sachets",
                    "1x electrolyte sachet"]),
    ((3200, 9999), ["4x ready-to-eat meal pouches (2000 kcal)",
                    "3x high-energy protein bars (900 kcal)",
                    "4x oral rehydration sachets",
                    "2x electrolyte sachets",
                    "1x vitamin C supplement"]),
]

MEDICAL_ACTIONS = {
    0: ["Monitor every 4 hours",
        "Standard first aid if needed",
        "Place in general rest area"],
    1: ["Medical assessment within 2 hours",
        "IV fluids if dehydration_level >= 2",
        "Pain relief if injury_severity >= 2",
        "Move to supervised rest area"],
    2: ["IMMEDIATE medical team evaluation",
        "Priority for evacuation transport",
        "Continuous vitals monitoring",
        "IV access — prepare for shock protocol",
        "Notify field hospital of incoming patient"],
}


def get_ration(cal: int) -> list:
    for (lo, hi), items in RATION_PACKS:
        if lo <= cal < hi:
            return items
    return RATION_PACKS[-1][1]


# ═════════════════════════════════════════════════════════════
# REPORT PRINTING
# ═════════════════════════════════════════════════════════════

def print_report(sid: str, vitals: dict, result: dict) -> None:
    pri      = result["priority_code"]
    lbl      = result["priority_label"]
    pcol     = PRIORITY_COLOR[pri]
    rad_lvl, rad_msg = result["radiation_alert"]
    rcol     = RED if rad_lvl in ("CRITICAL", "DANGER") \
               else YLW if rad_lvl == "WARNING" else GRN

    print(f"{BLD}{'='*54}{RST}")
    print(f"{BLD}  SURVIVOR TRIAGE REPORT — {sid}{RST}")
    print(f"{'='*54}")
    print(f"\n  {BLD}MEDICAL PRIORITY:{RST}  {pcol}{BLD}>>  {lbl}  <<{RST}\n")

    # Vitals
    print(f"  {BLD}-- Vitals --{RST}")
    fields = [
        ("Age",           f"{vitals.get('age', '?')} yrs"),
        ("Weight",        f"{vitals.get('weight_kg', '?')} kg"),
        ("Heart Rate",    f"{vitals.get('heart_rate_bpm', '?')} bpm"),
        ("SpO2",          f"{vitals.get('spo2_percent', '?')}%"),
        ("Temperature",   f"{vitals.get('temperature_c', '?')} C"),
        ("BP systolic",   f"{vitals.get('blood_pressure_sys_mmhg', '?')} mmHg"),
        ("Injury sev.",   f"{vitals.get('injury_severity', '?')}/3"),
        ("Days no food",  f"{vitals.get('days_without_food', '?')}"),
        ("Dehydration",   f"Level {vitals.get('dehydration_level', '?')}"),
        ("Radiation",     f"{vitals.get('radiation_mSv', '?')} mSv"),
    ]
    for label_, val in fields:
        print(f"    {label_:<20} {CYN}{val}{RST}")

    # Radiation
    print(f"\n  {BLD}-- Radiation Status --{RST}")
    print(f"    {rcol}{BLD}[{rad_lvl}]{RST}  {rad_msg}")

    # Ration plan
    print(f"\n  {BLD}-- 24-Hour Ration Plan --{RST}")
    print(f"    Target calories  : {CYN}{BLD}{result['calories_kcal']} kcal{RST}")
    print(f"    Water required   : {CYN}{BLD}{result['water_ml']} ml "
          f"({result['water_ml'] / 1000:.1f} L){RST}\n")
    for i, item in enumerate(get_ration(result["calories_kcal"]), 1):
        print(f"    {GRN}[{i}]{RST} {item}")

    # Medical actions
    print(f"\n  {BLD}-- Medical Actions --{RST}")
    for action in MEDICAL_ACTIONS[result["priority_code"]]:
        bullet = (f"{RED}>{RST}" if pri == 2
                  else f"{YLW}>{RST}" if pri == 1
                  else f"{GRN}>{RST}")
        print(f"    {bullet} {action}")

    # Comorbidities
    flags = []
    if vitals.get("has_diabetes"):
        flags.append(f"{YLW}! Diabetic — monitor blood glucose{RST}")
    if vitals.get("has_hypertension"):
        flags.append(f"{YLW}! Hypertensive — watch BP under stress{RST}")
    if vitals.get("has_respiratory"):
        flags.append(f"{RED}! Respiratory condition — SpO2 risk{RST}")
    if flags:
        print(f"\n  {BLD}-- Comorbidity Flags --{RST}")
        for fl in flags:
            print(f"    {fl}")

    print(f"\n{'='*54}\n")


# ═════════════════════════════════════════════════════════════
# RANKED SUMMARY
# ═════════════════════════════════════════════════════════════

def print_summary(results: list) -> None:
    ranked = sorted(results, key=lambda x: x["priority_code"], reverse=True)
    print(f"\n{BLD}{'='*62}{RST}")
    print(f"{BLD}  FINAL TRIAGE RANKING — ALL {len(ranked)} SURVIVORS{RST}")
    print(f"{'='*62}")
    print(f"  {'Rank':<5} {'Survivor':<16} {'Priority':<12} "
          f"{'Calories':>10} {'Water':>10}")
    print(f"  {'─'*56}")
    for rank, r in enumerate(ranked, 1):
        pcol = PRIORITY_COLOR[r["priority_code"]]
        print(f"  {rank:<5} {r['id']:<16} "
              f"{pcol}{r['priority_label']:<12}{RST}"
              f"{r['calories_kcal']:>10} kcal"
              f"{r['water_ml']:>8} ml")
    print(f"\n  {GRN}Triage complete. Reports ready for field medics.{RST}\n")


# ═════════════════════════════════════════════════════════════
# PIPELINE RUNNERS
# ═════════════════════════════════════════════════════════════

def run_demo(models: dict, fast: bool = False) -> list:
    """Kerala 2018 flood demo survivors."""
    DEMO_SURVIVORS = [
        {"_id": "KL-001",
         "age": 68, "weight_kg": 58.0, "height_cm": 165.0,
         "heart_rate_bpm": 112, "blood_pressure_sys_mmhg": 160,
         "spo2_percent": 89.0, "temperature_c": 38.4,
         "radiation_mSv": 0.0, "injury_severity": 3,
         "days_without_food": 3, "dehydration_level": 2,
         "has_diabetes": 1, "has_hypertension": 1, "has_respiratory": 0},

        {"_id": "KL-002",
         "age": 9, "weight_kg": 24.0, "height_cm": 128.0,
         "heart_rate_bpm": 130, "blood_pressure_sys_mmhg": 88,
         "spo2_percent": 94.5, "temperature_c": 39.1,
         "radiation_mSv": 0.0, "injury_severity": 2,
         "days_without_food": 2, "dehydration_level": 3,
         "has_diabetes": 0, "has_hypertension": 0, "has_respiratory": 1},

        {"_id": "KL-003",
         "age": 34, "weight_kg": 72.0, "height_cm": 174.0,
         "heart_rate_bpm": 98, "blood_pressure_sys_mmhg": 122,
         "spo2_percent": 96.0, "temperature_c": 37.2,
         "radiation_mSv": 340.0, "injury_severity": 2,
         "days_without_food": 1, "dehydration_level": 1,
         "has_diabetes": 0, "has_hypertension": 0, "has_respiratory": 1},

        {"_id": "KL-004",
         "age": 52, "weight_kg": 80.0, "height_cm": 168.0,
         "heart_rate_bpm": 75, "blood_pressure_sys_mmhg": 130,
         "spo2_percent": 97.0, "temperature_c": 36.8,
         "radiation_mSv": 12.0, "injury_severity": 0,
         "days_without_food": 0, "dehydration_level": 0,
         "has_diabetes": 0, "has_hypertension": 1, "has_respiratory": 0},
    ]

    print(f"\n{BLD}{CYN}{'='*54}{RST}")
    print(f"{BLD}{CYN}  CAMP TRIAGE & RATION OPTIMIZER{RST}")
    print(f"{BLD}{CYN}  Kerala Flood Relief 2018 — AI Triage{RST}")
    print(f"{BLD}{CYN}  Model: {models['label']}{RST}")
    print(f"{BLD}{CYN}{'='*54}{RST}\n")
    print(f"  {YLW}Processing {len(DEMO_SURVIVORS)} survivors...{RST}\n")

    results = []
    for i, s in enumerate(DEMO_SURVIVORS):
        sid    = s.pop("_id")
        vitals = dict(s)

        print(f"\n{BLD}  -- Survivor {sid} --{RST}")
        print(f"  Vitals received. Starting analysis ritual...\n")
        run_hourglass(duration=10, fast=fast)

        result = predict_survivor(vitals, models)
        results.append({"id": sid, **result, **vitals})
        print_report(sid, vitals, result)

        if not fast and i < len(DEMO_SURVIVORS) - 1:
            input(f"  {YLW}Press ENTER for next survivor...{RST}")

    return results


def run_batch(csv_path: Path, models: dict,
              fast: bool = False, max_n: int = None) -> list:
    """Process survivors from a CSV file."""
    df = pd.read_csv(csv_path)
    if max_n:
        df = df.head(max_n)

    missing = [c for c in FAIR_FEAT_COLS if c not in df.columns]
    if missing:
        print(f"{RED}[ERROR] CSV missing columns: {missing}{RST}")
        raise SystemExit(1)

    results = []
    total   = len(df)
    for idx, row in df.iterrows():
        sid    = str(row.get("survivor_id", f"SURV-{idx+1:04d}"))
        vitals = {c: row[c] for c in FAIR_FEAT_COLS}

        print(f"\n{BLD}  -- {sid} ({idx+1}/{total}) --{RST}")
        run_hourglass(duration=10, fast=fast)

        result = predict_survivor(vitals, models)
        results.append({"id": sid, **result, **vitals})
        print_report(sid, vitals, result)

        if not fast and idx < total - 1:
            input(f"  {YLW}Press ENTER for next survivor...{RST}")

    return results


def run_interactive(models: dict, fast: bool = False) -> list:
    """Enter a single survivor's vitals interactively."""
    print(f"\n{BLD}  INTERACTIVE INPUT{RST}")
    print(f"  Enter vitals. Press ENTER to use default.\n")

    DEFAULTS = {
        "age": 35, "weight_kg": 65.0, "height_cm": 170.0,
        "heart_rate_bpm": 85, "blood_pressure_sys_mmhg": 120,
        "spo2_percent": 96.0, "temperature_c": 37.0,
        "radiation_mSv": 50.0, "injury_severity": 1,
        "days_without_food": 2, "dehydration_level": 1,
        "has_diabetes": 0, "has_hypertension": 0, "has_respiratory": 0,
    }
    TYPES = {
        "age": int, "weight_kg": float, "height_cm": float,
        "heart_rate_bpm": int, "blood_pressure_sys_mmhg": int,
        "spo2_percent": float, "temperature_c": float,
        "radiation_mSv": float, "injury_severity": int,
        "days_without_food": int, "dehydration_level": int,
        "has_diabetes": int, "has_hypertension": int, "has_respiratory": int,
    }

    vitals = {}
    for col in FAIR_FEAT_COLS:
        default = DEFAULTS[col]
        raw     = input(f"  {col} [{default}]: ").strip()
        vitals[col] = TYPES[col](raw) if raw else default

    sid = input(f"\n  Survivor ID [SURV-0001]: ").strip() or "SURV-0001"

    print(f"\n  Vitals received. Starting analysis ritual...\n")
    run_hourglass(duration=10, fast=fast)
    result = predict_survivor(vitals, models)
    print_report(sid, vitals, result)

    return [{"id": sid, **result, **vitals}]


# ═════════════════════════════════════════════════════════════
# SAVE OUTPUTS
# ═════════════════════════════════════════════════════════════

def save_outputs(results: list) -> None:
    summary = [
        {
            "survivor_id":    r["id"],
            "priority_label": r["priority_label"],
            "priority_code":  r["priority_code"],
            "calories_kcal":  r["calories_kcal"],
            "water_ml":       r["water_ml"],
            "radiation_alert":r["radiation_alert"][0],
        }
        for r in results
    ]
    with open(OUTPUT_JSON, "w") as f:
        json.dump({"triage_results": summary}, f, indent=2)

    pd.DataFrame(summary).to_csv(OUTPUT_CSV_PATH, index=False)

    print(f"  Outputs saved:")
    print(f"    {OUTPUT_JSON}   "
          f"({OUTPUT_JSON.stat().st_size // 1024} KB)")
    print(f"    {OUTPUT_CSV_PATH}  "
          f"({OUTPUT_CSV_PATH.stat().st_size // 1024} KB)")


# ═════════════════════════════════════════════════════════════
# ENTRY POINT
# ═════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Camp Triage & Ration Optimizer — Phase 6 CLI")
    parser.add_argument("--csv",         type=str,
                        help="Path to survivors CSV for batch mode")
    parser.add_argument("--interactive", action="store_true",
                        help="Enter a single survivor interactively")
    parser.add_argument("--demo",        action="store_true",
                        help="Run Kerala flood demo (default)")
    parser.add_argument("--fast",        action="store_true",
                        help="Skip 10-second wait (for testing)")
    parser.add_argument("--biased",      action="store_true",
                        help="Use biased Phase 3 models (shows discrimination)")
    parser.add_argument("--max",         type=int, default=None,
                        help="Max rows to process from CSV")
    args = parser.parse_args()

    print(f"\n{'='*55}")
    print(f"{BLD}  Camp Triage  |  Phase 6: Orchestration CLI{RST}")
    print(f"{'='*55}\n")

    # Load (or auto-train) models
    models = load_models(use_fair=not args.biased)
    print(f"  Model : {GRN}{models['label']}{RST}")
    print(f"  Wait  : {'1s (fast mode)' if args.fast else '10s (mandatory ritual)'}\n")

    # Run chosen mode
    if args.interactive:
        results = run_interactive(models, fast=args.fast)
    elif args.csv:
        results = run_batch(Path(args.csv), models,
                            fast=args.fast, max_n=args.max)
    else:
        # Default / --demo
        results = run_demo(models, fast=args.fast)

    print_summary(results)
    save_outputs(results)

    print(f"\n{'='*55}")
    print(f"{GRN}{BLD}  Phase 6 complete.{RST}")
    print(f"  Next -> Phase 7: python phase7_dashboard.py")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()
