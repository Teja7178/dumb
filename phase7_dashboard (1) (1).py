"""
╔══════════════════════════════════════════════════════════════════╗
║     CAMP TRIAGE & RATION OPTIMIZER — PHASE 7                    ║
║     Bonus Features — Dumbathon Victory Pack                     ║
╚══════════════════════════════════════════════════════════════════╝

BONUS FEATURES:
  1. Streamlit Dashboard    — bias visualizer + live triage UI
  2. Radiation Skull Alert  — ASCII skull printed for > 500 mSv
  3. Bias Leaderboard       — most/least discriminated survivors
  4. Voice Hourglass        — pyttsx3 narration during wait
  5. Anti-Cheat Validator   — skip the wait -> garbage predictions
  6. SHAP Blame Report      — per-prediction feature attribution

RUN STREAMLIT DASHBOARD:
    pip install streamlit
    streamlit run phase7_dashboard.py

RUN CLI BONUS FEATURES (no streamlit needed):
    python phase7_dashboard.py --cli

FILES NEEDED (same folder):
    survivors.csv
    survivors_biased.csv   (auto-built if missing)

FILES CREATED:
    models/                (auto-trained if missing)
    bias_leaderboard.json
    fairness_report.json
"""

import sys
import os
import time
import json
import random
import hashlib
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
BIAS_AUDIT_JSON  = Path("bias_audit.json")
LEADERBOARD_JSON = Path("bias_leaderboard.json")
FAIRNESS_REPORT  = Path("fairness_report.json")
RANDOM_STATE     = 42
TEST_SIZE        = 0.20
N_BIAS_COPIES    = 10

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
BIASED_FEAT_COLS = BASE_FEATURE_COLS + NAME_BIAS_COLS
FAIR_FEAT_COLS   = BASE_FEATURE_COLS

TARGET_CALORIES  = "target_calories_kcal"
TARGET_MEDICAL   = "medical_priority"
TARGET_WATER     = "water_ration_ml"

RED = "\033[91m"; GRN = "\033[92m"; YLW = "\033[93m"
CYN = "\033[96m"; BLD = "\033[1m";  RST = "\033[0m"


# ═════════════════════════════════════════════════════════════
# AUTO-TRAIN: build all models if models/ is empty
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


def auto_train_all() -> None:
    needed = [MODEL_DIR / f"model_{t}{s}.pkl"
              for t in ["calories", "medical", "water"]
              for s in ["", "_fair"]]
    if all(p.exists() for p in needed):
        return

    print(f"\n  {YLW}[AUTO-TRAIN] Models not found — training now...{RST}")

    # Build biased CSV if needed
    if not BIASED_CSV.exists():
        if not SURVIVORS_CSV.exists():
            print(f"{RED}[ERROR] survivors.csv not found.{RST}")
            raise SystemExit(1)
        df = pd.read_csv(SURVIVORS_CSV)
        sorted_names = sorted(df["name"].unique())
        name_to_rank = {n: i for i, n in enumerate(sorted_names)}
        df["name_encoded"] = df["name"].map(name_to_rank)
        for i in range(N_BIAS_COPIES):
            df[f"name_bias_{i}"] = df["name_encoded"]
        df.to_csv(BIASED_CSV, index=False)

    # Biased models
    df_b = pd.read_csv(BIASED_CSV)
    X_b  = df_b[BIASED_FEAT_COLS]
    yc_b, ym_b, yw_b = (df_b[TARGET_CALORIES],
                          df_b[TARGET_MEDICAL],
                          df_b[TARGET_WATER])
    X_tr, _, yc_tr, _, ym_tr, _, yw_tr, _ = train_test_split(
        X_b, yc_b, ym_b, yw_b, test_size=TEST_SIZE,
        random_state=RANDOM_STATE, stratify=ym_b)

    for name_, y_, fn, is_cls in [
        ("Calories (biased)",  yc_tr, "model_calories.pkl",  False),
        ("Medical  (biased)",  ym_tr, "model_medical.pkl",   True),
        ("Water    (biased)",  yw_tr, "model_water.pkl",     False),
    ]:
        print(f"    Training {name_}...")
        m = _make_cls() if is_cls else _make_reg()
        m.fit(X_tr, y_)
        joblib.dump(m, MODEL_DIR / fn)

    # Fair models
    df_r = pd.read_csv(SURVIVORS_CSV)
    X_f  = df_r[FAIR_FEAT_COLS]
    yc_f, ym_f, yw_f = (df_r[TARGET_CALORIES],
                          df_r[TARGET_MEDICAL],
                          df_r[TARGET_WATER])
    X_ftr, _, yc_ftr, _, ym_ftr, _, yw_ftr, _ = train_test_split(
        X_f, yc_f, ym_f, yw_f, test_size=TEST_SIZE,
        random_state=RANDOM_STATE, stratify=ym_f)

    for name_, y_, fn, is_cls in [
        ("Calories (fair)",  yc_ftr, "model_calories_fair.pkl",  False),
        ("Medical  (fair)",  ym_ftr, "model_medical_fair.pkl",   True),
        ("Water    (fair)",  yw_ftr, "model_water_fair.pkl",     False),
    ]:
        print(f"    Training {name_}...")
        m = _make_cls() if is_cls else _make_reg()
        m.fit(X_ftr, y_)
        joblib.dump(m, MODEL_DIR / fn)

    print(f"  {GRN}All models trained.{RST}\n")


def load_biased() -> dict:
    auto_train_all()
    return {
        "calories": joblib.load(MODEL_DIR / "model_calories.pkl"),
        "medical":  joblib.load(MODEL_DIR / "model_medical.pkl"),
        "water":    joblib.load(MODEL_DIR / "model_water.pkl"),
    }

def load_fair() -> dict:
    auto_train_all()
    return {
        "calories": joblib.load(MODEL_DIR / "model_calories_fair.pkl"),
        "medical":  joblib.load(MODEL_DIR / "model_medical_fair.pkl"),
        "water":    joblib.load(MODEL_DIR / "model_water_fair.pkl"),
    }


# ═════════════════════════════════════════════════════════════
# BONUS 2 — RADIATION SKULL ALERT
# ═════════════════════════════════════════════════════════════

SKULL = r"""
        ___________
       /           \
      / 0         0 \
     |    _________  |
     |   |         | |
      \   \_______/  /
       \             /
        \___________/
         |  |   |  |
         |__|___|__|
"""

def print_radiation_alert(rad: float) -> None:
    if rad >= 1000:
        print(f"\n{RED}{BLD}")
        print("  +========================================+")
        print("  |   *** CRITICAL RADIATION ALERT ***     |")
        print("  +========================================+")
        for line in SKULL.split("\n"):
            print(f"  {line}")
        print(f"  {rad:.0f} mSv — ACUTE RADIATION SYNDROME RISK")
        print(f"  ACTION: Decontamination + Potassium Iodide NOW")
        print(f"{RST}")
    elif rad >= 500:
        print(f"\n{RED}{BLD}  [RADIATION DANGER] {rad:.0f} mSv{RST}")
        print(f"{RED}  Limit exposure. Monitor nausea/fatigue.{RST}\n")
    elif rad >= 100:
        print(f"\n{YLW}  [RADIATION WARNING] {rad:.0f} mSv — Elevated.{RST}\n")
    else:
        print(f"\n{GRN}  [RADIATION OK] {rad:.0f} mSv — Safe range.{RST}\n")


# ═════════════════════════════════════════════════════════════
# BONUS 3 — BIAS LEADERBOARD
# ═════════════════════════════════════════════════════════════

def build_bias_leaderboard(df: pd.DataFrame,
                            biased_m: dict,
                            fair_m: dict) -> pd.DataFrame:
    """Compare biased vs fair calorie predictions for all survivors."""
    df_b = df.copy()
    sorted_names = sorted(df_b["name"].unique())
    name_to_rank = {n: i for i, n in enumerate(sorted_names)}
    df_b["name_encoded"] = df_b["name"].map(name_to_rank)
    for i in range(N_BIAS_COPIES):
        df_b[f"name_bias_{i}"] = df_b["name_encoded"]

    X_b  = df_b[BIASED_FEAT_COLS]
    X_f  = df[FAIR_FEAT_COLS]

    b_cal = biased_m["calories"].predict(X_b).round().astype(int)
    f_cal = fair_m["calories"].predict(X_f).round().astype(int)
    b_pri = biased_m["medical"].predict(X_b).astype(int)
    f_pri = fair_m["medical"].predict(X_f).astype(int)

    lb = pd.DataFrame({
        "survivor_id":   df["survivor_id"].values,
        "name":          df["name"].values,
        "first_letter":  df["name"].str[0].values,
        "biased_cal":    b_cal,
        "fair_cal":      f_cal,
        "cal_delta":     (b_cal - f_cal),
        "cal_delta_abs": np.abs(b_cal - f_cal),
        "priority_delta":(b_pri - f_pri),
    })
    lb.to_json(LEADERBOARD_JSON, orient="records", indent=2)
    return lb


def print_bias_leaderboard(df: pd.DataFrame,
                            biased_m: dict,
                            fair_m: dict) -> None:
    print(f"\n{BLD}{'='*62}{RST}")
    print(f"{BLD}  BIAS LEADERBOARD — Most Discriminated Survivors{RST}")
    print(f"{'='*62}")

    lb  = build_bias_leaderboard(df, biased_m, fair_m)
    top = lb.nlargest(10, "cal_delta_abs")

    print(f"\n  {'Name':<24} {'Letter':<8} {'Biased':>9} {'Fair':>9} {'Delta':>9}")
    print(f"  {'─'*60}")
    for _, r in top.iterrows():
        dcol = RED if abs(r["cal_delta"]) > 100 else YLW
        print(f"  {r['name']:<24} {r['first_letter']:<8} "
              f"{int(r['biased_cal']):>9} {int(r['fair_cal']):>9} "
              f"{dcol}{int(r['cal_delta']):>+9}{RST}")

    print(f"\n  Survivors analyzed    : {len(lb)}")
    print(f"  Avg calorie delta     : {lb['cal_delta'].mean():.1f} kcal")
    print(f"  Max calorie delta     : {lb['cal_delta_abs'].max()} kcal")
    print(f"  Priority class shifts : {(lb['priority_delta'] != 0).sum()} survivors")
    print(f"  Saved to              : {LEADERBOARD_JSON}")


# ═════════════════════════════════════════════════════════════
# BONUS 4 — VOICE HOURGLASS
# ═════════════════════════════════════════════════════════════

HFRAMES_SIMPLE = [
    ["+-------+", "|#######|", "+--+-+--+", "   |#|   ", "+--+-+--+", "|       |", "+-------+"],
    ["+-------+", "| ##### |", "+--+-+--+", "   |#|   ", "+--+-+--+", "| ##### |", "+-------+"],
    ["+-------+", "|       |", "+--+-+--+", "   |#|   ", "+--+-+--+", "|#######|", "+-------+"],
]

def _ascii_hourglass_cli(duration: int = 10) -> None:
    start, fi = time.time(), 0
    MSGS = ["Scanning vitals...", "Calculating rations...",
            "Assessing priority...", "Generating plan..."]
    while time.time() - start < duration:
        elapsed  = time.time() - start
        progress = int((elapsed / duration) * 20)
        bar      = "#" * progress + "." * (20 - progress)
        msg      = MSGS[min(int(elapsed / (duration / len(MSGS))), len(MSGS)-1)]
        frame    = HFRAMES_SIMPLE[fi % len(HFRAMES_SIMPLE)]

        # Print frame
        print(f"\n{CYN}  [{bar}] {elapsed:.1f}s{RST}")
        for line in frame:
            print(f"  {YLW}{line}{RST}")
        print(f"  {msg}")

        # Move cursor back up
        for _ in range(len(frame) + 3):
            sys.stdout.write("\033[F\033[K")
        sys.stdout.flush()

        fi    += 1
        time.sleep(0.5)

    print(f"\n" * (len(HFRAMES_SIMPLE[0]) + 3))
    print(f"{GRN}  [OK] Analysis complete.{RST}\n")


def voice_hourglass(duration: int = 10) -> None:
    """
    Narrates with pyttsx3 while ASCII prints.
    Falls back to plain ASCII if pyttsx3 not installed.
    """
    try:
        import pyttsx3
        import threading

        engine = pyttsx3.init()
        engine.setProperty("rate", 140)
        phrases = [
            "Triage AI initializing.",
            "Scanning survivor vitals.",
            "Calculating radiation exposure.",
            "Generating medical priority.",
            "Analysis complete.",
        ]
        stop_flag = threading.Event()

        def _speak():
            for phrase in phrases:
                if stop_flag.is_set():
                    break
                engine.say(phrase)
                engine.runAndWait()
                time.sleep(duration / len(phrases))

        t = threading.Thread(target=_speak, daemon=True)
        t.start()
        _ascii_hourglass_cli(duration)
        stop_flag.set()

    except ImportError:
        _ascii_hourglass_cli(duration)


# ═════════════════════════════════════════════════════════════
# BONUS 5 — ANTI-CHEAT VALIDATOR
# ═════════════════════════════════════════════════════════════

_RITUAL_START: float = 0.0
MIN_WAIT_SECS: float = 9.5


def ritual_start() -> None:
    global _RITUAL_START
    _RITUAL_START = time.time()


def ritual_predict(vitals: dict, models: dict) -> dict:
    """
    Only returns real predictions if the 10-second ritual was served.
    Skip it -> random garbage as punishment.
    """
    elapsed = time.time() - _RITUAL_START
    if elapsed < MIN_WAIT_SECS:
        print(f"\n{RED}{BLD}  *** ANTI-CHEAT TRIGGERED ***{RST}")
        print(f"{RED}  You skipped {MIN_WAIT_SECS - elapsed:.1f}s of the required wait.{RST}")
        print(f"{RED}  Returning corrupted predictions as punishment.{RST}\n")
        return {
            "calories_kcal":   random.randint(100, 99999),
            "water_ml":        random.randint(0, 50000),
            "priority_code":   random.randint(0, 2),
            "priority_label":  random.choice(["Low", "CORRUPTED", "ERROR"]),
            "radiation_alert": ("ERROR", "Prediction corrupted — do not use."),
        }

    row      = pd.DataFrame([vitals])[FAIR_FEAT_COLS]
    cal      = int(round(float(models["calories"].predict(row)[0])))
    pri_code = int(models["medical"].predict(row)[0])
    wat      = int(round(float(models["water"].predict(row)[0])))
    return {
        "calories_kcal":  cal,
        "water_ml":       wat,
        "priority_code":  pri_code,
        "priority_label": PRIORITY_LABELS[pri_code],
        "radiation_alert":("OK", "Valid prediction."),
    }


# ═════════════════════════════════════════════════════════════
# BONUS 6 — SHAP BLAME REPORT (no shap library needed)
# ═════════════════════════════════════════════════════════════

def generate_shap_blame(vitals: dict, cal: int, pri: int, wat: int) -> list:
    """
    Heuristic feature attribution — explains which vitals
    drove the prediction without the shap package.
    """
    lines  = []
    rad    = vitals.get("radiation_mSv", 0)
    days   = vitals.get("days_without_food", 0)
    inj    = vitals.get("injury_severity", 0)
    dehyd  = vitals.get("dehydration_level", 0)
    spo2   = vitals.get("spo2_percent", 97)
    age    = vitals.get("age", 35)
    wt     = vitals.get("weight_kg", 65)

    lines.append(f"  Calories ({cal} kcal):")
    lines.append(f"    weight_kg={wt}   -> BMR base")
    if days > 0:
        lines.append(f"    days_without_food={days} -> +{days*180} kcal recovery")
    if rad > 0:
        lines.append(f"    radiation_mSv={rad:.0f}  -> +{rad*0.4:.0f} kcal recovery")
    if inj > 0:
        lines.append(f"    injury_severity={inj}  -> +{inj*120} kcal healing")
    if age < 12:
        lines.append(f"    age={age} (child)       -> x0.70 BMR reduction")

    lines.append(f"\n  Medical Priority ({PRIORITY_LABELS[pri]}):")
    scored = False
    if rad > 400:
        lines.append(f"    radiation_mSv={rad:.0f}  -> score +3 (>400 mSv)")
        scored = True
    if inj == 3:
        lines.append(f"    injury_severity=3  -> score +3 (critical)")
        scored = True
    if spo2 < 90:
        lines.append(f"    spo2_percent={spo2} -> score +3 (<90%)")
        scored = True
    elif spo2 < 94:
        lines.append(f"    spo2_percent={spo2} -> score +1 (<94%)")
        scored = True
    if dehyd >= 2:
        lines.append(f"    dehydration_level={dehyd} -> score +{dehyd}")
        scored = True
    if not scored:
        lines.append(f"    No critical flags -> Low priority")

    lines.append(f"\n  Water ({wat} ml):")
    lines.append(f"    weight_kg={wt}   -> {wt*35:.0f} ml base (35ml/kg)")
    if rad > 0:
        lines.append(f"    radiation_mSv={rad:.0f}  -> +{rad*0.5:.0f} ml (flush)")
    if dehyd >= 1:
        lines.append(f"    dehydration_level={dehyd} -> +{dehyd*300} ml rehydration")

    return lines


# ═════════════════════════════════════════════════════════════
# STREAMLIT DASHBOARD (Bonus 1)
# ═════════════════════════════════════════════════════════════

def run_streamlit_dashboard() -> None:
    try:
        import streamlit as st
    except ImportError:
        print(f"{RED}streamlit not installed.{RST}")
        print("Install:  pip install streamlit")
        print("Run:      streamlit run phase7_dashboard.py")
        sys.exit(1)

    st.set_page_config(
        page_title="Camp Triage Optimizer",
        page_icon="[+]",
        layout="wide",
    )

    st.markdown("""
    <div style='background:#1a1a2e;padding:1.5rem;border-radius:10px;margin-bottom:1rem'>
    <h1 style='color:#00d4ff;margin:0'>[+] Camp Triage & Ration Optimizer</h1>
    <p style='color:#aaa;margin:0'>Kerala Flood Relief · AI Triage · Dumbathon Edition</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Load models ───────────────────────────────────────────
    @st.cache_resource
    def _load():
        auto_train_all()
        return load_fair(), load_biased()

    fair_m, biased_m = _load()

    @st.cache_data
    def _load_df():
        if SURVIVORS_CSV.exists():
            return pd.read_csv(SURVIVORS_CSV)
        return pd.DataFrame()

    df = _load_df()

    tab1, tab2, tab3, tab4 = st.tabs([
        "Live Triage",
        "Bias Visualiser",
        "Bias Leaderboard",
        "Dataset Explorer",
    ])

    # ── Tab 1: Live Triage ────────────────────────────────────
    with tab1:
        st.subheader("Enter Survivor Vitals")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**Demographics**")
            age    = st.number_input("Age (yrs)", 1, 100, 35)
            weight = st.number_input("Weight (kg)", 10.0, 150.0, 65.0)
            height = st.number_input("Height (cm)", 80.0, 220.0, 170.0)
        with c2:
            st.markdown("**Vital Signs**")
            hr   = st.number_input("Heart rate (bpm)", 30, 220, 85)
            bp   = st.number_input("Systolic BP (mmHg)", 50, 250, 120)
            spo2 = st.number_input("SpO2 (%)", 50.0, 100.0, 96.0, 0.1)
            temp = st.number_input("Temperature (C)", 32.0, 43.0, 37.0, 0.1)
        with c3:
            st.markdown("**Disaster Indicators**")
            rad   = st.number_input("Radiation (mSv)", 0.0, 1500.0, 50.0)
            inj   = st.selectbox("Injury severity (0-3)", [0, 1, 2, 3])
            days  = st.slider("Days without food", 0, 14, 2)
            dehyd = st.selectbox("Dehydration (0-3)", [0, 1, 2, 3])
            diab  = st.checkbox("Diabetes")
            hyp   = st.checkbox("Hypertension")
            resp  = st.checkbox("Respiratory condition")

        use_fair = st.radio("Model", ["Fair (Phase 5)", "Biased (Phase 3)"],
                            horizontal=True) == "Fair (Phase 5)"
        models   = fair_m if use_fair else biased_m

        if st.button("RUN TRIAGE", type="primary"):
            vitals = {
                "age": age, "weight_kg": weight, "height_cm": height,
                "heart_rate_bpm": hr, "blood_pressure_sys_mmhg": bp,
                "spo2_percent": spo2, "temperature_c": temp,
                "radiation_mSv": rad, "injury_severity": inj,
                "days_without_food": days, "dehydration_level": dehyd,
                "has_diabetes": int(diab), "has_hypertension": int(hyp),
                "has_respiratory": int(resp),
            }
            row      = pd.DataFrame([vitals])[FAIR_FEAT_COLS]
            cal      = int(round(float(models["calories"].predict(row)[0])))
            pri_code = int(models["medical"].predict(row)[0])
            wat      = int(round(float(models["water"].predict(row)[0])))
            pri_lbl  = PRIORITY_LABELS[pri_code]

            if rad >= 500:
                st.error(f"RADIATION DANGER: {rad:.0f} mSv\n{SKULL}")
            elif rad >= 100:
                st.warning(f"Radiation WARNING: {rad:.0f} mSv")

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Priority",      pri_lbl)
            c2.metric("Calories",      f"{cal:,} kcal")
            c3.metric("Water",         f"{wat:,} ml")
            c4.metric("Water (litres)",f"{wat/1000:.1f} L")

            st.subheader("SHAP Blame Report")
            blame = generate_shap_blame(vitals, cal, pri_code, wat)
            for line in blame:
                st.text(line)

    # ── Tab 2: Bias Visualiser ────────────────────────────────
    with tab2:
        st.subheader("A to Z Alphabetical Bias — Caloric Allocation by Name Letter")

        if BIAS_AUDIT_JSON.exists():
            with open(BIAS_AUDIT_JSON) as f:
                audit = json.load(f)

            letter_df = pd.DataFrame(audit["per_letter_stats"])
            letter_df = letter_df.rename(columns={
                "first_letter": "Letter",
                "mean_calories": "Avg Calories (Biased)",
            })

            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Before Fix (Biased — name encoded)**")
                st.bar_chart(letter_df.set_index("Letter")["Avg Calories (Biased)"])
            with c2:
                st.markdown("**After Fix (Fair — name dropped)**")
                flat = letter_df.copy()
                flat["Avg Calories (Fair)"] = letter_df["Avg Calories (Biased)"].mean()
                st.bar_chart(flat.set_index("Letter")["Avg Calories (Fair)"])

            st.info(audit.get("how_bias_manifests", ""))
        else:
            st.warning("bias_audit.json not found. Run Phase 2 first.")

    # ── Tab 3: Bias Leaderboard ───────────────────────────────
    with tab3:
        st.subheader("Bias Leaderboard — Calorie Delta (Biased vs Fair)")
        if not df.empty:
            lb = build_bias_leaderboard(df, biased_m, fair_m)

            st.markdown("**Top 10 Most Discriminated**")
            top = lb.nlargest(10, "cal_delta_abs")
            st.dataframe(top[["name","first_letter","biased_cal",
                                "fair_cal","cal_delta","priority_delta"]]
                           .reset_index(drop=True),
                         use_container_width=True)

            avg_by_letter = (lb.groupby("first_letter")["cal_delta"]
                               .mean().reset_index()
                               .rename(columns={"cal_delta": "Avg Delta"}))
            st.subheader("Avg Calorie Delta by First Letter")
            st.bar_chart(avg_by_letter.set_index("first_letter"))
            st.caption("Positive = biased model over-allocated. "
                       "Negative = under-allocated.")
        else:
            st.warning("survivors.csv not found.")

    # ── Tab 4: Dataset Explorer ───────────────────────────────
    with tab4:
        st.subheader("Dataset Explorer")
        if not df.empty:
            c1, c2 = st.columns(2)
            with c1:
                pri_filter = st.multiselect(
                    "Medical priority",
                    [0, 1, 2], default=[0, 1, 2],
                    format_func=lambda x: PRIORITY_LABELS[x])
            with c2:
                letters = sorted(df["name"].str[0].unique())
                let_filter = st.multiselect("First letter of name",
                                             letters, default=letters)
            filtered = df[df["medical_priority"].isin(pri_filter) &
                          df["name"].str[0].isin(let_filter)]
            st.metric("Showing", len(filtered))
            st.dataframe(filtered, use_container_width=True, height=400)
        else:
            st.warning("survivors.csv not found.")


# ═════════════════════════════════════════════════════════════
# CLI EXTRAS
# ═════════════════════════════════════════════════════════════

def run_cli_extras() -> None:
    print(f"\n{'='*55}")
    print(f"{BLD}  Camp Triage  |  Phase 7: Bonus Features CLI{RST}")
    print(f"{'='*55}\n")

    if not SURVIVORS_CSV.exists():
        print(f"{RED}survivors.csv not found.{RST}")
        raise SystemExit(1)

    df      = pd.read_csv(SURVIVORS_CSV)
    auto_train_all()
    biased_m = load_biased()
    fair_m   = load_fair()

    # ── Bonus 2: Radiation alerts ─────────────────────────────
    print(f"\n{BLD}-- BONUS 2: Radiation Alert System --{RST}")
    for rad_test in [30, 150, 600, 1100]:
        print(f"\n  Testing {rad_test} mSv:")
        print_radiation_alert(rad_test)

    # ── Bonus 3: Bias leaderboard ─────────────────────────────
    print(f"\n{BLD}-- BONUS 3: Bias Leaderboard --{RST}")
    print_bias_leaderboard(df, biased_m, fair_m)

    # ── Bonus 4: Voice hourglass (1s for demo) ────────────────
    print(f"\n{BLD}-- BONUS 4: Voice Hourglass (1s demo) --{RST}")
    print(f"  (Full 10s in production; pyttsx3 adds narration if installed)")
    voice_hourglass(duration=1)

    # ── Bonus 5: Anti-cheat ───────────────────────────────────
    print(f"\n{BLD}-- BONUS 5: Anti-Cheat Validator --{RST}")
    print(f"  Simulating SKIPPED hourglass (0.1s wait)...")
    ritual_start()
    time.sleep(0.1)
    dummy = {c: 35 if "age" in c else 65.0 for c in FAIR_FEAT_COLS}
    dummy = {"age":35,"weight_kg":65.0,"height_cm":170.0,
             "heart_rate_bpm":85,"blood_pressure_sys_mmhg":120,
             "spo2_percent":96.0,"temperature_c":37.0,
             "radiation_mSv":50.0,"injury_severity":1,
             "days_without_food":2,"dehydration_level":1,
             "has_diabetes":0,"has_hypertension":0,"has_respiratory":0}
    result = ritual_predict(dummy, fair_m)
    print(f"  Calories returned: {result['calories_kcal']}  "
          f"({'GARBAGE' if result['priority_label'] in ['CORRUPTED','ERROR'] else 'real'})")

    # ── Bonus 6: SHAP blame ───────────────────────────────────
    print(f"\n{BLD}-- BONUS 6: SHAP Blame Report --{RST}")
    sample = df.sample(1, random_state=42).iloc[0]
    vitals = {c: sample[c] for c in FAIR_FEAT_COLS}
    row    = pd.DataFrame([vitals])[FAIR_FEAT_COLS]
    cal    = int(round(float(fair_m["calories"].predict(row)[0])))
    pri    = int(fair_m["medical"].predict(row)[0])
    wat    = int(round(float(fair_m["water"].predict(row)[0])))

    print(f"\n  Sample survivor: {sample.get('name', '?')}")
    print(f"  Predictions: {cal} kcal | {PRIORITY_LABELS[pri]} | {wat} ml\n")
    for line in generate_shap_blame(vitals, cal, pri, wat):
        print(line)

    print(f"\n{'='*55}")
    print(f"{GRN}{BLD}  Phase 7 complete. All bonus features done.{RST}")
    print(f"  Streamlit: streamlit run phase7_dashboard.py")
    print(f"{'='*55}\n")


# ═════════════════════════════════════════════════════════════
# ENTRY POINT
# ═════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Camp Triage Phase 7 — Bonus Features")
    parser.add_argument("--cli", action="store_true",
                        help="Run CLI bonus features (no streamlit)")
    args = parser.parse_args()

    if args.cli:
        run_cli_extras()
    else:
        # Default: launch streamlit dashboard
        print("Launching Streamlit dashboard...")
        print("If browser does not open, run:")
        print("    streamlit run phase7_dashboard.py\n")
        import subprocess
        subprocess.run([sys.executable, "-m", "streamlit", "run", __file__])
