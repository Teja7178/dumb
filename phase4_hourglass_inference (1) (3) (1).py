"""
╔══════════════════════════════════════════════════════════════════╗
║     CAMP TRIAGE & RATION OPTIMIZER — PHASE 4                    ║
║     10-Second Hourglass Inference Ritual                         ║
║                                                                  ║
║  Real-life scenario: Kerala Floods 2018 / Disaster Triage       ║
║                                                                  ║
║  What this phase does:                                           ║
║    1. Accepts survivor vitals as input                           ║
║    2. Plays a dramatic 10-second ASCII hourglass animation       ║
║       (orchestration pause — makes the "AI thinking" visible)   ║
║    3. Loads saved Phase 3 models                                 ║
║    4. Predicts calories, water, medical priority                 ║
║    5. Prints a ranked ration + medical action output             ║
║    6. Flags radiation danger (Kerala scenario)                   ║
╚══════════════════════════════════════════════════════════════════╝
"""

import time
import sys
import os
import joblib
import pandas as pd
import numpy as np

# ══════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════

MODEL_DIR = "."   # folder where .joblib files were saved in Phase 3

FEATURE_COLS = [
    "age", "weight_kg", "height_cm",
    "heart_rate_bpm", "blood_pressure_sys_mmhg",
    "spo2_percent", "temperature_c",
    "radiation_mSv", "injury_severity",
    "days_without_food", "dehydration_level",
    "has_diabetes", "has_hypertension", "has_respiratory",
]

PRIORITY_MAP   = {0: "LOW", 1: "MEDIUM", 2: "HIGH"}
PRIORITY_COLOR = {0: "\033[92m", 1: "\033[93m", 2: "\033[91m"}  # green/yellow/red
RESET          = "\033[0m"
BOLD           = "\033[1m"
CYAN           = "\033[96m"
YELLOW         = "\033[93m"
RED            = "\033[91m"
GREEN          = "\033[92m"
MAGENTA        = "\033[95m"

# ══════════════════════════════════════════════════════════════════
# SECTION 1 — ASCII HOURGLASS ANIMATION (the ritual)
# ══════════════════════════════════════════════════════════════════

HOURGLASS_FRAMES = [
    # Frame 0 — full top
    [
        "   ╔═══════╗   ",
        "   ║▓▓▓▓▓▓▓║   ",
        "   ║▓▓▓▓▓▓▓║   ",
        "   ╚══╦═╦══╝   ",
        "      ║▓║      ",
        "   ╔══╩═╩══╗   ",
        "   ║       ║   ",
        "   ║       ║   ",
        "   ╚═══════╝   ",
    ],
    # Frame 1
    [
        "   ╔═══════╗   ",
        "   ║▓▓▓▓▓▓▓║   ",
        "   ║  ▓▓▓  ║   ",
        "   ╚══╦═╦══╝   ",
        "      ║▓║      ",
        "   ╔══╩═╩══╗   ",
        "   ║   ▓   ║   ",
        "   ║       ║   ",
        "   ╚═══════╝   ",
    ],
    # Frame 2
    [
        "   ╔═══════╗   ",
        "   ║▓▓▓▓▓▓▓║   ",
        "   ║       ║   ",
        "   ╚══╦═╦══╝   ",
        "      ║▓║      ",
        "   ╔══╩═╩══╗   ",
        "   ║  ▓▓▓  ║   ",
        "   ║       ║   ",
        "   ╚═══════╝   ",
    ],
    # Frame 3
    [
        "   ╔═══════╗   ",
        "   ║ ▓▓▓▓▓ ║   ",
        "   ║       ║   ",
        "   ╚══╦═╦══╝   ",
        "      ║▓║      ",
        "   ╔══╩═╩══╗   ",
        "   ║ ▓▓▓▓▓ ║   ",
        "   ║       ║   ",
        "   ╚═══════╝   ",
    ],
    # Frame 4
    [
        "   ╔═══════╗   ",
        "   ║  ▓▓▓  ║   ",
        "   ║       ║   ",
        "   ╚══╦═╦══╝   ",
        "      ║▓║      ",
        "   ╔══╩═╩══╗   ",
        "   ║▓▓▓▓▓▓▓║   ",
        "   ║       ║   ",
        "   ╚═══════╝   ",
    ],
    # Frame 5
    [
        "   ╔═══════╗   ",
        "   ║   ▓   ║   ",
        "   ║       ║   ",
        "   ╚══╦═╦══╝   ",
        "      ║▓║      ",
        "   ╔══╩═╩══╗   ",
        "   ║▓▓▓▓▓▓▓║   ",
        "   ║  ▓▓▓  ║   ",
        "   ╚═══════╝   ",
    ],
    # Frame 6 — empty top, full bottom
    [
        "   ╔═══════╗   ",
        "   ║       ║   ",
        "   ║       ║   ",
        "   ╚══╦═╦══╝   ",
        "      ║ ║      ",
        "   ╔══╩═╩══╗   ",
        "   ║▓▓▓▓▓▓▓║   ",
        "   ║▓▓▓▓▓▓▓║   ",
        "   ╚═══════╝   ",
    ],
]

HOURGLASS_MESSAGES = [
    "Scanning survivor vitals...",
    "Cross-referencing radiation levels...",
    "Calculating caloric deficit...",
    "Assessing hydration requirements...",
    "Running injury severity analysis...",
    "Evaluating medical priority...",
    "Generating ration plan...",
]

def clear_line(n=1):
    """Move cursor up n lines and clear them."""
    for _ in range(n):
        sys.stdout.write("\033[F\033[K")
    sys.stdout.flush()

def print_hourglass(frame_idx, message, elapsed, total=10):
    """Print one hourglass frame with status message."""
    frame = HOURGLASS_FRAMES[frame_idx % len(HOURGLASS_FRAMES)]
    progress = int((elapsed / total) * 20)
    bar = "█" * progress + "░" * (20 - progress)

    print(f"\n{CYAN}{'─'*40}{RESET}")
    print(f"{BOLD}{CYAN}  ⚕  CAMP TRIAGE AI — ANALYZING SURVIVOR{RESET}")
    print(f"{CYAN}{'─'*40}{RESET}")
    for line in frame:
        print(f"  {YELLOW}{line}{RESET}")
    print(f"\n  {MAGENTA}▶ {message}{RESET}")
    print(f"  [{CYAN}{bar}{RESET}] {elapsed:.1f}s / {total}s")
    print(f"{CYAN}{'─'*40}{RESET}")

    # total lines printed = 2 + 1 + 1 + 9 + 1 + 1 + 1 + 1 = 17
    return 17

def run_hourglass(duration=10):
    """
    Runs the 10-second hourglass ritual.
    Prints frames sequentially — each frame overwrites the last.
    """
    print()
    frame_count   = 0
    frame_height  = None
    start         = time.time()
    msg_idx       = 0

    while True:
        elapsed = time.time() - start
        if elapsed >= duration:
            break

        # Clear previous frame
        if frame_height is not None:
            clear_line(frame_height + 1)  # +1 for the blank line before

        msg_idx = min(int(elapsed / (duration / len(HOURGLASS_MESSAGES))),
                      len(HOURGLASS_MESSAGES) - 1)

        frame_height = print_hourglass(
            frame_count,
            HOURGLASS_MESSAGES[msg_idx],
            elapsed,
            duration
        )
        frame_count += 1
        time.sleep(0.4)

    # Clear and show completion
    if frame_height is not None:
        clear_line(frame_height + 1)

    print(f"\n{GREEN}{'─'*40}{RESET}")
    print(f"{BOLD}{GREEN}  ✓  ANALYSIS COMPLETE — GENERATING REPORT{RESET}")
    print(f"{GREEN}{'─'*40}{RESET}\n")
    time.sleep(0.5)


# ══════════════════════════════════════════════════════════════════
# SECTION 2 — LOAD MODELS
# ══════════════════════════════════════════════════════════════════

def load_models():
    """Load all three Phase 3 models + scaler from disk."""
    try:
        cal_model   = joblib.load(os.path.join(MODEL_DIR, "model_calories.joblib"))
        water_model = joblib.load(os.path.join(MODEL_DIR, "model_water.joblib"))
        pri_model   = joblib.load(os.path.join(MODEL_DIR, "model_priority.joblib"))
        scaler      = joblib.load(os.path.join(MODEL_DIR, "scaler.joblib"))
        return cal_model, water_model, pri_model, scaler
    except FileNotFoundError as e:
        print(f"{RED}ERROR: Model file not found — {e}{RESET}")
        print(f"{YELLOW}Run phase3 (camp_triage_optimizer.py) first to train and save models.{RESET}")
        sys.exit(1)


# ══════════════════════════════════════════════════════════════════
# SECTION 3 — PREDICT
# ══════════════════════════════════════════════════════════════════

def predict(survivor: dict, cal_model, water_model, pri_model, scaler) -> dict:
    """Run all three models on a single survivor input."""
    row       = pd.DataFrame([survivor])[FEATURE_COLS]
    row_scaled = scaler.transform(row)

    calories  = int(round(cal_model.predict(row_scaled)[0]))
    water_ml  = int(round(water_model.predict(row_scaled)[0]))
    pri_code  = int(pri_model.predict(row_scaled)[0])
    pri_label = PRIORITY_MAP[pri_code]

    # ── Radiation danger flag (Kerala / nuclear scenario) ─────────
    # WHO threshold: >100 mSv = elevated risk
    rad       = survivor.get("radiation_mSv", 0)
    rad_alert = None
    if rad >= 1000:
        rad_alert = ("CRITICAL", f"≥1000 mSv — acute radiation syndrome risk. "
                                  "Immediate decontamination + potassium iodide.")
    elif rad >= 500:
        rad_alert = ("DANGER",   f"{rad:.0f} mSv — high radiation. "
                                  "Limit further exposure. Monitor for vomiting/fatigue.")
    elif rad >= 100:
        rad_alert = ("WARNING",  f"{rad:.0f} mSv — elevated radiation. "
                                  "Log exposure. Reassess in 24 hrs.")
    elif rad > 0:
        rad_alert = ("OK",       f"{rad:.0f} mSv — within tolerable range.")

    return {
        "calories_kcal": calories,
        "water_ml":      water_ml,
        "priority_code": pri_code,
        "priority_label": pri_label,
        "radiation_alert": rad_alert,
    }


# ══════════════════════════════════════════════════════════════════
# SECTION 4 — PRINT RATION REPORT
# ══════════════════════════════════════════════════════════════════

RATION_ITEMS = {
    # kcal ranges → suggested ration packs
    (0,    1200): ["1× emergency biscuit pack (400 kcal)",
                   "1× oral rehydration sachet",
                   "1× multivitamin tablet"],
    (1200, 1800): ["2× emergency biscuit packs (800 kcal)",
                   "1× ready-to-eat meal pouch (500 kcal)",
                   "2× oral rehydration sachets"],
    (1800, 2400): ["2× ready-to-eat meal pouches (1000 kcal)",
                   "1× high-energy protein bar (300 kcal)",
                   "2× oral rehydration sachets",
                   "1× electrolyte sachet"],
    (2400, 3200): ["3× ready-to-eat meal pouches (1500 kcal)",
                   "2× high-energy protein bars (600 kcal)",
                   "3× oral rehydration sachets",
                   "1× electrolyte sachet"],
    (3200, 9999): ["4× ready-to-eat meal pouches (2000 kcal)",
                   "3× high-energy protein bars (900 kcal)",
                   "4× oral rehydration sachets",
                   "2× electrolyte sachets",
                   "1× vitamin C supplement"],
}

MEDICAL_ACTIONS = {
    0: [  # LOW
        "Monitor every 4 hours",
        "Standard first aid if needed",
        "Place in general rest area",
    ],
    1: [  # MEDIUM
        "Medical assessment within 2 hours",
        "IV fluids if dehydration_level ≥ 2",
        "Pain relief if injury_severity ≥ 5",
        "Move to supervised rest area",
    ],
    2: [  # HIGH
        "IMMEDIATE medical team evaluation",
        "Priority for evacuation transport",
        "Continuous vitals monitoring",
        "IV access — prepare for shock protocol",
        "Notify field hospital of incoming",
    ],
}

def get_ration_items(calories: int) -> list:
    for (lo, hi), items in RATION_ITEMS.items():
        if lo <= calories < hi:
            return items
    return RATION_ITEMS[(3200, 9999)]

def print_report(survivor_id: str, survivor: dict, result: dict):
    """Print the full ranked ration + medical action report."""
    pri_code  = result["priority_code"]
    pri_label = result["priority_label"]
    color     = PRIORITY_COLOR[pri_code]

    print(f"{BOLD}{'═'*52}{RESET}")
    print(f"{BOLD}  SURVIVOR TRIAGE REPORT — {survivor_id}{RESET}")
    print(f"{'═'*52}")

    # ── Priority banner ───────────────────────────────────────────
    print(f"\n  {BOLD}MEDICAL PRIORITY:{RESET}  "
          f"{color}{BOLD}▶▶ {pri_label} ◀◀{RESET}")

    # ── Vitals summary ────────────────────────────────────────────
    print(f"\n  {BOLD}── Vitals Snapshot ──{RESET}")
    vitals = [
        ("Age",           f"{survivor['age']} yrs"),
        ("Weight",        f"{survivor['weight_kg']} kg"),
        ("Heart Rate",    f"{survivor['heart_rate_bpm']} bpm"),
        ("SpO2",          f"{survivor['spo2_percent']}%"),
        ("Temp",          f"{survivor['temperature_c']}°C"),
        ("Injury Sev.",   f"{survivor['injury_severity']}/10"),
        ("Days w/o food", f"{survivor['days_without_food']} days"),
        ("Dehydration",   f"Level {survivor['dehydration_level']}"),
        ("Radiation",     f"{survivor['radiation_mSv']} mSv"),
    ]
    for label, val in vitals:
        print(f"    {label:<18} {CYAN}{val}{RESET}")

    # ── Radiation alert ───────────────────────────────────────────
    rad_alert = result["radiation_alert"]
    if rad_alert:
        level, msg = rad_alert
        alert_color = RED if level in ("CRITICAL","DANGER") else YELLOW if level == "WARNING" else GREEN
        print(f"\n  {BOLD}── Radiation Status ──{RESET}")
        print(f"    {alert_color}{BOLD}[{level}]{RESET} {msg}")

    # ── Ration plan ───────────────────────────────────────────────
    print(f"\n  {BOLD}── 24-Hour Ration Plan ──{RESET}")
    print(f"    Target calories : {CYAN}{BOLD}{result['calories_kcal']} kcal{RESET}")
    print(f"    Water required  : {CYAN}{BOLD}{result['water_ml']} ml "
          f"({result['water_ml']/1000:.1f} litres){RESET}\n")

    items = get_ration_items(result["calories_kcal"])
    for i, item in enumerate(items, 1):
        print(f"    {GREEN}[{i}]{RESET} {item}")

    # ── Medical actions ───────────────────────────────────────────
    print(f"\n  {BOLD}── Medical Actions (Priority: {color}{pri_label}{RESET}{BOLD}) ──{RESET}")
    actions = MEDICAL_ACTIONS[pri_code]
    for i, action in enumerate(actions, 1):
        bullet = f"{RED}▶{RESET}" if pri_code == 2 else f"{YELLOW}▶{RESET}" if pri_code == 1 else f"{GREEN}▶{RESET}"
        print(f"    {bullet} {action}")

    # ── Comorbidity flags ─────────────────────────────────────────
    flags = []
    if survivor.get("has_diabetes"):    flags.append(f"{YELLOW}⚠ Diabetic — monitor blood glucose{RESET}")
    if survivor.get("has_hypertension"):flags.append(f"{YELLOW}⚠ Hypertensive — watch BP under stress{RESET}")
    if survivor.get("has_respiratory"): flags.append(f"{RED}⚠ Respiratory condition + low SpO2 risk{RESET}")

    if flags:
        print(f"\n  {BOLD}── Comorbidity Flags ──{RESET}")
        for f in flags:
            print(f"    {f}")

    print(f"\n{'═'*52}\n")


# ══════════════════════════════════════════════════════════════════
# SECTION 5 — ORCHESTRATION: BATCH TRIAGE PIPELINE
# ══════════════════════════════════════════════════════════════════

def run_triage_pipeline(survivors: list[dict]):
    """
    Full orchestration:
      For each survivor →
        1. Print their ID
        2. Run 10-second hourglass
        3. Predict
        4. Print ranked report
      Then print a final ranked summary table.
    """
    print(f"\n{BOLD}{CYAN}{'╔'+'═'*50+'╗'}{RESET}")
    print(f"{BOLD}{CYAN}║{'CAMP TRIAGE & RATION OPTIMIZER':^50}║{RESET}")
    print(f"{BOLD}{CYAN}║{'Kerala Flood Relief — AI Triage System':^50}║{RESET}")
    print(f"{BOLD}{CYAN}{'╚'+'═'*50+'╝'}{RESET}\n")
    print(f"  {YELLOW}Processing {len(survivors)} survivor(s)...{RESET}\n")

    models = load_models()
    results_summary = []

    for idx, survivor in enumerate(survivors):
        sid = survivor.pop("_id", f"SURV-{idx+1:04d}")
        print(f"\n{BOLD}  ── Survivor {sid} ──{RESET}")
        print(f"  Incoming vitals received. Starting analysis ritual...\n")

        # THE HOURGLASS RITUAL
        run_hourglass(duration=10)

        # PREDICT
        result = predict(survivor, *models)
        results_summary.append({
            "id":       sid,
            "priority": result["priority_code"],
            "priority_label": result["priority_label"],
            "calories": result["calories_kcal"],
            "water_ml": result["water_ml"],
        })

        # PRINT FULL REPORT
        print_report(sid, survivor, result)

        if idx < len(survivors) - 1:
            input(f"  {YELLOW}Press ENTER to process next survivor...{RESET}")

    # ── Final ranked summary ──────────────────────────────────────
    print(f"\n{BOLD}{'═'*52}{RESET}")
    print(f"{BOLD}  FINAL TRIAGE RANKING — ALL SURVIVORS{RESET}")
    print(f"{'═'*52}")

    # Sort: HIGH first (2), then MEDIUM (1), then LOW (0)
    ranked = sorted(results_summary, key=lambda x: x["priority"], reverse=True)

    print(f"  {'Rank':<6}{'Survivor':<14}{'Priority':<10}{'Calories':<12}{'Water':<10}")
    print(f"  {'─'*50}")
    for rank, r in enumerate(ranked, 1):
        color = PRIORITY_COLOR[r["priority"]]
        print(f"  {rank:<6}{r['id']:<14}"
              f"{color}{r['priority_label']:<10}{RESET}"
              f"{r['calories']:<12}{r['water_ml']} ml")

    print(f"\n  {GREEN}✓ Triage complete. Reports ready for field medics.{RESET}\n")


# ══════════════════════════════════════════════════════════════════
# SECTION 6 — KERALA FLOOD DEMO SURVIVORS
# ══════════════════════════════════════════════════════════════════
#
# Based on the 2018 Kerala floods — the worst in 100 years.
# 483 people died, 1 million displaced, 14 of 23 districts affected.
# Medical teams faced: trauma injuries, waterborne disease, elderly
# without chronic medications, and chemical/industrial contamination
# from flooded factories in Ernakulam/Thrissur districts.
#

KERALA_SURVIVORS = [
    {
        "_id":                      "KL-SURV-001",  # Elderly man, Ernakulam
        "age":                      68,
        "weight_kg":                58.0,
        "height_cm":                165.0,
        "heart_rate_bpm":           112,
        "blood_pressure_sys_mmhg":  160,
        "spo2_percent":             89.0,
        "temperature_c":            38.4,
        "radiation_mSv":            0.0,
        "injury_severity":          4,
        "days_without_food":        3,
        "dehydration_level":        2,
        "has_diabetes":             1,
        "has_hypertension":         1,
        "has_respiratory":          0,
    },
    {
        "_id":                      "KL-SURV-002",  # Child, Alappuzha
        "age":                      9,
        "weight_kg":                24.0,
        "height_cm":                128.0,
        "heart_rate_bpm":           130,
        "blood_pressure_sys_mmhg":  88,
        "spo2_percent":             94.5,
        "temperature_c":            39.1,
        "radiation_mSv":            0.0,
        "injury_severity":          2,
        "days_without_food":        2,
        "dehydration_level":        3,
        "has_diabetes":             0,
        "has_hypertension":         0,
        "has_respiratory":          1,
    },
    {
        "_id":                      "KL-SURV-003",  # Worker, chemical plant
        "age":                      34,
        "weight_kg":                72.0,
        "height_cm":                174.0,
        "heart_rate_bpm":           98,
        "blood_pressure_sys_mmhg":  122,
        "spo2_percent":             96.0,
        "temperature_c":            37.2,
        "radiation_mSv":            340.0,    # industrial contamination
        "injury_severity":          6,
        "days_without_food":        1,
        "dehydration_level":        1,
        "has_diabetes":             0,
        "has_hypertension":         0,
        "has_respiratory":          1,
    },
]


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    run_triage_pipeline(KERALA_SURVIVORS)
