"""
=============================================================
 Camp Triage & Ration Optimizer
 Phase 1 — Synthetic Survivor Data Generator
 Reverse-engineered from survivors.csv distributions
=============================================================

Columns produced (19 total):
  Identifiers  : survivor_id, name
  Demographics : age, weight_kg, height_cm
  Vital signs  : heart_rate_bpm, blood_pressure_sys_mmhg,
                 spo2_percent, temperature_c
  Disaster     : radiation_mSv, injury_severity,
                 days_without_food, dehydration_level
  Conditions   : has_diabetes, has_hypertension, has_respiratory
  TARGETS      : target_calories_kcal, medical_priority, water_ration_ml

Run:
    pip install pandas numpy
    python src/generate_data.py
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
SEED        = 42
N_SURVIVORS = 500
OUTPUT_DIR  = Path("data")
OUTPUT_DIR.mkdir(exist_ok=True)

np.random.seed(SEED)


# ─────────────────────────────────────────────────────────────
# NAME POOL
# Equal A-Z distribution (~19-20 per letter) so the
# alphabetical bias injected in Phase 2 spans all 26 letters.
# ─────────────────────────────────────────────────────────────
FIRST_NAMES_BY_LETTER = {
    "A": ["Aaron", "Alice", "Amir", "Amara", "Axel", "Anya"],
    "B": ["Bruno", "Bella", "Boris", "Beth", "Bao"],
    "C": ["Carlos", "Clara", "Chen", "Cora", "Cole"],
    "D": ["Diego", "Dana", "Demi", "Dario", "Dawn"],
    "E": ["Ethan", "Elena", "Emil", "Esme", "Ezra"],
    "F": ["Felix", "Fatima", "Finn", "Freya", "Faruk"],
    "G": ["Gina", "Gustav", "Gia", "Gael", "Grace"],
    "H": ["Hana", "Hugo", "Hira", "Hamid", "Holly"],
    "I": ["Ivan", "Isla", "Ines", "Idris", "Iris"],
    "J": ["Juno", "Jake", "Jana", "Joel", "Jin"],
    "K": ["Kai", "Kira", "Kofi", "Keiko", "Kurt"],
    "L": ["Lena", "Leon", "Layla", "Lars", "Lila"],
    "M": ["Mia", "Marco", "Maya", "Milo", "Mei"],
    "N": ["Nora", "Noah", "Nadia", "Nico", "Nina"],
    "O": ["Omar", "Olivia", "Otto", "Opal", "Orla"],
    "P": ["Pavel", "Petra", "Pita", "Phoebe", "Pedro"],
    "Q": ["Quinn", "Qara", "Quentin"],
    "R": ["Rosa", "Rex", "Riya", "Rolf", "Rae"],
    "S": ["Sara", "Sam", "Sven", "Sana", "Sofia"],
    "T": ["Tara", "Tom", "Tia", "Thor", "Temi"],
    "U": ["Uma", "Ugo", "Ursa", "Uri"],
    "V": ["Vera", "Vito", "Vanna", "Vijay"],
    "W": ["Wren", "Wade", "Wafa", "Walt"],
    "X": ["Xena", "Xavi", "Xio"],
    "Y": ["Yara", "Yuki", "Yemi", "Yves"],
    "Z": ["Zara", "Zeke", "Zola", "Zain"],
}

LAST_NAMES = [
    "Smith", "Khan", "Chen", "Garcia", "Okafor", "Petrov",
    "Nakamura", "Silva", "Mbeki", "Johansson", "Patel", "Torres",
    "Nguyen", "Mueller", "Diallo", "Romano", "Park", "Costa",
    "Ivanova", "Hassan", "Yamamoto", "Ferreira", "Owusu", "Lindqvist",
]


def generate_names(n):
    """Round-robin across 26 letters then shuffle. Yields ~19-20 per letter."""
    letters = list(FIRST_NAMES_BY_LETTER.keys())
    names = []
    for i in range(n):
        letter = letters[i % len(letters)]
        first  = np.random.choice(FIRST_NAMES_BY_LETTER[letter])
        last   = np.random.choice(LAST_NAMES)
        names.append(f"{first} {last}")
    np.random.shuffle(names)
    return names


# ─────────────────────────────────────────────────────────────
# FEATURE GENERATORS
# Calibrated to match uploaded survivors.csv distributions:
#   radiation : mean=91 mSv, median=31 mSv, max=1194 mSv
#   injury    : p=[.20, .45, .25, .10]
#   priority  : 194 Low / 179 Medium / 127 Critical
#   calories  : mean=2137, std=749, range 800-3817 kcal
#   water     : mean=2749, std=690, range 1000-4642 ml
# ─────────────────────────────────────────────────────────────

def gen_age(n):
    return np.random.randint(3, 85, n)


def gen_weight(n, age):
    base  = np.where(age < 15, 35.0, 68.0)
    noise = np.random.normal(0, 12, n)
    return np.clip(base + noise, 10, 140).round(1)


def gen_height(n, age):
    base  = np.where(age < 12, 130.0, np.where(age < 18, 160.0, 170.0))
    noise = np.random.normal(0, 10, n)
    return np.clip(base + noise, 80, 210).round(1)


def gen_radiation(n):
    """75% safe-zone Exp(30) | 25% fallout-zone Exp(300). Clipped 0-1200 mSv."""
    low  = np.random.exponential(30, n)
    high = np.random.exponential(300, n)
    mix  = np.where(np.random.random(n) < 0.75, low, high)
    return np.clip(mix, 0, 1200).round(1)


def gen_injury_severity(n):
    return np.random.choice([0, 1, 2, 3], n, p=[0.20, 0.45, 0.25, 0.10])


def gen_days_without_food(n):
    return np.random.randint(0, 13, n)


def gen_dehydration_level(n, days_without_food):
    """Ordered categorical 0-3, correlated with days_without_food."""
    base_prob = np.clip(days_without_food / 12, 0, 1)
    levels = []
    for p in base_prob:
        raw   = np.array([
            max(0.05, 0.70 - p * 0.60),
            max(0.05, 0.20 + p * 0.10),
            max(0.05, 0.07 + p * 0.20),
            max(0.05, 0.03 + p * 0.30),
        ])
        probs = raw / raw.sum()
        levels.append(np.random.choice([0, 1, 2, 3], p=probs))
    return np.array(levels, dtype=int)


def gen_heart_rate(n, injury_severity):
    base  = 75 + injury_severity * 15
    noise = np.random.normal(0, 12, n)
    return np.clip(base + noise, 35, 200).round(0).astype(int)


def gen_blood_pressure(n, injury_severity, radiation_mSv):
    base  = 120 - (radiation_mSv * 0.03) + injury_severity * 5
    noise = np.random.normal(0, 15, n)
    return np.clip(base + noise, 60, 200).round(0).astype(int)


def gen_spo2(n, radiation_mSv, injury_severity):
    base  = 97.0 - (radiation_mSv * 0.01) - injury_severity * 1.5
    noise = np.random.normal(0, 2, n)
    return np.clip(base + noise, 70, 100).round(1)


def gen_temperature(n, injury_severity, radiation_mSv):
    base  = 36.8 + injury_severity * 0.4 + (radiation_mSv > 200).astype(float) * 0.8
    noise = np.random.normal(0, 0.6, n)
    return np.clip(base + noise, 34.0, 42.0).round(1)


def gen_pre_existing_conditions(n):
    return {
        "has_diabetes":     np.random.choice([0, 1], n, p=[0.88, 0.12]),
        "has_hypertension": np.random.choice([0, 1], n, p=[0.75, 0.25]),
        "has_respiratory":  np.random.choice([0, 1], n, p=[0.90, 0.10]),
    }


# ─────────────────────────────────────────────────────────────
# LABEL / TARGET COMPUTATION
# ─────────────────────────────────────────────────────────────

def compute_target_calories(weight_kg, age, days_without_food,
                             radiation_mSv, injury_severity, dehydration_level):
    """
    Daily caloric allocation (kcal).
    Simplified Harris-Benedict + disaster multipliers.
    Output calibrated to CSV: mean~2137, range 800-5500.
    """
    bmr               = 10 * weight_kg + 300 - 4 * np.clip(age - 30, 0, 50)
    starvation_bonus  = days_without_food * 180
    radiation_bonus   = radiation_mSv * 0.4
    injury_bonus      = injury_severity * 120
    dehydration_bonus = dehydration_level * 60
    total = bmr + starvation_bonus + radiation_bonus + injury_bonus + dehydration_bonus
    total = np.where(age < 12, total * 0.70, total)
    return np.clip(total, 800, 5500).round(0).astype(int)


def compute_medical_priority(radiation_mSv, injury_severity,
                              spo2, heart_rate, dehydration_level):
    """
    Triage class: 0=Low, 1=Medium, 2=Critical.
    Score >= 4 -> Critical | 2-3 -> Medium | 0-1 -> Low.
    Calibrated to CSV: ~194 Low / 179 Medium / 127 Critical.
    """
    score = (
        (radiation_mSv > 400).astype(int) * 3 +
        (radiation_mSv > 100).astype(int) * 1 +
        (injury_severity == 3).astype(int) * 3 +
        (injury_severity == 2).astype(int) * 1 +
        (spo2 < 90).astype(int) * 3 +
        (spo2 < 94).astype(int) * 1 +
        (heart_rate > 150).astype(int) * 2 +
        (heart_rate < 45).astype(int) * 2 +
        dehydration_level
    )
    return np.where(score >= 4, 2, np.where(score >= 2, 1, 0)).astype(int)


def compute_water_ration(weight_kg, radiation_mSv, dehydration_level, injury_severity):
    """
    Daily water ration (ml).
    Calibrated to CSV: mean~2749, range 1000-6000 ml.
    """
    base              = weight_kg * 35
    radiation_extra   = radiation_mSv * 0.5
    dehydration_extra = dehydration_level * 300
    injury_extra      = injury_severity * 150
    return np.clip(base + radiation_extra + dehydration_extra + injury_extra, 1000, 6000).round(0).astype(int)


# ─────────────────────────────────────────────────────────────
# MAIN ASSEMBLY
# ─────────────────────────────────────────────────────────────

def generate_survivors(n=N_SURVIVORS, verbose=True):
    if verbose:
        print(f"\n{'='*55}")
        print(f"  Camp Triage  |  Phase 1: Data Generation")
        print(f"{'='*55}")
        print(f"  Seed            : {SEED}")
        print(f"  Survivors       : {n}")
        print(f"  Output dir      : {OUTPUT_DIR}/\n")

    survivor_ids = [f"SURV-{str(i+1).zfill(4)}" for i in range(n)]
    names        = generate_names(n)

    age               = gen_age(n)
    radiation_mSv     = gen_radiation(n)
    injury_severity   = gen_injury_severity(n)
    days_without_food = gen_days_without_food(n)
    dehydration_level = gen_dehydration_level(n, days_without_food)
    weight_kg         = gen_weight(n, age)
    height_cm         = gen_height(n, age)
    heart_rate        = gen_heart_rate(n, injury_severity)
    blood_pressure    = gen_blood_pressure(n, injury_severity, radiation_mSv)
    spo2              = gen_spo2(n, radiation_mSv, injury_severity)
    temperature_c     = gen_temperature(n, injury_severity, radiation_mSv)
    conditions        = gen_pre_existing_conditions(n)

    target_calories  = compute_target_calories(
        weight_kg, age, days_without_food,
        radiation_mSv, injury_severity, dehydration_level)
    medical_priority = compute_medical_priority(
        radiation_mSv, injury_severity, spo2, heart_rate, dehydration_level)
    water_ration_ml  = compute_water_ration(
        weight_kg, radiation_mSv, dehydration_level, injury_severity)

    df = pd.DataFrame({
        "survivor_id":              survivor_ids,
        "name":                     names,           # <- POISON COLUMN for Phase 2
        "age":                      age,
        "weight_kg":                weight_kg,
        "height_cm":                height_cm,
        "heart_rate_bpm":           heart_rate,
        "blood_pressure_sys_mmhg":  blood_pressure,
        "spo2_percent":             spo2,
        "temperature_c":            temperature_c,
        "radiation_mSv":            radiation_mSv,
        "injury_severity":          injury_severity,
        "days_without_food":        days_without_food,
        "dehydration_level":        dehydration_level,
        "has_diabetes":             conditions["has_diabetes"],
        "has_hypertension":         conditions["has_hypertension"],
        "has_respiratory":          conditions["has_respiratory"],
        "target_calories_kcal":     target_calories,
        "medical_priority":         medical_priority,
        "water_ration_ml":          water_ration_ml,
    })
    return df


# ─────────────────────────────────────────────────────────────
# VALIDATION
# ─────────────────────────────────────────────────────────────

def validate(df):
    print("  Validation Report")
    print("  " + "-" * 45)
    print(f"  Rows x Columns      : {df.shape[0]} x {df.shape[1]}")
    print(f"  Missing values      : {df.isnull().sum().sum()}")
    print(f"  Unique names        : {df['name'].nunique()}")
    print(f"  Age range           : {df['age'].min()}-{df['age'].max()}")
    print(f"  Radiation range     : {df['radiation_mSv'].min():.1f}-{df['radiation_mSv'].max():.1f} mSv")

    print(f"\n  Medical priority distribution:")
    labels = {0: "Low     ", 1: "Medium  ", 2: "Critical"}
    for k, v in df["medical_priority"].value_counts().sort_index().items():
        bar = "#" * (v // 8)
        print(f"    {labels[k]} ({k}): {bar} {v}")

    print(f"\n  Injury severity distribution:")
    slabels = {0: "None    ", 1: "Minor   ", 2: "Moderate", 3: "Critical"}
    for k, v in df["injury_severity"].value_counts().sort_index().items():
        bar = "#" * (v // 8)
        print(f"    {slabels[k]} ({k}): {bar} {v}")

    print(f"\n  Caloric allocation:")
    print(f"    Min  : {df['target_calories_kcal'].min():,} kcal")
    print(f"    Mean : {df['target_calories_kcal'].mean():,.0f} kcal")
    print(f"    Max  : {df['target_calories_kcal'].max():,} kcal")

    print(f"\n  Water ration:")
    print(f"    Min  : {df['water_ration_ml'].min():,} ml")
    print(f"    Mean : {df['water_ration_ml'].mean():,.0f} ml")
    print(f"    Max  : {df['water_ration_ml'].max():,} ml")

    letter_dist = df["name"].str[0].value_counts().sort_index()
    print(f"\n  Name first-letter coverage : {len(letter_dist)}/26 letters")
    print(f"  (Even spread needed for Phase 2 A-Z bias demo)")
    return df


# ─────────────────────────────────────────────────────────────
# SAVE
# ─────────────────────────────────────────────────────────────

FEATURE_COLS = [
    "survivor_id", "name",
    "age", "weight_kg", "height_cm",
    "heart_rate_bpm", "blood_pressure_sys_mmhg", "spo2_percent", "temperature_c",
    "radiation_mSv", "injury_severity", "days_without_food", "dehydration_level",
    "has_diabetes", "has_hypertension", "has_respiratory",
]
LABEL_COLS = ["survivor_id", "target_calories_kcal", "medical_priority", "water_ration_ml"]


def save(df):
    df.to_csv(OUTPUT_DIR / "survivors.csv", index=False)
    df[FEATURE_COLS].to_csv(OUTPUT_DIR / "survivors_features.csv", index=False)
    df[LABEL_COLS].to_csv(OUTPUT_DIR / "survivors_labels.csv", index=False)

    schema = {
        "generated_by":    "Phase 1 — generate_data.py",
        "seed":            SEED,
        "n_survivors":     len(df),
        "feature_cols":    FEATURE_COLS,
        "target_cols":     ["target_calories_kcal", "medical_priority", "water_ration_ml"],
        "priority_map":    {"0": "Low", "1": "Medium", "2": "Critical"},
        "severity_map":    {"0": "None", "1": "Minor", "2": "Moderate", "3": "Critical"},
        "dehydration_map": {"0": "Hydrated", "1": "Mild", "2": "Moderate", "3": "Severe"},
        "column_notes": {
            "name": (
                "WARNING - Phase 2 POISON COLUMN. "
                "Label-encoding alphabetically creates severe A-name bias. "
                "Phase 5 fix: hash-encode or drop entirely."
            ),
            "radiation_mSv":    "Acute radiation syndrome onset ~1000 mSv",
            "injury_severity":  "0=none 1=minor 2=moderate 3=critical",
            "dehydration_level":"0=hydrated 1=mild 2=moderate 3=severe",
            "medical_priority": "0=Low 1=Medium 2=Critical triage class",
        },
    }
    with open(OUTPUT_DIR / "schema.json", "w") as f:
        json.dump(schema, f, indent=2)

    print(f"\n  Saved files:")
    for fname in ["survivors.csv", "survivors_features.csv",
                  "survivors_labels.csv", "schema.json"]:
        fpath = OUTPUT_DIR / fname
        size  = fpath.stat().st_size
        print(f"    {fname:<30} {size/1024:6.1f} KB")

    print(f"\n{'='*55}")
    print(f"  Phase 1 complete.")
    print(f"  Next -> Phase 2:  python src/bias_encoder.py")
    print(f"{'='*55}\n")


# ─────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    df = generate_survivors(N_SURVIVORS, verbose=True)
    df = validate(df)
    save(df)
