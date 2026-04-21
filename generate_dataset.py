"""
Dataset Generator — models the structure of the Kaggle
'Sleep Efficiency Dataset' (kaggle.com/datasets/equilibriumm/sleep-efficiency)
which contains: Age, Gender, Bedtime, Wakeup time, Sleep duration,
Sleep efficiency, REM sleep %, Deep sleep %, Light sleep %,
Awakenings, Caffeine consumption, Alcohol consumption,
Smoking status, Exercise frequency, and a derived Sleep Quality score.

We also extend it with Screen time before bed (minutes) to align with
the user-facing predictor requirements.
"""

import numpy as np
import pandas as pd
import random

np.random.seed(42)
random.seed(42)

N = 1500

ages = np.random.randint(18, 70, N)
genders = np.random.choice(["Male", "Female"], N)
occupations = np.random.choice(
    ["Student", "Engineer", "Teacher", "Doctor", "Manager",
     "Nurse", "Artist", "Lawyer", "Retired", "Other"], N
)

# Bedtime hour (0–3 = midnight-3am, 21–23 = 9–11pm)
bedtime_hour = np.random.choice(
    [21, 22, 23, 0, 1, 2, 3],
    N,
    p=[0.10, 0.18, 0.22, 0.20, 0.12, 0.10, 0.08]
)
bedtime_minute = np.random.randint(0, 60, N)

sleep_duration = np.clip(
    np.random.normal(7.0, 1.2, N), 3.5, 10.5
)

wakeup_hour = (bedtime_hour + sleep_duration.astype(int)) % 24

caffeine = np.random.choice([0, 25, 50, 75, 100, 150, 200], N,
                             p=[0.25, 0.20, 0.20, 0.15, 0.10, 0.06, 0.04])
alcohol = np.random.choice([0, 1, 2, 3, 4, 5], N,
                            p=[0.40, 0.25, 0.15, 0.10, 0.06, 0.04])
smoking = np.random.choice([0, 1], N, p=[0.75, 0.25])  # 0=No, 1=Yes
exercise = np.random.choice([0, 1, 2, 3, 4, 5], N,
                             p=[0.15, 0.20, 0.25, 0.20, 0.12, 0.08])
screen_time = np.random.choice([0, 15, 30, 45, 60, 90, 120], N,
                                p=[0.10, 0.12, 0.18, 0.20, 0.20, 0.12, 0.08])
awakenings = np.random.choice([0, 1, 2, 3, 4], N,
                               p=[0.30, 0.30, 0.20, 0.12, 0.08])

# Derived sleep quality score (1–10 scale) based on research-backed relationships
def compute_sleep_quality(row):
    idx, age, gender, bedtime, sleep_dur, caff, alc, smoke, ex, screen, awakening = row

    score = 5.0  # baseline

    # Sleep duration effect (7–9 hrs optimal)
    if 7 <= sleep_dur <= 9:
        score += 2.0
    elif 6 <= sleep_dur < 7 or 9 < sleep_dur <= 10:
        score += 0.5
    elif sleep_dur < 6:
        score -= 2.0
    elif sleep_dur > 10:
        score -= 1.0

    # Bedtime effect (late = worse)
    if bedtime in [22, 23]:
        score += 0.5
    elif bedtime in [0, 1]:
        score -= 0.5
    elif bedtime in [2, 3]:
        score -= 1.5

    # Caffeine effect (mg consumed within 6h of bed)
    if caff == 0:
        score += 0.5
    elif caff <= 50:
        score -= 0.0
    elif caff <= 100:
        score -= 0.8
    elif caff <= 150:
        score -= 1.5
    else:
        score -= 2.5

    # Alcohol
    score -= alc * 0.3

    # Smoking
    if smoke:
        score -= 0.8

    # Exercise (days/week)
    if ex >= 4:
        score += 1.2
    elif ex >= 2:
        score += 0.5
    elif ex == 0:
        score -= 0.5

    # Screen time before bed
    if screen == 0:
        score += 0.8
    elif screen <= 30:
        score += 0.2
    elif screen <= 60:
        score -= 0.5
    elif screen <= 90:
        score -= 1.0
    else:
        score -= 1.8

    # Awakenings
    score -= awakening * 0.5

    # Age effect
    if age > 55:
        score -= 0.3

    # Add noise
    score += np.random.normal(0, 0.6)

    return round(float(np.clip(score, 1.0, 10.0)), 1)

rows = list(zip(
    range(N), ages, genders, bedtime_hour, sleep_duration,
    caffeine, alcohol, smoking, exercise, screen_time, awakenings
))

sleep_quality = [compute_sleep_quality(r) for r in rows]

# Sleep efficiency (% of time in bed actually sleeping)
sleep_efficiency = np.clip(
    np.array([
        85 - (caffeine[i] / 10) - (screen_time[i] / 10) - (awakenings[i] * 3)
        + (exercise[i] * 1.5) + np.random.normal(0, 3)
        for i in range(N)
    ]), 50, 99
)

rem_pct = np.clip(np.random.normal(22, 4, N), 10, 35)
deep_pct = np.clip(np.random.normal(18, 4, N), 5, 35)
light_pct = 100 - rem_pct - deep_pct

df = pd.DataFrame({
    "ID": range(1, N + 1),
    "Age": ages,
    "Gender": genders,
    "Occupation": occupations,
    "Bedtime_Hour": bedtime_hour,
    "Sleep_Duration_Hours": sleep_duration.round(2),
    "Wakeup_Hour": wakeup_hour,
    "Sleep_Efficiency_%": sleep_efficiency.round(1),
    "REM_Sleep_%": rem_pct.round(1),
    "Deep_Sleep_%": deep_pct.round(1),
    "Light_Sleep_%": light_pct.round(1),
    "Awakenings": awakenings,
    "Caffeine_mg": caffeine,
    "Alcohol_Units": alcohol,
    "Smoking": smoking,
    "Exercise_Days_Per_Week": exercise,
    "Screen_Time_Before_Bed_Min": screen_time,
    "Sleep_Quality_Score": sleep_quality,
})

# Introduce ~5% missing values in non-critical columns to simulate real data
for col in ["Caffeine_mg", "Alcohol_Units", "Exercise_Days_Per_Week", "Screen_Time_Before_Bed_Min", "Awakenings"]:
    mask = np.random.rand(N) < 0.05
    df.loc[mask, col] = np.nan

df.to_csv("sleep_dataset.csv", index=False)
print(f"Dataset saved: {len(df)} rows x {len(df.columns)} columns")
print(df.describe())
