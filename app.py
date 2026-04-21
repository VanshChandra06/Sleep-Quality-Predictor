"""
app.py — Flask web application for Sleep Quality Predictor
"""

import os, json
import numpy as np
import joblib
from flask import Flask, render_template, request, jsonify

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from tensorflow import keras

app = Flask(__name__)

# ── Load artefacts ──────────────────────────────────────────
scaler   = joblib.load(f"model_artifacts/scaler.pkl")
imputer  = joblib.load(f"model_artifacts/imputer.pkl")
features = joblib.load(f"model_artifacts/features.pkl")
inf_type = joblib.load(f"model_artifacts/inference_model_type.pkl")

if inf_type == "nn":
    model = keras.models.load_model(f"model_artifacts/neural_network.keras")
    model_name = "Neural Network"
else:
    model = joblib.load(f"model_artifacts/best_classical_model.pkl")
    model_name = "Gradient Boosting"

print(f"[app] Using {model_name} for inference.")

# ── Helper ──────────────────────────────────────────────────
def build_feature_vector(data: dict) -> np.ndarray:
    """
    Convert user inputs to the full feature vector used during training.
    """
    sleep_hours   = float(data["sleep_hours"])
    bedtime_hour  = int(data["bedtime_hour"])
    caffeine      = float(data["caffeine_mg"])
    exercise      = int(data["exercise_days"])
    screen_time   = float(data["screen_time_min"])
    gender_enc    = 1 if data.get("gender", "Female") == "Male" else 0
    age           = float(data.get("age", 30))
    alcohol       = float(data.get("alcohol_units", 0))
    smoking       = int(data.get("smoking", 0))
    awakenings    = float(data.get("awakenings", 1))

    # Derived
    sleep_efficiency = max(50, min(99, 85 - caffeine/10 - screen_time/10 - awakenings*3 + exercise*1.5))
    rem_pct   = 22.0
    deep_pct  = 18.0

    bedtime_lateness = bedtime_hour if bedtime_hour <= 6 else (24 - bedtime_hour)

    caff_bins  = [-1, 0, 50, 100, 200, 999]
    caff_risk  = int(np.digitize(caffeine, caff_bins[1:])) - 1
    caff_risk  = max(0, min(4, caff_risk))

    screen_bins = [-1, 0, 30, 60, 90, 9999]
    screen_risk = int(np.digitize(screen_time, screen_bins[1:])) - 1
    screen_risk = max(0, min(4, screen_risk))

    if 7 <= sleep_hours <= 9:   dur_score = 2
    elif 6 <= sleep_hours < 7 or 9 < sleep_hours <= 10: dur_score = 1
    elif sleep_hours < 6:       dur_score = -1
    else:                       dur_score = 0

    lifestyle = (exercise * 0.4 - caffeine/100
                 - alcohol * 0.3 - smoking * 0.5 - screen_time/60)

    wakeup_hour = (bedtime_hour + int(sleep_hours)) % 24

    feat_map = {
        "Age":                          age,
        "Gender_Enc":                   gender_enc,
        "Bedtime_Hour":                 bedtime_hour,
        "Sleep_Duration_Hours":         sleep_hours,
        "Sleep_Efficiency_%":           sleep_efficiency,
        "Awakenings":                   awakenings,
        "Caffeine_mg":                  caffeine,
        "Alcohol_Units":                alcohol,
        "Smoking":                      smoking,
        "Exercise_Days_Per_Week":       exercise,
        "Screen_Time_Before_Bed_Min":   screen_time,
        "Bedtime_Lateness":             bedtime_lateness,
        "Caffeine_Risk":                caff_risk,
        "Screen_Risk":                  screen_risk,
        "Sleep_Duration_Score":         dur_score,
        "Lifestyle_Score":              lifestyle,
        "REM_Sleep_%":                  rem_pct,
        "Deep_Sleep_%":                 deep_pct,
    }

    return np.array([[feat_map[f] for f in features]])


def generate_recommendations(data: dict, score: float) -> list:
    tips = []

    caffeine = float(data["caffeine_mg"])
    screen   = float(data["screen_time_min"])
    exercise = int(data["exercise_days"])
    bedtime  = int(data["bedtime_hour"])
    sleep_h  = float(data["sleep_hours"])

    if caffeine > 100:
        tips.append({"icon": "☕", "category": "Caffeine",
                     "tip": "You're consuming over 100 mg of caffeine close to bedtime. Try to cut off caffeine at least 6 hours before sleep."})
    elif caffeine > 50:
        tips.append({"icon": "☕", "category": "Caffeine",
                     "tip": "Moderate caffeine intake detected. Switching to herbal tea in the evening could improve sleep onset."})

    if screen > 60:
        tips.append({"icon": "📱", "category": "Screen Time",
                     "tip": f"Your {screen:.0f}-minute pre-bed screen session is suppressing melatonin. Try a 30-minute digital wind-down instead."})
    elif screen > 30:
        tips.append({"icon": "📱", "category": "Screen Time",
                     "tip": "Consider enabling Night Mode on your devices and reducing screen time before sleep."})

    if exercise == 0:
        tips.append({"icon": "🏃", "category": "Exercise",
                     "tip": "No exercise detected. Even a 20-minute walk daily can significantly improve deep sleep."})
    elif exercise <= 2:
        tips.append({"icon": "🏃", "category": "Exercise",
                     "tip": "Aim for at least 3–4 days of moderate activity per week for better sleep quality."})

    if bedtime in [0, 1, 2, 3]:
        tips.append({"icon": "🌙", "category": "Bedtime",
                     "tip": "Very late bedtime detected. Shifting sleep earlier by 1–2 hours can align your circadian rhythm and boost recovery."})

    if sleep_h < 6:
        tips.append({"icon": "😴", "category": "Sleep Duration",
                     "tip": "Less than 6 hours is insufficient for most adults. Aim for 7–9 hours to improve both mental and physical recovery."})
    elif sleep_h > 10:
        tips.append({"icon": "😴", "category": "Sleep Duration",
                     "tip": "Sleeping more than 10 hours can paradoxically reduce sleep quality. A consistent 7–9 hour schedule is optimal."})

    if score >= 7.5:
        tips.insert(0, {"icon": "🌟", "category": "Great Job",
                        "tip": "Your sleep habits are excellent! Maintain this routine for sustained health benefits."})
    elif score >= 5:
        if not tips:
            tips.append({"icon": "💡", "category": "General",
                         "tip": "Your sleep is decent. Small reductions in caffeine or screen time before bed could push your score higher."})

    return tips[:5]  # max 5 tips


@app.route("/")
def index():
    return render_template("index.html", model_name=model_name)


@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    try:
        X = build_feature_vector(data)
        X_imp = imputer.transform(X)
        X_sc  = scaler.transform(X_imp)

        if inf_type == "nn":
            raw_score = float(model.predict(X_sc, verbose=0).flatten()[0])
        else:
            raw_score = float(model.predict(X_sc)[0])

        score = round(max(1.0, min(10.0, raw_score)), 2)

        if score >= 8:     label, color = "Excellent",  "#27ae60"
        elif score >= 6.5: label, color = "Good",        "#2ecc71"
        elif score >= 5:   label, color = "Fair",        "#f39c12"
        elif score >= 3.5: label, color = "Poor",        "#e67e22"
        else:              label, color = "Very Poor",   "#e74c3c"

        recommendations = generate_recommendations(data, score)

        return jsonify({
            "score": score,
            "label": label,
            "color": color,
            "model": model_name,
            "recommendations": recommendations
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/plots")
def plots():
    plot_files = sorted([
        f for f in os.listdir("static/images")
        if f.endswith(".png")
    ])
    return jsonify(plot_files)


if __name__ == "__main__":
    app.run(debug=False, port=5050)


@app.route("/plot/<filename>")
def serve_plot(filename):
    from flask import send_from_directory
    return send_from_directory("static/images", filename)
