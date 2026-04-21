"""
train_pipeline.py — Full ML pipeline for Sleep Quality Predictor
Covers: preprocessing, feature engineering, EDA visualizations,
        traditional ML models, and a neural network.
"""

import os, warnings, joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor,
                               ExtraTreesRegressor)
from sklearn.linear_model import Ridge, Lasso
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline

import tensorflow as tf
from tensorflow import keras
# Access layers and callbacks from the already-imported keras to avoid
# static analysis issues with `tensorflow.keras` submodule resolution.
layers = keras.layers
callbacks = keras.callbacks

warnings.filterwarnings("ignore")
np.random.seed(42)
tf.random.set_seed(42)

PLOTS = "static/images"
ARTIFACTS = "model_artifacts"
os.makedirs(PLOTS, exist_ok=True)
os.makedirs(ARTIFACTS, exist_ok=True)

# ─────────────────────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────────────────────
print("=" * 60)
print("  SLEEP QUALITY PREDICTOR — ML PIPELINE")
print("=" * 60)

df = pd.read_csv("sleep_dataset.csv")
print(f"\n[1] Dataset loaded: {df.shape[0]} rows × {df.shape[1]} columns")
print(f"    Missing values per column:\n{df.isnull().sum()[df.isnull().sum()>0]}")

# ─────────────────────────────────────────────────────────────
# 2. PREPROCESSING & CLEANING
# ─────────────────────────────────────────────────────────────
print("\n[2] Preprocessing & Cleaning...")

num_cols = ["Caffeine_mg", "Alcohol_Units", "Exercise_Days_Per_Week",
            "Screen_Time_Before_Bed_Min", "Awakenings"]
for col in num_cols:
    mean_val = df[col].mean()
    df[col].fillna(mean_val, inplace=True)
    print(f"    Filled NaN in '{col}' with mean = {mean_val:.2f}")

# ─────────────────────────────────────────────────────────────
# 3. FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────
print("\n[3] Feature Engineering...")

# Bedtime lateness penalty (later = worse)
df["Bedtime_Lateness"] = df["Bedtime_Hour"].apply(
    lambda h: h if h <= 6 else (24 - h)
)

# Caffeine risk (high caffeine near bed = bad)
df["Caffeine_Risk"] = pd.cut(
    df["Caffeine_mg"], bins=[-1, 0, 50, 100, 200, 999],
    labels=[0, 1, 2, 3, 4]
).cat.add_categories([-1]).fillna(-1).astype(int).clip(0)

# Screen time risk
df["Screen_Risk"] = pd.cut(
    df["Screen_Time_Before_Bed_Min"], bins=[-1, 0, 30, 60, 90, 9999],
    labels=[0, 1, 2, 3, 4]
).cat.add_categories([-1]).fillna(-1).astype(int).clip(0)

# Sleep duration optimality
def sleep_dur_score(h):
    if 7 <= h <= 9: return 2
    if 6 <= h < 7 or 9 < h <= 10: return 1
    if h < 6: return -1
    return 0

df["Sleep_Duration_Score"] = df["Sleep_Duration_Hours"].apply(sleep_dur_score)

# Composite lifestyle index
df["Lifestyle_Score"] = (
    df["Exercise_Days_Per_Week"] * 0.4
    - df["Caffeine_mg"] / 100
    - df["Alcohol_Units"] * 0.3
    - df["Smoking"] * 0.5
    - df["Screen_Time_Before_Bed_Min"] / 60
)

# Encode gender
le = LabelEncoder()
df["Gender_Enc"] = le.fit_transform(df["Gender"])
joblib.dump(le, f"{ARTIFACTS}/gender_encoder.pkl")

print(f"    Engineered features: Bedtime_Lateness, Caffeine_Risk, Screen_Risk,")
print(f"    Sleep_Duration_Score, Lifestyle_Score, Gender_Enc")

# ─────────────────────────────────────────────────────────────
# 4. EDA VISUALIZATIONS
# ─────────────────────────────────────────────────────────────
print("\n[4] Generating EDA Visualizations...")

sns.set_theme(style="darkgrid", palette="muted")
FIGSIZE = (10, 6)

# 4a. Distribution of Sleep Quality
fig, ax = plt.subplots(figsize=FIGSIZE)
sns.histplot(df["Sleep_Quality_Score"], bins=30, kde=True, color="#4C72B0", ax=ax)
ax.set_title("Distribution of Sleep Quality Score", fontsize=14, fontweight="bold")
ax.set_xlabel("Sleep Quality Score (1–10)")
ax.set_ylabel("Count")
plt.tight_layout()
plt.savefig(f"{PLOTS}/01_sleep_quality_distribution.png", dpi=150)
plt.close()
print("    Saved: 01_sleep_quality_distribution.png")

# 4b. Correlation heatmap
fig, ax = plt.subplots(figsize=(12, 9))
num_df = df[[
    "Age", "Sleep_Duration_Hours", "Sleep_Efficiency_%",
    "Caffeine_mg", "Alcohol_Units", "Smoking",
    "Exercise_Days_Per_Week", "Screen_Time_Before_Bed_Min",
    "Awakenings", "Lifestyle_Score", "Sleep_Quality_Score"
]]
corr = num_df.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdYlGn",
            center=0, square=True, linewidths=0.5, ax=ax,
            annot_kws={"size": 8})
ax.set_title("Feature Correlation Heatmap", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{PLOTS}/02_correlation_heatmap.png", dpi=150)
plt.close()
print("    Saved: 02_correlation_heatmap.png")

# 4c. Sleep Quality vs Caffeine Intake (scatter)
fig, ax = plt.subplots(figsize=FIGSIZE)
scatter = ax.scatter(df["Caffeine_mg"], df["Sleep_Quality_Score"],
                     c=df["Sleep_Quality_Score"], cmap="RdYlGn",
                     alpha=0.5, edgecolors="none", s=25)
plt.colorbar(scatter, label="Sleep Quality")
ax.set_xlabel("Caffeine Intake Before Bed (mg)")
ax.set_ylabel("Sleep Quality Score")
ax.set_title("Sleep Quality vs Caffeine Intake", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{PLOTS}/03_quality_vs_caffeine.png", dpi=150)
plt.close()
print("    Saved: 03_quality_vs_caffeine.png")

# 4d. Sleep Quality vs Exercise (boxplot)
fig, ax = plt.subplots(figsize=FIGSIZE)
sns.boxplot(x="Exercise_Days_Per_Week", y="Sleep_Quality_Score",
            data=df, palette="Blues", ax=ax)
ax.set_title("Sleep Quality by Exercise Frequency", fontsize=14, fontweight="bold")
ax.set_xlabel("Exercise Days Per Week")
ax.set_ylabel("Sleep Quality Score")
plt.tight_layout()
plt.savefig(f"{PLOTS}/04_quality_vs_exercise.png", dpi=150)
plt.close()
print("    Saved: 04_quality_vs_exercise.png")

# 4e. Sleep Quality vs Screen Time (scatter + trend)
fig, ax = plt.subplots(figsize=FIGSIZE)
sns.regplot(x="Screen_Time_Before_Bed_Min", y="Sleep_Quality_Score",
            data=df, scatter_kws={"alpha": 0.3, "s": 20},
            line_kws={"color": "red", "lw": 2}, ax=ax)
ax.set_title("Sleep Quality vs Screen Time Before Bed", fontsize=14, fontweight="bold")
ax.set_xlabel("Screen Time Before Bed (minutes)")
ax.set_ylabel("Sleep Quality Score")
plt.tight_layout()
plt.savefig(f"{PLOTS}/05_quality_vs_screentime.png", dpi=150)
plt.close()
print("    Saved: 05_quality_vs_screentime.png")

# 4f. Sleep Duration vs Sleep Quality (scatter)
fig, ax = plt.subplots(figsize=FIGSIZE)
sns.scatterplot(x="Sleep_Duration_Hours", y="Sleep_Quality_Score",
                hue="Exercise_Days_Per_Week", palette="viridis",
                data=df, alpha=0.5, ax=ax)
ax.axvline(7, color="green", linestyle="--", alpha=0.7, label="Optimal start (7h)")
ax.axvline(9, color="green", linestyle="--", alpha=0.7, label="Optimal end (9h)")
ax.set_title("Sleep Quality vs Duration (colored by Exercise)", fontsize=14, fontweight="bold")
ax.set_xlabel("Sleep Duration (hours)")
ax.set_ylabel("Sleep Quality Score")
plt.tight_layout()
plt.savefig(f"{PLOTS}/06_quality_vs_duration.png", dpi=150)
plt.close()
print("    Saved: 06_quality_vs_duration.png")

# 4g. Bedtime distribution by Sleep Quality bucket
df["Quality_Bucket"] = pd.cut(df["Sleep_Quality_Score"],
                               bins=[0, 4, 7, 10],
                               labels=["Poor (<4)", "Average (4–7)", "Good (>7)"])
fig, ax = plt.subplots(figsize=FIGSIZE)
for bucket, color in zip(["Poor (<4)", "Average (4–7)", "Good (>7)"],
                          ["#e74c3c", "#f39c12", "#27ae60"]):
    subset = df[df["Quality_Bucket"] == bucket]["Bedtime_Hour"]
    ax.hist(subset, bins=7, alpha=0.6, label=bucket, color=color,
            range=(20, 27), align="left")
ax.set_xticks(range(21, 28))
ax.set_xticklabels(["9PM", "10PM", "11PM", "12AM", "1AM", "2AM", "3AM"])
ax.set_title("Bedtime Hour vs Sleep Quality Groups", fontsize=14, fontweight="bold")
ax.set_xlabel("Bedtime")
ax.set_ylabel("Count")
ax.legend()
plt.tight_layout()
plt.savefig(f"{PLOTS}/07_bedtime_vs_quality.png", dpi=150)
plt.close()
print("    Saved: 07_bedtime_vs_quality.png")

# 4h. Pairplot (subset)
pair_df = df[["Sleep_Duration_Hours", "Caffeine_mg",
              "Screen_Time_Before_Bed_Min", "Exercise_Days_Per_Week",
              "Sleep_Quality_Score"]].sample(400, random_state=42)
g = sns.pairplot(pair_df, diag_kind="kde", plot_kws={"alpha": 0.4, "s": 15},
                 vars=["Sleep_Duration_Hours", "Caffeine_mg",
                       "Screen_Time_Before_Bed_Min", "Exercise_Days_Per_Week",
                       "Sleep_Quality_Score"])
g.fig.suptitle("Pairplot: Key Features vs Sleep Quality", y=1.02, fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{PLOTS}/08_pairplot.png", dpi=120)
plt.close()
print("    Saved: 08_pairplot.png")

# ─────────────────────────────────────────────────────────────
# 5. FEATURE SELECTION & TRAIN/TEST SPLIT
# ─────────────────────────────────────────────────────────────
print("\n[5] Preparing features for model training...")

FEATURES = [
    "Age", "Gender_Enc", "Bedtime_Hour", "Sleep_Duration_Hours",
    "Sleep_Efficiency_%", "Awakenings", "Caffeine_mg", "Alcohol_Units",
    "Smoking", "Exercise_Days_Per_Week", "Screen_Time_Before_Bed_Min",
    "Bedtime_Lateness", "Caffeine_Risk", "Screen_Risk",
    "Sleep_Duration_Score", "Lifestyle_Score",
    "REM_Sleep_%", "Deep_Sleep_%",
]
TARGET = "Sleep_Quality_Score"

X = df[FEATURES].values
y = df[TARGET].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"    Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")

imputer = SimpleImputer(strategy="mean")
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)
joblib.dump(imputer, f"{ARTIFACTS}/imputer.pkl")

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)
joblib.dump(scaler, f"{ARTIFACTS}/scaler.pkl")
joblib.dump(FEATURES, f"{ARTIFACTS}/features.pkl")

# ─────────────────────────────────────────────────────────────
# 6. TRAIN ML MODELS
# ─────────────────────────────────────────────────────────────
print("\n[6] Training Classical ML Models...")

results = {}

models = {
    "Ridge Regression":        Ridge(alpha=1.0),
    "Random Forest":           RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1),
    "Gradient Boosting":       GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=4, random_state=42),
    "Extra Trees":             ExtraTreesRegressor(n_estimators=200, random_state=42, n_jobs=-1),
    "SVR":                     SVR(C=5, kernel="rbf"),
}

best_r2 = -999
best_model_name = None
best_model = None

for name, model in models.items():
    model.fit(X_train_sc, y_train)
    preds = model.predict(X_test_sc)
    r2   = r2_score(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae  = mean_absolute_error(y_test, preds)
    results[name] = {"R²": r2, "RMSE": rmse, "MAE": mae}
    print(f"    {name:<25} R²={r2:.4f}  RMSE={rmse:.4f}  MAE={mae:.4f}")
    if r2 > best_r2:
        best_r2 = r2
        best_model_name = name
        best_model = model

print(f"\n    Best classical model: {best_model_name} (R²={best_r2:.4f})")
joblib.dump(best_model, f"{ARTIFACTS}/best_classical_model.pkl")

# Feature importance plot (RF/GB/ET)
fi_model = models["Gradient Boosting"]
fi = fi_model.feature_importances_
idx = np.argsort(fi)[::-1]
fig, ax = plt.subplots(figsize=(10, 7))
colors = sns.color_palette("viridis", len(FEATURES))
ax.barh([FEATURES[i] for i in idx[::-1]], fi[idx[::-1]], color=colors)
ax.set_title("Feature Importances — Gradient Boosting", fontsize=14, fontweight="bold")
ax.set_xlabel("Importance Score")
plt.tight_layout()
plt.savefig(f"{PLOTS}/09_feature_importance.png", dpi=150)
plt.close()
print("    Saved: 09_feature_importance.png")

# ─────────────────────────────────────────────────────────────
# 7. NEURAL NETWORK
# ─────────────────────────────────────────────────────────────
print("\n[7] Training Neural Network...")

def build_nn(input_dim):
    inp = keras.Input(shape=(input_dim,))
    x = layers.Dense(128, activation="relu")(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(32, activation="relu")(x)
    out = layers.Dense(1, activation="linear")(x)
    model = keras.Model(inp, out)
    model.compile(optimizer=keras.optimizers.Adam(0.001),
                  loss="huber", metrics=["mae"])
    return model

nn = build_nn(X_train_sc.shape[1])
cb = [
    callbacks.EarlyStopping(patience=20, restore_best_weights=True, verbose=0),
    callbacks.ReduceLROnPlateau(factor=0.5, patience=8, verbose=0)
]
history = nn.fit(
    X_train_sc, y_train,
    validation_split=0.15,
    epochs=200, batch_size=64,
    callbacks=cb, verbose=0
)

nn_preds = nn.predict(X_test_sc, verbose=0).flatten()
nn_r2   = r2_score(y_test, nn_preds)
nn_rmse = np.sqrt(mean_squared_error(y_test, nn_preds))
nn_mae  = mean_absolute_error(y_test, nn_preds)
results["Neural Network"] = {"R²": nn_r2, "RMSE": nn_rmse, "MAE": nn_mae}
print(f"    Neural Network             R²={nn_r2:.4f}  RMSE={nn_rmse:.4f}  MAE={nn_mae:.4f}")

nn.save(f"{ARTIFACTS}/neural_network.keras")
print(f"    Neural network saved.")

# NN training curve
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].plot(history.history["loss"], label="Train Loss")
axes[0].plot(history.history["val_loss"], label="Val Loss")
axes[0].set_title("Neural Network — Loss Curve")
axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Huber Loss")
axes[0].legend()
axes[1].plot(history.history["mae"], label="Train MAE")
axes[1].plot(history.history["val_mae"], label="Val MAE")
axes[1].set_title("Neural Network — MAE Curve")
axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("MAE")
axes[1].legend()
plt.tight_layout()
plt.savefig(f"{PLOTS}/10_nn_training_curves.png", dpi=150)
plt.close()
print("    Saved: 10_nn_training_curves.png")

# ─────────────────────────────────────────────────────────────
# 8. MODEL COMPARISON PLOT
# ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 6))
metrics = ["R²", "RMSE", "MAE"]
colors_bar = sns.color_palette("Set2", len(results))

for ax, metric in zip(axes, metrics):
    vals = [results[m][metric] for m in results]
    bars = ax.bar(results.keys(), vals, color=colors_bar)
    ax.set_title(f"Model Comparison — {metric}", fontweight="bold")
    ax.set_xticklabels(results.keys(), rotation=30, ha="right", fontsize=8)
    ax.set_ylabel(metric)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{v:.3f}", ha="center", va="bottom", fontsize=7)

plt.tight_layout()
plt.savefig(f"{PLOTS}/11_model_comparison.png", dpi=150)
plt.close()
print("    Saved: 11_model_comparison.png")

# Predicted vs Actual scatter (best classical)
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, (name, preds) in zip(axes, [
    (best_model_name, best_model.predict(X_test_sc)),
    ("Neural Network", nn_preds)
]):
    ax.scatter(y_test, preds, alpha=0.4, s=20, c=preds, cmap="RdYlGn")
    ax.plot([1, 10], [1, 10], "r--", lw=2)
    ax.set_title(f"Predicted vs Actual — {name}", fontweight="bold")
    ax.set_xlabel("Actual Sleep Quality"); ax.set_ylabel("Predicted")
    r2_val = r2_score(y_test, preds)
    ax.text(0.05, 0.93, f"R²={r2_val:.3f}", transform=ax.transAxes,
            fontsize=10, color="blue")
plt.tight_layout()
plt.savefig(f"{PLOTS}/12_predicted_vs_actual.png", dpi=150)
plt.close()
print("    Saved: 12_predicted_vs_actual.png")

# Decide which model to use for inference (better R²)
use_nn = nn_r2 > best_r2
inference_model_type = "nn" if use_nn else "classical"
joblib.dump(inference_model_type, f"{ARTIFACTS}/inference_model_type.pkl")

print("\n" + "=" * 60)
print("  PIPELINE COMPLETE")
print(f"  Using '{('Neural Network' if use_nn else best_model_name)}' for inference")
print("=" * 60)
