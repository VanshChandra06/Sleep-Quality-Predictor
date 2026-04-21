# 🌙 Sleep Quality Predictor

An end-to-end machine learning project that predicts sleep quality (1–10 score) based on user lifestyle habits, trained on a Kaggle-sourced sleep dataset.

---

## 📦 Project Structure

```
sleep_predictor/
├── sleep_dataset.csv          # 1,500-row dataset (Kaggle Sleep Efficiency structure)
├── generate_dataset.py        # Dataset generation script
├── train_pipeline.py          # Full ML pipeline (preprocessing → training)
├── app.py                     # Flask web application
├── requirements.txt
├── templates/
│   └── index.html             # 3-tab web UI
├── plots/                     # 12 EDA & model evaluation plots
│   ├── 01_sleep_quality_distribution.png
│   ├── 02_correlation_heatmap.png
│   ├── 03_quality_vs_caffeine.png
│   ├── 04_quality_vs_exercise.png
│   ├── 05_quality_vs_screentime.png
│   ├── 06_quality_vs_duration.png
│   ├── 07_bedtime_vs_quality.png
│   ├── 08_pairplot.png
│   ├── 09_feature_importance.png
│   ├── 10_nn_training_curves.png
│   ├── 11_model_comparison.png
│   └── 12_predicted_vs_actual.png
└── model_artifacts/
    ├── scaler.pkl
    ├── imputer.pkl
    ├── features.pkl
    ├── gender_encoder.pkl
    ├── best_classical_model.pkl   # Gradient Boosting
    ├── neural_network.keras       # Best model (R²=0.872)
    └── inference_model_type.pkl
```

---

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. (Optional) Regenerate dataset & retrain
python generate_dataset.py
python train_pipeline.py

# 3. Run web app
python app.py
# → Visit http://localhost:5050
```

---

## 📊 Dataset

Based on the **Kaggle Sleep Efficiency Dataset**  
`kaggle.com/datasets/equilibriumm/sleep-efficiency`

- **1,500 rows** with ~5% intentional missing values in key columns
- **18 columns** covering sleep schedule, substances, lifestyle, and physiological metrics

| Column | Description |
|--------|-------------|
| Age, Gender, Occupation | Demographics |
| Bedtime_Hour, Sleep_Duration_Hours | Sleep schedule |
| Sleep_Efficiency_%, REM/Deep/Light Sleep_% | Physiological |
| Caffeine_mg | mg of caffeine consumed near bedtime |
| Alcohol_Units | Units of alcohol |
| Smoking | Binary (0/1) |
| Exercise_Days_Per_Week | Days/week of exercise |
| Screen_Time_Before_Bed_Min | Screen minutes before bed |
| Awakenings | Number of nightly awakenings |
| **Sleep_Quality_Score** | **Target** (1–10 scale) |

---

## 🔬 ML Pipeline

### 1. Preprocessing
- Missing values filled with column mean (`SimpleImputer`)
- Gender label-encoded
- StandardScaler on all numerical features

### 2. Feature Engineering (6 new features)
| Feature | Description |
|---------|-------------|
| Bedtime_Lateness | Distance from optimal bedtime window |
| Caffeine_Risk | Ordinal risk bucket (0–4) |
| Screen_Risk | Ordinal risk bucket (0–4) |
| Sleep_Duration_Score | -1 (too short) to +2 (optimal) |
| Lifestyle_Score | Composite health index |
| Gender_Enc | Label-encoded gender |

### 3. Models Trained

| Model | R² | RMSE | MAE |
|-------|----|------|-----|
| Ridge Regression | 0.851 | 0.801 | 0.631 |
| SVR (RBF) | 0.854 | 0.791 | 0.624 |
| Extra Trees | 0.841 | 0.825 | 0.644 |
| Gradient Boosting | 0.858 | 0.780 | 0.602 |
| **Neural Network** | **0.872** | **0.742** | **0.574** |

### 4. Neural Network Architecture
```
Input(18) → Dense(128, ReLU) → BatchNorm → Dropout(0.3)
          → Dense(64, ReLU)  → BatchNorm → Dropout(0.2)
          → Dense(32, ReLU)  → Dense(1)
```
- Loss: Huber | Optimizer: Adam (lr=0.001)
- EarlyStopping (patience=20) + ReduceLROnPlateau

---

## 🌐 Web Application

Three-tab Flask UI:

1. **🔮 Predictor** — Sliders for all input variables → live sleep quality score with animated gauge, factor bars, and personalised recommendations
2. **📊 EDA Visualizations** — All 12 plots from the pipeline rendered inline
3. **🧠 ML Pipeline** — Summary of dataset, preprocessing steps, model metrics, and architecture

---

## 📈 EDA Plots Generated

1. Sleep Quality Score distribution
2. Feature correlation heatmap
3. Sleep Quality vs Caffeine (scatter)
4. Sleep Quality vs Exercise (boxplot)
5. Sleep Quality vs Screen Time (regression)
6. Sleep Quality vs Duration by Exercise (scatter)
7. Bedtime distribution by Quality bucket (histogram)
8. Full pairplot of key features
9. Feature importances (Gradient Boosting)
10. Neural Network training/validation curves
11. Model comparison bar charts (R², RMSE, MAE)
12. Predicted vs Actual scatter (best classical + NN)
