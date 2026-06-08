# ============================================================
#   Car Evaluation System — SVM Classifier
#   Dataset: UCI Car Evaluation Dataset
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, ConfusionMatrixDisplay
)
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# 1. LOAD DATASET
# ─────────────────────────────────────────────
# UCI Car Evaluation Dataset (fetched directly)
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data"

columns = ["buying", "maint", "doors", "persons", "lug_boot", "safety", "class"]

print("=" * 55)
print("       CAR EVALUATION SYSTEM — SVM CLASSIFIER")
print("=" * 55)

try:
    df = pd.read_csv(url, names=columns)
    print("\n✅ Dataset loaded from UCI repository.")
except Exception:
    # Fallback: generate representative synthetic data
    print("\n⚠️  Could not fetch online dataset. Using built-in sample data.")
    from sklearn.datasets import make_classification
    np.random.seed(42)

    buying_vals   = ["vhigh", "high", "med", "low"]
    maint_vals    = ["vhigh", "high", "med", "low"]
    doors_vals    = ["2", "3", "4", "5more"]
    persons_vals  = ["2", "4", "more"]
    lug_vals      = ["small", "med", "big"]
    safety_vals   = ["low", "med", "high"]
    class_vals    = ["unacc", "acc", "good", "vgood"]

    n = 1728
    rng = np.random.default_rng(42)
    df = pd.DataFrame({
        "buying":   rng.choice(buying_vals,  n),
        "maint":    rng.choice(maint_vals,   n),
        "doors":    rng.choice(doors_vals,   n),
        "persons":  rng.choice(persons_vals, n),
        "lug_boot": rng.choice(lug_vals,     n),
        "safety":   rng.choice(safety_vals,  n),
        "class":    rng.choice(class_vals,   n, p=[0.70, 0.22, 0.04, 0.04]),
    })

# ─────────────────────────────────────────────
# 2. EXPLORATORY DATA ANALYSIS
# ─────────────────────────────────────────────
print("\n📊 Dataset Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())
print("\nClass Distribution:")
print(df["class"].value_counts())
print("\nMissing Values:", df.isnull().sum().sum())

# Plot class distribution
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle("Car Evaluation — EDA", fontsize=14, fontweight="bold")

df["class"].value_counts().plot(kind="bar", ax=axes[0], color=["#E74C3C","#3498DB","#2ECC71","#F39C12"],
                                 edgecolor="black", rot=0)
axes[0].set_title("Class Distribution")
axes[0].set_xlabel("Evaluation Class")
axes[0].set_ylabel("Count")

df["safety"].value_counts().plot(kind="bar", ax=axes[1], color=["#9B59B6","#1ABC9C","#E67E22"],
                                  edgecolor="black", rot=0)
axes[1].set_title("Safety Rating Distribution")
axes[1].set_xlabel("Safety")
axes[1].set_ylabel("Count")

plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/eda_plots.png", dpi=150, bbox_inches="tight")
plt.close()
print("\n✅ EDA plot saved.")

# ─────────────────────────────────────────────
# 3. PREPROCESSING
# ─────────────────────────────────────────────
le = LabelEncoder()
df_enc = df.copy()
for col in df_enc.columns:
    df_enc[col] = le.fit_transform(df_enc[col])

X = df_enc.drop("class", axis=1)
y = df_enc["class"]

# Feature scaling (important for SVM)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n🔀 Train size: {X_train.shape[0]} | Test size: {X_test.shape[0]}")

# ─────────────────────────────────────────────
# 4. SVM — BASELINE MODEL
# ─────────────────────────────────────────────
print("\n🤖 Training SVM (baseline — RBF kernel)...")
svm_base = SVC(kernel="rbf", random_state=42)
svm_base.fit(X_train, y_train)
y_pred_base = svm_base.predict(X_test)

print(f"   Baseline Accuracy: {accuracy_score(y_test, y_pred_base):.4f}")

# ─────────────────────────────────────────────
# 5. HYPERPARAMETER TUNING (GridSearchCV)
# ─────────────────────────────────────────────
print("\n🔍 Running GridSearchCV for hyperparameter tuning...")
param_grid = {
    "C":      [0.1, 1, 10, 100],
    "gamma":  ["scale", "auto", 0.1, 0.01],
    "kernel": ["rbf", "poly", "linear"]
}

grid_search = GridSearchCV(
    SVC(random_state=42),
    param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1,
    verbose=0
)
grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
print(f"   Best Params: {best_params}")
print(f"   Best CV Accuracy: {grid_search.best_score_:.4f}")

# ─────────────────────────────────────────────
# 6. FINAL MODEL
# ─────────────────────────────────────────────
best_svm = grid_search.best_estimator_
y_pred = best_svm.predict(X_test)

final_acc = accuracy_score(y_test, y_pred)
print(f"\n✅ Final Model Test Accuracy: {final_acc:.4f}")

# Cross-validation
cv_scores = cross_val_score(best_svm, X_scaled, y, cv=10, scoring="accuracy")
print(f"   10-Fold CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# ─────────────────────────────────────────────
# 7. EVALUATION METRICS
# ─────────────────────────────────────────────
target_names = ["acc", "good", "unacc", "vgood"]
print("\n📋 Classification Report:")
print(classification_report(y_test, y_pred, target_names=target_names))

# Confusion Matrix Plot
fig, ax = plt.subplots(figsize=(7, 5))
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
disp.plot(ax=ax, cmap="Blues", colorbar=False)
ax.set_title("SVM — Confusion Matrix (Tuned Model)", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.close()
print("✅ Confusion matrix saved.")

# ─────────────────────────────────────────────
# 8. KERNEL COMPARISON PLOT
# ─────────────────────────────────────────────
kernels = ["linear", "poly", "rbf", "sigmoid"]
kernel_acc = []
for k in kernels:
    m = SVC(kernel=k, C=best_params.get("C", 10), random_state=42)
    m.fit(X_train, y_train)
    kernel_acc.append(accuracy_score(y_test, m.predict(X_test)))

fig, ax = plt.subplots(figsize=(7, 4))
bars = ax.bar(kernels, kernel_acc, color=["#3498DB","#E74C3C","#2ECC71","#F39C12"], edgecolor="black", width=0.5)
ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=10)
ax.set_ylim(0.5, 1.05)
ax.set_title("SVM Kernel Comparison — Test Accuracy", fontsize=13, fontweight="bold")
ax.set_xlabel("Kernel")
ax.set_ylabel("Accuracy")
plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/kernel_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("✅ Kernel comparison plot saved.")

# ─────────────────────────────────────────────
# 9. PREDICTION FUNCTION
# ─────────────────────────────────────────────
feature_order   = ["buying", "maint", "doors", "persons", "lug_boot", "safety"]
category_maps   = {
    "buying":   {"low": 0, "med": 1, "high": 2, "vhigh": 3},
    "maint":    {"low": 0, "med": 1, "high": 2, "vhigh": 3},
    "doors":    {"2": 0, "3": 1, "4": 2, "5more": 3},
    "persons":  {"2": 0, "4": 1, "more": 2},
    "lug_boot": {"small": 0, "med": 1, "big": 2},
    "safety":   {"low": 0, "med": 1, "high": 2},
}
class_map_inv   = {0: "acc", 1: "good", 2: "unacc", 3: "vgood"}

def predict_car(buying, maint, doors, persons, lug_boot, safety):
    row = [
        category_maps["buying"][buying],
        category_maps["maint"][maint],
        category_maps["doors"][doors],
        category_maps["persons"][persons],
        category_maps["lug_boot"][lug_boot],
        category_maps["safety"][safety],
    ]
    row_scaled = scaler.transform([row])
    pred = best_svm.predict(row_scaled)[0]
    return class_map_inv[pred]

# Example prediction
print("\n🚗 Sample Prediction:")
sample = dict(buying="low", maint="low", doors="4", persons="more", lug_boot="big", safety="high")
result = predict_car(**sample)
print(f"   Input: {sample}")
print(f"   Predicted Class: {result.upper()}")

print("\n" + "=" * 55)
print("  Project complete! All outputs saved to outputs/")
print("=" * 55)
