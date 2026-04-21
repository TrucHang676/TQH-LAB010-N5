"""
═══════════════════════════════════════════════════════════════════════════
  train_model1.py — Huấn luyện & đánh giá Model 1 (Regression)
───────────────────────────────────────────────────────────────────────────
  Mục tiêu: Dự đoán log1p(sold_count) từ đặc trưng sản phẩm mỹ phẩm Tiki
  Chạy: cd dashboard && python train_model1.py
  Output (trong dashboard/ml_models/):
    - model1_regressor.joblib            (pipeline cuối)
    - model1_metrics.json                (metrics tổng hợp)
    - model1_predictions.csv             (test set + predicted)
    - model1_feature_importance.csv      (permutation importance)
    - model1_learning_curve.csv          (learning curve data)
    - model1_partial_dependence.csv      (partial dependence price)
    - model1_feature_meta.json           (metadata cho What-if form)
═══════════════════════════════════════════════════════════════════════════
"""
import json
import os
import warnings
import numpy as np
import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import partial_dependence, permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import (
    KFold,
    cross_val_score,
    learning_curve,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────
# Script nằm ở dashboard/ → data ở ../data/ → artifacts ở ./ml_models/
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.abspath(os.path.join(BASE_DIR, "..", "data",
                                         "tiki_cosmetics_processed.csv"))
OUT_DIR = os.path.join(BASE_DIR, "ml_models")
os.makedirs(OUT_DIR, exist_ok=True)

RANDOM_STATE = 42

# ═══════════════════════════════════════════════════════════════════
#  1. LOAD & CHUẨN BỊ DỮ LIỆU
# ═══════════════════════════════════════════════════════════════════
print("\n▶ BƯỚC 1 · Load dữ liệu")
df = pd.read_csv(DATA_PATH)
print(f"  → {len(df):,} sản phẩm × {df.shape[1]} cột")

# Loại outlier cực đoan
df = df[df["is_extreme_outlier"] == False].reset_index(drop=True)
print(f"  → sau khi loại outlier cực đoan: {len(df):,} sản phẩm")

# Feature engineering
df["log_price"] = np.log1p(df["price"])
df["has_rating"] = (df["rating"] > 0).astype(int)

# Gộp brand thưa thành "Other"
TOP_BRAND_N = 20
top_brands = df["brand_name"].value_counts().head(TOP_BRAND_N).index
df["brand_grouped"] = df["brand_name"].where(df["brand_name"].isin(top_brands), "Other")

# Target: log1p(sold_count)
df["target"] = np.log1p(df["sold_count"])

# ── Chọn features ─────────────────────────────────────────────────
# LƯU Ý: Không dùng review_count / review_ratio vì có leakage
# (review chỉ có sau khi bán, không phản ánh khả năng dự đoán sản phẩm mới)

NUMERIC_FEATURES = [
    "log_price",
    "discount_rate",
    "rating",
    "has_rating",
    "is_official_store",
    "tiki_verified",
    "has_authentic_badge",
]
CATEGORICAL_FEATURES = [
    "origin_class_corrected",
    "product_type",
    "price_segment",
    "brand_grouped",
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Ép kiểu số cho boolean columns
for col in ["is_official_store", "tiki_verified", "has_authentic_badge"]:
    df[col] = df[col].astype(int)

X = df[FEATURES].copy()
y = df["target"].copy()

print(f"  → features: {len(FEATURES)} ({len(NUMERIC_FEATURES)} num + {len(CATEGORICAL_FEATURES)} cat)")

# ═══════════════════════════════════════════════════════════════════
#  2. TRAIN/TEST SPLIT — stratify theo popularity_tier để giữ phân phối
# ═══════════════════════════════════════════════════════════════════
print("\n▶ BƯỚC 2 · Split 70/15/15 (train/val/test)")

strat = df["popularity_tier"]
X_trainval, X_test, y_trainval, y_test, strat_trainval, _ = train_test_split(
    X, y, strat, test_size=0.15, stratify=strat, random_state=RANDOM_STATE
)
X_train, X_val, y_train, y_val = train_test_split(
    X_trainval,
    y_trainval,
    test_size=0.176,  # 15/(100-15) = 0.1764
    stratify=strat_trainval,
    random_state=RANDOM_STATE,
)
print(f"  → train {len(X_train):,} · val {len(X_val):,} · test {len(X_test):,}")

# ═══════════════════════════════════════════════════════════════════
#  3. PIPELINE TIỀN XỬ LÝ
# ═══════════════════════════════════════════════════════════════════
# Compat với sklearn cũ/mới (tham số đổi từ sparse → sparse_output ở 1.2)
try:
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
except TypeError:
    ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), NUMERIC_FEATURES),
        ("cat", ohe, CATEGORICAL_FEATURES),
    ],
    remainder="drop",
)

# ═══════════════════════════════════════════════════════════════════
#  4. ĐỊNH NGHĨA MODELS (có baseline)
# ═══════════════════════════════════════════════════════════════════
print("\n▶ BƯỚC 3 · Train & đánh giá 4 models")

models = {
    "Dummy (mean)": Pipeline(
        [("prep", preprocessor), ("reg", DummyRegressor(strategy="mean"))]
    ),
    "Ridge": Pipeline(
        [("prep", preprocessor), ("reg", Ridge(alpha=10.0, random_state=RANDOM_STATE))]
    ),
    "RandomForest": Pipeline(
        [
            ("prep", preprocessor),
            (
                "reg",
                RandomForestRegressor(
                    n_estimators=200,
                    max_depth=10,
                    min_samples_leaf=3,
                    n_jobs=-1,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    ),
    "GradientBoosting": Pipeline(
        [
            ("prep", preprocessor),
            (
                "reg",
                GradientBoostingRegressor(
                    n_estimators=250,
                    learning_rate=0.05,
                    max_depth=4,
                    min_samples_leaf=5,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    ),
}

# ═══════════════════════════════════════════════════════════════════
#  5. HÀM METRIC
# ═══════════════════════════════════════════════════════════════════
def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def eval_model(pipe, X_tr, y_tr, X_ev, y_ev):
    """Trả dict metric trên log-scale và scale gốc."""
    pipe.fit(X_tr, y_tr)
    y_pred_log = pipe.predict(X_ev)

    # Đảm bảo predict log hợp lệ (không âm sau inverse vì sold_count >= 0)
    y_pred_orig = np.expm1(y_pred_log).clip(min=0)
    y_true_orig = np.expm1(y_ev)

    return {
        "r2_log": float(r2_score(y_ev, y_pred_log)),
        "rmse_log": rmse(y_ev, y_pred_log),
        "mae_log": float(mean_absolute_error(y_ev, y_pred_log)),
        "rmse_orig": rmse(y_true_orig, y_pred_orig),
        "mae_orig": float(mean_absolute_error(y_true_orig, y_pred_orig)),
        "spearman": float(pd.Series(y_pred_log).corr(pd.Series(y_ev.values), method="spearman")),
    }


# ═══════════════════════════════════════════════════════════════════
#  6. TRAIN + ĐÁNH GIÁ TRÊN VAL SET
# ═══════════════════════════════════════════════════════════════════
val_results = {}
for name, pipe in models.items():
    m = eval_model(pipe, X_train, y_train, X_val, y_val)
    val_results[name] = m
    print(
        f"  {name:20s} | R²={m['r2_log']:+.3f} · RMSE_log={m['rmse_log']:.3f} · "
        f"MAE_log={m['mae_log']:.3f} · Spearman={m['spearman']:+.3f}"
    )

# Chọn model tốt nhất theo R² log
best_name = max(val_results, key=lambda n: val_results[n]["r2_log"])
print(f"\n  ★ Best model (theo R² log trên val): {best_name}")

# ═══════════════════════════════════════════════════════════════════
#  7. 5-FOLD CROSS-VALIDATION
# ═══════════════════════════════════════════════════════════════════
print("\n▶ BƯỚC 4 · 5-Fold Cross-Validation trên train+val")

X_trval = pd.concat([X_train, X_val], axis=0)
y_trval = pd.concat([y_train, y_val], axis=0)

cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
cv_results = {}
for name, pipe in models.items():
    scores = cross_val_score(pipe, X_trval, y_trval, cv=cv, scoring="r2", n_jobs=-1)
    cv_results[name] = {
        "mean": float(scores.mean()),
        "std": float(scores.std()),
        "folds": [float(s) for s in scores],
    }
    print(f"  {name:20s} | R² = {scores.mean():+.3f} ± {scores.std():.3f}")

# ═══════════════════════════════════════════════════════════════════
#  8. ROBUSTNESS — TRAIN LẠI VỚI 3 RANDOM SEED
# ═══════════════════════════════════════════════════════════════════
print("\n▶ BƯỚC 5 · Robustness (3 random seeds) cho best model")

SEEDS = [42, 7, 2024]
robust_r2 = []
for seed in SEEDS:
    X_tr_s, X_te_s, y_tr_s, y_te_s = train_test_split(
        X, y, test_size=0.15, stratify=strat, random_state=seed
    )
    pipe_s = models[best_name]
    pipe_s.fit(X_tr_s, y_tr_s)
    pred = pipe_s.predict(X_te_s)
    r2_s = r2_score(y_te_s, pred)
    robust_r2.append(float(r2_s))
    print(f"  seed={seed:4d} → R² = {r2_s:+.3f}")

print(f"  → mean ± std = {np.mean(robust_r2):+.3f} ± {np.std(robust_r2):.3f}")

# ═══════════════════════════════════════════════════════════════════
#  9. ĐÁNH GIÁ CUỐI TRÊN TEST SET
# ═══════════════════════════════════════════════════════════════════
print("\n▶ BƯỚC 6 · Đánh giá cuối trên TEST set (hold-out)")

final_pipe = models[best_name]
final_pipe.fit(X_trval, y_trval)

test_metrics = eval_model(final_pipe, X_trval, y_trval, X_test, y_test)
print(
    f"  R² log = {test_metrics['r2_log']:+.3f}   RMSE log = {test_metrics['rmse_log']:.3f}"
)
print(
    f"  RMSE gốc (sold_count) = {test_metrics['rmse_orig']:.1f}   MAE gốc = {test_metrics['mae_orig']:.1f}"
)

dummy_r2 = val_results["Dummy (mean)"]["r2_log"]
print(f"  Baseline (Dummy mean) R² log = {dummy_r2:+.3f}")

# ═══════════════════════════════════════════════════════════════════
#  10. PERMUTATION IMPORTANCE
# ═══════════════════════════════════════════════════════════════════
print("\n▶ BƯỚC 7 · Permutation Importance (n_repeats=10)")

perm = permutation_importance(
    final_pipe,
    X_test,
    y_test,
    n_repeats=10,
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

fi_df = pd.DataFrame(
    {
        "feature": FEATURES,
        "importance_mean": perm.importances_mean,
        "importance_std": perm.importances_std,
    }
).sort_values("importance_mean", ascending=False)

print(fi_df.to_string(index=False))

# ═══════════════════════════════════════════════════════════════════
#  11. LEARNING CURVE
# ═══════════════════════════════════════════════════════════════════
print("\n▶ BƯỚC 8 · Learning curve")

train_sizes, train_scores, val_scores = learning_curve(
    final_pipe,
    X_trval,
    y_trval,
    cv=5,
    scoring="r2",
    train_sizes=np.linspace(0.1, 1.0, 8),
    n_jobs=-1,
    random_state=RANDOM_STATE,
)
lc_df = pd.DataFrame(
    {
        "train_size": train_sizes,
        "train_r2_mean": train_scores.mean(axis=1),
        "train_r2_std": train_scores.std(axis=1),
        "val_r2_mean": val_scores.mean(axis=1),
        "val_r2_std": val_scores.std(axis=1),
    }
)
print(lc_df.round(3).to_string(index=False))

# ═══════════════════════════════════════════════════════════════════
#  12. PARTIAL DEPENDENCE — log_price
# ═══════════════════════════════════════════════════════════════════
print("\n▶ BƯỚC 9 · Partial dependence của log_price")
pd_result = partial_dependence(final_pipe, X_trval, features=["log_price"], kind="average")
pd_df = pd.DataFrame(
    {
        "log_price": pd_result["grid_values"][0],
        "price_vnd": np.expm1(pd_result["grid_values"][0]),
        "predicted_log_sold": pd_result["average"][0],
        "predicted_sold": np.expm1(pd_result["average"][0]),
    }
)
print(pd_df.head(10).to_string(index=False))

# ═══════════════════════════════════════════════════════════════════
#  13. RESIDUAL TRÊN TEST SET
# ═══════════════════════════════════════════════════════════════════
y_pred_test_log = final_pipe.predict(X_test)
y_pred_test_orig = np.expm1(y_pred_test_log).clip(min=0)
y_test_orig = np.expm1(y_test.values)
residuals_log = y_test.values - y_pred_test_log

test_predictions_df = pd.DataFrame(
    {
        "actual_sold": y_test_orig,
        "predicted_sold": y_pred_test_orig,
        "actual_log": y_test.values,
        "predicted_log": y_pred_test_log,
        "residual_log": residuals_log,
    }
)

# ═══════════════════════════════════════════════════════════════════
#  14. LƯU KẾT QUẢ
# ═══════════════════════════════════════════════════════════════════
print(f"\n▶ BƯỚC 10 · Lưu kết quả vào {OUT_DIR}")

joblib.dump(final_pipe, os.path.join(OUT_DIR, "model1_regressor.joblib"))
print("  ✓ model1_regressor.joblib")

feature_meta = {
    "numeric_features": NUMERIC_FEATURES,
    "categorical_features": CATEGORICAL_FEATURES,
    "all_features": FEATURES,
    "top_brands": list(top_brands),
    "product_types": sorted(df["product_type"].dropna().unique().tolist()),
    "price_segments": sorted(df["price_segment"].dropna().unique().tolist()),
    "origin_classes": sorted(df["origin_class_corrected"].dropna().unique().tolist()),
}
with open(os.path.join(OUT_DIR, "model1_feature_meta.json"), "w", encoding="utf-8") as f:
    json.dump(feature_meta, f, ensure_ascii=False, indent=2)
print("  ✓ model1_feature_meta.json")

all_metrics = {
    "best_model": best_name,
    "random_state": RANDOM_STATE,
    "n_train": int(len(X_train)),
    "n_val": int(len(X_val)),
    "n_test": int(len(X_test)),
    "val_results": val_results,
    "cv_results": cv_results,
    "robustness": {
        "seeds": SEEDS,
        "r2_each": robust_r2,
        "r2_mean": float(np.mean(robust_r2)),
        "r2_std": float(np.std(robust_r2)),
    },
    "test_metrics": test_metrics,
    "baseline_r2_log": dummy_r2,
}
with open(os.path.join(OUT_DIR, "model1_metrics.json"), "w", encoding="utf-8") as f:
    json.dump(all_metrics, f, ensure_ascii=False, indent=2)
print("  ✓ model1_metrics.json")

test_predictions_df.to_csv(os.path.join(OUT_DIR, "model1_predictions.csv"), index=False)
fi_df.to_csv(os.path.join(OUT_DIR, "model1_feature_importance.csv"), index=False)
lc_df.to_csv(os.path.join(OUT_DIR, "model1_learning_curve.csv"), index=False)
pd_df.to_csv(os.path.join(OUT_DIR, "model1_partial_dependence.csv"), index=False)
print("  ✓ model1_predictions.csv · model1_feature_importance.csv")
print("  ✓ model1_learning_curve.csv · model1_partial_dependence.csv")

print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
print(f"  Xong! Model = {best_name}")
print(
    f"  Test R² (log) = {test_metrics['r2_log']:+.3f}    "
    f"Test RMSE gốc = {test_metrics['rmse_orig']:.1f}"
)
print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
