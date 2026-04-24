"""
═══════════════════════════════════════════════════════════════════════════
  train_model2.py — Huấn luyện & đánh giá Model 2 (Clustering)
───────────────────────────────────────────────────────────────────────────
  Mục tiêu: Phân khúc thị trường mỹ phẩm Tiki thành K cụm tự nhiên
            → Trả lời: "Nội/ngoại phân bố thế nào giữa các phân khúc?"

  Chiến lược đánh giá 7 lớp:
    1. Internal metrics (Inertia + Silhouette + Davies-Bouldin)
    2. Cluster stability (ARI across 10 random seeds)
    3. ARI vs product_type (có tái phát hiện phân loại có sẵn không?)
    4. PCA 2D visualization
    5. Profile từng cluster (mean features, % VN/NK, top brand)
    6. Business naming + strategy suggestion
    7. Cross với Model 1 (R² regression trong từng cluster)

  Features (6): log_price · rating · review_ratio · discount_rate
               · log_sold_count · tiki_verified
  KHÔNG đưa origin_class vào features — để kiểm tra nó có tự phân biệt
  giữa các cluster hay không (câu hỏi nghiên cứu cốt lõi).

  Audience: người quản lý ngành hàng của sàn + analyst brand.

  Chạy: cd dashboard && python train_model2.py
═══════════════════════════════════════════════════════════════════════════
"""
import os
import json
import warnings
import numpy as np
import pandas as pd
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import (silhouette_score, davies_bouldin_score,
                             adjusted_rand_score, r2_score)
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")
RNG = 42

# ─────────────────────────────────────────────────────────────────────────
# PATH CONFIG
# ─────────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.abspath(os.path.join(BASE_DIR, "..", "data",
                                         "tiki_cosmetics_processed.csv"))
OUT_DIR = os.path.join(BASE_DIR, "ml_models")
os.makedirs(OUT_DIR, exist_ok=True)


# ═════════════════════════════════════════════════════════════════════════
# BƯỚC 1 · Load & Feature Engineering
# ═════════════════════════════════════════════════════════════════════════
print("▶ BƯỚC 1 · Load dữ liệu & Feature Engineering")
df = pd.read_csv(DATA_PATH)
print(f"  Dataset gốc: {df.shape}")

# Loại outlier cực đoan
df = df[~df["is_extreme_outlier"]].copy().reset_index(drop=True)
print(f"  Sau khi loại is_extreme_outlier: {df.shape}")

# Trim bổ sung: loại các sản phẩm ở đuôi 99.5% của price và sold_count
# (K-Means rất nhạy với outlier — 1 sản phẩm ngoại lệ có thể tạo cluster 1-phần-tử)
p995_price = df["price"].quantile(0.995)
p995_sold = df["sold_count"].quantile(0.995)
before = len(df)
df = df[(df["price"] <= p995_price) & (df["sold_count"] <= p995_sold)].copy().reset_index(drop=True)
print(f"  Sau khi trim đuôi 99.5% (price≤{p995_price:,.0f} · sold≤{p995_sold:,.0f}): "
      f"{df.shape}  (bỏ {before - len(df)} dòng)")

# Feature engineering
df["log_price"] = np.log1p(df["price"])
df["log_sold_count"] = np.log1p(df["sold_count"])

# Chuẩn hoá discount_rate về [0, 1] (hiện đang 0-73%)
df["discount_rate_norm"] = df["discount_rate"] / 100.0

# ── Lưu bản FULL (chưa filter) để compute correlation RAW + stats ──
df_full = df.copy()
n_total = len(df_full)
print(f"  [FULL] {n_total:,} sản phẩm (trước filter)")

# ── HƯỚNG A: Lọc về active + rated products ──
# Lý do: rating=0 là confounding signal (chưa có review, không phải rating thấp).
# Chỉ cluster sản phẩm có data đầy đủ để tránh artifact.
mask_active_rated = (df["rating"] > 0) & (df["sold_count"] > 0)
df_excluded = df[~mask_active_rated].copy()
df = df[mask_active_rated].copy().reset_index(drop=True)
print(f"  [FILTERED] Active+Rated: {len(df):,} sản phẩm ({len(df)/n_total*100:.1f}%)")
print(f"  [EXCLUDED] Loại: {len(df_excluded):,} sản phẩm ({len(df_excluded)/n_total*100:.1f}%)")

# Features cho clustering (bỏ review_ratio vì corr cao với rating)
FEATURE_COLS = [
    "log_price",
    "rating",
    "discount_rate_norm",
    "log_sold_count",
    "tiki_verified",
]

# All 6 features cho correlation RAW (bao gồm review_ratio)
RAW_FEATURE_COLS = [
    "log_price", "rating", "review_ratio",
    "discount_rate_norm", "log_sold_count", "tiki_verified",
]

X = df[FEATURE_COLS].values
print(f"  Features ({len(FEATURE_COLS)}): {FEATURE_COLS}")

# ═════════════════════════════════════════════════════════════════════════
# BƯỚC 2 · Scale features + Correlation Analysis
# ═════════════════════════════════════════════════════════════════════════
print("\n▶ BƯỚC 2 · StandardScaler + Correlation Analysis")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print(f"  Mean sau scale ≈ 0: {X_scaled.mean(axis=0).round(3)}")
print(f"  Std sau scale  ≈ 1: {X_scaled.std(axis=0).round(3)}")

# ── Correlation matrix RAW (trên df_full, 6 features gồm review_ratio) ──
raw_feats = {col: df_full[col] for col in RAW_FEATURE_COLS}
corr_raw = pd.DataFrame(raw_feats).corr(method="pearson")
corr_raw.to_csv(os.path.join(OUT_DIR, "model2_corr_raw.csv"))

# ── Correlation matrix FILTERED (trên df đã lọc, 5 features) ──
filt_feats = {col: df[col] for col in FEATURE_COLS}
corr_filtered = pd.DataFrame(filt_feats).corr(method="pearson")
corr_filtered.to_csv(os.path.join(OUT_DIR, "model2_corr_filtered.csv"))

def top_pairs(corr_df, n=3):
    feats = corr_df.columns.tolist()
    pairs = []
    for i in range(len(feats)):
        for j in range(i + 1, len(feats)):
            pairs.append({"f1": feats[i], "f2": feats[j],
                          "r": float(corr_df.iloc[i, j]),
                          "abs_r": abs(float(corr_df.iloc[i, j]))})
    pairs.sort(key=lambda x: x["abs_r"], reverse=True)
    return pairs[:n]

with open(os.path.join(OUT_DIR, "model2_corr_insights.json"), "w", encoding="utf-8") as f:
    json.dump({
        "raw": {"top_pairs": top_pairs(corr_raw, 3),
                "max_abs_corr": float(top_pairs(corr_raw, 1)[0]["abs_r"])},
        "filtered": {"top_pairs": top_pairs(corr_filtered, 3),
                     "max_abs_corr": float(top_pairs(corr_filtered, 1)[0]["abs_r"])},
    }, f, indent=2, ensure_ascii=False)
print(f"  Correlation RAW max|r| = {top_pairs(corr_raw,1)[0]['abs_r']:.3f}")
print(f"  Correlation FILTERED max|r| = {top_pairs(corr_filtered,1)[0]['abs_r']:.3f}")

# ── Feature quality stats (cho Section 1) ──
feature_quality = {
    "n_total": int(n_total),
    "rating": {
        "pct_zero": round((df_full["rating"] == 0).sum() / n_total * 100, 1),
        "pct_rated": round((df_full["rating"] > 0).sum() / n_total * 100, 1),
        "mean_rated_only": round(float(df_full[df_full["rating"] > 0]["rating"].mean()), 2),
    },
    "sold_count": {
        "pct_zero": round((df_full["sold_count"] == 0).sum() / n_total * 100, 1),
        "pct_active": round((df_full["sold_count"] > 0).sum() / n_total * 100, 1),
        "median_active_only": int(df_full[df_full["sold_count"] > 0]["sold_count"].median()),
    },
    "tiki_verified": {
        "pct_true": round(df_full["tiki_verified"].mean() * 100, 1),
    },
    "filtering_decision": {
        "n_clustered": len(df),
        "pct_clustered": round(len(df) / n_total * 100, 1),
        "n_excluded": len(df_excluded),
        "pct_excluded": round(len(df_excluded) / n_total * 100, 1),
    },
}
with open(os.path.join(OUT_DIR, "model2_feature_quality.json"), "w", encoding="utf-8") as f:
    json.dump(feature_quality, f, indent=2, ensure_ascii=False)
print(f"  Feature quality: rating=0 → {feature_quality['rating']['pct_zero']}%")

# ── Stats cho catalog dormant (Section 8) ──
exc = df_excluded.copy()
n_exc = len(exc)

exc["exclude_reason"] = "unrated_only"  # rating=0, sold>0
exc.loc[(exc["rating"] == 0) & (exc["sold_count"] == 0), "exclude_reason"] = "both_zero"
exc.loc[(exc["rating"] > 0) & (exc["sold_count"] == 0), "exclude_reason"] = "dead_only"

reason_counts = exc["exclude_reason"].value_counts().to_dict()
top_brands_exc = exc["brand_name"].value_counts().head(10).to_dict()
top_types_exc = exc["product_type"].value_counts().to_dict()

dormant_stats = {
    "n_excluded": int(n_exc),
    "pct_of_total": round(n_exc / n_total * 100, 1),
    "reasons": {
        "both_zero": {
            "count": int(reason_counts.get("both_zero", 0)),
            "pct": round(reason_counts.get("both_zero", 0) / max(n_exc, 1) * 100, 1),
            "desc": "Vừa chưa rated vừa chưa bán"},
        "unrated_only": {
            "count": int(reason_counts.get("unrated_only", 0)),
            "pct": round(reason_counts.get("unrated_only", 0) / max(n_exc, 1) * 100, 1),
            "desc": "Chưa có rating nhưng đã có giao dịch"},
        "dead_only": {
            "count": int(reason_counts.get("dead_only", 0)),
            "pct": round(reason_counts.get("dead_only", 0) / max(n_exc, 1) * 100, 1),
            "desc": "Có rating nhưng chưa có giao dịch"},
    },
    "top_brands": [{"name": str(k), "count": int(v)} for k, v in top_brands_exc.items()],
    "top_product_types": [{"name": str(k), "count": int(v)} for k, v in top_types_exc.items()],
    "pct_verified": round(exc["tiki_verified"].mean() * 100, 1),
}
with open(os.path.join(OUT_DIR, "model2_dormant_stats.json"), "w", encoding="utf-8") as f:
    json.dump(dormant_stats, f, indent=2, ensure_ascii=False)
print(f"  Dormant stats: {n_exc:,} excluded ({dormant_stats['pct_of_total']}%)")


# ═════════════════════════════════════════════════════════════════════════
# BƯỚC 3 · Chọn K tối ưu bằng Elbow + Silhouette + Davies-Bouldin
# ═════════════════════════════════════════════════════════════════════════
print("\n▶ BƯỚC 3 · Scan K từ 2..10 (Elbow + Silhouette + DB)")

K_RANGE = list(range(2, 11))
k_metrics = []

# Dùng sample 3000 rows cho silhouette để tránh quá chậm
SIL_SAMPLE = 3000
rng = np.random.RandomState(RNG)
sil_idx = rng.choice(len(X_scaled), size=min(SIL_SAMPLE, len(X_scaled)), replace=False)

for k in K_RANGE:
    km = KMeans(n_clusters=k, n_init=10, random_state=RNG)
    labels = km.fit_predict(X_scaled)
    inertia = km.inertia_
    sil = silhouette_score(X_scaled[sil_idx], labels[sil_idx])
    db = davies_bouldin_score(X_scaled, labels)
    k_metrics.append({
        "k": k,
        "inertia": float(inertia),
        "silhouette": float(sil),
        "davies_bouldin": float(db),
    })
    print(f"  K={k}: inertia={inertia:,.0f}  silhouette={sil:.3f}  DB={db:.3f}")

km_df = pd.DataFrame(k_metrics)
km_df.to_csv(os.path.join(OUT_DIR, "model2_elbow_silhouette.csv"), index=False)

# Chọn K: composite score = silhouette − 0.15 × DB
# Ưu tiên silhouette CAO + Davies-Bouldin THẤP. Range 3..6 để cluster dễ diễn giải.
candidate_k = km_df[(km_df["k"] >= 3) & (km_df["k"] <= 6)].copy()
candidate_k["score"] = candidate_k["silhouette"] - 0.15 * candidate_k["davies_bouldin"]
BEST_K = int(candidate_k.loc[candidate_k["score"].idxmax(), "k"])
print(f"\n  ⇒ Chọn K* = {BEST_K} "
      f"(composite score = silhouette − 0.15·DB, cao nhất trong 3-6)")
print(f"     score table:\n{candidate_k[['k','silhouette','davies_bouldin','score']].round(3).to_string(index=False)}")


# ═════════════════════════════════════════════════════════════════════════
# BƯỚC 4 · Fit K-Means với K* + cluster stability (ARI across 10 seeds)
# ═════════════════════════════════════════════════════════════════════════
print(f"\n▶ BƯỚC 4 · Fit K-Means (K={BEST_K}) + stability test")

final_km = KMeans(n_clusters=BEST_K, n_init=20, random_state=RNG)
cluster_labels = final_km.fit_predict(X_scaled)

# Post-process: cluster quá nhỏ (< 20 phần tử) thường là outlier —
# merge vào centroid gần nhất khác nó
MIN_CLUSTER_SIZE = 20
cluster_sizes = pd.Series(cluster_labels).value_counts()
tiny_clusters = cluster_sizes[cluster_sizes < MIN_CLUSTER_SIZE].index.tolist()
if tiny_clusters:
    print(f"  ⚠ Phát hiện {len(tiny_clusters)} cluster < {MIN_CLUSTER_SIZE} "
          f"phần tử → merge vào cluster gần nhất")
    centroids = final_km.cluster_centers_
    for tiny_c in tiny_clusters:
        idx = np.where(cluster_labels == tiny_c)[0]
        for i in idx:
            # Tính khoảng cách tới tất cả centroid, trừ centroid của chính nó
            dists = np.linalg.norm(centroids - X_scaled[i], axis=1)
            dists[tiny_c] = np.inf
            # Cũng loại các centroid của tiny cluster khác
            for t in tiny_clusters:
                dists[t] = np.inf
            cluster_labels[i] = int(np.argmin(dists))
    # Re-label để cluster IDs liên tục 0..K'-1
    unique_clusters = sorted(set(cluster_labels))
    relabel_map = {old: new for new, old in enumerate(unique_clusters)}
    cluster_labels = np.array([relabel_map[l] for l in cluster_labels])
    BEST_K = len(unique_clusters)
    print(f"  → K hiệu dụng sau merge = {BEST_K}")

# Stability: chạy thêm 9 lần với seed khác, so sánh ARI với lần đầu
stability_aris = []
for seed in [7, 11, 23, 42, 77, 101, 202, 303, 404]:
    km_s = KMeans(n_clusters=BEST_K, n_init=10, random_state=seed)
    labels_s = km_s.fit_predict(X_scaled)
    ari = adjusted_rand_score(cluster_labels, labels_s)
    stability_aris.append(ari)
stability_mean = float(np.mean(stability_aris))
stability_std = float(np.std(stability_aris))
print(f"  Stability ARI: mean={stability_mean:.3f} · std={stability_std:.3f}")
print(f"  {'✓ Ổn định' if stability_mean >= 0.75 else '⚠ Ổn định trung bình' if stability_mean >= 0.5 else '✗ Kém ổn định'}")


# ═════════════════════════════════════════════════════════════════════════
# BƯỚC 5 · ARI vs product_type (cluster có chỉ tái phát hiện phân loại không?)
# ═════════════════════════════════════════════════════════════════════════
print("\n▶ BƯỚC 5 · So sánh cluster với product_type (ARI)")

product_type_codes = df["product_type"].astype("category").cat.codes.values
ari_vs_type = float(adjusted_rand_score(cluster_labels, product_type_codes))
print(f"  ARI(cluster, product_type) = {ari_vs_type:.3f}")
if ari_vs_type > 0.7:
    print("  ⚠ Cluster gần trùng với product_type → insight ít mới")
elif ari_vs_type > 0.3:
    print("  ~ Cluster có liên hệ vừa phải với product_type")
else:
    print("  ✓ Cluster thể hiện chiều khác với product_type → có insight mới")


# ═════════════════════════════════════════════════════════════════════════
# BƯỚC 6 · PCA 2D cho visualization
# ═════════════════════════════════════════════════════════════════════════
print("\n▶ BƯỚC 6 · PCA 2D")
pca = PCA(n_components=2, random_state=RNG)
X_pca = pca.fit_transform(X_scaled)
var_explained = pca.explained_variance_ratio_.sum()
print(f"  Variance explained (2D): {var_explained:.2%}")


# ═════════════════════════════════════════════════════════════════════════
# BƯỚC 7 · Profile từng cluster + business naming + strategy
# ═════════════════════════════════════════════════════════════════════════
print(f"\n▶ BƯỚC 7 · Profile {BEST_K} clusters")

df["cluster"] = cluster_labels
df["pca_x"] = X_pca[:, 0]
df["pca_y"] = X_pca[:, 1]

profiles = []
for c in range(BEST_K):
    mask = df["cluster"] == c
    sub = df[mask]
    n = int(mask.sum())

    prof = {
        "cluster_id": c,
        "size": n,
        "pct_of_total": round(n / len(df) * 100, 1),

        # Feature means (scale gốc để dễ đọc)
        "mean_price": int(sub["price"].mean()),
        "median_price": int(sub["price"].median()),
        "mean_rating": round(float(sub["rating"].mean()), 2),
        "mean_sold_count": int(sub["sold_count"].mean()),
        "median_sold_count": int(sub["sold_count"].median()),
        "mean_discount_rate": round(float(sub["discount_rate"].mean()), 1),
        "mean_review_ratio": round(float(sub["review_ratio"].mean()), 3),
        "pct_tiki_verified": round(float(sub["tiki_verified"].mean()) * 100, 1),

        # VN / NK breakdown
        "pct_vn": round((sub["origin_class_corrected"] == "Trong nước").mean() * 100, 1),
        "pct_nk": round((sub["origin_class_corrected"] == "Ngoài nước").mean() * 100, 1),

        # Top brand / product_type
        "top_brands": sub["brand_name"].value_counts().head(5).index.tolist(),
        "top_product_types": sub["product_type"].value_counts().head(3).index.tolist(),
    }
    profiles.append(prof)

# Business naming dựa trên profile
def assign_name_and_strategy(p):
    """Tạo tên + chiến lược dựa vào đặc trưng cluster."""
    price = p["mean_price"]
    rating = p["mean_rating"]
    sold = p["median_sold_count"]
    discount = p["mean_discount_rate"]
    verified = p["pct_tiki_verified"]

    # Price tier
    if price > 800_000:
        price_tier = "Cao cấp"
    elif price > 300_000:
        price_tier = "Tầm trung"
    else:
        price_tier = "Bình dân"

    # Activity tier
    if sold >= 10:
        activity = "Bán chạy"
    elif sold >= 2:
        activity = "Ổn định"
    else:
        activity = "Kén khách"

    # Trust tier
    trust = "Uy tín cao" if verified >= 80 else "Uy tín trung bình"

    # Discount tier
    promo = "Săn sale" if discount >= 15 else ("Ít sale" if discount < 3 else "")

    # Compose name
    name_parts = [price_tier, activity, trust]
    if promo:
        name_parts.append(promo)
    name = " · ".join(name_parts)

    # Strategy logic (for sàn / brand managers)
    strategies = []
    if price_tier == "Cao cấp":
        strategies += [
            "Đầu tư nhiều hơn vào gian hàng chính hãng + huy hiệu authentic",
            "Tập trung cày rating và UGC review chất lượng cao",
            "Campaign định hướng: Authentic Week, Brand Day",
        ]
    elif price_tier == "Bình dân":
        strategies += [
            "Ưu tiên ngân sách Freeship Extra + Flash Sale",
            "Tận dụng bundle deal (mua combo) để tăng AOV",
            "Campaign định hướng: 11/11, 12/12, Deal Hot cuối tuần",
        ]
    else:  # Tầm trung
        strategies += [
            "Cân bằng giữa rating và promotion — cả 2 đều quan trọng",
            "Cross-sell với sản phẩm cao cấp để khách trade up",
            "Campaign định hướng: Voucher theo danh mục",
        ]

    if activity == "Kén khách":
        strategies.append(
            "Xem lại hình ảnh + mô tả sản phẩm, cân nhắc re-list với giá mới"
        )
    if verified < 50:
        strategies.append(
            "Đăng ký Tiki Trading / Official Store để tăng độ tin cậy"
        )

    return name, strategies


for p in profiles:
    name, strategies = assign_name_and_strategy(p)
    p["name"] = name
    p["strategies"] = strategies

# In ra bảng tóm tắt
print("\n  ┌─────┬─────────────────────────────────────┬──────┬──────┬────────┬──────┐")
print("  │  #  │ Tên cluster                         │ Size │ % VN │ Giá TB │ Rate │")
print("  ├─────┼─────────────────────────────────────┼──────┼──────┼────────┼──────┤")
for p in profiles:
    print(f"  │  {p['cluster_id']}  │ {p['name'][:35]:<35} │ {p['size']:>4} │ {p['pct_vn']:>4.1f} │ {p['mean_price']//1000:>5}K │ {p['mean_rating']:>4.2f} │")
print("  └─────┴─────────────────────────────────────┴──────┴──────┴────────┴──────┘")


# ═════════════════════════════════════════════════════════════════════════
# BƯỚC 8 · Cross với Model 1: R² regression trong từng cluster
# ═════════════════════════════════════════════════════════════════════════
print("\n▶ BƯỚC 8 · Cross-validation với Model 1")

# Load Model 1 preprocessor + retrain 1 GBR đơn giản trong mỗi cluster
# để đo R². Đây là approximation — không re-train full pipeline.

cross_results = []
reg_features_num = ["log_price", "rating", "discount_rate", "review_ratio"]
reg_features_cat = ["product_type", "origin_class_corrected"]

for c in range(BEST_K):
    sub = df[df["cluster"] == c].copy()
    if len(sub) < 100:
        cross_results.append({"cluster_id": c, "r2": None, "n_train": len(sub)})
        continue

    y = np.log1p(sub["sold_count"].values)
    if y.std() < 1e-6:
        cross_results.append({"cluster_id": c, "r2": None, "n_train": len(sub)})
        continue

    # Basic preprocessor inline
    pre = ColumnTransformer([
        ("num", "passthrough", reg_features_num),
        ("cat", OneHotEncoder(handle_unknown="ignore"), reg_features_cat),
    ])
    reg = Pipeline([
        ("pre", pre),
        ("gbr", GradientBoostingRegressor(
            n_estimators=100, learning_rate=0.08, max_depth=3,
            random_state=RNG
        )),
    ])
    X_reg = sub[reg_features_num + reg_features_cat]
    try:
        Xtr, Xte, ytr, yte = train_test_split(X_reg, y, test_size=0.25, random_state=RNG)
        reg.fit(Xtr, ytr)
        r2 = float(r2_score(yte, reg.predict(Xte)))
    except Exception as e:
        r2 = None

    cross_results.append({"cluster_id": c, "r2": r2, "n_train": int(len(sub))})
    print(f"  Cluster {c}: n={len(sub):>4} · R² log_sold_count = "
          f"{r2:+.3f}" if r2 is not None else f"  Cluster {c}: n<100, skip")


# ═════════════════════════════════════════════════════════════════════════
# BƯỚC 8.5 · Hybrid sampling — chọn sản phẩm tiêu biểu cho mỗi cluster
# ═════════════════════════════════════════════════════════════════════════
print("\n▶ BƯỚC 8.5 · Chọn sản phẩm tiêu biểu cho mỗi cluster")

# Tính khoảng cách Euclidean từ mỗi sản phẩm tới centroid (trong scaled space)
centroids_final = np.zeros((BEST_K, X_scaled.shape[1]))
for c in range(BEST_K):
    mask_c = cluster_labels == c
    centroids_final[c] = X_scaled[mask_c].mean(axis=0)

all_distances = np.zeros(len(df))
for c in range(BEST_K):
    mask_c = cluster_labels == c
    all_distances[mask_c] = np.linalg.norm(
        X_scaled[mask_c] - centroids_final[c], axis=1
    )

df["distance_to_centroid"] = all_distances


def hybrid_sample(cluster_df, distances_to_centroid, n_total=10):
    """
    cluster_df: DataFrame các sản phẩm thuộc cluster đó
    distances_to_centroid: array khoảng cách từ mỗi product tới centroid

    Returns: DataFrame ≤ n_total dòng với cột thêm 'sample_type'
    """
    # Edge case: cluster nhỏ → lấy hết
    if len(cluster_df) <= n_total:
        result = cluster_df.copy()
        result["sample_type"] = "representative"
        return result

    # 3 closest-to-centroid (đại diện trung bình của cluster)
    closest_idx = np.argsort(distances_to_centroid)[:3]
    closest = cluster_df.iloc[closest_idx].copy()
    closest["sample_type"] = "representative"

    # 5 top-selling (loại trừ những cái đã có ở closest)
    remaining = cluster_df[~cluster_df["product_id"].isin(closest["product_id"])]
    top_sold = remaining.nlargest(5, "sold_count").copy()
    top_sold["sample_type"] = "bestseller"

    # 2 highest-rated (rating × log(review_count+1) để ưu tiên rating cao có nhiều review)
    remaining2 = remaining[~remaining["product_id"].isin(top_sold["product_id"])]
    remaining2 = remaining2[remaining2["rating"] > 0]  # chỉ lấy có rating

    if len(remaining2) >= 2:
        remaining2 = remaining2.copy()
        remaining2["_score"] = remaining2["rating"] * np.log1p(remaining2["review_count"])
        top_rated = remaining2.nlargest(2, "_score").drop(columns="_score").copy()
        top_rated["sample_type"] = "top_rated"
    elif len(remaining2) == 1:
        remaining2 = remaining2.copy()
        remaining2["_score"] = remaining2["rating"] * np.log1p(remaining2["review_count"])
        top_rated = remaining2.nlargest(1, "_score").drop(columns="_score").copy()
        top_rated["sample_type"] = "top_rated"
        # Bù thêm 1 bestseller
        extra_remaining = cluster_df[
            ~cluster_df["product_id"].isin(
                pd.concat([closest, top_sold, top_rated])["product_id"]
            )
        ]
        if len(extra_remaining) > 0:
            extra = extra_remaining.nlargest(1, "sold_count").copy()
            extra["sample_type"] = "bestseller"
            top_sold = pd.concat([top_sold, extra], ignore_index=True)
    else:
        # Không có sản phẩm nào có rating > 0 → lấy thêm bestseller cho đủ
        top_rated = pd.DataFrame()
        extra_remaining = cluster_df[
            ~cluster_df["product_id"].isin(
                pd.concat([closest, top_sold])["product_id"]
            )
        ]
        extra = extra_remaining.nlargest(2, "sold_count").copy()
        extra["sample_type"] = "bestseller"
        top_sold = pd.concat([top_sold, extra], ignore_index=True)

    parts = [closest, top_sold]
    if len(top_rated) > 0:
        parts.append(top_rated)
    return pd.concat(parts, ignore_index=True)


all_samples = []
for c in range(BEST_K):
    mask_c = df["cluster"] == c
    cluster_df = df[mask_c].reset_index(drop=True)
    dists = all_distances[mask_c]

    samples = hybrid_sample(cluster_df, dists, n_total=10)
    samples["cluster_id"] = c
    all_samples.append(samples)
    print(f"  Cluster {c}: chọn {len(samples)} sản phẩm "
          f"({(samples['sample_type'].value_counts().to_dict())})")

samples_df = pd.concat(all_samples, ignore_index=True)

# Chỉ giữ các cột cần thiết
SAMPLE_COLS = [
    "cluster_id", "sample_type", "product_id", "name_clean", "brand_name",
    "price", "sold_count", "rating", "review_count", "tiki_verified",
    "distance_to_centroid",
]
samples_df = samples_df[SAMPLE_COLS]
print(f"  → Tổng: {len(samples_df)} sản phẩm tiêu biểu")


# ═════════════════════════════════════════════════════════════════════════
# BƯỚC 9 · Lưu tất cả artifacts
# ═════════════════════════════════════════════════════════════════════════
print(f"\n▶ BƯỚC 9 · Lưu artifacts vào {OUT_DIR}")

# (1) Model + scaler
joblib.dump(final_km, os.path.join(OUT_DIR, "model2_kmeans.joblib"))
joblib.dump(scaler, os.path.join(OUT_DIR, "model2_scaler.joblib"))
joblib.dump(pca, os.path.join(OUT_DIR, "model2_pca.joblib"))

# (2) Metrics JSON
metrics = {
    "best_k": BEST_K,
    "k_range_scan": k_metrics,
    "final_silhouette": float(silhouette_score(X_scaled[sil_idx], cluster_labels[sil_idx])),
    "final_davies_bouldin": float(davies_bouldin_score(X_scaled, cluster_labels)),
    "final_inertia": float(final_km.inertia_),
    "stability_ari_mean": stability_mean,
    "stability_ari_std": stability_std,
    "stability_ari_runs": stability_aris,
    "ari_vs_product_type": ari_vs_type,
    "pca_variance_explained": float(var_explained),
    "n_samples": int(len(df)),
    "features": FEATURE_COLS,
}
with open(os.path.join(OUT_DIR, "model2_metrics.json"), "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2, ensure_ascii=False)

# (3) Cluster profiles + strategies
with open(os.path.join(OUT_DIR, "model2_profile.json"), "w", encoding="utf-8") as f:
    json.dump(profiles, f, indent=2, ensure_ascii=False)

# (4) Cross regression results
with open(os.path.join(OUT_DIR, "model2_cross_regression.json"), "w", encoding="utf-8") as f:
    json.dump(cross_results, f, indent=2, ensure_ascii=False)

# (5) Per-product labels + PCA coords
out_df = df[[
    "product_id", "name_clean", "brand_name", "product_type",
    "origin_class_corrected", "price", "rating", "sold_count",
    "discount_rate", "tiki_verified", "cluster", "pca_x", "pca_y"
]].copy()
out_df.to_csv(os.path.join(OUT_DIR, "model2_cluster_labels.csv"), index=False)

# (6) Cluster representative samples
samples_df.to_csv(os.path.join(OUT_DIR, "model2_cluster_samples.csv"), index=False)

# (7) Baseline comparison — K-Means vs Naive Groupby
print("\n  Computing baseline comparison...")
baseline_groups = df_full.groupby(
    ["price_segment", "popularity_tier", "rating_tier"],
    observed=True
).size().reset_index(name="count")

n_segments = df_full["price_segment"].nunique()
n_pop = df_full["popularity_tier"].nunique()
n_rating = df_full["rating_tier"].nunique()
n_possible = n_segments * n_pop * n_rating

n_actual = len(baseline_groups)
n_large = int((baseline_groups["count"] >= 100).sum())
n_tiny = int((baseline_groups["count"] <= 2).sum())
baseline_size_std = float(baseline_groups["count"].std())
baseline_size_mean = float(baseline_groups["count"].mean())

top_baseline = baseline_groups.nlargest(10, "count")
top_baseline_records = []
for _, row in top_baseline.iterrows():
    top_baseline_records.append({
        "price_segment": str(row["price_segment"]),
        "popularity_tier": str(row["popularity_tier"]),
        "rating_tier": str(row["rating_tier"]),
        "count": int(row["count"]),
    })

kmeans_sizes = pd.Series([p["size"] for p in profiles])
kmeans_silhouette = float(silhouette_score(X_scaled[sil_idx], cluster_labels[sil_idx]))

baseline_data = {
    "n_possible_combinations": int(n_possible),
    "baseline": {
        "n_actual_groups": int(n_actual),
        "n_large_groups": n_large,
        "n_tiny_groups": n_tiny,
        "pct_large": round(n_large / max(n_actual, 1) * 100, 1),
        "pct_tiny": round(n_tiny / max(n_actual, 1) * 100, 1),
        "cv": round(baseline_size_std / max(baseline_size_mean, 1e-9), 2),
        "top_groups": top_baseline_records,
    },
    "kmeans": {
        "n_groups": len(profiles),
        "silhouette": round(kmeans_silhouette, 3),
        "cv": round(float(kmeans_sizes.std()) / max(float(kmeans_sizes.mean()), 1e-9), 2),
        "sizes": [int(s) for s in kmeans_sizes],
        "names": [p["name"] for p in profiles],
        "pct_of_total": round(len(df) / n_total * 100, 1),
    },
}
with open(os.path.join(OUT_DIR, "model2_baseline_comparison.json"), "w", encoding="utf-8") as f:
    json.dump(baseline_data, f, indent=2, ensure_ascii=False)

print("  ✓ model2_kmeans.joblib · model2_scaler.joblib · model2_pca.joblib")
print("  ✓ model2_metrics.json · model2_profile.json · model2_cross_regression.json")
print("  ✓ model2_elbow_silhouette.csv · model2_cluster_labels.csv")
print("  ✓ model2_cluster_samples.csv")
print("  ✓ model2_corr_raw.csv · model2_corr_filtered.csv · model2_corr_insights.json")
print("  ✓ model2_feature_quality.json · model2_dormant_stats.json")
print("  ✓ model2_baseline_comparison.json")

print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
print(f"  Xong! K = {BEST_K} clusters · Silhouette = {metrics['final_silhouette']:.3f}")
print(f"  Filtered: {len(df):,}/{n_total:,} sản phẩm ({len(df)/n_total*100:.1f}% catalog)")
print(f"  Stability ARI = {stability_mean:.3f} · ARI vs product_type = {ari_vs_type:.3f}")
print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
