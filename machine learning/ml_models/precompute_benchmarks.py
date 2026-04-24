"""
precompute_benchmarks.py
========================
Precompute percentile benchmarks cho ML Module 1 (Regression dashboard).

Output: dashboardML/ml_models/percentile_benchmarks.json

Dùng bởi: page_ml_regression.py (load ở module-level, dùng trong Section 3
callback để sinh 3 block phụ: Percentile · Burn Ratio · Revenue Benchmark).

Design decisions:
- Peer group key = `product_type` (5 nhóm: Skincare, Body Care, Hair Care,
  Makeup, Fragrance). Mỗi nhóm có n > 370 — đủ lớn để percentile đáng tin cậy.
- Chỉ tính benchmark trên SP **active** (sold_count > 0) — vì 41.9% dataset
  có sold=0, nếu gộp vào thì percentile bị lệch về bottom.
- Mẫu 500 điểm `sold_values_sample` (sorted) để runtime vẽ ECDF + tính
  percentile rank qua np.searchsorted.
- Lưu luôn `global_burn` — benchmark burn ratio chung cho Box 2.

Chạy 1 lần sau mỗi lần cập nhật dataset:
    python dashboardML/ml_models/precompute_benchmarks.py
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH    = PROJECT_ROOT / "data" / "tiki_cosmetics_processed.csv"
OUTPUT_PATH  = PROJECT_ROOT / "dashboardML" / "ml_models" / "percentile_benchmarks.json"

# ─────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────
PERCENTILES     = [25, 50, 75, 90, 95, 99]
ECDF_SAMPLE_N   = 500       # số điểm sampled để runtime tính percentile + vẽ ECDF
MIN_PEER_SIZE   = 30        # dưới mức này sẽ warning (hiện không xảy ra với product_type)
RANDOM_SEED     = 42        # cho sampling reproducible


# ─────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────
def _sample_sorted(values: np.ndarray, n: int, seed: int) -> list[float]:
    """Lấy n mẫu ngẫu nhiên (với thay thế nếu cần) rồi sort tăng dần.
    Dùng làm lookup CDF tại runtime."""
    rng = np.random.default_rng(seed)
    if len(values) <= n:
        sample = values.copy()
    else:
        sample = rng.choice(values, size=n, replace=False)
    sample = np.sort(sample)
    return [float(x) for x in sample]


def _compute_group_benchmark(sub: pd.DataFrame, group_name: str) -> dict:
    """Tính toàn bộ thống kê benchmark cho 1 peer group."""
    sold = sub["sold_count"].values.astype(float)
    revenue = sub["estimated_revenue"].values.astype(float)

    benchmark = {
        "n_peers": int(len(sub)),
        "sold": {
            "mean":   float(np.mean(sold)),
            **{f"p{p}": float(np.percentile(sold, p)) for p in PERCENTILES},
        },
        "revenue": {
            "median":  float(np.median(revenue)),
            "p75":     float(np.percentile(revenue, 75)),
            "p90":     float(np.percentile(revenue, 90)),
            "p95":     float(np.percentile(revenue, 95)),
        },
        "sold_values_sample": _sample_sorted(sold, ECDF_SAMPLE_N, RANDOM_SEED),
    }

    # Cảnh báo nếu peer group nhỏ
    if benchmark["n_peers"] < MIN_PEER_SIZE:
        benchmark["warning"] = (
            f"Peer group '{group_name}' chỉ có {benchmark['n_peers']} SP — "
            f"percentile có thể không đáng tin cậy."
        )

    return benchmark


def _compute_global_burn(df: pd.DataFrame) -> dict:
    """Benchmark burn ratio toàn dataset (tất cả SP có KM)."""
    disc = df.loc[df["discount_rate"] > 0, "discount_rate"].values / 100.0
    # burn ratio = discount / (1 - discount), bỏ các case discount=1 (hiếm/hỏng)
    disc = disc[disc < 1.0]
    burn = disc / (1 - disc)
    return {
        "n_products_with_discount": int(len(disc)),
        "discount_rate_pct": {
            "median": float(np.median(disc) * 100),
            "p75":    float(np.percentile(disc, 75) * 100),
            "p90":    float(np.percentile(disc, 90) * 100),
        },
        "burn_ratio": {
            "median": float(np.median(burn)),
            "p75":    float(np.percentile(burn, 75)),
            "p90":    float(np.percentile(burn, 90)),
        },
        "interpretation": (
            "burn_ratio = discount / (1 - discount) — số đồng margin hy sinh "
            "trên mỗi 1 đồng doanh thu thu về. Ví dụ: median 0.28 nghĩa là "
            "SP median đang hy sinh 28đ/100đ doanh thu."
        ),
    }


# ─────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────
def main() -> None:
    print(f"[1/4] Load dataset từ {DATA_PATH.relative_to(PROJECT_ROOT)}")
    df = pd.read_csv(DATA_PATH)
    print(f"      → {len(df):,} rows · {df['product_type'].nunique()} product_types")

    # Lọc SP active (sold > 0) để benchmark
    df_active = df[df["sold_count"] > 0].copy()
    print(f"      → SP active (sold>0): {len(df_active):,} "
          f"({len(df_active) / len(df) * 100:.1f}% dataset)")

    # ── Per product_type ──
    print(f"\n[2/4] Tính benchmark cho từng product_type...")
    per_type: dict[str, dict] = {}
    for ptype, sub in df_active.groupby("product_type"):
        bench = _compute_group_benchmark(sub, ptype)
        per_type[str(ptype)] = bench
        print(f"      {ptype:<12} n={bench['n_peers']:>4}  "
              f"P50={bench['sold']['p50']:>6.0f}  "
              f"P90={bench['sold']['p90']:>8.0f}  "
              f"P95={bench['sold']['p95']:>9.0f}")

    # ── Global active (fallback khi product_type không khớp) ──
    print(f"\n[3/4] Tính benchmark GLOBAL (fallback)...")
    global_bench = _compute_group_benchmark(df_active, "ALL")
    print(f"      n={global_bench['n_peers']}  "
          f"P50={global_bench['sold']['p50']:.0f}  "
          f"P90={global_bench['sold']['p90']:.0f}")

    # ── Global burn ──
    print(f"\n[4/4] Tính benchmark burn ratio...")
    burn_bench = _compute_global_burn(df)
    print(f"      n_discount={burn_bench['n_products_with_discount']}  "
          f"burn_median={burn_bench['burn_ratio']['median']:.3f}  "
          f"burn_p90={burn_bench['burn_ratio']['p90']:.3f}")

    # ── Payload final ──
    payload = {
        "version": "1.0",
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source": {
            "dataset_path": str(DATA_PATH.relative_to(PROJECT_ROOT)),
            "n_total": int(len(df)),
            "n_active": int(len(df_active)),
            "active_ratio": round(len(df_active) / len(df), 4),
        },
        "config": {
            "peer_group_key": "product_type",
            "min_peer_size_warning": MIN_PEER_SIZE,
            "ecdf_sample_n": ECDF_SAMPLE_N,
            "random_seed": RANDOM_SEED,
        },
        "by_product_type": per_type,
        "global": global_bench,
        "global_burn": burn_bench,
        "notes": {
            "peer_filter": "Chỉ tính trên SP active (sold_count > 0) — "
                           "tránh bias vì 41.9% dataset có sold=0.",
            "usage": "Dùng trong page_ml_regression.py Section 3 callback để "
                     "sinh Block B (Percentile), Block C (Burn Ratio), "
                     "Block D (Revenue Benchmark).",
            "runtime_percentile": "percentile_rank(x, G) = "
                                   "np.searchsorted(G['sold_values_sample'], x, "
                                   "side='right') / len(sample) * 100",
        },
    }

    # ── Write ──
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    size_kb = OUTPUT_PATH.stat().st_size / 1024
    print(f"\n✅ Đã lưu {OUTPUT_PATH.relative_to(PROJECT_ROOT)} ({size_kb:.1f} KB)")
    print(f"   → {len(per_type)} peer groups · "
          f"{sum(b['n_peers'] for b in per_type.values())} total peers")


if __name__ == "__main__":
    main()
