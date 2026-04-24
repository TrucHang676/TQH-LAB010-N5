# data_loader.py — Load & cache dữ liệu dùng chung cho tất cả pages
# Import: from data_loader import load_data, df, df_vn, df_nn

import pandas as pd
import numpy as np
import os

# ── Đường dẫn tới file CSV ───────────────────────────────────
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(_BASE_DIR, 'data', 'tiki_cosmetics_processed.csv')

PRICE_ORDER = ['Dưới 100k', '100k – 300k', '300k – 700k', '700k – 2tr', 'Trên 2tr']

_cache = {}   # cache đơn giản, tránh đọc file nhiều lần

def load_data():
    """Load và trả về DataFrame đã xử lý sẵn. Có cache."""
    if 'df' in _cache:
        return _cache['df'], _cache['df_vn'], _cache['df_nn']

    df = pd.read_csv(DATA_PATH)

    # Restore ordered categorical
    df['price_segment'] = pd.Categorical(
        df['price_segment'], categories=PRICE_ORDER, ordered=True
    )

    # Đảm bảo kiểu số đúng
    for col in ['sold_count', 'review_count', 'estimated_revenue', 'price']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    df['is_official_store'] = df['is_official_store'].astype(bool)
    df['is_extreme_seller'] = df['is_extreme_seller'].astype(bool)

    df_vn = df[df['origin_class_corrected'] == 'Trong nước'].copy()
    df_nn = df[df['origin_class_corrected'] == 'Ngoài nước'].copy()

    _cache['df']    = df
    _cache['df_vn'] = df_vn
    _cache['df_nn'] = df_nn

    print(f'Data loaded: {len(df):,} sản phẩm | '
          f'Nội: {len(df_vn):,} | Ngoại: {len(df_nn):,}')
    return df, df_vn, df_nn


# Preload khi import
df, df_vn, df_nn = load_data()
