# pages/page_ml_regression.py
# ╔══════════════════════════════════════════════════════════════╗
# ║  Trang ML — Dự đoán lượt bán (Regression Model)              ║
# ║  Dark navy theme · Đồng bộ với các page khác                 ║
# ║  S1: Hiệu năng mô hình (Actual vs Predicted · Learning curve)║
# ║  S2: Yếu tố ảnh hưởng (Permutation Importance · PDP giá)     ║
# ║  S3: Dự đoán tương tác (What-if form)                        ║
# ║  S4: Phân tích phần dư (Residual plot)                       ║
# ╚══════════════════════════════════════════════════════════════╝

import dash
from dash import html, dcc, callback, Input, Output, State, no_update
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from data_loader import load_data

import joblib

dash.register_page(
    __name__,
    path='/',
    name='Machine Learning',
    title='Machine Learning | Mỹ phẩm Tiki',
    order=5,
)

# ── Data ─────────────────────────────────────────────────────
df_full, df_vn_full, df_nn_full = load_data()

# ── Load model artifacts ─────────────────────────────────────
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ML_DIR = os.path.abspath(os.path.join(_BASE_DIR, '..', 'ml_models'))


def _safe_load():
    """Load ML artifacts. Nếu chưa train thì trả None để page hiển thị hướng dẫn."""
    try:
        model = joblib.load(os.path.join(ML_DIR, 'model1_regressor.joblib'))
        with open(os.path.join(ML_DIR, 'model1_metrics.json'), encoding='utf-8') as f:
            metrics = json.load(f)
        with open(os.path.join(ML_DIR, 'model1_feature_meta.json'), encoding='utf-8') as f:
            feat_meta = json.load(f)
        preds = pd.read_csv(os.path.join(ML_DIR, 'model1_predictions.csv'))
        fi = pd.read_csv(os.path.join(ML_DIR, 'model1_feature_importance.csv'))
        lc = pd.read_csv(os.path.join(ML_DIR, 'model1_learning_curve.csv'))
        pdp = pd.read_csv(os.path.join(ML_DIR, 'model1_partial_dependence.csv'))
        return model, metrics, feat_meta, preds, fi, lc, pdp
    except Exception as e:
        print(f'[page_ml_regression] Chưa load được artifact: {e}')
        return None, None, None, None, None, None, None


MODEL, METRICS, FEAT_META, PREDS, FI, LC, PDP = _safe_load()
ML_READY = MODEL is not None

# ── Load percentile benchmarks ───────────────────────────────
from pathlib import Path as _Path
_BENCH_PATH = _Path(__file__).resolve().parents[1] / 'ml_models' / 'percentile_benchmarks.json'
try:
    with open(_BENCH_PATH, encoding='utf-8') as _f:
        BENCHMARKS = json.load(_f)
except Exception:
    BENCHMARKS = None

# ── Ranking metrics (computed once) ──────────────────────────
def _compute_hit_at_k(y_true, y_pred, k_pct: float) -> float:
    """Tỷ lệ model chọn đúng top k% thực tế."""
    n = len(y_true)
    k = max(1, int(n * k_pct))
    top_true = set(np.argsort(y_true)[-k:])
    top_pred = set(np.argsort(y_pred)[-k:])
    return len(top_true & top_pred) / k * 100

if ML_READY and PREDS is not None:
    HIT_AT_5  = _compute_hit_at_k(PREDS['actual_sold'].values, PREDS['predicted_sold'].values, 0.05)
    HIT_AT_10 = _compute_hit_at_k(PREDS['actual_sold'].values, PREDS['predicted_sold'].values, 0.10)
    SPEARMAN  = METRICS['test_metrics']['spearman']
else:
    HIT_AT_5, HIT_AT_10, SPEARMAN = 0, 0, 0

# ── Precompute accuracy-sorted auto-fill options (run once) ──
_AUTOFILL_OPTIONS = []
if ML_READY:
    try:
        _top = df_full[df_full['sold_count'] > 0].copy()
        _top['log_price'] = np.log1p(_top['price'])
        _top['has_rating'] = (_top['rating'] > 0).astype(int)
        _top_brands = FEAT_META['top_brands'] + ['Other']
        _top['brand_grouped'] = _top['brand_name'].where(
            _top['brand_name'].isin(_top_brands), 'Other')
        for _c in ['is_official_store', 'tiki_verified', 'has_authentic_badge']:
            _top[_c] = _top[_c].astype(int)

        # Price segment
        _top['price_segment'] = pd.cut(
            _top['price'],
            bins=[0, 100_000, 300_000, 700_000, 2_000_000, float('inf')],
            labels=['Dưới 100k', '100k – 300k', '300k – 700k', '700k – 2tr', 'Trên 2tr'],
            right=False,
        ).astype(str)

        _feats = ['log_price', 'discount_rate', 'rating', 'has_rating',
                  'is_official_store', 'tiki_verified', 'has_authentic_badge',
                  'origin_class_corrected', 'product_type', 'price_segment', 'brand_grouped']
        _X = _top[_feats]
        _pred_log = MODEL.predict(_X)
        _top['_pred_sold'] = np.expm1(_pred_log)
        _top['_abs_log_err'] = np.abs(np.log1p(_top['sold_count'].values) - _pred_log)

        # Sort: small error first, then high sold_count as tiebreaker
        _top = _top.sort_values(['_abs_log_err', 'sold_count'], ascending=[True, False]).head(1000)

        _AUTOFILL_OPTIONS = []
        for _idx, _r in _top.iterrows():
            _err = _r['_abs_log_err']
            _icon = '🎯' if _err < 0.5 else ('⚡' if _err < 1.0 else '📊')
            _label = f"{_icon} {_r['name'][:75]}... (Đã bán: {_r['sold_count']:,.0f})"
            _AUTOFILL_OPTIONS.append({'label': _label, 'value': _idx})
        print(f'[Auto-fill] {len(_AUTOFILL_OPTIONS)} sản phẩm sorted by accuracy')
    except Exception as _e:
        print(f'[Auto-fill] Fallback to sold_count sort: {_e}')
        _top_fallback = df_full.sort_values('sold_count', ascending=False).head(1000)
        _AUTOFILL_OPTIONS = [
            {'label': f"{r['name'][:80]}... (Đã bán: {r['sold_count']:,.0f})", 'value': i}
            for i, r in _top_fallback.iterrows()
        ]

# ── Dark palette (đồng bộ với page3) ─────────────────────────
BG       = '#172542'
SURFACE  = '#1A2A4D'
CARD     = '#1C2D55'
BORDER   = 'rgba(255,255,255,0.07)'
BORDER2  = 'rgba(255,255,255,0.13)'
GRID     = 'rgba(255,255,255,0.05)'
TXT      = '#F0F6FF'
SUBTXT   = '#94A3B8'
MUTED    = '#4B6178'

# ML page accent: violet (AI/intelligence) + amber (warning/importance)
VIOLET   = '#A78BFA'
INDIGO   = '#818CF8'
AMBER    = '#FBBF24'
EMERALD  = '#34D399'
ROSE     = '#FB7185'
CYAN     = '#22D3EE'
BLUE     = '#60A5FA'
ORANGE   = '#FB923C'

C_DOM    = '#38BDF8'
C_IMP    = '#F87171'


# ══════════════════════════════════════════════════════════════
#  CHART THEME HELPER
# ══════════════════════════════════════════════════════════════
def hex_to_rgba(hex_color, alpha=1):
    hex_color = hex_color.lstrip('#')
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return f'rgba({r},{g},{b},{alpha})'


_SUP = str.maketrans('0123456789-', '⁰¹²³⁴⁵⁶⁷⁸⁹⁻')


def fmt_sci(v, precision=0):
    """Format số rất nhỏ dạng khoa học với Unicode superscript.

    Ví dụ: -4.66e-07 → '−5×10⁻⁷' (gọn, chuẩn học thuật).
    Dùng cho các con số gần 0 để tránh hiển thị '−0.000' gây hiểu lầm.
    """
    s = f'{v:.{precision}e}'
    mant, exp = s.split('e')
    mant = mant.replace('-', '−')           # minus sign đẹp
    exp_sup = str(int(exp)).translate(_SUP)
    return f'{mant}×10{exp_sup}'


def _theme(fig, height=320, **kw):
    margin = kw.pop('margin', dict(l=14, r=14, t=28, b=14))

    fig.update_layout(
        height=height,
        font=dict(
            family="'Space Grotesk','Segoe UI',sans-serif",
            size=12,
            color=TXT
        ),
        paper_bgcolor=CARD,
        plot_bgcolor=CARD,
        margin=margin,
        hoverlabel=dict(
            bgcolor=SURFACE,
            bordercolor=BORDER2,
            font=dict(size=12, color=TXT)
        ),
        **kw,
    )
    fig.update_xaxes(
        gridcolor=GRID, gridwidth=1,
        linecolor='rgba(255,255,255,0.06)', linewidth=1,
        tickfont=dict(size=10, color=SUBTXT),
        title_font=dict(size=11, color=SUBTXT),
        zeroline=False,
    )
    fig.update_yaxes(
        gridcolor=GRID, gridwidth=1,
        linecolor='rgba(0,0,0,0)',
        tickfont=dict(size=10, color=SUBTXT),
        title_font=dict(size=11, color=SUBTXT),
        zeroline=False,
    )
    return fig


def _leg(**kw):
    return dict(orientation='h', yanchor='bottom', y=1.03,
                xanchor='right', x=1, font=dict(size=10, color=SUBTXT),
                bgcolor='rgba(0,0,0,0)', **kw)


def _module_switcher(active='module1'):
    return html.Div([
        dcc.Link(
            'Module 1 · Regression',
            href='/',
            className='ml-switcher-link active' if active == 'module1' else 'ml-switcher-link'
        ),
        dcc.Link(
            'Module 2 · Clustering',
            href='/market-segmentation',
            className='ml-switcher-link active' if active == 'module2' else 'ml-switcher-link'
        ),
    ], className='ml-switcher')


# ══════════════════════════════════════════════════════════════
#  FIGURES — HIỆU NĂNG MÔ HÌNH
# ══════════════════════════════════════════════════════════════

def make_actual_vs_predicted():
    """Hexbin density + binned-mean calibration line (thay cho scatter).

    Lý do: sold_count là số nguyên → log1p ra giá trị rời rạc tại
    log(1)=0, log(2)=0.69, log(3)=1.10... Dùng scatter sẽ tạo ra
    các cột thẳng đứng do overplotting. Dùng 2D histogram density
    để thể hiện mật độ đúng cách.
    """
    if PREDS is None:
        return go.Figure()

    act = PREDS['actual_log'].values
    pre = PREDS['predicted_log'].values
    lims = [min(act.min(), pre.min()) - 0.2, max(act.max(), pre.max()) + 0.2]

    # ── Binned-mean calibration line ───────────────────────────
    # Chia x thành 12 bin bằng nhau, tính mean/std predicted trong
    # mỗi bin → thấy rõ model bị bias ở dải giá trị nào
    N_BINS = 12
    bin_edges = np.linspace(lims[0] + 0.2, lims[1] - 0.2, N_BINS + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_ids = np.digitize(act, bin_edges) - 1
    bin_ids = np.clip(bin_ids, 0, N_BINS - 1)

    mean_pred = np.full(N_BINS, np.nan)
    std_pred = np.full(N_BINS, np.nan)
    for b in range(N_BINS):
        mask = bin_ids == b
        if mask.sum() >= 5:       # chỉ plot bin có >=5 điểm
            mean_pred[b] = pre[mask].mean()
            std_pred[b] = pre[mask].std()

    valid = ~np.isnan(mean_pred)
    bc = bin_centers[valid]
    mp = mean_pred[valid]
    sp = std_pred[valid]

    fig = go.Figure()

    # ── Layer 1: 2D histogram density (hexbin-style) ───────────
    fig.add_trace(go.Histogram2d(
        x=act, y=pre,
        nbinsx=32, nbinsy=32,
        xbins=dict(start=lims[0], end=lims[1]),
        ybins=dict(start=lims[0], end=lims[1]),
        colorscale=[
            [0.0, 'rgba(0,0,0,0)'],           # ô rỗng trong suốt
            [0.02, hex_to_rgba(INDIGO, 0.35)],
            [0.25, hex_to_rgba(VIOLET, 0.70)],
            [0.55, hex_to_rgba(AMBER, 0.85)],
            [1.0, hex_to_rgba(EMERALD, 1.0)],
        ],
        colorbar=dict(
            title=dict(text='Mật độ', font=dict(size=9, color=SUBTXT)),
            tickfont=dict(size=9, color=SUBTXT),
            thickness=10, len=0.7,
            bgcolor='rgba(0,0,0,0)', outlinecolor='rgba(0,0,0,0)',
        ),
        hovertemplate=('<b>Actual (log):</b> %{x:.2f}<br>'
                       '<b>Predicted (log):</b> %{y:.2f}<br>'
                       '<b>Số sản phẩm:</b> %{z}<extra></extra>'),
        name='Mật độ sản phẩm',
        showlegend=False,
    ))

    # ── Layer 2: đường y=x (lý tưởng) ──────────────────────────
    fig.add_trace(go.Scatter(
        x=lims, y=lims, mode='lines',
        line=dict(color=AMBER, width=1.8, dash='dash'),
        name='y = x (lý tưởng)',
        hoverinfo='skip',
    ))

    # ── Layer 3: dải ±1σ quanh binned-mean ─────────────────────
    fig.add_trace(go.Scatter(
        x=np.concatenate([bc, bc[::-1]]),
        y=np.concatenate([mp + sp, (mp - sp)[::-1]]),
        fill='toself', fillcolor=hex_to_rgba(ROSE, 0.18),
        line=dict(color='rgba(0,0,0,0)'),
        name='±1σ theo bin', hoverinfo='skip',
        showlegend=True,
    ))

    # ── Layer 4: binned-mean line (calibration) ────────────────
    fig.add_trace(go.Scatter(
        x=bc, y=mp, mode='lines+markers',
        line=dict(color=ROSE, width=2.5),
        marker=dict(size=7, color=ROSE,
                    line=dict(color='rgba(255,255,255,0.3)', width=1)),
        name='Mean dự đoán / bin',
        hovertemplate=('<b>Actual ≈ %{x:.2f}</b><br>'
                       'Mean dự đoán = %{y:.2f}<extra></extra>'),
    ))

    _theme(fig, height=320,
           showlegend=True,
           legend=_leg(),
           xaxis=dict(title='log(1 + lượt bán thực tế)', range=lims),
           yaxis=dict(title='log(1 + lượt bán dự đoán)', range=lims))
    return fig


def make_learning_curve():
    """Learning curve: train vs validation R²."""
    if LC is None:
        return go.Figure()

    x = LC['train_size'].values
    tr_m = LC['train_r2_mean'].values
    tr_s = LC['train_r2_std'].values
    va_m = LC['val_r2_mean'].values
    va_s = LC['val_r2_std'].values

    fig = go.Figure()
    # Train band
    fig.add_trace(go.Scatter(
        x=np.concatenate([x, x[::-1]]),
        y=np.concatenate([tr_m + tr_s, (tr_m - tr_s)[::-1]]),
        fill='toself', fillcolor=hex_to_rgba(VIOLET, 0.14),
        line=dict(color='rgba(0,0,0,0)'), showlegend=False, hoverinfo='skip',
    ))
    fig.add_trace(go.Scatter(
        x=x, y=tr_m, mode='lines+markers', name='Train R²',
        line=dict(color=VIOLET, width=2.3),
        marker=dict(size=7, color=VIOLET,
                    line=dict(color='rgba(255,255,255,0.3)', width=1)),
        hovertemplate='Train size: %{x}<br>R² = %{y:.3f}<extra></extra>',
    ))
    # Val band
    fig.add_trace(go.Scatter(
        x=np.concatenate([x, x[::-1]]),
        y=np.concatenate([va_m + va_s, (va_m - va_s)[::-1]]),
        fill='toself', fillcolor=hex_to_rgba(AMBER, 0.14),
        line=dict(color='rgba(0,0,0,0)'), showlegend=False, hoverinfo='skip',
    ))
    fig.add_trace(go.Scatter(
        x=x, y=va_m, mode='lines+markers', name='Validation R²',
        line=dict(color=AMBER, width=2.3),
        marker=dict(size=7, color=AMBER,
                    line=dict(color='rgba(255,255,255,0.3)', width=1)),
        hovertemplate='Train size: %{x}<br>R² = %{y:.3f}<extra></extra>',
    ))
    _theme(fig, height=320,
           showlegend=True, legend=_leg(),
           xaxis=dict(title='Số mẫu train'),
           yaxis=dict(title='R² score', range=[0, 1]))
    return fig


# ══════════════════════════════════════════════════════════════
#  FIGURES — YẾU TỐ ẢNH HƯỞNG
# ══════════════════════════════════════════════════════════════

FEATURE_LABEL_VN = {
    'rating': 'Rating',
    'log_price': 'Giá (log)',
    'discount_rate': 'Mức giảm giá',
    'tiki_verified': 'Tiki Verified',
    'has_rating': 'Có đánh giá',
    'brand_grouped': 'Thương hiệu',
    'origin_class_corrected': 'Xuất xứ',
    'product_type': 'Ngành hàng',
    'has_authentic_badge': 'Authentic badge',
    'price_segment': 'Phân khúc giá',
    'is_official_store': 'Official store',
}


def make_feature_importance():
    """Permutation importance horizontal bar with error bars."""
    if FI is None:
        return go.Figure()

    # Sort ascending để hiển thị top lên trên
    top = FI.sort_values('importance_mean', ascending=True)
    labels = [FEATURE_LABEL_VN.get(f, f) for f in top['feature']]
    means = top['importance_mean'].values
    stds = top['importance_std'].values

    # Color gradient theo importance
    max_m = max(abs(means).max(), 1e-6)
    colors = [
        hex_to_rgba(VIOLET, 0.35 + 0.65 * (abs(v) / max_m))
        for v in means
    ]

    label_offset = max(float(np.max(stds)) * 1.2, max_m * 0.02)
    x_max = float(np.max(means + stds) + label_offset + max_m * 0.04)

    fig = go.Figure(go.Bar(
        x=means, y=labels, orientation='h',
        marker=dict(color=colors, line=dict(color=VIOLET, width=0.5)),
        error_x=dict(type='data', array=stds, color=AMBER, thickness=1.2, width=3),
        hovertemplate='<b>%{y}</b><br>Importance: %{x:.3f} ± <extra></extra>',
        cliponaxis=False,
    ))

    # Đặt nhãn số lệch phải khỏi error bar để không bị đường vàng che.
    for y_lbl, mean_val, std_val in zip(labels, means, stds):
        fig.add_annotation(
            x=float(mean_val + std_val + label_offset),
            y=y_lbl,
            xref='x', yref='y',
            text=f'{mean_val:+.3f}',
            showarrow=False,
            xanchor='left',
            align='left',
            font=dict(size=10, color=TXT),
        )

    fig.add_vline(x=0, line_color=BORDER2, line_width=1)
    fig.add_annotation(
        xref='paper', yref='paper', x=0, y=1.16,
        text='● Thanh lỗi = độ lệch chuẩn qua 10 lần shuffle · thanh càng dài càng quan trọng',
        showarrow=False,
        align='left',
        font=dict(size=10, color=MUTED),
        bgcolor='rgba(0,0,0,0)',
    )
    _theme(fig, height=340,
           showlegend=False,
            xaxis=dict(
             title='Permutation Importance (↓ R² khi shuffle feature)',
             showgrid=True,
             range=[min(0, float(np.min(means)) - 0.02), x_max],
            ),
           yaxis=dict(showgrid=False, tickfont=dict(size=10.5)),
            margin=dict(l=14, r=125, t=20, b=28))
    return fig


def make_pdp_price():
    """Partial Dependence Plot của giá (trên scale tiền thật)."""
    if PDP is None:
        return go.Figure()

    x = PDP['price_vnd'].values / 1000  # về nghìn VNĐ
    y_log = PDP['predicted_log_sold'].values
    y_orig = PDP['predicted_sold'].values

    fig = go.Figure()
    # Area fill
    fig.add_trace(go.Scatter(
        x=x, y=y_log,
        mode='lines', name='log(1+lượt bán) dự đoán',
        line=dict(color=VIOLET, width=2.5, shape='spline'),
        fill='tozeroy', fillcolor=hex_to_rgba(VIOLET, 0.12),
        customdata=y_orig,
        hovertemplate='Giá: %{x:,.0f}k VNĐ<br>'
                      'log(1+lượt bán): %{y:.2f}<br>'
                      'lượt bán ước: %{customdata:.1f}<extra></extra>',
    ))
    _theme(fig, height=300,
           showlegend=False,
           xaxis=dict(title='Giá sản phẩm (nghìn VNĐ)', type='log'),
           yaxis=dict(title='log(1 + lượt bán) dự đoán'))
    return fig


# ══════════════════════════════════════════════════════════════
#  FIGURES — PHÂN TÍCH PHẦN DƯ (RESIDUAL)
# ══════════════════════════════════════════════════════════════

def make_residual_plot():
    """Residual vs predicted — hexbin density + running-mean ±1σ.

    Lý do thay scatter bằng hexbin: residual = actual − predicted, mà
    actual_log là rời rạc (log1p của số nguyên sold_count) → với mỗi
    actual cố định, khi predicted biến thiên ta có residual = hằng số
    − predicted ⇒ tạo đường thẳng dốc −1. Nhiều giá trị actual rời rạc
    ⇒ nhiều đường chéo song song (artifact). Dùng Histogram2d density
    + running-mean để bias được đọc ra ngay, không bị artifact che.
    """
    if PREDS is None:
        return go.Figure()

    y_pred = PREDS['predicted_log'].values
    resid = PREDS['residual_log'].values

    x_lims = [y_pred.min() - 0.2, y_pred.max() + 0.2]
    y_abs = max(abs(resid.min()), abs(resid.max())) + 0.3
    y_lims = [-y_abs, y_abs]

    # ── Binned-mean residual (12 bin theo predicted) ───────────
    N_BINS = 12
    bin_edges = np.linspace(x_lims[0] + 0.2, x_lims[1] - 0.2, N_BINS + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_ids = np.digitize(y_pred, bin_edges) - 1
    bin_ids = np.clip(bin_ids, 0, N_BINS - 1)

    mean_res = np.full(N_BINS, np.nan)
    std_res = np.full(N_BINS, np.nan)
    for b in range(N_BINS):
        mask = bin_ids == b
        if mask.sum() >= 5:
            mean_res[b] = resid[mask].mean()
            std_res[b] = resid[mask].std()

    valid = ~np.isnan(mean_res)
    bc = bin_centers[valid]
    mr = mean_res[valid]
    sr = std_res[valid]

    fig = go.Figure()

    # ── Layer 1: 2D density (hexbin-style) ─────────────────────
    fig.add_trace(go.Histogram2d(
        x=y_pred, y=resid,
        nbinsx=32, nbinsy=32,
        xbins=dict(start=x_lims[0], end=x_lims[1]),
        ybins=dict(start=y_lims[0], end=y_lims[1]),
        colorscale=[
            [0.0, 'rgba(0,0,0,0)'],
            [0.02, hex_to_rgba(INDIGO, 0.35)],
            [0.25, hex_to_rgba(VIOLET, 0.70)],
            [0.55, hex_to_rgba(AMBER, 0.85)],
            [1.0, hex_to_rgba(EMERALD, 1.0)],
        ],
        colorbar=dict(
            title=dict(text='Mật độ', font=dict(size=9, color=SUBTXT)),
            tickfont=dict(size=9, color=SUBTXT),
            thickness=10, len=0.7,
            bgcolor='rgba(0,0,0,0)', outlinecolor='rgba(0,0,0,0)',
        ),
        hovertemplate=('<b>Predicted (log):</b> %{x:.2f}<br>'
                       '<b>Residual:</b> %{y:.2f}<br>'
                       '<b>Số sản phẩm:</b> %{z}<extra></extra>'),
        name='Mật độ sản phẩm',
        showlegend=False,
    ))

    # ── Layer 2: đường y=0 (residual lý tưởng) ─────────────────
    fig.add_trace(go.Scatter(
        x=x_lims, y=[0, 0], mode='lines',
        line=dict(color=AMBER, width=1.8, dash='dash'),
        name='y = 0 (không bias)',
        hoverinfo='skip',
    ))

    # ── Layer 3: dải ±1σ quanh mean residual ───────────────────
    fig.add_trace(go.Scatter(
        x=np.concatenate([bc, bc[::-1]]),
        y=np.concatenate([mr + sr, (mr - sr)[::-1]]),
        fill='toself', fillcolor=hex_to_rgba(ROSE, 0.18),
        line=dict(color='rgba(0,0,0,0)'),
        name='±1σ theo bin', hoverinfo='skip',
        showlegend=True,
    ))

    # ── Layer 4: binned-mean residual line ─────────────────────
    fig.add_trace(go.Scatter(
        x=bc, y=mr, mode='lines+markers',
        line=dict(color=ROSE, width=2.5),
        marker=dict(size=7, color=ROSE,
                    line=dict(color='rgba(255,255,255,0.3)', width=1)),
        name='Mean residual / bin',
        hovertemplate=('<b>Predicted ≈ %{x:.2f}</b><br>'
                       'Mean residual = %{y:.3f}<extra></extra>'),
    ))

    _theme(fig, height=320,
           showlegend=True,
           legend=_leg(),
           xaxis=dict(title='Giá trị dự đoán (log)', range=x_lims),
           yaxis=dict(title='Phần dư (thực tế − dự đoán)', range=y_lims))
    return fig


# ══════════════════════════════════════════════════════════════
#  UI COMPONENTS (tái dùng class "p3-*" để nhất quán style)
# ══════════════════════════════════════════════════════════════

def kpi(icon, val, lbl, color, bg):
    return html.Div([
        html.Div(icon, className='p3-kpi-dot', style={'background': bg}),
        html.Div([
            html.P(val, className='p3-kpi-val', style={'color': color}),
            html.P(lbl, className='p3-kpi-lbl'),
        ]),
    ], className='p3-kpi-item')


def sec(badge_txt, title, variant='c1'):
    return html.Div([
        html.Span(badge_txt, className='p3-sec-badge'),
        html.P(title, className='p3-sec-text'),
    ], className=f'p3-sec-label {variant}')


def card(title, children, flex='1', min_w='280px',
         glow='g-violet', extra_style=None):
    s = {'flex': flex, 'minWidth': min_w}
    if extra_style:
        s.update(extra_style)
    return html.Div([
        html.P([html.Span(className='dot'), f' {title}'],
               className='p3-card-title'),
        children,
    ], className=f'p3-card p3-card-glow {glow}', style=s)


def insight(icon, text, variant='ic'):
    return html.Div([
        html.Span(icon, className='p3-insight-ico'),
        html.Span(text, style={'fontSize': '12px'}),
    ], className=f'p3-insight {variant}')


def mini_stat(val, lbl, color):
    return html.Div([
        html.P(val, className='p3-mini-val', style={'color': color}),
        html.P(lbl, className='p3-mini-lbl'),
    ], className='p3-mini-stat')


# ══════════════════════════════════════════════════════════════
#  BLOCK HELPERS — Section 3 result grid (4 blocks)
# ══════════════════════════════════════════════════════════════

def _fmt_vnd(x: float) -> str:
    """Format VNĐ with K/M/tỷ suffix."""
    if x >= 1e9:
        return f"{x/1e9:.2f} tỷ"
    if x >= 1e6:
        return f"{x/1e6:.1f}M"
    if x >= 1e3:
        return f"{x/1e3:.0f}K"
    return f"{x:.0f}"


def _block_prediction(pred_orig, low, high):
    """Block A — Prediction result (violet)."""
    return html.Div([
        html.P('🎯 DỰ ĐOÁN LƯỢT BÁN', className='ml-block-header'),
        html.P(f'{pred_orig:,.1f}',
               style={
                   'fontFamily': "'Rajdhani', sans-serif",
                   'fontSize': '40px',
                   'fontWeight': 700,
                   'color': VIOLET,
                   'margin': '0',
                   'lineHeight': '1',
               }),
        html.P(f'(khoảng: {low:,.1f} – {high:,.1f})',
               className='ml-block-detail'),
        html.P('Dự đoán dựa trên log-scale · Khoảng ±1 std residual',
               className='ml-block-note'),
    ], className='ml-block ml-block-a')


def _block_percentile(pred, ptype, actual=None):
    """Block B — Market positioning / Percentile benchmark (amber)."""
    if BENCHMARKS is None:
        return html.Div('Chưa có benchmarks', className='ml-block ml-block-b')

    if ptype in BENCHMARKS.get('by_product_type', {}):
        bench = BENCHMARKS['by_product_type'][ptype]
    else:
        bench = BENCHMARKS.get('global', BENCHMARKS.get('by_product_type', {}).get('Skincare', {}))

    sample = np.array(bench['sold_values_sample'])
    n_peers = bench['n_peers']

    # Percentile rank
    pct_pred = np.searchsorted(sample, pred, side='right') / len(sample) * 100
    top_pred = 100 - pct_pred

    # Tier label
    if top_pred <= 5:
        tier, tier_color = "🔥 TOP PERFORMER", EMERALD
    elif top_pred <= 10:
        tier, tier_color = "🔥 TOP 10%", EMERALD
    elif top_pred <= 25:
        tier, tier_color = "✨ Above Average", EMERALD
    elif top_pred <= 50:
        tier, tier_color = "📊 Average", AMBER
    elif top_pred <= 75:
        tier, tier_color = "⚠️ Below Average", AMBER
    else:
        tier, tier_color = "🚨 Underperforming", ROSE

    children = [
        html.P(f'📍 ĐỊNH VỊ CẠNH TRANH — {ptype} · {n_peers} peers',
               className='ml-block-header'),
        html.P(tier, className='ml-block-tier', style={'color': tier_color}),
        html.P(f'● Model dự đoán: {pred:,.1f} → Top {top_pred:.1f}%',
               className='ml-block-detail'),
    ]

    # Actual (conditional) — compare TIER instead of raw gap
    if actual is not None and actual > 0:
        pct_actual = np.searchsorted(sample, actual, side='right') / len(sample) * 100
        top_actual = 100 - pct_actual

        # Determine actual tier
        def _get_tier_idx(top_pct):
            if top_pct <= 10: return 0   # Top performer
            if top_pct <= 25: return 1   # Above average
            if top_pct <= 50: return 2   # Average
            if top_pct <= 75: return 3   # Below average
            return 4                      # Underperforming

        tier_pred_idx = _get_tier_idx(top_pred)
        tier_actual_idx = _get_tier_idx(top_actual)

        children.append(
            html.P(f'▲ Thực tế SP gốc: {actual:,.0f} → Top {top_actual:.1f}%',
                   className='ml-block-detail', style={'color': CYAN})
        )

        if tier_pred_idx == tier_actual_idx:
            children.append(
                html.P('✅ Model xếp hạng chính xác (cùng tier)',
                       className='ml-block-detail',
                       style={'color': EMERALD, 'fontWeight': 600})
            )
        elif abs(tier_pred_idx - tier_actual_idx) == 1:
            children.append(
                html.P('⚠️ Model lệch 1 bậc tier (chấp nhận được)',
                       className='ml-block-detail',
                       style={'color': AMBER, 'fontWeight': 600})
            )
        else:
            children.append(
                html.P('⚠️ Model lệch tier — con số tuyệt đối khác xa thực tế',
                       className='ml-block-detail',
                       style={'color': ROSE, 'fontWeight': 600})
            )

    # Benchmark line
    benchmark_line = (
        f"P50={bench['sold']['p50']:.0f} · P75={bench['sold']['p75']:.0f} · "
        f"P90={bench['sold']['p90']:.0f} · P95={bench['sold']['p95']:.0f}"
    )
    children.append(html.P(benchmark_line, className='ml-block-benchmark'))

    return html.Div(children, className='ml-block ml-block-b')


def _block_burn(discount_pct):
    """Block C — Burn ratio / Marketing cost analysis (rose)."""
    if discount_pct is None or discount_pct <= 0:
        return html.Div([
            html.P('💸 BURN RATIO (hy sinh margin)', className='ml-block-header'),
            html.P('✅ Không có khuyến mãi', className='ml-block-tier',
                   style={'color': EMERALD, 'fontSize': '18px'}),
            html.P('Bán nguyên giá → không hy sinh margin',
                   className='ml-block-detail'),
        ], className='ml-block ml-block-c')

    disc = discount_pct / 100.0
    disc = min(disc, 0.99)
    burn = disc / (1 - disc)

    # Compare with global benchmarks
    if BENCHMARKS and 'global_burn' in BENCHMARKS:
        gb = BENCHMARKS['global_burn']['burn_ratio']
        med = gb['median']
        p75 = gb['p75']
        p90 = gb['p90']
    else:
        med, p75, p90 = 0.282, 0.587, 1.0

    if burn > p90:
        flag, flag_color = "🚨 ĐỐT QUÁ NHIỀU", ROSE
    elif burn > p75:
        flag, flag_color = "⚠️ Cao hơn 75% ngành", AMBER
    else:
        flag, flag_color = "✅ Hợp lý", EMERALD

    diff = (burn - med) * 100

    return html.Div([
        html.P('💸 BURN RATIO (hy sinh margin)', className='ml-block-header'),
        html.P(f'{burn*100:.1f}đ / 100đ doanh thu',
               className='ml-block-tier', style={'color': flag_color}),
        html.P(flag, className='ml-block-detail',
               style={'color': flag_color, 'fontWeight': 600}),
        html.P(
            f'Ngành median {med*100:.1f}đ → bạn '
            f'{"cao" if diff > 0 else "thấp"} hơn {abs(diff):.1f}đ',
            className='ml-block-detail'),
        html.P(
            f'Burn ratio = discount/(1−discount). '
            f'Cứ 100đ thu về thì đốt {burn*100:.1f}đ margin.',
            className='ml-block-note'),
    ], className='ml-block ml-block-c')


def _block_revenue(pred, price, ptype):
    """Block D — Revenue lifetime benchmark (emerald)."""
    revenue = pred * price

    if BENCHMARKS and ptype in BENCHMARKS.get('by_product_type', {}):
        rev_bench = BENCHMARKS['by_product_type'][ptype].get('revenue', {})
    elif BENCHMARKS and 'global' in BENCHMARKS:
        rev_bench = BENCHMARKS['global'].get('revenue', {})
    else:
        rev_bench = {}

    children = [
        html.P('💰 DOANH THU LIFETIME (benchmark)', className='ml-block-header'),
        html.P(f'{_fmt_vnd(revenue)} VNĐ',
               className='ml-block-tier', style={'color': EMERALD}),
    ]

    if rev_bench:
        med_rev = rev_bench.get('median', 0)
        p90_rev = rev_bench.get('p90', 0)
        children.append(
            html.P(f'Median ngành: {_fmt_vnd(med_rev)} · P90: {_fmt_vnd(p90_rev)}',
                   className='ml-block-benchmark')
        )

    children.append(
        html.P('⚠️ Tổng vòng đời, KHÔNG phải doanh thu tháng',
               className='ml-block-note')
    )

    return html.Div(children, className='ml-block ml-block-d')


# ══════════════════════════════════════════════════════════════
#  WHAT-IF FORM INPUTS
# ══════════════════════════════════════════════════════════════

INPUT_STYLE = {
    'background': SURFACE,
    'border': f'1px solid {BORDER2}',
    'borderRadius': '8px',
    'padding': '8px 12px',
    'color': TXT,
    'fontSize': '12px',
    'fontFamily': "'Space Grotesk', 'Segoe UI', sans-serif",
    'width': '100%',
}

LABEL_STYLE = {
    'fontSize': '10.5px',
    'fontWeight': 600,
    'color': SUBTXT,
    'textTransform': 'uppercase',
    'letterSpacing': '0.06em',
    'margin': '0 0 5px 0',
    'display': 'block',
}


def _form_field(label, component, width='1'):
    return html.Div([
        html.Label(label, style=LABEL_STYLE),
        component,
    ], style={'flex': width, 'minWidth': '140px'})


def whatif_form():
    if not ML_READY:
        return html.Div('Chưa có model — chạy train_model1.py trước.',
                        style={'color': SUBTXT, 'padding': '16px'})

    brands = FEAT_META['top_brands'] + ['Other']
    ptypes = FEAT_META['product_types']
    origins = FEAT_META['origin_classes']

    # Sử dụng danh sách đã tính sẵn (sorted by accuracy)
    product_options = _AUTOFILL_OPTIONS

    return html.Div([
        # Row 0: Tìm kiếm sản phẩm Auto-fill
        html.Div([
            _form_field('🔍 Chọn nhanh sản phẩm từ CSDL (Auto-fill)', dcc.Dropdown(
                id='ml-select-product',
                options=product_options,
                placeholder='Gõ tên sản phẩm để tìm kiếm...',
                clearable=True,
                className='ml-dd',
            ), width='1'),
        ], style={'display': 'flex', 'gap': '12px', 'marginBottom': '20px'}),

        # Row 1: Origin · Product type · Brand
        html.Div([
            _form_field('Xuất xứ', dcc.Dropdown(
                id='ml-input-origin',
                options=[{'label': o, 'value': o} for o in origins],
                value=origins[0], clearable=False, className='ml-dd',
            )),
            _form_field('Ngành hàng', dcc.Dropdown(
                id='ml-input-ptype',
                options=[{'label': p, 'value': p} for p in ptypes],
                value=ptypes[0], clearable=False, className='ml-dd',
            )),
            _form_field('Thương hiệu', dcc.Dropdown(
                id='ml-input-brand',
                options=[{'label': b, 'value': b} for b in brands],
                value='Other', clearable=False, className='ml-dd',
            )),
        ], style={'display': 'flex', 'gap': '12px', 'flexWrap': 'wrap',
                  'marginBottom': '12px'}),

        # Row 2: Price · Discount · Rating
        html.Div([
            _form_field('Giá (VNĐ)', dcc.Input(
                id='ml-input-price', type='number', value=300000,
                min=1000, max=20000000, step=1000, style=INPUT_STYLE,
            )),
            _form_field('Mức giảm giá (%)', dcc.Input(
                id='ml-input-discount', type='number', value=10,
                min=0, max=100, step=1, style=INPUT_STYLE,
            )),
            _form_field('Rating (0–5, 0=chưa có)', dcc.Input(
                id='ml-input-rating', type='number', value=4.5,
                min=0, max=5, step=0.1, style=INPUT_STYLE,
            )),
        ], style={'display': 'flex', 'gap': '12px', 'flexWrap': 'wrap',
                  'marginBottom': '12px'}),

        # Row 3: Badges
        html.Div([
            _form_field('Tiki Verified', dcc.Dropdown(
                id='ml-input-verified',
                options=[{'label': 'Có', 'value': 1}, {'label': 'Không', 'value': 0}],
                value=1, clearable=False, className='ml-dd',
            )),
            _form_field('Official Store', dcc.Dropdown(
                id='ml-input-official',
                options=[{'label': 'Có', 'value': 1}, {'label': 'Không', 'value': 0}],
                value=0, clearable=False, className='ml-dd',
            )),
            _form_field('Authentic badge', dcc.Dropdown(
                id='ml-input-authentic',
                options=[{'label': 'Có', 'value': 1}, {'label': 'Không', 'value': 0}],
                value=1, clearable=False, className='ml-dd',
            )),
        ], style={'display': 'flex', 'gap': '12px', 'flexWrap': 'wrap',
                  'marginBottom': '16px'}),

        # Predict button + Output
        html.Div([
            html.Button(
                '✨ Dự đoán lượt bán',
                id='ml-predict-btn', n_clicks=0,
                style={
                    'background': f'linear-gradient(135deg, {VIOLET}, {INDIGO})',
                    'color': 'white',
                    'border': 'none',
                    'borderRadius': '10px',
                    'padding': '10px 22px',
                    'fontWeight': 700,
                    'fontSize': '13px',
                    'letterSpacing': '0.04em',
                    'cursor': 'pointer',
                    'boxShadow': f'0 6px 18px {hex_to_rgba(VIOLET, 0.35)}',
                    'transition': 'transform .15s',
                },
            ),
            html.Div(id='ml-predict-output', style={
                'marginTop': '14px',
                'minHeight': '60px',
            }),
        ], style={'textAlign': 'center'}),
    ])


# ══════════════════════════════════════════════════════════════
#  LAYOUT
# ══════════════════════════════════════════════════════════════

def _not_ready_layout():
    return html.Div([
        html.Div([
            html.Div([
                html.P('MACHINE LEARNING', className='p3-page-label'),
                html.H1('Model chưa được huấn luyện', className='p3-page-title'),
            ], className='p3-topbar-left'),
        ], className='p3-topbar'),
        html.Div([
            html.P('Vui lòng chạy lệnh sau từ thư mục dashboard/:',
                   style={'color': SUBTXT, 'marginBottom': '12px'}),
            html.Pre(
                'cd dashboard && python train_model1.py',
                style={
                    'background': CARD,
                    'padding': '14px 18px',
                    'borderRadius': '10px',
                    'border': f'1px solid {BORDER2}',
                    'color': AMBER,
                    'fontSize': '13px',
                },
            ),
            html.P('Sau khi chạy xong, refresh trang này để xem kết quả.',
                   style={'color': SUBTXT, 'marginTop': '12px', 'fontSize': '12px'}),
        ], style={'padding': '28px'}),
    ], className='p3-page')


def layout():
    if not ML_READY:
        return _not_ready_layout()

    # KPI values
    test_r2  = METRICS['test_metrics']['r2_log']
    test_rmse_o = METRICS['test_metrics']['rmse_orig']
    test_mae_o  = METRICS['test_metrics']['mae_orig']
    best = METRICS['best_model']
    cv_mean = METRICS['cv_results'][best]['mean']
    cv_std  = METRICS['cv_results'][best]['std']
    base_r2 = METRICS['baseline_r2_log']
    uplift = test_r2 - base_r2
    n_train = METRICS['n_train'] + METRICS['n_val']
    n_test = METRICS['n_test']

    # Config for all graphs
    cfg = {'displayModeBar': False}

    return html.Div([

        _module_switcher('module1'),

        # ── TOP BAR ─────────────────────────────────────────
        html.Div([
            html.Div([
                html.P('MACHINE LEARNING · REGRESSION', className='p3-page-label'),
                html.H1('Dự đoán lượt bán sản phẩm · T3/2026',
                        className='p3-page-title'),
            ], className='p3-topbar-left'),
            html.Div([
                html.Span(f'🧠 {best}', className='p3-topchip violet'),
                html.Span(f'✅ R² = {test_r2:+.3f}', className='p3-topchip amber'),
                html.Span(f'📊 n = {n_train+n_test:,}', className='p3-topchip dim'),
                html.Span('Nhóm 05 · FIT-HCMUS', className='p3-topchip dim'),
            ], className='p3-topbar-chips'),
        ], className='p3-topbar'),

        # ── KPI STRIP — Classification metrics ─────────────
        html.P('Classification metrics · Absolute accuracy',
               className='ml-kpi-row-label'),
        html.Div([
            kpi('🎯', f'{test_r2:+.3f}', 'R² · Test', VIOLET,
                hex_to_rgba(VIOLET, .15)),
            kpi('🚀', f'+{uplift:.3f}', 'So với baseline', EMERALD,
                hex_to_rgba(EMERALD, .12)),
            kpi('📉', f'{test_rmse_o:,.1f}', 'RMSE (lượt bán)', AMBER,
                hex_to_rgba(AMBER, .12)),
            kpi('📐', f'{test_mae_o:,.1f}', 'MAE (lượt bán)', CYAN,
                hex_to_rgba(CYAN, .12)),
            kpi('🔁', f'{cv_mean:+.3f}±{cv_std:.3f}', 'CV R² (5-fold)', INDIGO,
                hex_to_rgba(INDIGO, .12)),
            kpi('🧪', f'{n_test:,}', 'Mẫu test', ROSE,
                hex_to_rgba(ROSE, .12)),
        ], className='p3-kpi-strip'),

        # ── KPI STRIP — Ranking metrics ──────────────────────
        html.P('Ranking metrics · Benchmarking use case (Section 3)',
               className='ml-kpi-row-label'),
        html.Div([
            kpi('📈', f'{SPEARMAN:.3f}', 'Spearman rank corr', AMBER,
                hex_to_rgba(AMBER, .15)),
            kpi('🎯', f'{HIT_AT_10:.1f}%', 'Hit@top 10%', EMERALD,
                hex_to_rgba(EMERALD, .12)),
            kpi('🎯', f'{HIT_AT_5:.1f}%', 'Hit@top 5%', INDIGO,
                hex_to_rgba(INDIGO, .12)),
        ], className='p3-kpi-strip'),

        # ─── inner content wrapper ───────────────────────────
        html.Div([

            # ══════════════════════════════════════════════════
            #  S1 — HIỆU NĂNG MÔ HÌNH
            # ══════════════════════════════════════════════════
            sec('SECTION 01', 'Hiệu năng mô hình trên tập test', 'cml1'),

            html.Div([
                card('ACTUAL vs PREDICTED (HEXBIN DENSITY)',
                     html.Div([
                         html.P('● Ô càng sáng = càng nhiều sản phẩm · '
                                'Đường amber = y=x (dự đoán hoàn hảo) · '
                                'Đường rose = mean dự đoán theo bin actual ±1σ',
                                style={'fontSize': '10px', 'color': MUTED,
                                       'margin': '0 0 8px 0'}),
                         dcc.Graph(figure=make_actual_vs_predicted(), config=cfg),
                     ]),
                     flex='1', min_w='320px', glow='g-violet'),
                card('LEARNING CURVE',
                     html.Div([
                         html.P('● Khoảng cách giữa Train và Val cho biết mức overfit',
                                style={'fontSize': '10px', 'color': MUTED,
                                       'margin': '0 0 8px 0'}),
                         dcc.Graph(figure=make_learning_curve(), config=cfg),
                     ]),
                     flex='1', min_w='320px', glow='g-violet'),
                # Mini insight panel
                html.Div([
                    html.Div([
                        mini_stat(f'{test_r2:.3f}', 'R² Test', VIOLET),
                        mini_stat(f'{cv_std:.3f}', 'CV std', AMBER),
                    ], className='p3-mini-stats'),
                    html.Div(style={'height': '12px'}),
                    insight('✅', html.Span([
                        html.Strong(f'Hơn baseline {uplift:.3f} điểm R²'), ' — '
                        f'Dummy mean đạt R² ≈ {fmt_sci(base_r2, 0)} (~0), '
                        f'mô hình giải thích được ~{test_r2*100:.1f}% biến thiên '
                        f'log(sold_count).',
                    ]), 'ic'),
                    insight('🔁', html.Span([
                        html.Strong(f'5-fold CV: {cv_mean:+.3f} ± {cv_std:.3f}'), ' — '
                        'độ lệch thấp cho thấy model ổn định giữa các fold.',
                    ]), 'ic'),
                    insight('🎲', html.Span([
                        html.Strong(f'Robustness 3 seeds: '
                                    f'{METRICS["robustness"]["r2_mean"]:+.3f} ± '
                                    f'{METRICS["robustness"]["r2_std"]:.3f}'),
                        ' — kết quả không phụ thuộc may mắn của split.',
                    ]), 'ia'),
                    insight('💡', html.Span([
                        html.Strong(f'Spearman ({SPEARMAN:.3f}) cao hơn Pearson (~0.66)'),
                        ' → model đúng thứ tự xếp hạng tốt hơn đúng con số '
                        'tuyệt đối — phù hợp với use case benchmarking ở Section 3.',
                    ]), 'ie'),
                ], className='p3-card', style={
                    'flex': '0.75', 'minWidth': '230px',
                    'borderLeft': f'2px solid {VIOLET}',
                }),
            ], className='p3-row'),

            html.Div(className='p3-divider'),

            # ══════════════════════════════════════════════════
            #  S2 — YẾU TỐ ẢNH HƯỞNG
            # ══════════════════════════════════════════════════
            sec('SECTION 02',
                'Yếu tố ảnh hưởng đến lượt bán (Permutation Importance · n_repeats=10)',
                'cml2'),

            html.Div([
                card('PERMUTATION IMPORTANCE',
                     html.Div([
                         html.P('● Biểu đồ cho thấy mức quan trọng của từng feature',
                                style={'fontSize': '10px', 'color': MUTED,
                                       'margin': '0 0 8px 0'}),
                         dcc.Graph(figure=make_feature_importance(), config=cfg),
                     ]),
                     flex='1.4', min_w='360px', glow='g-amber'),
                card('ẢNH HƯỞNG CỦA GIÁ (PDP)',
                     html.Div([
                         html.P('● Cho các feature khác bằng mean, chỉ đổi giá → model dự đoán ra sao',
                                style={'fontSize': '10px', 'color': MUTED,
                                       'margin': '0 0 8px 0'}),
                         dcc.Graph(figure=make_pdp_price(), config=cfg),
                     ]),
                     flex='1', min_w='300px', glow='g-amber'),
            ], className='p3-row'),

            # S2 insight strip
            html.Div([
                insight('⭐', html.Span([
                    html.Strong(f'Rating là yếu tố mạnh nhất'), ' — '
                    f'Permutation importance {FI.iloc[0]["importance_mean"]:.3f} '
                    f'(gấp ~{FI.iloc[0]["importance_mean"] / max(FI.iloc[1]["importance_mean"], 1e-6):.0f}× feature '
                    f'thứ 2). Sản phẩm có rating cao là tín hiệu mua hàng rõ nhất.',
                ]), 'ia'),
                insight('🌏', html.Span([
                    html.Strong('Xuất xứ chỉ xếp hạng giữa bảng'), ' — '
                    f'importance ~'
                    f'{FI[FI["feature"]=="origin_class_corrected"]["importance_mean"].values[0]:.3f}, '
                    'nhỏ hơn nhiều so với rating & giá. Khi kiểm soát các yếu tố khác, '
                    '"nội/ngoại" không phải là yếu tố quyết định lượt bán.',
                ]), 'ic'),
            ], style={'marginBottom': '14px'}),

            html.Div(className='p3-divider'),

            # ══════════════════════════════════════════════════
            #  S3 — DỰ ĐOÁN TƯƠNG TÁC
            # ══════════════════════════════════════════════════
            sec('SECTION 03', 'Dự đoán tương tác · What-if analysis', 'cml3'),

            # Hidden store for auto-fill snapshot
            dcc.Store(id='ml-autofill-snapshot', data=None),

            html.Div([
                card('NHẬP THÔNG TIN SẢN PHẨM GIẢ ĐỊNH',
                     whatif_form(),
                     flex='1', min_w='480px', glow='g-emerald'),
            ], className='p3-row'),

            # Methodology disclaimer (collapsible)
            html.Details([
                html.Summary('⚠️ Methodology Note — Giới hạn & cách diễn giải'),
                html.Div([
                    html.P([
                        html.Strong('Target variable sold_count'),
                        ' là doanh số tích lũy vòng đời (lifetime accumulated), '
                        'không phải chu kỳ. Do đó mô hình được định hướng như ',
                        html.Strong('BENCHMARKING TOOL'),
                        ' (chấm điểm vị thế), KHÔNG PHẢI forecasting tool.',
                    ]),
                    html.P(html.Strong('Giới hạn đã biết:'), style={'marginTop': '8px'}),
                    html.Ul([
                        html.Li('Cold-start với SP pre-launch chưa có rating'),
                        html.Li('Heavy-tail outliers (top 1%) — model undersize ở đuôi phân phối'),
                        html.Li('Peer group theo product_type (5 nhóm) — trade tin cậy lấy granularity'),
                    ]),
                    html.P([
                        html.Strong('Metrics phù hợp: '),
                        f'Spearman {SPEARMAN:.3f} (ranking) > Pearson ~0.66 (absolute).',
                    ]),
                ]),
            ], className='ml-methodology-card'),

            html.Div(className='p3-divider'),

            # ══════════════════════════════════════════════════
            #  S4 — PHÂN TÍCH PHẦN DƯ
            # ══════════════════════════════════════════════════
            sec('SECTION 04', 'Phân tích phần dư · Model có bỏ sót pattern nào không?',
                'cml4'),

            html.Div([
                card('RESIDUAL vs PREDICTED',
                     html.Div([
                         html.P('● Ô càng sáng = càng nhiều sản phẩm · '
                                'Đường amber = y=0 (không bias) · '
                                'Đường rose = mean residual theo bin ±1σ',
                                style={'fontSize': '10px', 'color': MUTED,
                                       'margin': '0 0 8px 0'}),
                         dcc.Graph(figure=make_residual_plot(), config=cfg),
                     ]),
                     flex='1.4', min_w='380px', glow='g-rose'),

                html.Div([
                    html.Div([
                        mini_stat(
                            f'{np.mean(PREDS["residual_log"].values):+.3f}',
                            'Mean residual', EMERALD
                            if abs(np.mean(PREDS["residual_log"].values)) < 0.05
                            else AMBER),
                        mini_stat(
                            f'{np.std(PREDS["residual_log"].values):.3f}',
                            'Std residual', VIOLET),
                    ], className='p3-mini-stats'),
                    html.Div(style={'height': '12px'}),
                    insight('📊', html.Span([
                        html.Strong('Mean residual ≈ 0'), ' — '
                        'model không bị thiên lệch dự đoán quá cao/thấp hệ thống.',
                    ]), 'ie'),
                    insight('⚠️', html.Span([
                        html.Strong('Std residual ~0.8 log unit'), ' — '
                        'tương đương sai số ×e^0.8 ≈ 2.2× lượt bán thực tế ở phân khúc '
                        'sold_count > 10. Model mạnh ở xu hướng chung, yếu ở bestseller cá biệt.',
                    ]), 'ia'),
                ], className='p3-card', style={
                    'flex': '1', 'minWidth': '260px',
                    'borderLeft': f'2px solid {ROSE}',
                }),
            ], className='p3-row'),

        ], style={'padding': '16px 20px 8px'}),

        # ── FOOTER ──────────────────────────────────────────
        html.Div([
            html.Span('🧠 Model: ' + best),
            html.Span(f'Target: log1p(sold_count) · {len(FI)} features'),
            html.Span('Nhóm 05 · FIT-HCMUS'),
        ], className='p3-footer', style={'margin': '0 20px'}),

    ], className='p3-page', style={'padding': '0'})


# ══════════════════════════════════════════════════════════════
#  CALLBACK — WHAT-IF PREDICTION (4-BLOCK OUTPUT)
# ══════════════════════════════════════════════════════════════

@callback(
    Output('ml-predict-output', 'children'),
    Input('ml-predict-btn', 'n_clicks'),
    State('ml-input-origin', 'value'),
    State('ml-input-ptype', 'value'),
    State('ml-input-brand', 'value'),
    State('ml-input-price', 'value'),
    State('ml-input-discount', 'value'),
    State('ml-input-rating', 'value'),
    State('ml-input-verified', 'value'),
    State('ml-input-official', 'value'),
    State('ml-input-authentic', 'value'),
    State('ml-autofill-snapshot', 'data'),
    prevent_initial_call=True,
)
def on_predict(n, origin, ptype, brand, price, discount, rating,
               verified, official, authentic, snapshot):
    if not ML_READY or n is None or n == 0:
        return no_update

    try:
        # Tính các feature engineered
        log_price = float(np.log1p(price or 0))
        has_rating = 1 if (rating or 0) > 0 else 0

        # Xác định price_segment từ price
        p = price or 0
        if p < 100_000:
            p_seg = 'Dưới 100k'
        elif p < 300_000:
            p_seg = '100k – 300k'
        elif p < 700_000:
            p_seg = '300k – 700k'
        elif p < 2_000_000:
            p_seg = '700k – 2tr'
        else:
            p_seg = 'Trên 2tr'

        # Đảm bảo brand hợp lệ
        valid_brands = FEAT_META['top_brands'] + ['Other']
        brand_clean = brand if brand in valid_brands else 'Other'

        X_input = pd.DataFrame([{
            'log_price': log_price,
            'discount_rate': float(discount or 0),
            'rating': float(rating or 0),
            'has_rating': has_rating,
            'is_official_store': int(official or 0),
            'tiki_verified': int(verified or 0),
            'has_authentic_badge': int(authentic or 0),
            'origin_class_corrected': origin,
            'product_type': ptype,
            'price_segment': p_seg,
            'brand_grouped': brand_clean,
        }])

        pred_log = float(MODEL.predict(X_input)[0])
        pred_orig = max(0, float(np.expm1(pred_log)))

        # Confidence interval thô: ± 1 std residual
        std_resid = float(np.std(PREDS['residual_log'].values))
        low = max(0, float(np.expm1(pred_log - std_resid)))
        high = float(np.expm1(pred_log + std_resid))

        # ── Determine actual sold (conditional on snapshot match) ──
        actual_sold = None
        modified_note = None
        if snapshot is not None:
            # Check if user has modified any field vs snapshot
            current_vals = {
                'origin': origin, 'ptype': ptype, 'brand': brand,
                'price': price, 'discount': discount, 'rating': rating,
                'verified': verified, 'official': official, 'authentic': authentic,
            }
            snap_vals = {
                'origin': snapshot.get('origin'), 'ptype': snapshot.get('ptype'),
                'brand': snapshot.get('brand'), 'price': snapshot.get('price'),
                'discount': snapshot.get('discount'), 'rating': snapshot.get('rating'),
                'verified': snapshot.get('verified'), 'official': snapshot.get('official'),
                'authentic': snapshot.get('authentic'),
            }
            if current_vals == snap_vals:
                actual_sold = snapshot.get('actual_sold')
            else:
                modified_note = "Đã chỉnh sửa thông số gốc → không còn so sánh với thực tế"

        # ── Build 4-block result grid ──
        block_a = _block_prediction(pred_orig, low, high)
        block_b = _block_percentile(pred_orig, ptype, actual=actual_sold)
        block_c = _block_burn(discount or 0)
        block_d = _block_revenue(pred_orig, price or 0, ptype)

        result_children = [
            html.Div([block_a, block_b, block_c, block_d],
                     className='ml-result-grid'),
        ]

        # Show modification note if applicable
        if modified_note:
            result_children.append(
                html.P(f'ℹ️ {modified_note}',
                       style={'fontSize': '11px', 'color': AMBER,
                              'fontStyle': 'italic', 'marginTop': '10px',
                              'textAlign': 'center'})
            )

        return html.Div(result_children)

    except Exception as e:
        return html.Div(f'⚠️ Lỗi dự đoán: {e}',
                        style={'color': ROSE, 'fontSize': '12px', 'padding': '14px',
                               'background': hex_to_rgba(ROSE, 0.08),
                               'borderRadius': '8px',
                               'border': f'1px solid {hex_to_rgba(ROSE, 0.3)}'})


# ══════════════════════════════════════════════════════════════
#  CALLBACK — AUTO-FILL TỪ CSDL (with snapshot)
# ══════════════════════════════════════════════════════════════

@callback(
    Output('ml-input-origin', 'value'),
    Output('ml-input-ptype', 'value'),
    Output('ml-input-brand', 'value'),
    Output('ml-input-price', 'value'),
    Output('ml-input-discount', 'value'),
    Output('ml-input-rating', 'value'),
    Output('ml-input-verified', 'value'),
    Output('ml-input-official', 'value'),
    Output('ml-input-authentic', 'value'),
    Output('ml-autofill-snapshot', 'data'),
    Input('ml-select-product', 'value'),
    prevent_initial_call=True,
)
def on_product_select(selected_idx):
    _no = no_update
    if selected_idx is None:
        return _no, _no, _no, _no, _no, _no, _no, _no, _no, None

    try:
        row = df_full.loc[selected_idx]

        origin = row['origin_class_corrected']
        ptype = row['product_type']

        # Map brand
        valid_brands = FEAT_META['top_brands'] + ['Other']
        brand = row['brand_name'] if row['brand_name'] in valid_brands else 'Other'

        price = float(row['price'])
        discount = float(row['discount_rate'])
        rating = float(row['rating'])

        verified = int(row['tiki_verified'])
        official = int(row['is_official_store'])
        authentic = int(row['has_authentic_badge'])

        # Save snapshot for comparison in on_predict
        snapshot = {
            'product_id': int(selected_idx),
            'origin': origin, 'ptype': ptype, 'brand': brand,
            'price': price, 'discount': discount, 'rating': rating,
            'verified': verified, 'official': official, 'authentic': authentic,
            'actual_sold': int(row['sold_count']),
        }

        return origin, ptype, brand, price, discount, rating, verified, official, authentic, snapshot
    except Exception as e:
        print(f"Lỗi Auto-fill: {e}")
        return _no, _no, _no, _no, _no, _no, _no, _no, _no, _no


