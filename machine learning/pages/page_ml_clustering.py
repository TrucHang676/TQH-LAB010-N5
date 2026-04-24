# pages/page_ml_clustering.py
# ╔══════════════════════════════════════════════════════════════╗
# ║  Trang ML — Phân khúc thị trường (K-Means Clustering)        ║
# ║  Dark navy theme · Đồng bộ với page_ml_regression.py         ║
# ║  S1: Chọn K tối ưu (Elbow + Silhouette + Davies-Bouldin)     ║
# ║  S2: Bản đồ phân khúc (PCA 2D) · Phân bố VN/NK per cluster   ║
# ║  S3: Profile từng cluster (Heatmap giá trị gốc + normalized) ║
# ║  S4: Chiến lược gợi ý cho từng phân khúc                     ║
# ║  S5: Cross với Model 1 (R² regression trong từng cluster)    ║
# ╚══════════════════════════════════════════════════════════════╝

import dash
from dash import html, dcc, callback, Input, Output, State, ALL, ctx
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

dash.register_page(
    __name__,
    path='/market-segmentation',
    name='Market Segmentation',
    title='Phân khúc thị trường | Mỹ phẩm Tiki',
    order=6,
)

# ── Load model artifacts ─────────────────────────────────────
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ML_DIR = os.path.abspath(os.path.join(_BASE_DIR, '..', 'ml_models'))


def _load_json(name):
    p = os.path.join(ML_DIR, name)
    if not os.path.exists(p):
        return None
    with open(p, encoding='utf-8') as f:
        return json.load(f)

def _load_csv(name, **kw):
    p = os.path.join(ML_DIR, name)
    return pd.read_csv(p, **kw) if os.path.exists(p) else None

def _safe_load():
    try:
        metrics = _load_json('model2_metrics.json')
        profiles = _load_json('model2_profile.json')
        cross = _load_json('model2_cross_regression.json')
        labels = _load_csv('model2_cluster_labels.csv')
        ks = _load_csv('model2_elbow_silhouette.csv')
        samples = _load_csv('model2_cluster_samples.csv')
        corr_raw = _load_csv('model2_corr_raw.csv', index_col=0)
        corr_filtered = _load_csv('model2_corr_filtered.csv', index_col=0)
        corr_insights = _load_json('model2_corr_insights.json')
        fq = _load_json('model2_feature_quality.json')
        dormant = _load_json('model2_dormant_stats.json')
        baseline = _load_json('model2_baseline_comparison.json')
        return (metrics, profiles, cross, labels, ks, samples,
                corr_raw, corr_filtered, corr_insights, fq, dormant, baseline)
    except Exception as e:
        print(f'[page_ml_clustering] Chưa load được artifact: {e}')
        return tuple([None] * 12)

(METRICS, PROFILES, CROSS, LABELS, KS, SAMPLES,
 CORR_RAW, CORR_FILTERED, CORR_INSIGHTS,
 FEATURE_QUALITY, DORMANT_STATS, BASELINE_CMP) = _safe_load()
ML_READY = METRICS is not None

# ── Dark palette (đồng bộ với page_ml_regression) ────────────
BG       = '#172542'
SURFACE  = '#1A2A4D'
CARD     = '#1C2D55'
BORDER   = 'rgba(255,255,255,0.07)'
BORDER2  = 'rgba(255,255,255,0.13)'
GRID     = 'rgba(255,255,255,0.05)'
TXT      = '#F0F6FF'
SUBTXT   = '#94A3B8'
MUTED    = '#4B6178'

VIOLET   = '#A78BFA'
INDIGO   = '#818CF8'
AMBER    = '#FBBF24'
EMERALD  = '#34D399'
ROSE     = '#FB7185'
CYAN     = '#22D3EE'
BLUE     = '#60A5FA'
ORANGE   = '#FB923C'

C_DOM    = '#38BDF8'   # nội địa (VN)
C_IMP    = '#F87171'   # nhập khẩu (NK)

# Palette màu cho các cluster (tối đa 6 cluster)
CLUSTER_COLORS = [VIOLET, EMERALD, AMBER, ROSE, CYAN, INDIGO]


# ══════════════════════════════════════════════════════════════
#  CHART THEME HELPER
# ══════════════════════════════════════════════════════════════
def hex_to_rgba(hex_color, alpha=1):
    hex_color = hex_color.lstrip('#')
    return f'rgba({int(hex_color[0:2],16)},{int(hex_color[2:4],16)},{int(hex_color[4:6],16)},{alpha})'


def hex_to_rgb_str(hex_color):
    h = hex_color.lstrip('#')
    return f'{int(h[0:2],16)},{int(h[2:4],16)},{int(h[4:6],16)}'


def _theme(fig, height=320, **kw):
    margin = kw.pop('margin', dict(l=14, r=14, t=28, b=14))
    fig.update_layout(
        height=height,
        font=dict(family="'Space Grotesk','Segoe UI',sans-serif",
                  size=12, color=TXT),
        paper_bgcolor=CARD, plot_bgcolor=CARD, margin=margin,
        hoverlabel=dict(bgcolor=SURFACE, bordercolor=BORDER2,
                        font=dict(size=12, color=TXT)),
        **kw,
    )
    fig.update_xaxes(gridcolor=GRID, gridwidth=1,
                     linecolor='rgba(255,255,255,0.06)', linewidth=1,
                     tickfont=dict(size=10, color=SUBTXT),
                     title_font=dict(size=11, color=SUBTXT), zeroline=False)
    fig.update_yaxes(gridcolor=GRID, gridwidth=1,
                     linecolor='rgba(0,0,0,0)',
                     tickfont=dict(size=10, color=SUBTXT),
                     title_font=dict(size=11, color=SUBTXT), zeroline=False)
    return fig


def _leg(**kw):
    return dict(orientation='h', yanchor='bottom', y=1.03,
                xanchor='right', x=1, font=dict(size=10, color=SUBTXT),
                bgcolor='rgba(0,0,0,0)', **kw)


def _wrap_cluster_label(s):
    """Xuống dòng theo dấu phân tách để hiển thị đủ tên cluster."""
    return s.replace(' · ', ' ·<br>')


def _module_switcher(active='module2'):
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
#  FIGURES — S1: ELBOW + SILHOUETTE
# ══════════════════════════════════════════════════════════════

def make_elbow_silhouette():
    """2 trục: inertia (elbow) bên trái, silhouette bên phải."""
    if KS is None:
        return go.Figure()

    fig = go.Figure()

    # Inertia line (Elbow) — primary y
    fig.add_trace(go.Scatter(
        x=KS['k'], y=KS['inertia'], mode='lines+markers',
        name='Inertia (Elbow)',
        line=dict(color=VIOLET, width=2.5),
        marker=dict(size=8, color=VIOLET,
                    line=dict(color='rgba(255,255,255,0.3)', width=1)),
        yaxis='y',
        hovertemplate='K=%{x} · Inertia=%{y:,.0f}<extra></extra>',
    ))

    # Silhouette line — secondary y
    fig.add_trace(go.Scatter(
        x=KS['k'], y=KS['silhouette'], mode='lines+markers',
        name='Silhouette',
        line=dict(color=AMBER, width=2.5),
        marker=dict(size=8, color=AMBER,
                    line=dict(color='rgba(255,255,255,0.3)', width=1)),
        yaxis='y2',
        hovertemplate='K=%{x} · Silhouette=%{y:.3f}<extra></extra>',
    ))

    # Davies-Bouldin line — secondary y
    fig.add_trace(go.Scatter(
        x=KS['k'], y=KS['davies_bouldin'], mode='lines+markers',
        name='Davies-Bouldin',
        line=dict(color=ROSE, width=2, dash='dot'),
        marker=dict(size=7, color=ROSE),
        yaxis='y2',
        hovertemplate='K=%{x} · DB=%{y:.3f}<extra></extra>',
    ))

    # Chosen K vertical line
    chosen_k = METRICS.get('best_k', 5) if METRICS else 5
    # vẽ bằng line 2 điểm thay vì add_vline để tránh lỗi range
    fig.add_shape(type='line',
                  x0=chosen_k, x1=chosen_k,
                  y0=0, y1=KS['inertia'].max(),
                  line=dict(color=EMERALD, width=2, dash='dash'),
                  yref='y')
    fig.add_annotation(x=chosen_k, y=KS['inertia'].max(),
                       text=f'K*={chosen_k}', showarrow=False,
                       yanchor='bottom', font=dict(color=EMERALD, size=11),
                       bgcolor='rgba(0,0,0,0)')

    _theme(fig, height=320, showlegend=True, legend=_leg(),
           xaxis=dict(title='Số cluster K', dtick=1),
           yaxis=dict(title=dict(text='Inertia (Elbow)',
                                 font=dict(color=VIOLET)),
                      tickfont=dict(color=VIOLET)),
           yaxis2=dict(title=dict(text='Silhouette / DB',
                                  font=dict(color=AMBER)),
                       tickfont=dict(color=AMBER),
                       overlaying='y', side='right',
                       gridcolor='rgba(0,0,0,0)'))
    return fig


# ══════════════════════════════════════════════════════════════
#  FIGURES — S2: PCA 2D SCATTER + VN/NK STACK
# ══════════════════════════════════════════════════════════════

def make_pca_scatter():
    """Scatter 2D theo PCA, màu theo cluster.

    Lý do jitter + giảm size: 6 features có nhiều giá trị rời rạc
    (tiki_verified binary, rating 11 mức, discount_rate có repeat)
    → PCA tổ hợp tuyến tính kéo dữ liệu thành các "rails" / đường
    chéo song song. Đường chéo là cấu trúc thật nhưng gây rối mắt.
    Thêm jitter Gaussian nhẹ (~4% std) chỉ lúc render để mờ rails,
    giữ nguyên hình dạng cluster. Hover vẫn xem được tên/brand/giá
    vì customdata dùng giá trị gốc, không dùng toạ độ jittered.
    """
    if LABELS is None:
        return go.Figure()

    # Downsample để render nhanh
    n_show = min(3000, len(LABELS))
    sample = (LABELS.sample(n=n_show, random_state=42)
              if len(LABELS) > n_show else LABELS)

    # Jitter scale tính theo std toàn bộ (ổn định giữa các lần render)
    jx_scale = 0.04 * LABELS['pca_x'].std()
    jy_scale = 0.04 * LABELS['pca_y'].std()
    rng = np.random.default_rng(42)

    fig = go.Figure()
    n_clusters = LABELS['cluster'].max() + 1
    for c in range(n_clusters):
        sub = sample[sample['cluster'] == c]
        prof_name = (PROFILES[c]['name']
                     if PROFILES and c < len(PROFILES) else f'Cluster {c}')

        # ── Jitter chỉ cho toạ độ hiển thị ──────────────────────
        x_plot = sub['pca_x'].values + rng.normal(0, jx_scale, size=len(sub))
        y_plot = sub['pca_y'].values + rng.normal(0, jy_scale, size=len(sub))

        fig.add_trace(go.Scatter(
            x=x_plot, y=y_plot,
            mode='markers',
            name=f'[{c + 1}] {prof_name}',
            marker=dict(
                size=4, color=CLUSTER_COLORS[c % len(CLUSTER_COLORS)],
                opacity=0.45,
                line=dict(color='rgba(255,255,255,0.12)', width=0.3),
            ),
            hovertemplate=(
                '<b>%{customdata[0]}</b><br>'
                'Brand: %{customdata[1]}<br>'
                'Giá: %{customdata[2]:,.0f}đ · Rating: %{customdata[3]:.1f}<br>'
                'Đã bán: %{customdata[4]:,} · Xuất xứ: %{customdata[5]}'
                '<extra></extra>'
            ),
            customdata=np.stack([
                sub['name_clean'].str[:50].values,
                sub['brand_name'].values,
                sub['price'].values,
                sub['rating'].values,
                sub['sold_count'].values,
                sub['origin_class_corrected'].values,
            ], axis=-1),
        ))

    _theme(fig, height=400, showlegend=True,
           legend=dict(orientation='v', yanchor='top', y=0.98,
                       xanchor='left', x=1.01,
                       font=dict(size=9, color=TXT),
                       bgcolor='rgba(26,42,77,0.78)',
                       bordercolor=BORDER2, borderwidth=1),
           margin=dict(l=14, r=280, t=28, b=14),
           xaxis=dict(title='PC1'), yaxis=dict(title='PC2'))
    return fig


def make_vn_nk_stack():
    """Stacked bar: % VN vs % NK trong mỗi cluster."""
    if PROFILES is None:
        return go.Figure()

    names = [f"[{p['cluster_id'] + 1}] {p['name']}" for p in PROFILES]
    names_wrapped = [_wrap_cluster_label(n) for n in names]
    pct_vn = [p['pct_vn'] for p in PROFILES]
    pct_nk = [p['pct_nk'] for p in PROFILES]
    sizes = [p['size'] for p in PROFILES]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=names_wrapped, x=pct_vn, orientation='h', name='VN (nội địa)',
        marker=dict(color=C_DOM, line=dict(color='rgba(255,255,255,0.15)', width=0.5)),
        text=[f'{v:.0f}%' for v in pct_vn], textposition='inside',
        textfont=dict(color='white', size=11),
        hovertemplate='VN: %{x:.1f}%<extra></extra>',
    ))
    fig.add_trace(go.Bar(
        y=names_wrapped, x=pct_nk, orientation='h', name='NK (nhập khẩu)',
        marker=dict(color=C_IMP, line=dict(color='rgba(255,255,255,0.15)', width=0.5)),
        text=[f'{v:.0f}%' for v in pct_nk], textposition='inside',
        textfont=dict(color='white', size=11),
        hovertemplate='NK: %{x:.1f}%<extra></extra>',
    ))

    _theme(fig, height=max(300, 78 * len(names_wrapped) + 50),
           barmode='stack', showlegend=True, legend=_leg(),
           xaxis=dict(title='Tỷ lệ (%)', range=[0, 100]),
           yaxis=dict(title='', autorange='reversed', automargin=True),
           margin=dict(l=165, r=24, t=28, b=20))
    return fig


# ══════════════════════════════════════════════════════════════
#  FIGURES — S3: PROFILE HEATMAP (cluster profile)
# ══════════════════════════════════════════════════════════════
#  Ghi chú: đổi từ radar → heatmap vì radar bị collapse khi có
#  cluster "dead products" (rating=0, sold=0, review=0). Radar
#  polygon thành wedge sliver ở 1-2 góc, che mất insight chính.
#  Heatmap hiển thị cả màu (normalized) lẫn giá trị gốc trong ô
#  → đọc trực tiếp mọi con số, không cần đoán.

def make_profile_heatmap():
    """Heatmap 5 cluster × 5 feature — thay radar tránh polygon collapse.

    - Cell color = min-max normalized (so sánh tương đối giữa cluster).
    - Text trong ô = giá trị gốc (format phù hợp từng feature).
    - Cluster 0/1 "dead" sẽ lộ ra thành vùng tối ở 4 cột giữa,
      tương phản rõ với cluster 2 (bán chạy) sáng hẳn.
    """
    if PROFILES is None:
        return go.Figure()

    # ── Định nghĩa 5 feature + format hiển thị giá trị gốc ─────
    feat_specs = [
        ('median_price',       'Giá (median)',       lambda v: f'{v/1000:.0f}K'),
        ('mean_rating',        'Rating TB',          lambda v: f'{v:.2f}'),
        ('median_sold_count',  'Lượt bán (median)',  lambda v: f'{v:,.0f}'),
        ('mean_discount_rate', 'Discount %',         lambda v: f'{v:.1f}%'),
        ('pct_tiki_verified',  '% Tiki Verified',    lambda v: f'{v:.0f}%'),
    ]
    feat_keys = [s[0] for s in feat_specs]
    feat_labels = [s[1] for s in feat_specs]
    fmt_fns = [s[2] for s in feat_specs]

    cluster_labels = [f'[{p["cluster_id"] + 1}] {p["name"]}' for p in PROFILES]
    cluster_labels_wrapped = [_wrap_cluster_label(lbl) for lbl in cluster_labels]

    # Raw values: shape (n_cluster, n_feat)
    raw_mat = np.array([[p[k] for k in feat_keys] for p in PROFILES],
                       dtype=float)

    # Min-max normalize theo từng cột (feature) để vẽ màu
    col_min = raw_mat.min(axis=0)
    col_max = raw_mat.max(axis=0)
    rng = np.where((col_max - col_min) < 1e-9, 1.0, col_max - col_min)
    norm_mat = (raw_mat - col_min) / rng

    # Text overlay: giá trị gốc, format theo feature
    text_mat = [[fmt_fns[j](raw_mat[i, j]) for j in range(len(feat_keys))]
                for i in range(len(PROFILES))]

    fig = go.Figure(data=go.Heatmap(
        z=norm_mat,
        x=feat_labels,
        y=cluster_labels_wrapped,
        text=text_mat,
        texttemplate='%{text}',
        textfont=dict(size=12, color='white',
                      family="'Space Grotesk','Segoe UI',sans-serif"),
        colorscale=[
            [0.0, hex_to_rgba(INDIGO, 0.35)],
            [0.3, hex_to_rgba(VIOLET, 0.70)],
            [0.6, hex_to_rgba(AMBER, 0.85)],
            [1.0, hex_to_rgba(EMERALD, 1.0)],
        ],
        colorbar=dict(
            title=dict(text='Normalized<br>(0-1)',
                       font=dict(size=9, color=SUBTXT)),
            tickfont=dict(size=9, color=SUBTXT),
            thickness=10, len=0.75,
            bgcolor='rgba(0,0,0,0)', outlinecolor='rgba(0,0,0,0)',
        ),
        hovertemplate='<b>%{y}</b><br>%{x}: %{text}<br>'
                      'Normalized: %{z:.2f}<extra></extra>',
        zmin=0, zmax=1,
        xgap=3, ygap=3,           # khe hở giữa các ô cho gọn
    ))

    _theme(fig, height=max(340, 72 * len(cluster_labels_wrapped) + 70),
           margin=dict(l=210, r=40, t=50, b=20),
           showlegend=False,
           xaxis=dict(side='top', tickfont=dict(size=11, color=TXT),
                 tickangle=0, showgrid=False),
           yaxis=dict(tickfont=dict(size=11, color=TXT),
                      autorange='reversed', showgrid=False,
                      automargin=True))
    return fig


def make_cross_regression_bar():
    """Bar: R² của Model 1 trong từng cluster."""
    if CROSS is None or PROFILES is None:
        return go.Figure()

    valid = [r for r in CROSS if r.get('r2') is not None]
    names = []
    r2s = []
    colors = []
    for r in valid:
        cid = r['cluster_id']
        names.append(_wrap_cluster_label(f'[{cid + 1}] {PROFILES[cid]["name"]}'))
        r2s.append(r['r2'])
        colors.append(CLUSTER_COLORS[cid % len(CLUSTER_COLORS)])

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=names, x=r2s, orientation='h',
        marker=dict(color=colors,
                    line=dict(color='rgba(255,255,255,0.2)', width=0.5)),
        text=[f'{v:+.3f}' for v in r2s], textposition='outside',
        textfont=dict(color=TXT, size=11),
        hovertemplate='R² (log) = %{x:+.3f}<extra></extra>',
    ))
    _theme(fig, height=max(280, 72 * len(names) + 50),
           showlegend=False,
           xaxis=dict(title='R² của Model 1 (log_sold_count)',
                      range=[min(0, min(r2s) - 0.05), max(1, max(r2s) + 0.1)]),
           yaxis=dict(title='', autorange='reversed', automargin=True),
           margin=dict(l=220, r=28, t=28, b=20))
    return fig
# ══════════════════════════════════════════════════════════════
#  NEW — Section 1: Feature Analysis Charts
# ══════════════════════════════════════════════════════════════

def make_problematic_features_chart():
    """3 mini bar charts: rating, sold_count, verified distribution."""
    if FEATURE_QUALITY is None:
        return _empty_fig()
    fq = FEATURE_QUALITY
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=(
            f"<b>Rating</b><br><span style='color:#FB7185;font-size:11px'>"
            f"{fq['rating']['pct_zero']}% = 0</span>",
            f"<b>Sold Count</b><br><span style='color:#FB7185;font-size:11px'>"
            f"{fq['sold_count']['pct_zero']}% = 0</span>",
            f"<b>Tiki Verified</b><br><span style='color:#FBBF24;font-size:11px'>"
            f"{fq['tiki_verified']['pct_true']}% TRUE</span>",
        ),
        horizontal_spacing=0.08,
    )
    fig.add_trace(go.Bar(
        x=['= 0', '> 0'], y=[fq['rating']['pct_zero'], fq['rating']['pct_rated']],
        marker=dict(color=[ROSE, EMERALD]),
        text=[f"{fq['rating']['pct_zero']}%", f"{fq['rating']['pct_rated']}%"],
        textposition='outside', showlegend=False,
    ), row=1, col=1)
    fig.add_trace(go.Bar(
        x=['= 0', '> 0'], y=[fq['sold_count']['pct_zero'], fq['sold_count']['pct_active']],
        marker=dict(color=[ROSE, EMERALD]),
        text=[f"{fq['sold_count']['pct_zero']}%", f"{fq['sold_count']['pct_active']}%"],
        textposition='outside', showlegend=False,
    ), row=1, col=2)
    fig.add_trace(go.Bar(
        x=['Yes', 'No'],
        y=[fq['tiki_verified']['pct_true'], 100 - fq['tiki_verified']['pct_true']],
        marker=dict(color=[AMBER, MUTED]),
        text=[f"{fq['tiki_verified']['pct_true']}%",
              f"{100-fq['tiki_verified']['pct_true']:.1f}%"],
        textposition='outside', showlegend=False,
    ), row=1, col=3)
    fig.update_yaxes(range=[0, 110], title_text='%', tickfont=dict(size=10))
    return _theme(fig, height=280, margin=dict(l=40, r=20, t=60, b=40))


def make_corr_heatmap(corr_df):
    """Correlation heatmap for any corr matrix."""
    if corr_df is None:
        return _empty_fig()
    lm = {'log_price': 'Giá (log)', 'rating': 'Rating', 'review_ratio': 'Review ratio',
          'discount_rate_norm': 'Discount', 'log_sold_count': 'Sold (log)',
          'tiki_verified': 'Verified'}
    cols = [c for c in corr_df.columns if c in lm]
    labels = [lm[c] for c in cols]
    sub = corr_df.loc[cols, cols]
    fig = go.Figure(go.Heatmap(
        z=sub.values, x=labels, y=labels,
        colorscale='RdBu_r', zmin=-1, zmax=1,
        text=sub.values, texttemplate='%{text:.2f}',
        textfont={'size': 11}, xgap=2, ygap=2,
        colorbar=dict(title=dict(text='r', font=dict(color=TXT, size=10)),
                      tickfont=dict(color=TXT, size=10), thickness=12, len=0.7),
    ))
    return _theme(fig, height=280, margin=dict(l=80, r=20, t=20, b=70))


def build_decision_card():
    """Card explaining the filtering decision."""
    if FEATURE_QUALITY is None or CORR_INSIGHTS is None:
        return html.Div()
    fq = FEATURE_QUALITY
    raw_top = CORR_INSIGHTS['raw']['top_pairs'][0]
    lm = {'log_price': 'Giá', 'rating': 'Rating', 'review_ratio': 'Review ratio',
          'discount_rate_norm': 'Discount', 'log_sold_count': 'Sold', 'tiki_verified': 'Verified'}
    cs = {'background': 'rgba(251,113,133,0.15)', 'padding': '2px 6px',
          'borderRadius': '4px', 'color': ROSE}
    return html.Div([
        html.Div([
            html.Span('⚠️', style={'fontSize': '20px', 'marginRight': '8px'}),
            html.Strong('VẤN ĐỀ PHÁT HIỆN', style={'color': ROSE, 'fontSize': '12px',
                                                     'letterSpacing': '0.08em'}),
        ], style={'marginBottom': '10px'}),
        html.Ul([
            html.Li([html.Code('rating = 0', style=cs),
                     f" — {fq['rating']['pct_zero']}% sản phẩm chưa có review → confounding signal"],
                    style={'fontSize': '12px', 'marginBottom': '6px', 'color': TXT}),
            html.Li([f"{fq['sold_count']['pct_zero']}% sản phẩm có ",
                     html.Code('sold = 0', style=cs), " → review_ratio luôn = 0"],
                    style={'fontSize': '12px', 'marginBottom': '6px', 'color': TXT}),
            html.Li([f"Multicollinearity: ",
                     html.Code(f"{lm.get(raw_top['f1'],raw_top['f1'])} ↔ "
                               f"{lm.get(raw_top['f2'],raw_top['f2'])}", style=cs),
                     f" r = {raw_top['r']:+.2f}"],
                    style={'fontSize': '12px', 'marginBottom': '6px', 'color': TXT}),
        ], style={'paddingLeft': '20px', 'margin': '0 0 16px 0'}),
        html.Div([
            html.Span('🎯', style={'fontSize': '20px', 'marginRight': '8px'}),
            html.Strong('QUYẾT ĐỊNH: LỌC ACTIVE + RATED',
                       style={'color': EMERALD, 'fontSize': '12px', 'letterSpacing': '0.08em'}),
        ], style={'marginBottom': '10px'}),
        html.Ul([
            html.Li(f"Cluster {fq['filtering_decision']['n_clustered']:,} sản phẩm "
                    f"({fq['filtering_decision']['pct_clustered']}% catalog)",
                   style={'fontSize': '12px', 'marginBottom': '6px', 'color': TXT}),
            html.Li(f"Loại bỏ confounding signal rating=0 và review_ratio",
                   style={'fontSize': '12px', 'marginBottom': '6px', 'color': TXT}),
            html.Li(f"Catalog bị loại ({fq['filtering_decision']['n_excluded']:,} SP) không đưa vào clustering",
                   style={'fontSize': '12px', 'marginBottom': '6px', 'color': TXT}),
        ], style={'paddingLeft': '20px', 'margin': '0 0 16px 0'}),
        html.Div([
            html.Span('⚖️', style={'fontSize': '14px', 'marginRight': '6px'}),
            html.Strong('Trade-off: ', style={'color': SUBTXT, 'fontSize': '11px'}),
            html.Span(f"Cluster trên {fq['filtering_decision']['pct_clustered']}% catalog "
                     "nhưng cluster có ý nghĩa thị trường, không phải artifact.",
                     style={'color': SUBTXT, 'fontSize': '11px'}),
        ], style={'background': 'rgba(255,255,255,0.03)', 'padding': '10px 12px',
                  'borderRadius': '8px', 'borderLeft': f'3px solid {AMBER}'}),
    ])


# ══════════════════════════════════════════════════════════════
#  NEW — Section 7: Baseline Comparison
# ══════════════════════════════════════════════════════════════

def make_baseline_comparison_chart():
    if BASELINE_CMP is None:
        return _empty_fig()
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            f"<b>Naive Groupby</b>: {BASELINE_CMP['baseline']['n_actual_groups']} nhóm (top 10)",
            f"<b>K-Means</b>: {BASELINE_CMP['kmeans']['n_groups']} cluster",
        ),
        column_widths=[0.6, 0.4], horizontal_spacing=0.12,
    )
    top = BASELINE_CMP['baseline']['top_groups']
    labels_b = [f"{r['price_segment'][:10]}·{r['popularity_tier'][:8]}·{r['rating_tier'][:8]}"
                for r in top]
    counts_b = [r['count'] for r in top]
    fig.add_trace(go.Bar(
        y=labels_b[::-1], x=counts_b[::-1], orientation='h',
        marker=dict(color=AMBER, opacity=0.7),
        text=[f'{c:,}' for c in counts_b[::-1]], textposition='outside',
        showlegend=False,
    ), row=1, col=1)
    names_k = BASELINE_CMP['kmeans']['names']
    sizes_k = BASELINE_CMP['kmeans']['sizes']
    fig.add_trace(go.Bar(
        y=[f"#{i+1} {n[:22]}" for i, n in enumerate(names_k)],
        x=sizes_k, orientation='h',
        marker=dict(color=CLUSTER_COLORS[:len(names_k)]),
        text=[f'{s:,}' for s in sizes_k], textposition='outside',
        showlegend=False,
    ), row=1, col=2)
    fig.update_xaxes(title_text='Số SP', row=1, col=1)
    fig.update_xaxes(title_text='Số SP', row=1, col=2)
    return _theme(fig, height=380, margin=dict(l=20, r=40, t=50, b=40))


def build_comparison_table():
    if BASELINE_CMP is None:
        return html.Div()
    b, k = BASELINE_CMP['baseline'], BASELINE_CMP['kmeans']
    np_ = BASELINE_CMP['n_possible_combinations']
    rows_data = [
        ('Số nhóm', f"{b['n_actual_groups']}/{np_}", f"{k['n_groups']}", 'kmeans'),
        ('Nhóm ≥ 100 SP', f"{b['pct_large']}%", '100%', 'kmeans'),
        ('Nhóm ≤ 2 SP', f"{b['pct_tiny']}%", '0%', 'kmeans'),
        ('CV (size)', f"{b['cv']}", f"{k['cv']}", 'kmeans'),
        ('Silhouette', '—', f"{k['silhouette']}", 'kmeans'),
        ('Ngưỡng động', '❌', '✅', 'kmeans'),
        ('Dễ giải thích', '✅', '⚠️', 'baseline'),
    ]
    ths = {'textAlign': 'left', 'padding': '10px 12px',
           'borderBottom': f'2px solid {BORDER2}', 'color': SUBTXT,
           'fontSize': '11px', 'textTransform': 'uppercase',
           'letterSpacing': '0.05em', 'fontWeight': 700}
    tds = lambda c=TXT, w=400: {'padding': '10px 12px',
           'borderBottom': f'1px solid {BORDER2}', 'color': c,
           'fontSize': '12px', 'fontWeight': w}
    body = []
    for criterion, bv, kv, winner in rows_data:
        badge = (html.Span('K-Means', style={'background': 'rgba(167,139,250,0.18)',
                 'color': VIOLET, 'padding': '3px 8px', 'borderRadius': '6px',
                 'fontSize': '11px', 'fontWeight': 700})
                 if winner == 'kmeans'
                 else html.Span('Baseline', style={'background': 'rgba(251,191,36,0.18)',
                 'color': AMBER, 'padding': '3px 8px', 'borderRadius': '6px',
                 'fontSize': '11px', 'fontWeight': 700}))
        body.append(html.Tr([
            html.Td(criterion, style=tds(w=600)),
            html.Td(bv, style=tds(c=SUBTXT)),
            html.Td(kv, style=tds()),
            html.Td(badge, style=tds()),
        ]))
    return html.Table([
        html.Thead(html.Tr([html.Th(h, style=ths) for h in
                            ['Tiêu chí', 'Naive Groupby', 'K-Means', '']])),
        html.Tbody(body),
    ], style={'width': '100%', 'borderCollapse': 'collapse'})


# ══════════════════════════════════════════════════════════════
#  NEW — Section 8: Dormant Analysis Charts
# ══════════════════════════════════════════════════════════════

def make_dormant_breakdown_chart():
    if DORMANT_STATS is None:
        return _empty_fig()
    reasons = DORMANT_STATS['reasons']
    labels = [r['desc'] for r in reasons.values()]
    values = [r['count'] for r in reasons.values()]
    fig = go.Figure(go.Pie(
        labels=labels, values=values,
        marker=dict(colors=[ROSE, AMBER, MUTED], line=dict(color=BG, width=2)),
        textinfo='percent+value', textfont=dict(size=11, color=TXT),
        hole=0.55,
    ))
    fig.add_annotation(
        text=f"<b>{DORMANT_STATS['n_excluded']:,}</b><br><span style='font-size:10px'>SP</span>",
        x=0.5, y=0.5, showarrow=False, font=dict(color=TXT, size=18))
    return _theme(fig, height=300, margin=dict(l=20, r=20, t=20, b=20))


def make_dormant_brands_chart():
    if DORMANT_STATS is None:
        return _empty_fig()
    brands = DORMANT_STATS['top_brands']
    fig = go.Figure(go.Bar(
        y=[b['name'] for b in brands][::-1],
        x=[b['count'] for b in brands][::-1],
        orientation='h', marker=dict(color=ROSE, opacity=0.7),
        text=[f"{b['count']:,}" for b in brands][::-1], textposition='outside',
    ))
    return _theme(fig, height=300, margin=dict(l=120, r=40, t=10, b=30))


# ══════════════════════════════════════════════════════════════
#  HELPERS (copy từ page_ml_regression để đồng bộ)
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
#  SAMPLE TABLE BUILDER + BADGES
# ══════════════════════════════════════════════════════════════

SAMPLE_TYPE_BADGES = {
    'representative': ('🎯', 'Đại diện', VIOLET),
    'bestseller':     ('🔥', 'Bán chạy', AMBER),
    'top_rated':      ('⭐', 'Đánh giá cao', EMERALD),
}


def build_samples_table(df):
    """Render bảng sản phẩm tiêu biểu cho modal."""
    headers = ['#', 'Tên sản phẩm', 'Brand', 'Giá', 'Đã bán', 'Rating']

    rows = []
    for i, (_, r) in enumerate(df.iterrows(), 1):
        name_clean = str(r.get('name_clean', ''))
        rating_val = r.get('rating', 0)
        review_val = r.get('review_count', 0)
        rating_str = (f"⭐ {rating_val:.1f} ({int(review_val)})"
                      if rating_val > 0 else '—')

        rows.append(html.Tr([
            html.Td(i, style={'color': SUBTXT}),
            html.Td(
                name_clean[:50] + ('...' if len(name_clean) > 50 else ''),
                style={'color': TXT, 'fontWeight': 500, 'maxWidth': '300px'},
            ),
            html.Td(r.get('brand_name', ''), style={'color': SUBTXT}),
            html.Td(f"{int(r['price']/1000)}K", style={'color': TXT}),
            html.Td(f"{int(r['sold_count']):,}", style={'color': TXT}),
            html.Td(rating_str, style={'color': TXT}),
        ]))

    return html.Table([
        html.Thead(html.Tr([html.Th(h, style={
            'textAlign': 'left', 'padding': '10px 12px',
            'borderBottom': f'1px solid {BORDER2}',
            'color': SUBTXT, 'fontSize': '11px',
            'textTransform': 'uppercase', 'letterSpacing': '0.05em',
        }) for h in headers])),
        html.Tbody(rows, style={'fontSize': '12px'}),
    ], style={'width': '100%', 'borderCollapse': 'collapse'})


def strategy_card(p):
    """Card gợi ý chiến lược cho 1 cluster."""
    cid = p['cluster_id']
    display_id = cid + 1  # Hiển thị 1-based cho user
    color = CLUSTER_COLORS[cid % len(CLUSTER_COLORS)]
    return html.Div([
        # Header
        html.Div([
            html.Span(f'#{display_id}', style={
                'background': color, 'color': 'white',
                'padding': '4px 10px', 'borderRadius': '6px',
                'fontWeight': 700, 'fontSize': '11px',
                'letterSpacing': '0.05em',
            }),
            html.H3(p['name'], style={
                'color': TXT, 'fontSize': '14px',
                'margin': '0 0 0 10px', 'display': 'block',
                'lineHeight': '1.25',
                'fontWeight': 600,
            }),
        ], style={
            'marginBottom': '10px',
            'display': 'flex',
            'alignItems': 'flex-start',
            'minHeight': '56px',
        }),

        # Stats row
        html.Div([
            mini_stat(f"{p['size']:,}", f"Sản phẩm ({p['pct_of_total']:.1f}%)", color),
            mini_stat(f"{p['median_price']//1000}K", 'Giá median', VIOLET),
            mini_stat(f"{p['pct_vn']:.0f}%", 'Hàng VN', C_DOM),
            mini_stat(f"{p['mean_rating']:.2f}", 'Rating TB', AMBER),
        ], className='p3-mini-stats', style={'marginBottom': '10px'}),

        # Top brands
        html.P([
            html.Strong('Top brand: ', style={'color': SUBTXT}),
            ', '.join(p['top_brands'][:3]),
        ], style={'fontSize': '11px', 'color': TXT, 'margin': '6px 0'}),
        html.P([
            html.Strong('Ngành hàng chính: ', style={'color': SUBTXT}),
            ', '.join(p['top_product_types'][:3]),
        ], style={'fontSize': '11px', 'color': TXT, 'margin': '6px 0'}),

        # Strategies
        html.P('CHIẾN LƯỢC GỢI Ý', style={
            'color': SUBTXT, 'fontSize': '10px',
            'letterSpacing': '0.08em', 'margin': '10px 0 6px 0',
            'fontWeight': 700,
        }),
        html.Ul([
            html.Li(s, style={'fontSize': '12px', 'color': TXT,
                              'marginBottom': '4px', 'lineHeight': '1.5'})
            for s in p['strategies']
        ], style={'paddingLeft': '18px', 'margin': 0}),

        # View samples button — pushed to bottom with marginTop:auto
        html.Button(
            [html.Span('🔍 ', style={'marginRight': '4px'}),
             'Xem sản phẩm tiêu biểu'],
            id={'type': 'open-samples-btn', 'cluster': cid},
            n_clicks=0,
            title=f'Xem 10 sản phẩm tiêu biểu của Phân khúc #{display_id}',
            style={
                'marginTop': 'auto',
                'paddingTop': '12px',
                'background': 'transparent',
                'color': color,
                'border': f'1px solid {color}',
                'borderRadius': '8px',
                'padding': '8px 14px',
                'fontSize': '12px',
                'fontWeight': 600,
                'cursor': 'pointer',
                'width': '100%',
                'transition': 'all 0.15s ease',
            }
        ),
    ], className='p3-card', style={
        'flex': '1', 'minWidth': '280px',
        'borderLeft': f'3px solid {color}',
        'display': 'flex', 'flexDirection': 'column',
    })


def _modal_component():
    """Modal overlay for displaying cluster sample products."""
    return html.Div([
        html.Div([
            # Header
            html.Div([
                html.H3(id='modal-title', style={'margin': 0, 'color': TXT}),
                html.Button('✕', id='close-modal-btn',
                            title='Đóng',
                            style={'background': 'none', 'border': 'none',
                                   'color': SUBTXT, 'fontSize': '24px',
                                   'cursor': 'pointer', 'padding': '4px 8px'}),
            ], style={'display': 'flex', 'justifyContent': 'space-between',
                      'alignItems': 'center', 'marginBottom': '16px'}),

            # Subtitle
            html.P(id='modal-subtitle', style={'color': SUBTXT, 'fontSize': '13px',
                                                'marginBottom': '20px'}),

            # Table container
            html.Div(id='modal-table-container'),
        ], id='modal-content-box', style={
            'background': CARD,
            'borderRadius': '14px',
            'padding': '24px',
            'maxWidth': '900px',
            'width': '100%',
            'maxHeight': '80vh',
            'overflow': 'auto',
            'border': f'1px solid {BORDER2}',
            'boxShadow': '0 20px 60px rgba(0,0,0,0.5)',
        }),
    ], id='modal-overlay',
       role='dialog',
       **{'aria-modal': 'true'},
       n_clicks=0,
       style={
           'display': 'none',
           'position': 'fixed',
           'top': 0, 'left': 0, 'right': 0, 'bottom': 0,
           'background': 'rgba(0,0,0,0.7)',
           'zIndex': 9999,
           'justifyContent': 'center',
           'alignItems': 'center',
           'padding': '20px',
       })


# ══════════════════════════════════════════════════════════════
#  LAYOUT
# ══════════════════════════════════════════════════════════════

def _not_ready_layout():
    return html.Div([
        html.Div([
            html.Div([
                html.P('MACHINE LEARNING · CLUSTERING', className='p3-page-label'),
                html.H1('Model 2 chưa được huấn luyện', className='p3-page-title'),
            ], className='p3-topbar-left'),
        ], className='p3-topbar'),
        html.Div([
            html.P('Vui lòng chạy lệnh sau từ thư mục machine learning/:',
                   style={'color': SUBTXT, 'marginBottom': '12px'}),
            html.Pre(
                "cd 'machine learning' && python train_model2.py",
                style={'background': CARD, 'padding': '14px 18px',
                       'borderRadius': '10px',
                       'border': f'1px solid {BORDER2}',
                       'color': AMBER, 'fontSize': '13px'},
            ),
            html.P('Sau khi chạy xong, refresh trang này để xem kết quả.',
                   style={'color': SUBTXT, 'marginTop': '12px', 'fontSize': '12px'}),
        ], style={'padding': '28px'}),
    ], className='p3-page')


def layout():
    if not ML_READY:
        return _not_ready_layout()

    K = METRICS['best_k']
    # K hiệu dụng có thể nhỏ hơn best_k nếu đã merge cluster tí hon
    K_eff = len(PROFILES)
    sil = METRICS['final_silhouette']
    db = METRICS['final_davies_bouldin']
    stab_mean = METRICS['stability_ari_mean']
    stab_std = METRICS['stability_ari_std']
    ari_pt = METRICS['ari_vs_product_type']
    n_samples = METRICS['n_samples']
    var_pca = METRICS['pca_variance_explained']

    cfg = {'displayModeBar': False}

    return html.Div([

        _module_switcher('module2'),

        # ── TOP BAR ─────────────────────────────────────────
        html.Div([
            html.Div([
                html.P('MACHINE LEARNING · CLUSTERING', className='p3-page-label'),
                html.H1('Phân khúc thị trường mỹ phẩm · K-Means',
                        className='p3-page-title'),
            ], className='p3-topbar-left'),
            html.Div([
                html.Span(f'🧩 K = {K_eff} clusters', className='p3-topchip violet'),
                html.Span(f'📐 Silhouette = {sil:.3f}', className='p3-topchip amber'),
                html.Span(f'📊 n = {n_samples:,}', className='p3-topchip dim'),
                html.Span('Nhóm 05 · FIT-HCMUS', className='p3-topchip dim'),
            ], className='p3-topbar-chips'),
        ], className='p3-topbar'),

        # ── KPI STRIP ───────────────────────────────────────
        html.Div([
            kpi('🧩', f'{K_eff}', 'Số cluster', VIOLET,
                hex_to_rgba(VIOLET, .15)),
            kpi('📐', f'{sil:.3f}', 'Silhouette', AMBER,
                hex_to_rgba(AMBER, .12)),
            kpi('📉', f'{db:.3f}', 'Davies-Bouldin', ROSE,
                hex_to_rgba(ROSE, .12)),
            kpi('🎯', f'{stab_mean:.3f}', 'Stability ARI', EMERALD,
                hex_to_rgba(EMERALD, .12)),
            kpi('🔬', f'{ari_pt:.3f}', 'ARI vs product_type', CYAN,
                hex_to_rgba(CYAN, .12)),
            kpi('🗺️', f'{var_pca:.0%}', 'PCA variance (2D)', INDIGO,
                hex_to_rgba(INDIGO, .12)),
        ], className='p3-kpi-strip'),

        html.Div([

            # ══════════════════════════════════════════════════
            #  S1 — FEATURE ANALYSIS & FILTERING DECISION
            # ══════════════════════════════════════════════════
            sec('SECTION 01',
                'Phân tích chất lượng dữ liệu & Xử lý nhiễu · Lọc Active + Rated',
                'cml1'),

            html.Div([
                card('PHÂN BỐ CÁC FEATURE TRÊN TOÀN BỘ CATALOG',
                     dcc.Graph(figure=make_problematic_features_chart(), config=cfg),
                     flex='1.5', min_w='400px', glow='g-rose'),
                card('CORRELATION MATRIX (TRƯỚC KHI LỌC)',
                     html.Div([
                         html.P(f'● Tồn tại multicollinearity mạnh (max |r| = {CORR_INSIGHTS["raw"]["max_abs_corr"]:.2f}) '
                                'do rating=0 và sold=0',
                                style={'fontSize': '10px', 'color': MUTED, 'margin': '0 0 8px 0'}),
                         dcc.Graph(figure=make_corr_heatmap(CORR_RAW), config=cfg),
                     ]),
                     flex='1', min_w='350px', glow='g-amber'),
                card('QUYẾT ĐỊNH XỬ LÝ',
                     build_decision_card(),
                     flex='1', min_w='300px', glow='g-emerald', extra_style={'padding': '16px'}),
            ], className='p3-row'),

            html.Div(className='p3-divider'),

            # ══════════════════════════════════════════════════
            #  S2 — CHỌN K TỐI ƯU & KIỂM TRA LẠI CORRELATION
            # ══════════════════════════════════════════════════
            sec('SECTION 02',
                f'Tổng quan Model · Chọn K* = {K} và Kiểm tra tính độc lập của Features',
                'cml2'),

            html.Div([
                card('ELBOW & SILHOUETTE & DAVIES-BOULDIN',
                     html.Div([
                         html.P('● Chọn K bằng composite score (silhouette − 0.15·DB) trong range 3–6',
                                style={'fontSize': '10px', 'color': MUTED, 'margin': '0 0 8px 0'}),
                         dcc.Graph(figure=make_elbow_silhouette(), config=cfg),
                     ]),
                     flex='1.5', min_w='380px', glow='g-violet'),

                card('CORRELATION MATRIX (SAU KHI LỌC)',
                     html.Div([
                         html.P(f'● Max |r| giảm còn {CORR_INSIGHTS["filtered"]["max_abs_corr"]:.2f} · '
                                'Features đã đủ tính độc lập',
                                style={'fontSize': '10px', 'color': MUTED, 'margin': '0 0 8px 0'}),
                         dcc.Graph(figure=make_corr_heatmap(CORR_FILTERED), config=cfg),
                     ]),
                     flex='1.2', min_w='350px', glow='g-emerald'),

                html.Div([
                    html.Div([
                        mini_stat(f'{K_eff}', 'K hiệu dụng', VIOLET),
                        mini_stat(f'{stab_mean:.2f}', 'Stability', EMERALD),
                    ], className='p3-mini-stats'),
                    html.Div(style={'height': '12px'}),
                    insight('🎲', html.Span([
                        html.Strong(f'Stability ARI = {stab_mean:.3f} ± {stab_std:.3f}'),
                        ' — chạy 10 lần với random_state khác nhau, cluster ổn định.',
                    ]), 'ia'),
                    insight('🔬', html.Span([
                        html.Strong(f'ARI vs product_type = {ari_pt:.3f}'),
                        ' — cluster ≠ phân loại ngành hàng. Tách thị trường theo giá/uy tín.',
                    ]), 'ic'),
                ], className='p3-card', style={
                    'flex': '1', 'minWidth': '220px',
                    'borderLeft': f'2px solid {VIOLET}',
                }),
            ], className='p3-row'),

            html.Div(className='p3-divider'),

            # ══════════════════════════════════════════════════
            #  S3 — BẢN ĐỒ PHÂN KHÚC
            # ══════════════════════════════════════════════════
            sec('SECTION 03',
                f'Bản đồ phân khúc · PCA 2D ({var_pca:.1%} variance) + phân bố VN/NK',
                'cml3'),

            html.Div([
                card('PCA 2D · MỖI ĐIỂM = 1 SẢN PHẨM',
                     html.Div([
                         html.P('● Hover điểm để xem chi tiết · Jitter ~4% std để mờ rails',
                                style={'fontSize': '10px', 'color': MUTED, 'margin': '0 0 8px 0'}),
                         dcc.Graph(figure=make_pca_scatter(), config=cfg),
                     ]),
                     flex='1.4', min_w='400px', glow='g-amber'),
                card('PHÂN BỐ VN ↔ NK TRONG MỖI CLUSTER',
                     html.Div([
                         html.P('● Tỷ lệ hàng nội/ngoại trong từng phân khúc',
                                style={'fontSize': '10px', 'color': MUTED, 'margin': '0 0 8px 0'}),
                         dcc.Graph(figure=make_vn_nk_stack(), config=cfg),
                     ]),
                     flex='1', min_w='320px', glow='g-amber'),
            ], className='p3-row'),

            _s2_findings_insight(),

            html.Div(className='p3-divider'),

            # ══════════════════════════════════════════════════
            #  S4 — PROFILE TỪNG CLUSTER (HEATMAP)
            # ══════════════════════════════════════════════════
            sec('SECTION 04',
                'Profile từng cluster · Heatmap 5 features (giá trị chuẩn hóa màu)',
                'cml4'),

            html.Div([
                card('HEATMAP · SO SÁNH ĐẶC TRƯNG CÁC CLUSTER',
                     html.Div([
                         html.P('● Màu ô = giá trị chuẩn hoá 0-1 · Số trong ô = giá trị gốc',
                                style={'fontSize': '10px', 'color': MUTED, 'margin': '0 0 8px 0'}),
                         dcc.Graph(figure=make_profile_heatmap(), config=cfg),
                     ]),
                     flex='1', min_w='520px', glow='g-emerald'),
            ], className='p3-row'),

            html.Div(className='p3-divider'),

            # ══════════════════════════════════════════════════
            #  S5 — CHIẾN LƯỢC GỢI Ý TỪNG CLUSTER
            # ══════════════════════════════════════════════════
            sec('SECTION 05',
                'Chiến lược gợi ý cho từng phân khúc · Dành cho quản lý ngành hàng',
                'cml5'),

            html.Div([strategy_card(p) for p in PROFILES],
                     className='p3-row', style={'flexWrap': 'wrap'}),

            html.Div(className='p3-divider'),

            # ══════════════════════════════════════════════════
            #  S6 — CROSS VỚI MODEL 1
            # ══════════════════════════════════════════════════
            sec('SECTION 06',
                'Model 1 dự đoán trong từng cluster · Phân khúc nào dễ / khó dự đoán hơn?',
                'cml6'),

            html.Div([
                card('R² CỦA MODEL 1 (GradientBoosting) THEO CLUSTER',
                     html.Div([
                         html.P('● Train Model 1 riêng trong mỗi cluster (75/25 split, target: log_sold_count)',
                                style={'fontSize': '10px', 'color': MUTED, 'margin': '0 0 8px 0'}),
                         dcc.Graph(figure=make_cross_regression_bar(), config=cfg),
                     ]),
                     flex='1', min_w='400px', glow='g-violet'),
                html.Div([
                    insight('🔗', html.Span([
                        html.Strong('Hai model bổ trợ lẫn nhau'),
                        ' — Model 2 chia sàn thành các phân khúc, Model 1 '
                        'học pattern bán hàng bên TRONG mỗi phân khúc.',
                    ]), 'ic'),
                    insight('📊', html.Span([
                        'R² đo lường khả năng dự đoán doanh số. '
                        'Cluster có R² cao chứng tỏ hành vi khách hàng ổn định, dễ dự đoán.',
                    ]), 'ia'),
                ], className='p3-card', style={
                    'flex': '1', 'minWidth': '260px',
                    'borderLeft': f'2px solid #F472B6',
                }),
            ], className='p3-row'),

        ], className='p3-inner'),

        # Footer note
        html.Div([
            html.P([
                html.Strong('Audience: '), 'Category manager của sàn TMĐT và '
                'analyst/brand team — cần nhìn toàn thị trường. ',
                html.Strong('Limit: '), 'Cluster không dùng cho sản phẩm '
                'sắp ra mắt (thiếu rating/sold_count). Inference nên áp '
                'cho sản phẩm đã có trên sàn.',
            ], style={'color': SUBTXT, 'fontSize': '11px',
                      'textAlign': 'center', 'margin': '20px 0 0 0',
                      'fontStyle': 'italic'}),
        ]),

        # Modal overlay (default hidden)
        _modal_component(),

    ], className='p3-page')


# ══════════════════════════════════════════════════════════════
#  CALLBACK — OPEN / CLOSE MODAL
# ══════════════════════════════════════════════════════════════

_MODAL_HIDDEN = {
    'display': 'none',
    'position': 'fixed',
    'top': 0, 'left': 0, 'right': 0, 'bottom': 0,
    'background': 'rgba(0,0,0,0.7)',
    'zIndex': 9999,
    'justifyContent': 'center',
    'alignItems': 'center',
    'padding': '20px',
}
_MODAL_SHOWN = {**_MODAL_HIDDEN, 'display': 'flex'}


@callback(
    [Output('modal-overlay', 'style'),
     Output('modal-title', 'children'),
     Output('modal-subtitle', 'children'),
     Output('modal-table-container', 'children')],
    [Input({'type': 'open-samples-btn', 'cluster': ALL}, 'n_clicks'),
     Input('close-modal-btn', 'n_clicks'),
     Input('modal-overlay', 'n_clicks')],
    [State('modal-overlay', 'style')],
    prevent_initial_call=True
)
def toggle_modal(open_clicks, close_click, overlay_click, current_style):
    triggered = ctx.triggered_id

    # Close button
    if triggered == 'close-modal-btn':
        return _MODAL_HIDDEN, '', '', None

    # Click on overlay (outside content box) → close
    if triggered == 'modal-overlay':
        return _MODAL_HIDDEN, '', '', None

    # Open button — get cluster_id from pattern-matching id
    if isinstance(triggered, dict) and triggered.get('type') == 'open-samples-btn':
        # Check that at least one button was actually clicked
        if not any(open_clicks):
            raise dash.exceptions.PreventUpdate

        cid = triggered['cluster']

        # Handle case where samples data is not available
        if SAMPLES is None:
            return (
                _MODAL_SHOWN,
                f'Phân khúc #{cid + 1}',
                'Dữ liệu sản phẩm tiêu biểu chưa có.',
                html.Div([
                    html.P('⚠️ File model2_cluster_samples.csv chưa tồn tại.',
                           style={'color': AMBER, 'fontSize': '14px',
                                  'marginBottom': '12px'}),
                    html.P('Vui lòng chạy lại train_model2.py để tạo dữ liệu mẫu:',
                           style={'color': SUBTXT, 'marginBottom': '8px'}),
                    html.Pre(
                        'cd dashboardML && python train_model2.py',
                        style={'background': BG, 'padding': '12px 16px',
                               'borderRadius': '8px', 'color': AMBER,
                               'fontSize': '13px',
                               'border': f'1px solid {BORDER2}'}),
                ], style={'padding': '12px 0'})
            )

        cluster_profile = next(
            (p for p in PROFILES if p['cluster_id'] == cid), None)
        if cluster_profile is None:
            raise dash.exceptions.PreventUpdate

        cluster_samples = SAMPLES[SAMPLES['cluster_id'] == cid]

        title = f"Phân khúc #{cid + 1} · {cluster_profile['name']}"
        subtitle = (f"Hiển thị {len(cluster_samples)} sản phẩm tiêu biểu / "
                    f"{cluster_profile['size']:,} sản phẩm trong cluster")

        table = build_samples_table(cluster_samples)

        return _MODAL_SHOWN, title, subtitle, table

    raise dash.exceptions.PreventUpdate


def _s2_findings_insight():
    """Sinh insight tự động từ profiles: cluster nào VN chiếm cao/thấp nhất."""
    if PROFILES is None:
        return html.Div()

    # Highest & lowest VN %
    by_vn = sorted(PROFILES, key=lambda p: p['pct_vn'])
    lowest = by_vn[0]
    highest = by_vn[-1]

    return html.Div([
        insight('🇻🇳', html.Span([
            'Cluster ', html.Strong(f'"{highest["name"]}"'),
            f' có tỷ lệ hàng VN cao nhất: ', html.Strong(f'{highest["pct_vn"]:.1f}%'),
            f' ({highest["size"]:,} sản phẩm, giá median '
            f'{highest["median_price"]//1000}K, rating TB {highest["mean_rating"]:.2f}).',
        ]), 'ia'),
        insight('🌏', html.Span([
            'Ngược lại, cluster ', html.Strong(f'"{lowest["name"]}"'),
            f' chỉ có ', html.Strong(f'{lowest["pct_vn"]:.1f}% hàng VN'),
            f' — phân khúc này bị chiếm lĩnh bởi hàng NK '
            f'(giá median {lowest["median_price"]//1000}K). '
            f'Đây là khoảng trống thị trường cho brand VN muốn mở rộng.',
        ]), 'ic'),
    ], style={'marginBottom': '14px'})
