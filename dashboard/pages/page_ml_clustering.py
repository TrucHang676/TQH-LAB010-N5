# pages/page_ml_clustering.py
# ╔══════════════════════════════════════════════════════════════╗
# ║  Trang ML — Phân khúc thị trường (K-Means Clustering)        ║
# ║  Dark navy theme · Đồng bộ với page_ml_regression.py         ║
# ║  S1: Chọn K tối ưu (Elbow + Silhouette + Davies-Bouldin)     ║
# ║  S2: Bản đồ phân khúc (PCA 2D) · Phân bố VN/NK per cluster   ║
# ║  S3: Profile từng cluster (Radar + table)                    ║
# ║  S4: Chiến lược gợi ý cho từng phân khúc                     ║
# ║  S5: Cross với Model 1 (R² regression trong từng cluster)    ║
# ╚══════════════════════════════════════════════════════════════╝

import dash
from dash import html, dcc
import plotly.graph_objects as go
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


def _safe_load():
    try:
        with open(os.path.join(ML_DIR, 'model2_metrics.json'), encoding='utf-8') as f:
            metrics = json.load(f)
        with open(os.path.join(ML_DIR, 'model2_profile.json'), encoding='utf-8') as f:
            profiles = json.load(f)
        with open(os.path.join(ML_DIR, 'model2_cross_regression.json'), encoding='utf-8') as f:
            cross = json.load(f)
        labels = pd.read_csv(os.path.join(ML_DIR, 'model2_cluster_labels.csv'))
        ks = pd.read_csv(os.path.join(ML_DIR, 'model2_elbow_silhouette.csv'))
        return metrics, profiles, cross, labels, ks
    except Exception as e:
        print(f'[page_ml_clustering] Chưa load được artifact: {e}')
        return None, None, None, None, None


METRICS, PROFILES, CROSS, LABELS, KS = _safe_load()
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
            name=f'[{c}] {prof_name[:32]}',
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
           legend=dict(orientation='v', yanchor='top', y=1,
                       xanchor='left', x=1.02,
                       font=dict(size=9, color=SUBTXT),
                       bgcolor='rgba(0,0,0,0)'),
           margin=dict(l=14, r=220, t=28, b=14),
           xaxis=dict(title='PC1'), yaxis=dict(title='PC2'))
    return fig


def make_vn_nk_stack():
    """Stacked bar: % VN vs % NK trong mỗi cluster."""
    if PROFILES is None:
        return go.Figure()

    names = [f"[{p['cluster_id']}] {p['name'][:24]}" for p in PROFILES]
    pct_vn = [p['pct_vn'] for p in PROFILES]
    pct_nk = [p['pct_nk'] for p in PROFILES]
    sizes = [p['size'] for p in PROFILES]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=names, x=pct_vn, orientation='h', name='VN (nội địa)',
        marker=dict(color=C_DOM, line=dict(color='rgba(255,255,255,0.15)', width=0.5)),
        text=[f'{v:.0f}%' for v in pct_vn], textposition='inside',
        textfont=dict(color='white', size=11),
        hovertemplate='VN: %{x:.1f}%<extra></extra>',
    ))
    fig.add_trace(go.Bar(
        y=names, x=pct_nk, orientation='h', name='NK (nhập khẩu)',
        marker=dict(color=C_IMP, line=dict(color='rgba(255,255,255,0.15)', width=0.5)),
        text=[f'{v:.0f}%' for v in pct_nk], textposition='inside',
        textfont=dict(color='white', size=11),
        hovertemplate='NK: %{x:.1f}%<extra></extra>',
    ))

    _theme(fig, height=max(240, 60 * len(names) + 40),
           barmode='stack', showlegend=True, legend=_leg(),
           xaxis=dict(title='Tỷ lệ (%)', range=[0, 100]),
           yaxis=dict(title='', autorange='reversed'))
    return fig


# ══════════════════════════════════════════════════════════════
#  FIGURES — S3: RADAR CHART (cluster profile)
# ══════════════════════════════════════════════════════════════

def make_radar():
    """Radar chart 6 features (đã min-max scale 0-1) so sánh cluster."""
    if PROFILES is None:
        return go.Figure()

    # Lấy raw values từ profiles, rồi min-max normalize để vẽ radar
    feat_keys = [
        ('median_price', 'Giá (median)'),
        ('mean_rating', 'Rating TB'),
        ('median_sold_count', 'Lượt bán (median)'),
        ('mean_discount_rate', 'Discount %'),
        ('pct_tiki_verified', '% Tiki Verified'),
        ('mean_review_ratio', 'Review ratio'),
    ]

    # Min-max normalize per feature across clusters
    raw_mat = np.array([[p[k] for k, _ in feat_keys] for p in PROFILES], dtype=float)
    col_min = raw_mat.min(axis=0)
    col_max = raw_mat.max(axis=0)
    rng = np.where((col_max - col_min) < 1e-9, 1, col_max - col_min)
    norm_mat = (raw_mat - col_min) / rng

    fig = go.Figure()
    for c, p in enumerate(PROFILES):
        vals = list(norm_mat[c]) + [norm_mat[c][0]]   # close polygon
        labels = [lbl for _, lbl in feat_keys] + [feat_keys[0][1]]
        fig.add_trace(go.Scatterpolar(
            r=vals, theta=labels, fill='toself',
            name=f'[{c}] {p["name"][:24]}',
            line=dict(color=CLUSTER_COLORS[c % len(CLUSTER_COLORS)], width=2),
            fillcolor=hex_to_rgba(CLUSTER_COLORS[c % len(CLUSTER_COLORS)], 0.18),
            hovertemplate='%{theta}: %{r:.2f}<extra></extra>',
        ))

    fig.update_layout(
        polar=dict(
            bgcolor=CARD,
            radialaxis=dict(visible=True, range=[0, 1],
                            gridcolor=GRID, tickfont=dict(size=9, color=SUBTXT)),
            angularaxis=dict(gridcolor=GRID, tickfont=dict(size=10, color=TXT)),
        ),
        font=dict(family="'Space Grotesk','Segoe UI',sans-serif",
                  size=11, color=TXT),
        paper_bgcolor=CARD, plot_bgcolor=CARD,
        height=420,
        margin=dict(l=30, r=30, t=30, b=30),
        showlegend=True,
        legend=dict(orientation='v', yanchor='middle', y=0.5,
                    xanchor='left', x=1.02,
                    font=dict(size=9, color=SUBTXT),
                    bgcolor='rgba(0,0,0,0)'),
        hoverlabel=dict(bgcolor=SURFACE, bordercolor=BORDER2,
                        font=dict(size=11, color=TXT)),
    )
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
        names.append(f'[{cid}] {PROFILES[cid]["name"][:24]}')
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
    _theme(fig, height=max(220, 50 * len(names) + 40),
           showlegend=False,
           xaxis=dict(title='R² của Model 1 (log_sold_count)',
                      range=[min(0, min(r2s) - 0.05), max(1, max(r2s) + 0.1)]),
           yaxis=dict(title='', autorange='reversed'))
    return fig


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


def strategy_card(p):
    """Card gợi ý chiến lược cho 1 cluster."""
    cid = p['cluster_id']
    color = CLUSTER_COLORS[cid % len(CLUSTER_COLORS)]
    return html.Div([
        # Header
        html.Div([
            html.Span(f'#{cid}', style={
                'background': color, 'color': 'white',
                'padding': '4px 10px', 'borderRadius': '6px',
                'fontWeight': 700, 'fontSize': '11px',
                'letterSpacing': '0.05em',
            }),
            html.H3(p['name'], style={
                'color': TXT, 'fontSize': '14px',
                'margin': '0 0 0 10px', 'display': 'inline-block',
                'fontWeight': 600,
            }),
        ], style={'marginBottom': '10px'}),

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
    ], className='p3-card', style={
        'flex': '1', 'minWidth': '280px',
        'borderLeft': f'3px solid {color}',
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
            html.P('Vui lòng chạy lệnh sau từ thư mục dashboard/:',
                   style={'color': SUBTXT, 'marginBottom': '12px'}),
            html.Pre(
                'cd dashboard && python train_model2.py',
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
            #  S1 — CHỌN K TỐI ƯU
            # ══════════════════════════════════════════════════
            sec('SECTION 01',
                f'Chọn K tối ưu · Elbow + Silhouette + Davies-Bouldin (chọn K* = {K})',
                'cml1'),

            html.Div([
                card('ELBOW & SILHOUETTE & DAVIES-BOULDIN',
                     html.Div([
                         html.P('● Violet = Inertia (Elbow) · Amber = Silhouette (cao là tốt) · '
                                'Rose = Davies-Bouldin (thấp là tốt) · Xanh = K* được chọn',
                                style={'fontSize': '10px', 'color': MUTED,
                                       'margin': '0 0 8px 0'}),
                         dcc.Graph(figure=make_elbow_silhouette(), config=cfg),
                     ]),
                     flex='1.5', min_w='380px', glow='g-violet'),

                # Mini insight panel
                html.Div([
                    html.Div([
                        mini_stat(f'{K_eff}', 'K hiệu dụng', VIOLET),
                        mini_stat(f'{stab_mean:.2f}', 'Stability', EMERALD),
                    ], className='p3-mini-stats'),
                    html.Div(style={'height': '12px'}),
                    insight('🧩', html.Span([
                        html.Strong(f'Chọn K = {K} bằng composite score'),
                        ' (silhouette − 0.15·DB) trong range 3–6 '
                        'để cluster dễ diễn giải. Các cluster quá nhỏ '
                        '(< 20 sản phẩm) được merge vào cluster gần nhất.',
                    ]), 'ic'),
                    insight('🎲', html.Span([
                        html.Strong(f'Stability ARI = {stab_mean:.3f} ± {stab_std:.3f}'),
                        ' — chạy 10 lần với random_state khác nhau, '
                        f'cluster {"rất ổn định" if stab_mean >= 0.9 else "ổn định"}.',
                    ]), 'ia'),
                    insight('🔬', html.Span([
                        html.Strong(f'ARI vs product_type = {ari_pt:.3f}'),
                        ' — cluster ≠ phân loại ngành hàng. '
                        'Model đang tách thị trường theo chiều khác '
                        '(giá × rating × uy tín), không tái phát hiện '
                        'product_type sẵn có.',
                    ]), 'ic'),
                ], className='p3-card', style={
                    'flex': '1', 'minWidth': '250px',
                    'borderLeft': f'2px solid {VIOLET}',
                }),
            ], className='p3-row'),

            html.Div(className='p3-divider'),

            # ══════════════════════════════════════════════════
            #  S2 — BẢN ĐỒ PHÂN KHÚC
            # ══════════════════════════════════════════════════
            sec('SECTION 02',
                f'Bản đồ phân khúc · PCA 2D ({var_pca:.1%} variance) + phân bố VN/NK',
                'cml2'),

            html.Div([
                card('PCA 2D · MỖI ĐIỂM = 1 SẢN PHẨM',
                     html.Div([
                         html.P('● Hover điểm để xem tên, brand, giá · '
                                'Downsample 3000 sản phẩm · '
                                'Jitter ~4% std để mờ rails do feature rời rạc',
                                style={'fontSize': '10px', 'color': MUTED,
                                       'margin': '0 0 8px 0'}),
                         dcc.Graph(figure=make_pca_scatter(), config=cfg),
                     ]),
                     flex='1.4', min_w='400px', glow='g-amber'),
                card('PHÂN BỐ VN ↔ NK TRONG MỖI CLUSTER',
                     html.Div([
                         html.P('● Mục tiêu: xem hàng nội/ngoại có tụ tập ở '
                                'phân khúc cụ thể nào không',
                                style={'fontSize': '10px', 'color': MUTED,
                                       'margin': '0 0 8px 0'}),
                         dcc.Graph(figure=make_vn_nk_stack(), config=cfg),
                     ]),
                     flex='1', min_w='320px', glow='g-amber'),
            ], className='p3-row'),

            # S2 insight strip — findings about VN/NK
            _s2_findings_insight(),

            html.Div(className='p3-divider'),

            # ══════════════════════════════════════════════════
            #  S3 — PROFILE TỪNG CLUSTER (RADAR)
            # ══════════════════════════════════════════════════
            sec('SECTION 03',
                'Profile từng cluster · Radar chart 6 features (min-max scale)',
                'cml3'),

            html.Div([
                card('RADAR · SO SÁNH ĐẶC TRƯNG CÁC CLUSTER',
                     html.Div([
                         html.P('● Giá trị normalize 0-1 trong từng feature '
                                '(min ở clusters = 0, max = 1) · '
                                'Cluster nào có polygon rộng nhất ở 1 góc = '
                                'mạnh nhất ở feature đó',
                                style={'fontSize': '10px', 'color': MUTED,
                                       'margin': '0 0 8px 0'}),
                         dcc.Graph(figure=make_radar(), config=cfg),
                     ]),
                     flex='1', min_w='400px', glow='g-emerald'),
            ], className='p3-row'),

            html.Div(className='p3-divider'),

            # ══════════════════════════════════════════════════
            #  S4 — CHIẾN LƯỢC GỢI Ý TỪNG CLUSTER
            # ══════════════════════════════════════════════════
            sec('SECTION 04',
                'Chiến lược gợi ý cho từng phân khúc · '
                'Dành cho quản lý ngành hàng / brand team',
                'cml4'),

            html.Div([strategy_card(p) for p in PROFILES],
                     className='p3-row', style={'flexWrap': 'wrap'}),

            html.Div(className='p3-divider'),

            # ══════════════════════════════════════════════════
            #  S5 — CROSS VỚI MODEL 1
            # ══════════════════════════════════════════════════
            sec('SECTION 05',
                'Model 1 dự đoán trong từng cluster · '
                'Phân khúc nào dễ / khó dự đoán hơn?',
                'cml1'),

            html.Div([
                card('R² CỦA MODEL 1 (GradientBoosting) THEO CLUSTER',
                     html.Div([
                         html.P('● Train Model 1 riêng trong mỗi cluster '
                                '(75/25 split, log_sold_count target) · '
                                'Cluster có R² cao = pattern rõ ràng, '
                                'cluster R² thấp = nhiễu/đa dạng',
                                style={'fontSize': '10px', 'color': MUTED,
                                       'margin': '0 0 8px 0'}),
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
                        'Cluster ', html.Strong('Bán chạy · Uy tín cao'),
                        ' thường có R² cao nhất — pattern rõ ràng, '
                        'feature quyết định sức bán (rating, giá) đủ mạnh.',
                    ]), 'ia'),
                    insight('⚠️', html.Span([
                        'Cluster ', html.Strong('Kén khách · Rating 0'),
                        ' có R² thấp — đa số sản phẩm chưa có đánh giá → '
                        'tín hiệu chính (rating) bị dập tắt, '
                        'model chỉ có giá để bấu víu.',
                    ]), 'ic'),
                ], className='p3-card', style={
                    'flex': '1', 'minWidth': '260px',
                    'borderLeft': f'2px solid {VIOLET}',
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

    ], className='p3-page')


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
