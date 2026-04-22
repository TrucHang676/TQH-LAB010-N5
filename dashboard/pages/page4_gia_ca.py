# pages/page4_gia_ca.py
# ╔══════════════════════════════════════════════════════════════╗
# ║  Trang 4 — Phân khúc giá & Chiến lược khuyến mãi            ║
# ║  MT1: Sweet Spot — Trung vị lượt bán + Hit Rate              ║
# ║  MT2: Hiệu quả giảm giá sâu (> 30%) theo phân khúc          ║
# ╚══════════════════════════════════════════════════════════════╝

import dash
from dash import html, dcc, callback, Input, Output
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from data_loader import load_data

dash.register_page(
    __name__,
    path='/gia-ca',
    name='Giá cả',
    title='Giá cả | Mỹ phẩm Tiki',
    order=4,
)

# ── Data ─────────────────────────────────────────────────────
df_full, df_vn_full, df_nn_full = load_data()

PRICE_ORDER  = ['Dưới 100k', '100k – 300k', '300k – 700k', '700k – 2tr', 'Trên 2tr']
PRICE_LABELS = ['< 100k', '100k–300k', '300k–700k', '700k–2tr', '> 2tr']

PRODUCT_TYPES_ALL  = sorted(df_full['product_type'].dropna().unique().tolist())
ORIGIN_OPTIONS = [
    {'label': 'Tất cả',     'value': 'all'},
    {'label': 'Trong nước', 'value': 'domestic'},
    {'label': 'Ngoài nước', 'value': 'import'},
]
PRICE_SEGMENTS_ALL = [p for p in PRICE_ORDER if p in df_full['price_segment'].unique()]

# ── Palette ──────────────────────────────────────────────────
BG      = '#172542'
SURFACE = '#1A2A4D'
CARD    = '#1C2D55'
BORDER2 = 'rgba(255,255,255,0.13)'
GRID    = 'rgba(255,255,255,0.05)'
TXT     = '#F0F6FF'
SUBTXT  = '#94A3B8'
MUTED   = '#4B6178'

TEAL    = '#14B8A6'
TEAL_L  = '#7EE8D6'
GOLD    = '#F59E0B'
GOLD_L  = '#FCD34D'
EMERALD = '#34D399'
ROSE    = '#FB7185'
BLUE    = '#60A5FA'

C_DOM   = '#38BDF8'   # xanh dương nhạt — hàng trong nước (giữ nhất quán toàn app)
C_IMP   = '#F87171'   # đỏ nhạt         — hàng ngoài nước

# Màu 4 nhóm trong biểu đồ discount comparison
C_VN_LO = '#93C5FD'   # nội ≤ 30%  (xanh pastel)
C_VN_HI = C_DOM       # nội > 30%  (xanh đậm)
C_NN_LO = '#FCA5A5'   # ngoại ≤30% (đỏ pastel)
C_NN_HI = C_IMP       # ngoại >30% (đỏ đậm)

# ── Insights ─────────────────────────────────────────────────
INSIGHTS = {
    'p4-c-median':    ('it', 'Trung vị lượt bán là chỉ số bền vững hơn trung bình vì loại bỏ ảnh hưởng của các sản phẩm viral. Phân khúc nào có trung vị cao thì "lượt bán điển hình" cao — không chỉ vài sản phẩm ngôi sao kéo lên.'),
    'p4-c-hitrate':   ('it', 'Hit rate (tỉ lệ sản phẩm có ít nhất 1 lượt bán) đo mức độ "thanh khoản" của phân khúc. Hit rate cao = người mua dễ tìm thấy sản phẩm phù hợp, cạnh tranh trong phân khúc đó khỏe mạnh.'),
    'p4-c-disc-comp': ('ig', 'Biểu đồ so sánh trực tiếp trung vị lượt bán giữa nhóm giảm ≤30% và >30% cho từng phân khúc × nguồn gốc. Nếu cột >30% thấp hơn ≤30%, giảm sâu không đem lại hiệu quả tốt hơn.'),
    'p4-c-disc-pen':  ('ig', 'Tỉ lệ sản phẩm áp dụng giảm giá sâu (>30%) cho thấy chiến lược khuyến mãi của từng nhóm. Tỉ lệ cao ở phân khúc rẻ có thể là dấu hiệu cạnh tranh về giá thay vì chất lượng sản phẩm.'),
}


# ══════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════

def hex_to_rgba(hex_color, alpha=1):
    h = hex_color.lstrip('#')
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f'rgba({r},{g},{b},{alpha})'


def apply_filters(product_type='all', origin='all'):
    df = df_full.copy()
    if product_type and product_type != 'all':
        df = df[df['product_type'] == product_type]
    if origin == 'domestic':
        df = df[df['origin_class_corrected'] == 'Trong nước']
    elif origin == 'import':
        df = df[df['origin_class_corrected'] == 'Ngoài nước']
    return df


def _theme(fig, height=340, title_text=None, **kw):
    margin = kw.pop('margin', dict(l=14, r=14, t=64, b=14))
    fig.update_layout(
        height=height,
        font=dict(family="'Space Grotesk','Segoe UI',sans-serif", size=12, color=TXT),
        paper_bgcolor=CARD,
        plot_bgcolor=CARD,
        margin=margin,
        hoverlabel=dict(bgcolor=SURFACE, bordercolor=BORDER2,
                        font=dict(size=12, color=TXT)),
        **kw,
    )
    if title_text:
        fig.update_layout(title=dict(
            text=f'<b>{title_text}</b>',
            x=0.5, xanchor='center',
            font=dict(size=17, color=TXT,
                      family="'Space Grotesk','Segoe UI',sans-serif"),
            pad=dict(t=0, b=10),
            y=0.97, yanchor='top',
        ))
    fig.update_xaxes(
        gridcolor=GRID, gridwidth=1,
        linecolor='rgba(255,255,255,0.06)', linewidth=1,
        tickfont=dict(size=10, color=SUBTXT),
        title_font=dict(size=11, color=SUBTXT),
        zeroline=False, automargin=True,
    )
    fig.update_yaxes(
        gridcolor=GRID, gridwidth=1,
        linecolor='rgba(0,0,0,0)',
        tickfont=dict(size=10, color=SUBTXT),
        title_font=dict(size=11, color=SUBTXT),
        zeroline=False, automargin=True,
    )
    return fig


def _leg(**kw):
    return dict(orientation='h', yanchor='bottom', y=1.03,
                xanchor='right', x=1,
                font=dict(size=10, color=SUBTXT),
                bgcolor='rgba(0,0,0,0)', **kw)


def _empty_fig(msg='Không có dữ liệu với bộ lọc hiện tại'):
    fig = go.Figure()
    fig.update_layout(
        paper_bgcolor=CARD, plot_bgcolor=CARD, height=340,
        annotations=[dict(text=msg, x=0.5, y=0.5,
                          xref='paper', yref='paper',
                          showarrow=False,
                          font=dict(size=14, color=SUBTXT))],
        xaxis=dict(visible=False), yaxis=dict(visible=False),
    )
    return fig


# ══════════════════════════════════════════════════════════════
#  DATA PROCESSING
# ══════════════════════════════════════════════════════════════

def _iqr_cap(s):
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    return s.clip(upper=q3 + 1.5 * (q3 - q1))


def _price_analysis(df):
    """Trung vị lượt bán + hit rate theo phân khúc × nguồn gốc."""
    if len(df) == 0:
        return None

    # Trung vị lượt bán (sau IQR cap)
    df = df.copy()
    df['sold_capped'] = (df.groupby(['price_segment', 'origin_class_corrected'],
                                    observed=True)['sold_count']
                         .transform(_iqr_cap))
    med = (df.groupby(['price_segment', 'origin_class_corrected'], observed=True)
           ['sold_count'].median().reset_index())
    med.columns = ['segment', 'origin', 'median_sold']

    def _get(origin):
        return (med[med['origin'] == origin]
                .set_index('segment').reindex(PRICE_ORDER)['median_sold']
                .fillna(0).values)

    # Hit rate
    hr = (df.groupby(['price_segment', 'origin_class_corrected'], observed=True)
          ['sold_count'].agg(lambda s: (s > 0).mean() * 100).reset_index())
    hr.columns = ['segment', 'origin', 'hit_rate']

    def _hr(origin):
        return (hr[hr['origin'] == origin]
                .set_index('segment').reindex(PRICE_ORDER)['hit_rate']
                .fillna(0).values)

    return {
        'med_vn': _get('Trong nước'), 'med_nn': _get('Ngoài nước'),
        'hr_vn':  _hr('Trong nước'),  'hr_nn':  _hr('Ngoài nước'),
    }


def _discount_analysis(df):
    """Hiệu quả giảm giá sâu (>30%) theo phân khúc × nguồn gốc."""
    if len(df) == 0 or 'discount_rate' not in df.columns:
        return None

    df = df.copy()
    df['disc_bin'] = df['discount_rate'].apply(
        lambda x: 'Giảm > 30%' if x > 30 else 'Giảm ≤ 30%'
    )
    med = (df.groupby(['price_segment', 'origin_class_corrected', 'disc_bin'],
                      observed=True)['sold_count']
           .median().reset_index())
    med.columns = ['segment', 'origin', 'disc_bin', 'median_sold']

    def _get(origin, disc):
        return (med[(med['origin'] == origin) & (med['disc_bin'] == disc)]
                .set_index('segment').reindex(PRICE_ORDER)['median_sold']
                .fillna(0).values)

    disc_pct = (df.groupby(['price_segment', 'origin_class_corrected'], observed=True)
                .apply(lambda g: (g['discount_rate'] > 30).mean() * 100)
                .reset_index())
    disc_pct.columns = ['segment', 'origin', 'pct']

    def _dp(origin):
        return (disc_pct[disc_pct['origin'] == origin]
                .set_index('segment').reindex(PRICE_ORDER)['pct']
                .fillna(0).values)

    return {
        'vn_lo': _get('Trong nước', 'Giảm ≤ 30%'),
        'vn_hi': _get('Trong nước', 'Giảm > 30%'),
        'nn_lo': _get('Ngoài nước', 'Giảm ≤ 30%'),
        'nn_hi': _get('Ngoài nước', 'Giảm > 30%'),
        'dp_vn': _dp('Trong nước'),
        'dp_nn': _dp('Ngoài nước'),
    }


# ══════════════════════════════════════════════════════════════
#  CHART BUILDERS
# ══════════════════════════════════════════════════════════════

def make_median_chart(df):
    d = _price_analysis(df)
    if d is None:
        return _empty_fig()

    fig = go.Figure()
    groups = [g for g in ['Trong nước', 'Ngoài nước']
              if g in df['origin_class_corrected'].unique()]

    for origin, vals, color in [
        ('Trong nước', d['med_vn'], C_DOM),
        ('Ngoài nước', d['med_nn'], C_IMP),
    ]:
        if origin not in groups:
            continue
        fig.add_trace(go.Bar(
            name=origin, x=PRICE_LABELS, y=vals,
            marker=dict(color=color, line=dict(color='rgba(0,0,0,0)')),
            text=[f'{v:.0f}' for v in vals],
            textposition='outside',
            textfont=dict(size=10, weight=700, color=TXT),
            cliponaxis=False,
            hovertemplate=f'<b>%{{x}}</b><br>{origin}: %{{y:.0f}} lượt<extra></extra>',
        ))

    _theme(fig, height=340,
           title_text='Trung vị lượt bán theo phân khúc giá',
           barmode='group',
           xaxis=dict(showgrid=False),
           yaxis=dict(title='Trung vị lượt bán'),
           legend=_leg(),
           margin=dict(l=14, r=14, t=68, b=14))
    return fig


def make_hitrate_chart(df):
    d = _price_analysis(df)
    if d is None:
        return _empty_fig()

    fig = go.Figure()
    groups = [g for g in ['Trong nước', 'Ngoài nước']
              if g in df['origin_class_corrected'].unique()]

    for origin, vals, color in [
        ('Trong nước', d['hr_vn'], C_DOM),
        ('Ngoài nước', d['hr_nn'], C_IMP),
    ]:
        if origin not in groups:
            continue
        fig.add_trace(go.Bar(
            name=origin, x=PRICE_LABELS, y=vals,
            marker=dict(color=color, line=dict(color='rgba(0,0,0,0)')),
            text=[f'{v:.0f}%' for v in vals],
            textposition='outside',
            textfont=dict(size=10, weight=700, color=TXT),
            cliponaxis=False,
            hovertemplate=f'<b>%{{x}}</b><br>{origin}: %{{y:.1f}}%<extra></extra>',
        ))

    _theme(fig, height=340,
           title_text='Hit rate: Tỉ lệ sản phẩm có lượt bán > 0',
           barmode='group',
           xaxis=dict(showgrid=False),
           yaxis=dict(title='Tỉ lệ (%)', range=[0, 115]),
           legend=_leg(),
           margin=dict(l=14, r=14, t=68, b=14))
    return fig


def make_disc_comparison_chart(df):
    d = _discount_analysis(df)
    if d is None:
        return _empty_fig('Không có dữ liệu discount_rate')

    fig = go.Figure()

    for name, vals, color in [
        ('Nội ≤ 30%',    d['vn_lo'], C_VN_LO),
        ('Nội > 30%',    d['vn_hi'], C_VN_HI),
        ('Ngoại ≤ 30%',  d['nn_lo'], C_NN_LO),
        ('Ngoại > 30%',  d['nn_hi'], C_NN_HI),
    ]:
        fig.add_trace(go.Bar(
            name=name, x=PRICE_LABELS, y=vals,
            marker=dict(color=color, line=dict(color='rgba(0,0,0,0)')),
            hovertemplate=f'<b>%{{x}}</b><br>{name}: %{{y:.0f}} lượt<extra></extra>',
        ))

    _theme(fig, height=360,
           title_text='Lượt bán theo mức giảm giá & nguồn gốc',
           barmode='group',
           xaxis=dict(showgrid=False),
           yaxis=dict(title='Trung vị lượt bán'),
           legend=_leg(),
           margin=dict(l=14, r=14, t=68, b=14))
    return fig


def make_disc_penetration_chart(df):
    d = _discount_analysis(df)
    if d is None:
        return _empty_fig('Không có dữ liệu discount_rate')

    groups = [g for g in ['Trong nước', 'Ngoài nước']
              if g in df['origin_class_corrected'].unique()]
    fig = go.Figure()

    for origin, vals, color in [
        ('Trong nước', d['dp_vn'], C_DOM),
        ('Ngoài nước', d['dp_nn'], C_IMP),
    ]:
        if origin not in groups:
            continue
        fig.add_trace(go.Bar(
            name=origin, x=PRICE_LABELS, y=vals,
            marker=dict(color=color, line=dict(color='rgba(0,0,0,0)')),
            text=[f'{v:.0f}%' for v in vals],
            textposition='outside',
            textfont=dict(size=10, color=TXT),
            cliponaxis=False,
            hovertemplate=f'<b>%{{x}}</b><br>{origin}: %{{y:.1f}}%<extra></extra>',
        ))

    _theme(fig, height=340,
           title_text='Tỉ lệ sản phẩm áp dụng giảm giá sâu (> 30%)',
           barmode='group',
           xaxis=dict(showgrid=False),
           yaxis=dict(title='Tỉ lệ (%)', range=[0, 115]),
           legend=_leg(),
           margin=dict(l=14, r=14, t=68, b=14))
    return fig


# ══════════════════════════════════════════════════════════════
#  KPI
# ══════════════════════════════════════════════════════════════

KPI_TIPS = {
    'sweet_spot':  'Phân khúc giá có trung vị lượt bán cao nhất trong tập dữ liệu đang lọc.',
    'hit_rate':    'Tỉ lệ sản phẩm có ít nhất 1 lượt bán trên toàn bộ tập dữ liệu đang lọc.',
    'disc_avg':    'Mức giảm giá trung bình của tất cả sản phẩm trong tập đang lọc.',
    'deep_disc':   'Tỉ lệ sản phẩm áp dụng giảm giá sâu hơn 30% so với giá gốc.',
}


def _compute_kpis(df):
    if len(df) == 0:
        return 'N/A', 0, 0, 0

    med = (df.groupby('price_segment', observed=True)['sold_count']
           .median().reindex(PRICE_ORDER).fillna(0))
    sweet = PRICE_LABELS[int(med.values.argmax())] if med.sum() > 0 else 'N/A'

    hr = (df['sold_count'] > 0).mean() * 100
    disc_avg = df['discount_rate'].mean() if 'discount_rate' in df.columns else 0
    deep = (df['discount_rate'] > 30).mean() * 100 if 'discount_rate' in df.columns else 0

    return sweet, hr, disc_avg, deep


def kpi(icon, val, lbl, color, bg, tip_key):
    return html.Div([
        html.Div(icon, className='p4-kpi-dot', style={'background': bg}),
        html.Div([
            html.P(val, className='p4-kpi-val', style={'color': color}),
            html.P(lbl, className='p4-kpi-lbl'),
        ]),
        html.Span(KPI_TIPS[tip_key], className='p4-kpi-tip'),
    ], className='p4-kpi-item', style={'cursor': 'help'})


def make_kpi_row(df):
    sweet, hr, disc_avg, deep = _compute_kpis(df)
    return [
        kpi('🎯', str(sweet),           'Sweet Spot',           TEAL,    'rgba(20,184,166,.15)',  'sweet_spot'),
        kpi('✅', f'{hr:.1f}%',          'Hit Rate',             EMERALD, 'rgba(52,211,153,.12)',  'hit_rate'),
        kpi('💰', f'{disc_avg:.1f}%',    'Giảm giá TB',          GOLD,    'rgba(245,158,11,.12)',  'disc_avg'),
        kpi('🔥', f'{deep:.1f}%',        'Giảm sâu > 30%',       C_IMP,   'rgba(248,113,113,.12)', 'deep_disc'),
    ]


# ══════════════════════════════════════════════════════════════
#  CHART PANEL (giống chart_panel_3 của page3)
# ══════════════════════════════════════════════════════════════

def chart_panel_4(graph_id, figure, question,
                  icon='📊', icon_bg='rgba(20,184,166,0.15)',
                  flex='1', min_w='280px', glow='g-teal'):
    insight_class, insight_text = INSIGHTS.get(graph_id, ('it', ''))
    btn_id = f'{graph_id}-ibtn'
    box_id = f'{graph_id}-ibox'
    return html.Div([
        html.Div([
            html.Div([
                html.Div(icon, className='p4-card-icon', style={'background': icon_bg}),
                html.P(question, className='p4-card-subtitle'),
            ], style={'display': 'flex', 'alignItems': 'center', 'gap': '10px', 'flex': '1'}),
            html.Button('i', id=btn_id, className='p4-insight-btn', n_clicks=0),
        ], className='p4-card-header-row', style={'justifyContent': 'space-between'}),
        html.Div(insight_text, id=box_id,
                 className=f'p4-insight {insight_class}',
                 style={'display': 'none'}),
        dcc.Graph(id=graph_id, figure=figure, config={'displayModeBar': False}),
    ], className=f'p4-card p4-card-glow {glow}',
       style={'flex': flex, 'minWidth': min_w})


# ══════════════════════════════════════════════════════════════
#  LAYOUT
# ══════════════════════════════════════════════════════════════

def layout():
    df = apply_filters()

    return html.Div([

        # ── HERO ──────────────────────────────────────────────
        html.Div([
            html.Div(style={
                'position': 'absolute', 'width': '280px', 'height': '280px',
                'borderRadius': '50%', 'background': 'rgba(20,184,166,0.10)',
                'top': '-90px', 'right': '-60px', 'pointerEvents': 'none',
            }),
            html.Div(style={
                'position': 'absolute', 'width': '150px', 'height': '150px',
                'borderRadius': '50%', 'background': 'rgba(245,158,11,0.08)',
                'bottom': '-40px', 'right': '180px', 'pointerEvents': 'none',
            }),

            html.Div([
                html.H1('Phân khúc giá & Chiến lược khuyến mãi · T3/2026', style={
                    'margin': '0 0 6px 0', 'fontSize': '30px', 'fontWeight': '700',
                    'color': '#FFFFFF', 'letterSpacing': '-0.02em', 'lineHeight': '1.15',
                }),
                html.P('Phân tích Sweet Spot phân khúc giá và hiệu quả chiến dịch giảm giá sâu (T3/2026)', style={
                    'margin': '0', 'fontSize': '13px',
                    'color': 'rgba(255,255,255,0.55)', 'fontWeight': '400',
                }),
            ], style={'marginBottom': '20px'}),

            # Filter bar
            html.Div([
                html.Div([
                    html.Span('Lọc:', style={
                        'fontSize': '12px', 'color': 'rgba(255,255,255,0.6)',
                        'fontWeight': '600', 'marginRight': '8px',
                        'textTransform': 'uppercase', 'letterSpacing': '0.06em',
                    }),
                    html.Div([
                        dcc.Dropdown(
                            id='p4-filter-type',
                            options=[{'label': 'Tất cả ngành', 'value': 'all'}] +
                                    [{'label': t, 'value': t} for t in PRODUCT_TYPES_ALL],
                            value='all', multi=False, clearable=False,
                            className='p4-filter-pill-select',
                            style={
                                'color': '#111827', 'backgroundColor': '#FFFFFF',
                                'borderRadius': '8px', 'fontSize': '16px',
                                'minWidth': '155px',
                            }
                        ),
                    ], style={'minWidth': '165px', 'display': 'flex'}),
                    html.Div([
                        dcc.Dropdown(
                            id='p4-filter-price',
                            options=[{'label': 'Tất cả giá', 'value': '__all__'}] +
                                    [{'label': p, 'value': p} for p in PRICE_SEGMENTS_ALL],
                            value='__all__', multi=False, clearable=False,
                            className='p4-filter-pill-select',
                            style={
                                'color': '#111827',
                                'backgroundColor': '#FFFFFF',
                                'borderRadius': '8px',
                                'fontSize': '16px',
                                'minWidth': '150px',
                            }
                        ),
                    ], style={'minWidth': '160px', 'display': 'flex'}),
                ], style={'display': 'flex', 'alignItems': 'center', 'gap': '12px'}),

                html.Div([
                    html.Span('Xuất xứ:', style={
                        'fontSize': '13px', 'color': 'rgba(255,255,255,0.6)',
                        'fontWeight': '600', 'marginRight': '8px',
                        'textTransform': 'uppercase', 'letterSpacing': '0.06em',
                    }),
                    dcc.RadioItems(
                        id='p4-filter-origin',
                        options=ORIGIN_OPTIONS,
                        value='all', inline=True,
                        style={'display': 'flex', 'gap': '14px'},
                        className='p4-filter-radio',
                    ),
                ], style={'display': 'flex', 'alignItems': 'center', 'gap': '13px'}),
            ], style={
                'display': 'flex', 'alignItems': 'center', 'gap': '20px',
                'paddingTop': '16px', 'borderTop': '1px solid rgba(255,255,255,0.12)',
                'flexWrap': 'wrap',
            }),
        ], className='p4-hero', style={'paddingBottom': '20px'}),

        # ── KPI ROW ───────────────────────────────────────────
        html.Div(
            id='p4-kpi-row',
            children=make_kpi_row(df),
            className='p4-kpi-row',
            style={'marginBottom': '28px'},
        ),

        # ── CHARTS ────────────────────────────────────────────
        html.Div([

            # MT1 — Sweet Spot
            html.Div([
                chart_panel_4(
                    'p4-c-median',
                    make_median_chart(df),
                    'Phân khúc giá nào mang lại lượt bán ổn định nhất?',
                    icon='📈', icon_bg='rgba(20,184,166,0.15)',
                    flex='1', min_w='280px', glow='g-teal',
                ),
                chart_panel_4(
                    'p4-c-hitrate',
                    make_hitrate_chart(df),
                    'Phân khúc nào có tỉ lệ sản phẩm bán được cao nhất?',
                    icon='🎯', icon_bg='rgba(20,184,166,0.15)',
                    flex='1', min_w='280px', glow='g-teal',
                ),
            ], className='p4-row', style={'marginBottom': '16px', 'gap': '16px'}),

            # MT2 — Discount
            html.Div([
                chart_panel_4(
                    'p4-c-disc-comp',
                    make_disc_comparison_chart(df),
                    'Giảm giá sâu có thực sự tăng lượt bán không?',
                    icon='💰', icon_bg='rgba(245,158,11,0.15)',
                    flex='1', min_w='280px', glow='g-gold',
                ),
                chart_panel_4(
                    'p4-c-disc-pen',
                    make_disc_penetration_chart(df),
                    'Nhóm nào đang dùng khuyến mãi sâu nhiều nhất?',
                    icon='🔥', icon_bg='rgba(245,158,11,0.15)',
                    flex='1', min_w='280px', glow='g-gold',
                ),
            ], className='p4-row', style={'marginBottom': '16px', 'gap': '16px'}),

        ], style={'padding': '0'}),

        # ── FOOTER ────────────────────────────────────────────
        html.Div([
            html.Span('⚠️ Dữ liệu crawl từ Tiki tháng 3/2026'),
            html.Span('Trung vị lượt bán đã qua IQR capping loại outlier'),
            html.Span('Nhóm 05 · FIT-HCMUS'),
        ], className='p4-footer', style={'margin': '24px 0 0 0'}),

    ], className='p4-page', style={'padding': '0'})


# ══════════════════════════════════════════════════════════════
#  CALLBACKS
# ══════════════════════════════════════════════════════════════

@callback(
    [
        Output('p4-kpi-row',       'children'),
        Output('p4-c-median',      'figure'),
        Output('p4-c-hitrate',     'figure'),
        Output('p4-c-disc-comp',   'figure'),
        Output('p4-c-disc-pen',    'figure'),
    ],
    [
        Input('p4-filter-type',   'value'),
        Input('p4-filter-origin', 'value'),
    ]
)
def update_p4(selected_type, selected_origin):
    df = apply_filters(selected_type, selected_origin)
    return (
        make_kpi_row(df),
        make_median_chart(df),
        make_hitrate_chart(df),
        make_disc_comparison_chart(df),
        make_disc_penetration_chart(df),
    )


# ── Insight toggle callbacks ──────────────────────────────────
def _make_toggle(gid):
    @callback(
        Output(f'{gid}-ibox', 'style'),
        Input(f'{gid}-ibtn', 'n_clicks'),
        prevent_initial_call=True,
    )
    def _toggle(n):
        return {'display': 'flex'} if n % 2 == 1 else {'display': 'none'}
    return _toggle

for _gid in INSIGHTS:
    _make_toggle(_gid)