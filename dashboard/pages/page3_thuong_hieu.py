# pages/page3_chat_luong.py
# ╔══════════════════════════════════════════════════════════════╗
# ║  Trang 3 — Thương hiệu & Hệ sinh thái Gian hàng             ║
# ║  MT1: Tiki Verified (stacked bar + grouped bar)              ║
# ║  MT2: Top 10 brands combined + Bubble combined               ║
# ║  MT3: Quốc gia nhập khẩu (donut + dual bar)                 ║
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
    path='/thuong-hieu',
    name='Thương hiệu',
    title='Thương hiệu | Mỹ phẩm Tiki',
    order=3,
)

# ── Data ─────────────────────────────────────────────────────
df_full, df_vn_full, df_nn_full = load_data()
PRICE_ORDER = ['Dưới 100k', '100k – 300k', '300k – 700k', '700k – 2tr', 'Trên 2tr']

PRODUCT_TYPES_ALL = sorted(df_full['product_type'].dropna().unique().tolist())
ORIGIN_OPTIONS = [
    {'label': 'Tất cả', 'value': 'all'},
    {'label': 'Trong nước', 'value': 'domestic'},
    {'label': 'Ngoài nước', 'value': 'import'},
]
PRICE_SEGMENTS_ALL = [p for p in PRICE_ORDER if p in df_full['price_segment'].unique()]

# ── Dark palette ─────────────────────────────────────────────
BG      = '#172542'
SURFACE = '#1A2A4D'
CARD    = '#1C2D55'
BORDER2 = 'rgba(255,255,255,0.13)'
GRID    = 'rgba(255,255,255,0.05)'
TXT     = '#F0F6FF'
SUBTXT  = '#94A3B8'
MUTED   = '#4B6178'

CYAN    = '#22D3EE'
BLUE    = '#60A5FA'
INDIGO  = '#818CF8'
AMBER   = '#FBBF24'
EMERALD = '#34D399'
ROSE    = '#FB7185'
VIOLET  = '#A78BFA'
ORANGE  = '#FB923C'
C_DOM   = '#38BDF8'
C_IMP   = '#F87171'

COUNTRY_COLS = [CYAN, AMBER, EMERALD, VIOLET, ORANGE, ROSE, BLUE, INDIGO,
                '#E879F9', '#4ADE80']


# ══════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════

def hex_to_rgba(hex_color, alpha=1):
    h = hex_color.lstrip('#')
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f'rgba({r},{g},{b},{alpha})'


def apply_filters(product_type='all', origin='all', price_segments=None):
    df = df_full.copy()
    if product_type and product_type != 'all':
        df = df[df['product_type'] == product_type]
    if price_segments and len(price_segments) > 0:
        df = df[df['price_segment'].isin(price_segments)]
    if origin == 'domestic':
        df = df[df['origin_class_corrected'] == 'Trong nước']
    elif origin == 'import':
        df = df[df['origin_class_corrected'] == 'Ngoài nước']
    df_vn = df[df['origin_class_corrected'] == 'Trong nước']
    df_nn = df[df['origin_class_corrected'] == 'Ngoài nước']
    return df, df_vn, df_nn


def _theme(fig, height=320, title_text=None, **kw):
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
                xanchor='right', x=1, font=dict(size=10, color=SUBTXT),
                bgcolor='rgba(0,0,0,0)', **kw)


def _brand_profile(df_sub, n=5):
    if len(df_sub) == 0:
        return pd.DataFrame()
    p = df_sub.groupby('brand_name').agg(
        revenue=('estimated_revenue', 'sum'),
        products=('product_id', 'count'),
        sold_total=('sold_count', 'sum'),
        avg_price=('price', 'mean'),
    ).sort_values('revenue', ascending=False).head(n)
    p['rev_B'] = p['revenue'] / 1e9
    return p


# ══════════════════════════════════════════════════════════════
#  MT1 — TIKI VERIFIED
# ══════════════════════════════════════════════════════════════

def make_verified_stacked(df):
    groups = ['Trong nước', 'Ngoài nước']
    rates_v, rates_n = [], []
    for g in groups:
        sub = df[df['origin_class_corrected'] == g]
        rv = sub['tiki_verified'].mean() * 100 if len(sub) else 0
        rates_v.append(rv); rates_n.append(100 - rv)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name='Verified', x=groups, y=rates_v,
        marker=dict(color=[C_DOM, C_IMP], line=dict(color='rgba(0,0,0,0)')),
        text=[f'{v:.1f}%' for v in rates_v],
        textposition='inside', textfont=dict(size=14, weight=700, color='white'),
        hovertemplate='<b>%{x}</b><br>Verified: %{y:.1f}%<extra></extra>',
    ))
    fig.add_trace(go.Bar(
        name='Chưa Verified', x=groups, y=rates_n,
        marker=dict(color=[hex_to_rgba(C_DOM, 0.19), hex_to_rgba(C_IMP, 0.19)],
                    line=dict(color='rgba(0,0,0,0)')),
        text=[f'{v:.1f}%' for v in rates_n],
        textposition='inside', textfont=dict(size=13, color=SUBTXT),

        hovertemplate='<b>%{x}</b><br>Chưa Verified: %{y:.1f}%<extra></extra>',
    ))
    _theme(fig, height=310, title_text='Tỉ lệ Tiki Verified theo nguồn gốc',
           barmode='stack',
           xaxis=dict(showgrid=False),
           yaxis=dict(range=[0, 115], title='Tỉ lệ (%)'),
           legend=dict(orientation='h', yanchor='bottom', y=1.02,
                       xanchor='right', x=1,
                       font=dict(size=10, color=SUBTXT), bgcolor='rgba(0,0,0,0)'),
           showlegend=True,
           margin=dict(l=14, r=14, t=68, b=14))
    return fig


def make_verified_impact(df):
    df_sold = df[df['sold_count'] > 0].copy()
    groups = ['Trong nước', 'Ngoài nước']
    ver_sold, not_sold = [], []
    for g in groups:
        sub = df_sold[df_sold['origin_class_corrected'] == g]
        sv = sub[sub['tiki_verified'] == 1]['sold_count'].mean() if len(sub[sub['tiki_verified'] == 1]) else 0
        sn = sub[sub['tiki_verified'] == 0]['sold_count'].mean() if len(sub[sub['tiki_verified'] == 0]) else 0
        ver_sold.append(sv); not_sold.append(sn)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name='Chưa Verified', x=groups, y=not_sold,
        marker=dict(color=[hex_to_rgba(C_DOM, 0.27), hex_to_rgba(C_IMP, 0.27)],
                    line=dict(color='rgba(0,0,0,0)')),
        text=[f'{v:,.0f}' for v in not_sold],
        textposition='outside', textfont=dict(size=11, color=SUBTXT),
        hovertemplate='<b>%{x} · Chưa Verified</b><br>TB lượt bán: %{y:,.0f}<extra></extra>',
    ))
    fig.add_trace(go.Bar(
        name='Verified', x=groups, y=ver_sold,
        marker=dict(color=[C_DOM, C_IMP], line=dict(color='rgba(0,0,0,0)')),
        text=[f'{v:,.0f}' for v in ver_sold],
        textposition='outside', textfont=dict(size=11, weight=700, color=TXT),
        hovertemplate='<b>%{x} · Verified</b><br>TB lượt bán: %{y:,.0f}<extra></extra>',
    ))
    _theme(fig, height=310, title_text='Ảnh hưởng của Tiki Verified lên lượt bán',
           barmode='group',
           xaxis=dict(showgrid=False),
           yaxis=dict(title='Lượt bán TB'),
           legend=dict(orientation='h', yanchor='bottom', y=1.02,
                       xanchor='right', x=1,
                       font=dict(size=10, color=SUBTXT), bgcolor='rgba(0,0,0,0)'),
           showlegend=True,
           margin=dict(l=14, r=14, t=68, b=14))
    return fig


# ══════════════════════════════════════════════════════════════
#  MT2 — TOP 10 BRANDS
# ══════════════════════════════════════════════════════════════

def make_top10_combined(df_vn, df_nn):
    top_vn = _brand_profile(df_vn, 10)
    top_nn = _brand_profile(df_nn, 10)
    fig = go.Figure()

    for df_b, color, name in [(top_vn, C_DOM, 'Trong nước'),
                               (top_nn, C_IMP, 'Ngoài nước')]:
        if len(df_b) == 0:
            continue
        n = len(df_b)
        brands  = df_b.index.tolist()[::-1]
        revs    = df_b['rev_B'].tolist()[::-1]
        soleds  = df_b['sold_total'].tolist()[::-1]
        gcols   = [f'rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},'
                   f'{0.45+0.55*(i/max(n-1,1)):.2f})' for i in range(n)]
        fig.add_trace(go.Bar(
            name=name,
            x=revs, y=brands, orientation='h',
            marker=dict(color=gcols[::-1], line=dict(color='rgba(0,0,0,0)')),
            text=[f'{v:.1f} tỉ' for v in revs],
            textposition='outside',
            textfont=dict(size=9.5, weight=700, color=TXT),
            customdata=[[s] for s in soleds],
            hovertemplate='<b>%{y}</b><br>Doanh thu: %{x:.1f} tỉ<br>Lượt bán: %{customdata[0]:,}<extra></extra>',
            cliponaxis=False,
        ))

    _theme(fig, height=520, title_text='Top 10 thương hiệu: Nội địa vs Quốc tế',
           showlegend=True, barmode='overlay',
           legend=_leg(),
           xaxis=dict(title='Doanh thu (tỉ VNĐ)', showgrid=True),
           yaxis=dict(showgrid=False, tickfont=dict(size=10)),
           margin=dict(l=14, r=70, t=68, b=28))
    return fig


def make_bubble_combined(df_vn, df_nn):
    fig = go.Figure()
    for df_b, color, name in [(df_vn, C_DOM, 'Trong nước'),
                               (df_nn, C_IMP, 'Ngoài nước')]:
        top = _brand_profile(df_b, 10)
        if len(top) == 0:
            continue
        rev_sqrt = np.sqrt(top['rev_B'].clip(lower=0.1))
        # Smaller bubbles: max diameter 38 to reduce overlap
        sizes    = (rev_sqrt / rev_sqrt.max() * 38 + 6).tolist()
        fig.add_trace(go.Scatter(
            name=name,
            x=top['avg_price'] / 1000,
            y=top['sold_total'],
            mode='markers+text',
            marker=dict(
                size=sizes,
                color=hex_to_rgba(color, 0.65),
                line=dict(color=color, width=1.2),
                sizemode='diameter', sizemin=6,
            ),
        text=[''] * len(top),   # ẩn hết text mặc định
        textposition='top center',
        textfont=dict(size=8, color=TXT, weight=700),
            customdata=top['rev_B'].tolist(),
            hovertemplate='<b>%{text}</b><br>Giá TB: %{x:.0f}k<br>'
                          'Lượt bán: %{y:,}<br>Doanh thu: %{customdata:.1f} tỉ<extra></extra>',
        ))

    # Gộp tất cả brand để annotate, chỉ show brand có size > 12
    for df_b, color in [(df_vn, C_DOM), (df_nn, C_IMP)]:
        top = _brand_profile(df_b, 10)
        if len(top) == 0:
            continue
        rev_sqrt = np.sqrt(top['rev_B'].clip(lower=0.1))
        sizes = (rev_sqrt / rev_sqrt.max() * 38 + 6).tolist()
        for i, (brand, sz) in enumerate(zip(top.index, sizes)):
            if sz < 15:   # bỏ qua bubble quá nhỏ
                continue
            fig.add_annotation(
                x=top['avg_price'].iloc[i] / 1000,
                y=top['sold_total'].iloc[i],
                text=f'<b>{brand}</b>',
                showarrow=False,
                yshift=sz / 2 + 6,   # đẩy lên trên bubble
                font=dict(size=8, color=TXT),
                bgcolor='rgba(0,0,0,0)',
                xanchor='center',
            )


    _theme(fig, height=460, title_text='Định vị chiến lược: Nội địa vs Quốc tế',
           showlegend=True, legend=_leg(),
           xaxis=dict(title='Giá TB (nghìn VNĐ)', showgrid=True),
           yaxis=dict(title='Tổng lượt bán', showgrid=True),
           margin=dict(l=14, r=14, t=68, b=28))
    fig.update_yaxes(tickformat=',')
    return fig


# ══════════════════════════════════════════════════════════════
#  MT3 — QUỐC GIA NHẬP KHẨU
# ══════════════════════════════════════════════════════════════

def _country_data(df_nn):
    if len(df_nn) == 0:
        return pd.DataFrame()
    cp = df_nn.groupby('origin_corrected').agg(
        products=('product_id', 'count'),
        revenue=('estimated_revenue', 'sum'),
        sold=('sold_count', 'sum'),
        brands=('brand_name', 'nunique'),
    ).sort_values('products', ascending=False)
    cp['rev_B']   = cp['revenue'] / 1e9
    cp['pct_sp']  = cp['products'] / cp['products'].sum() * 100
    cp['pct_rev'] = cp['revenue']  / cp['revenue'].sum()  * 100
    return cp


def make_country_donut(df_nn):
    cp = _country_data(df_nn)
    if len(cp) == 0:
        return go.Figure()
    top7   = cp.head(7)
    others = cp.iloc[7:]['products'].sum() if len(cp) > 7 else 0
    labels = list(top7.index)
    values = list(top7['products'])
    if others > 0:
        labels.append('Các nước khác'); values.append(int(others))
    colors = COUNTRY_COLS[:len(top7)] + ([MUTED] if others > 0 else [])
    total  = int(cp['products'].sum())
    # Ẩn % cho slice < 4% tránh chồng chéo
    text_vals = [f'{v/total*100:.0f}%' if v / total * 100 >= 4 else '' for v in values]

    fig = go.Figure(go.Pie(
        values=values, labels=labels,
        marker=dict(colors=colors, line=dict(color=CARD, width=2.5)),
        hole=0.62,
        text=text_vals, textinfo='text',
        textfont=dict(size=11, weight=700),
        hovertemplate='<b>%{label}</b><br>%{value:,} SP (%{percent})<extra></extra>',
        pull=[0.04] + [0] * (len(values) - 1),
        sort=False, direction='clockwise', textposition='inside',
    ))
    fig.add_annotation(
        text=f'<b style="font-size:20px">{total:,}</b><br>'
             f'<span style="font-size:10px;color:{SUBTXT}">SP nhập khẩu</span>',
        x=0.5, y=0.5, showarrow=False, font=dict(color=TXT),
        xref='paper', yref='paper',
    )
    _theme(fig, height=360, title_text='Tỷ trọng sản phẩm nhập khẩu theo quốc gia',
           showlegend=True,
           legend=dict(orientation='h', yanchor='top', y=-0.06,
                       xanchor='center', x=0.5,
                       font=dict(size=9.5, color=SUBTXT), bgcolor='rgba(0,0,0,0)'),
           margin=dict(l=14, r=14, t=68, b=60))
    return fig


def make_country_compare(df_vn, df_nn):
    cp = _country_data(df_nn)
    if len(cp) == 0:
        return go.Figure()
    TOP3       = cp.sort_values('revenue', ascending=False).head(3).index.tolist()
    compare    = TOP3 + ['Việt Nam']
    bar_colors = COUNTRY_COLS[:3] + [C_DOM]
    rev_vals, sold_vals = [], []
    for c in compare:
        if c == 'Việt Nam':
            rev_vals.append(df_vn['estimated_revenue'].sum() / 1e9)
            sold_vals.append(df_vn['sold_count'].sum() / 1e3)
        else:
            sub = df_nn[df_nn['origin_corrected'] == c]
            rev_vals.append(sub['estimated_revenue'].sum() / 1e9)
            sold_vals.append(sub['sold_count'].sum() / 1e3)
    vn_rev = rev_vals[-1]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name='Doanh thu (tỉ VNĐ)', x=compare, y=rev_vals,
        marker=dict(color=bar_colors, line=dict(color='rgba(0,0,0,0)')),
        text=[f'{v:.0f}' for v in rev_vals],
        textposition='auto', textfont=dict(size=11, weight=700, color=TXT),
        hovertemplate='<b>%{x}</b><br>Doanh thu: %{y:.1f} tỉ<extra></extra>',
        yaxis='y',
    ))
    fig.add_trace(go.Bar(
        name='Lượt bán (nghìn)', x=compare, y=sold_vals,
        marker=dict(color=[hex_to_rgba(c, 0.38) for c in bar_colors],
                    line=dict(color='rgba(0,0,0,0)', width=0)),
        text=[f'{v:,.0f}k' for v in sold_vals],
        textposition='inside', textfont=dict(size=10, color=SUBTXT),
        hovertemplate='<b>%{x}</b><br>Lượt bán: %{y:,.0f}k<extra></extra>',
        yaxis='y2',
    ))
    fig.add_hline(y=vn_rev, line_dash='dot', line_color=C_DOM,
                  line_width=1.5, opacity=0.55, yref='y')
    _theme(fig, height=380, title_text='Top 3 quốc gia nhập khẩu vs Việt Nam',
           barmode='group',
           xaxis=dict(showgrid=False),
           yaxis=dict(title='Doanh thu (tỉ VNĐ)', side='left', showgrid=True),
           yaxis2=dict(title='Lượt bán (nghìn)', side='right',
                       overlaying='y', showgrid=False),
           legend=_leg(), showlegend=True,
           margin=dict(l=14, r=60, t=68, b=28))
    return fig


# ══════════════════════════════════════════════════════════════
#  KPI
# ══════════════════════════════════════════════════════════════

KPI_TIPS = {
    'brands':      'Số thương hiệu có ít nhất 1 sản phẩm trong tập dữ liệu đang hiển thị.',
    'verified':    'Tỉ lệ % sản phẩm có nhãn Tiki Verified trên tổng sản phẩm đang lọc.',
    'countries':   'Số quốc gia xuất xứ thương hiệu khác nhau trong nhóm hàng nhập khẩu.',
    'top_brand':   'Thương hiệu nhập khẩu có doanh thu ước tính cao nhất trong tập đang lọc.',
    'top_country': 'Quốc gia nhập khẩu có tổng doanh thu ước tính lớn nhất.',
}


def kpi(icon, val, lbl, color, bg, tip_key):
    return html.Div([
        html.Div(icon, className='p3-kpi-dot', style={'background': bg}),
        html.Div([
            html.P(val,  className='p3-kpi-val', style={'color': color}),
            html.P(lbl, className='p3-kpi-lbl'),
        ]),
        html.Span(KPI_TIPS[tip_key], className='p3-kpi-tip'),
    ], className='p3-kpi-item', style={'cursor': 'help'})


def _compute_kpis(df, df_vn, df_nn):
    n_brands    = df['brand_name'].nunique()
    n_countries = df_nn['origin_corrected'].nunique() if len(df_nn) else 0
    pct_ver     = df['tiki_verified'].mean() * 100 if len(df) else 0
    top_brand   = _brand_profile(df, 1).index[0] if len(df) else 'N/A'
    top_country = (df_nn.groupby('origin_corrected')['estimated_revenue']
                   .sum().idxmax() if len(df_nn) else 'N/A')
    return n_brands, n_countries, pct_ver, top_brand, top_country


def make_kpi_row(n_brands, n_countries, pct_ver, top_brand, top_country):
    return [
        kpi('🏷️', f'{n_brands:,}',     'Thương hiệu',        CYAN,    'rgba(34,211,238,.15)',  'brands'),
        kpi('✅', f'{pct_ver:.1f}%',    'Tiki Verified',       EMERALD, 'rgba(52,211,153,.12)',  'verified'),
        kpi('🌏', f'{n_countries}',      'Quốc gia nhập khẩu', VIOLET,  'rgba(167,139,250,.12)', 'countries'),
        kpi('🥇', top_brand[:12],        'Top brand nhập khẩu', C_DOM,   'rgba(56,189,248,.12)',  'top_brand'),
        kpi('🌟', top_country,           'Top quốc gia nhập khẩu',     AMBER,   'rgba(251,191,36,.12)',  'top_country'),
    ]


# ══════════════════════════════════════════════════════════════
#  CHART CARD with icon + question
# ══════════════════════════════════════════════════════════════

def chart_panel_3(graph_id, figure, question,
                  icon='📊', icon_bg='rgba(34,211,238,0.15)',
                  flex='1', min_w='280px', glow='g-cyan'):
    return html.Div([
        html.Div([
            html.Div(icon, className='p3-card-icon', style={'background': icon_bg}),
            html.P(question, className='p3-card-subtitle'),
        ], className='p3-card-header-row'),
        dcc.Graph(id=graph_id, figure=figure, config={'displayModeBar': False}),
    ], className=f'p3-card p3-card-glow {glow}',
       style={'flex': flex, 'minWidth': min_w})


# ══════════════════════════════════════════════════════════════
#  LAYOUT
# ══════════════════════════════════════════════════════════════

def layout():
    df, df_vn, df_nn = apply_filters()
    n_brands, n_countries, pct_ver, top_brand, top_country = _compute_kpis(df, df_vn, df_nn)

    return html.Div([

        # ── HERO ──────────────────────────────────────────────
        html.Div([
            html.Div(style={
                'position': 'absolute', 'width': '280px', 'height': '280px',
                'borderRadius': '50%', 'background': 'rgba(34,211,238,0.10)',
                'top': '-90px', 'right': '-60px', 'pointerEvents': 'none',
            }),
            html.Div(style={
                'position': 'absolute', 'width': '150px', 'height': '150px',
                'borderRadius': '50%', 'background': 'rgba(96,165,250,0.08)',
                'bottom': '-40px', 'right': '180px', 'pointerEvents': 'none',
            }),

            html.Div([
                html.H1('Thương hiệu & Hệ sinh thái · T3/2026', style={
                    'margin': '0 0 6px 0', 'fontSize': '30px', 'fontWeight': '700',
                    'color': '#FFFFFF', 'letterSpacing': '-0.02em', 'lineHeight': '1.15',
                }),
                html.P('Phân tích thương hiệu và định vị chiến lược trên Tiki (T3/2026)', style={
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
                            id='p3-filter-type',
                            options=[{'label': 'Tất cả ngành', 'value': 'all'}] +
                                    [{'label': t, 'value': t} for t in PRODUCT_TYPES_ALL],
                            value='all', multi=False, clearable=False,
                            className='p3-filter-pill-select',
                        ),
                    ], style={'minWidth': '165px', 'display': 'flex'}),
                    html.Div([
                        dcc.Dropdown(
                            id='p3-filter-price',
                            options=[{'label': 'Tất cả giá', 'value': '__all__'}] +
                                    [{'label': p, 'value': p} for p in PRICE_SEGMENTS_ALL],
                            value='__all__', multi=False, clearable=False,
                            className='p3-filter-pill-select',
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
                        id='p3-filter-origin',
                        options=ORIGIN_OPTIONS,
                        value='all', inline=True,
                        style={'display': 'flex', 'gap': '14px'},
                        className='p3-filter-radio',
                    ),
                ], style={'display': 'flex', 'alignItems': 'center', 'gap': '13px'}),
            ], style={
                'display': 'flex', 'alignItems': 'center', 'gap': '20px',
                'paddingTop': '16px', 'borderTop': '1px solid rgba(255,255,255,0.12)',
                'flexWrap': 'wrap',
            }),
        ], className='p3-hero', style={'paddingBottom': '20px'}),

        # ── KPI ROW ───────────────────────────────────────────
        html.Div(
            id='p3-kpi-row',
            children=make_kpi_row(n_brands, n_countries, pct_ver, top_brand, top_country),
            className='p3-kpi-row',
            style={'marginBottom': '28px'},
        ),

        # ── CHARTS ────────────────────────────────────────────
        html.Div([

            # MT1
            html.Div([
                chart_panel_3(
                    'p3-c-verified-stacked',
                    make_verified_stacked(df),
                    'Nhóm nào đầu tư xây dựng uy tín nền tảng nhiều hơn?',
                    icon='✅', icon_bg='rgba(34,211,238,0.15)',
                    flex='1', min_w='280px', glow='g-cyan',
                ),
                chart_panel_3(
                    'p3-c-verified-impact',
                    make_verified_impact(df),
                    'Tiki Verified có thực sự tạo ra lợi thế bán hàng không?',
                    icon='📈', icon_bg='rgba(34,211,238,0.15)',
                    flex='1', min_w='280px', glow='g-cyan',
                ),
            ], className='p3-row', style={'marginBottom': '16px', 'gap': '16px'}),

            # MT2
            html.Div([
                chart_panel_3(
                    'p3-c-top10',
                    make_top10_combined(df_vn, df_nn),
                    'Thương hiệu nào đang dẫn dắt thị phần doanh thu?',
                    icon='🏆', icon_bg='rgba(251,191,36,0.15)',
                    flex='1', min_w='300px', glow='g-amber',
                ),
                chart_panel_3(
                    'p3-c-bubble',
                    make_bubble_combined(df_vn, df_nn),
                    'Ai đang cạnh tranh bằng giá, ai bằng thương hiệu?',
                    icon='🎯', icon_bg='rgba(251,191,36,0.15)',
                    flex='1', min_w='300px', glow='g-amber',
                ),
            ], className='p3-row', style={'marginBottom': '16px', 'gap': '16px'}),

            # MT3
            html.Div([
                chart_panel_3(
                    'p3-c-donut',
                    make_country_donut(df_nn),
                    'Quốc gia nào chiếm nhiều "kệ hàng" nhập khẩu nhất?',
                    icon='🌍', icon_bg='rgba(52,211,153,0.15)',
                    flex='1', min_w='300px', glow='g-emerald',
                ),
                chart_panel_3(
                    'p3-c-compare',
                    make_country_compare(df_vn, df_nn),
                    'Top 3 đối thủ nhập khẩu so với Việt Nam như thế nào?',
                    icon='⚔️', icon_bg='rgba(52,211,153,0.15)',
                    flex='1.4', min_w='360px', glow='g-emerald',
                ),
            ], className='p3-row', style={'marginBottom': '16px', 'gap': '16px'}),

        ], style={'padding': '0'}),

        # ── FOOTER ────────────────────────────────────────────
        html.Div([
            html.Span('⚠️ Dữ liệu crawl từ Tiki tháng 3/2026'),
            html.Span('Doanh thu = sold_count × price (ước tính)'),
            html.Span('Nhóm 05 · FIT-HCMUS'),
        ], className='p3-footer', style={'margin': '24px 0 0 0'}),

    ], className='p3-page', style={'padding': '0'})


# ══════════════════════════════════════════════════════════════
#  CALLBACK
# ══════════════════════════════════════════════════════════════

@callback(
    [
        Output('p3-kpi-row',              'children'),
        Output('p3-c-verified-stacked',   'figure'),
        Output('p3-c-verified-impact',    'figure'),
        Output('p3-c-top10',              'figure'),
        Output('p3-c-bubble',             'figure'),
        Output('p3-c-donut',              'figure'),
        Output('p3-c-compare',            'figure'),
    ],
    [
        Input('p3-filter-type',   'value'),
        Input('p3-filter-origin', 'value'),
        Input('p3-filter-price',  'value'),
    ]
)
def update_p3(selected_type, selected_origin, selected_price):
    price_segs = None if (not selected_price or selected_price == '__all__') else [selected_price]
    df, df_vn, df_nn = apply_filters(selected_type, selected_origin, price_segs)
    n_brands, n_countries, pct_ver, top_brand, top_country = _compute_kpis(df, df_vn, df_nn)
    return (
        make_kpi_row(n_brands, n_countries, pct_ver, top_brand, top_country),
        make_verified_stacked(df),
        make_verified_impact(df),
        make_top10_combined(df_vn, df_nn),
        make_bubble_combined(df_vn, df_nn),
        make_country_donut(df_nn),
        make_country_compare(df_vn, df_nn),
    )