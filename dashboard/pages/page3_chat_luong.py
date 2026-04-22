# pages/page3_chat_luong.py
# ╔══════════════════════════════════════════════════════════════╗
# ║  Trang 3 — Thương hiệu & Hệ sinh thái Gian hàng             ║
# ║  Dark navy theme · Charts-first · Minimal text              ║
# ║  MT1: Tiki Verified (stacked bar + grouped bar)              ║
# ║  MT2: Top 10 brands (hbar + bubble)                         ║
# ║  MT3: Quốc gia nhập khẩu (donut + dual bar)                 ║
# ╚══════════════════════════════════════════════════════════════╝

import dash
from dash import html, dcc
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

# ── Dark palette ─────────────────────────────────────────────
BG       = '#172542'
SURFACE  = '#1A2A4D'
CARD     = '#1C2D55'
BORDER   = 'rgba(255,255,255,0.07)'
BORDER2  = 'rgba(255,255,255,0.13)'
GRID     = 'rgba(255,255,255,0.05)'
TXT      = '#F0F6FF'
SUBTXT   = '#94A3B8'
MUTED    = '#4B6178'

CYAN     = '#22D3EE'
BLUE     = '#60A5FA'
INDIGO   = '#818CF8'
AMBER    = '#FBBF24'
EMERALD  = '#34D399'
ROSE     = '#FB7185'
VIOLET   = '#A78BFA'
ORANGE   = '#FB923C'
C_DOM    = '#38BDF8'   # bright sky for domestic on dark bg
C_IMP    = '#F87171'   # bright red for import on dark bg

COUNTRY_COLS = [CYAN, AMBER, EMERALD, VIOLET, ORANGE, ROSE, BLUE, INDIGO,
                '#E879F9', '#4ADE80']


# ══════════════════════════════════════════════════════════════
#  CHART THEME HELPER
# ══════════════════════════════════════════════════════════════
def hex_to_rgba(hex_color, alpha=1):
    hex_color = hex_color.lstrip('#')
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return f'rgba({r},{g},{b},{alpha})'

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


# ══════════════════════════════════════════════════════════════
#  MT1 — TIKI VERIFIED
# ══════════════════════════════════════════════════════════════

def _verified_data():
    vs = df_full.groupby('origin_class_corrected')['tiki_verified'].agg(
        verified='sum', total='count'
    ).reset_index()
    vs['rate_v'] = vs['verified'] / vs['total'] * 100
    vs['rate_n'] = 100 - vs['rate_v']
    return vs


def make_verified_stacked():
    vs = _verified_data()
    groups = ['Trong nước', 'Ngoài nước']
    colors = [C_DOM, C_IMP]

    fig = go.Figure()
    # Verified bars
    rates_v = [vs.loc[vs['origin_class_corrected'] == g, 'rate_v'].values[0] for g in groups]
    rates_n = [vs.loc[vs['origin_class_corrected'] == g, 'rate_n'].values[0] for g in groups]

    fig.add_trace(go.Bar(
        name='Verified', x=groups, y=rates_v,
        marker=dict(color=[C_DOM, C_IMP],
                    line=dict(color='rgba(0,0,0,0)')),
        text=[f'{v:.1f}%' for v in rates_v],
        textposition='inside', textfont=dict(size=14, weight=700, color='white'),
        insidetextanchor='middle',
        hovertemplate='<b>%{x}</b><br>Verified: %{y:.1f}%<extra></extra>',
    ))
    fig.add_trace(go.Bar(
        name='Chưa Verified', x=groups, y=rates_n,
        marker=dict(
        color=[hex_to_rgba(C_DOM, 0.19), hex_to_rgba(C_IMP, 0.19)],
        line=dict(color='rgba(0,0,0,0)')
    ),
        text=[f'{v:.1f}%' for v in rates_n],
        textposition='inside', textfont=dict(size=13, color=SUBTXT),
        insidetextanchor='middle',
        hovertemplate='<b>%{x}</b><br>Chưa Verified: %{y:.1f}%<extra></extra>',
    ))

    _theme(fig, height=310,
           barmode='stack',
           xaxis=dict(showgrid=False),
           yaxis=dict(range=[0, 115], title='Tỉ lệ (%)'),
           legend=dict(orientation='h', yanchor='bottom', y=1.02,
                      xanchor='center', x=0.5, font=dict(size=10, color=SUBTXT),
                      bgcolor='rgba(0,0,0,0)'),
           showlegend=True)
    fig.update_layout(margin=dict(l=14, r=14, t=40, b=14))
    return fig


def make_verified_impact():
    df_sold = df_full[df_full['sold_count'] > 0].copy()
    impact = df_sold.groupby(['origin_class_corrected', 'tiki_verified']).agg(
        avg_sold=('sold_count', 'mean'),
    ).reset_index()

    groups = ['Trong nước', 'Ngoài nước']
    ver_sold = []
    not_sold = []
    for g in groups:
        sub_v = impact[(impact['origin_class_corrected'] == g) & (impact['tiki_verified'] == 1)]
        sub_n = impact[(impact['origin_class_corrected'] == g) & (impact['tiki_verified'] == 0)]
        ver_sold.append(sub_v['avg_sold'].values[0] if len(sub_v) else 0)
        not_sold.append(sub_n['avg_sold'].values[0] if len(sub_n) else 0)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name='Verified', x=groups, y=ver_sold,
        marker=dict(color=[C_DOM, C_IMP],
                    line=dict(color='rgba(0,0,0,0)')),
        text=[f'{v:,.0f}' for v in ver_sold],
        textposition='outside', textfont=dict(size=11, weight=700, color=TXT),
        hovertemplate='<b>%{x} · Verified</b><br>TB lượt bán: %{y:,.0f}<extra></extra>',
    ))
    fig.add_trace(go.Bar(
        name='Chưa Verified', x=groups, y=not_sold,
        marker=dict(
        color=[hex_to_rgba(C_DOM, 0.27), hex_to_rgba(C_IMP, 0.27)],
        line=dict(color='rgba(0,0,0,0)')
    ),
        text=[f'{v:,.0f}' for v in not_sold],
        textposition='outside', textfont=dict(size=11, color=SUBTXT),
        hovertemplate='<b>%{x} · Chưa Verified</b><br>TB lượt bán: %{y:,.0f}<extra></extra>',
    ))

    _theme(fig, height=310,
           barmode='group',
           xaxis=dict(showgrid=False),
           yaxis=dict(title='Lượt bán TB'),
           legend=dict(orientation='h', yanchor='bottom', y=1.02,
                      xanchor='center', x=0.5, font=dict(size=10, color=SUBTXT),
                      bgcolor='rgba(0,0,0,0)'),
           showlegend=True)
    return fig


# ══════════════════════════════════════════════════════════════
#  MT2 — TOP 10 BRANDS
# ══════════════════════════════════════════════════════════════

def _brand_profile(df_sub, n=10):
    p = df_sub.groupby('brand_name').agg(
        revenue=('estimated_revenue', 'sum'),
        products=('product_id', 'count'),
        sold_total=('sold_count', 'sum'),
        avg_price=('price', 'mean'),
        avg_rating=('rating', lambda x: x[x > 0].mean() if (x > 0).any() else 0),
    ).sort_values('revenue', ascending=False).head(n)
    p['rev_B'] = p['revenue'] / 1e9
    return p


def make_top10_bar(is_domestic=True):
    df_sub = df_vn_full if is_domestic else df_nn_full
    color  = C_DOM if is_domestic else C_IMP
    top10  = _brand_profile(df_sub, 10)

    brands  = top10.index.tolist()[::-1]
    revs    = top10['rev_B'].tolist()[::-1]
    soleds  = top10['sold_total'].tolist()[::-1]

    # Gradient opacity
    n = len(brands)
    bar_colors = [f'rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},{0.45 + 0.55*(i/max(n-1,1)):.2f})'
                  for i in range(n)]

    fig = go.Figure(go.Bar(
        x=revs, y=brands, orientation='h',
        marker=dict(color=bar_colors,
                    line=dict(color='rgba(0,0,0,0)')),
        text=[f'{v:.1f} tỉ' for v in revs],
        textposition='outside',
        textfont=dict(size=10, weight=700, color=TXT),
        customdata=[[s] for s in soleds],
        hovertemplate='<b>%{y}</b><br>Doanh thu: %{x:.1f} tỉ<br>Lượt bán: %{customdata[0]:,}<extra></extra>',
        cliponaxis=False,
    ))
    _theme(fig, height=340,
           showlegend=False,
           xaxis=dict(title='Doanh thu (tỉ VNĐ)', showgrid=True),
           yaxis=dict(showgrid=False, tickfont=dict(size=10.5)),
           margin=dict(l=14, r=55, t=20, b=28))
    return fig


def make_bubble(is_domestic=True):
    df_sub = df_vn_full if is_domestic else df_nn_full
    color  = C_DOM if is_domestic else C_IMP
    top10  = _brand_profile(df_sub, 10)

    fig = go.Figure(go.Scatter(
        x=top10['avg_price'] / 1000,
        y=top10['sold_total'],
        mode='markers+text',
        marker=dict(
            size=top10['rev_B'] * (18 if is_domestic else 5),
            color=top10['rev_B'],
            colorscale=[[0, hex_to_rgba(color, 0.25)], [1, color]],
            showscale=True,
            colorbar=dict(
                title=dict(text='Doanh thu (tỉ)', font=dict(size=9, color=SUBTXT)),
                tickfont=dict(size=9, color=SUBTXT),
                thickness=10,
                len=0.7,
                bgcolor='rgba(0,0,0,0)',
                outlinecolor='rgba(0,0,0,0)',
            ),
            line=dict(color=color, width=1.5),
            sizemode='area',
            sizemin=6,
        ),
        text=top10.index.tolist(),
        textposition='top center',
        textfont=dict(size=9, color=TXT, weight=700),
        hovertemplate='<b>%{text}</b><br>Giá TB: %{x:.0f}k VNĐ<br>Lượt bán: %{y:,}<extra></extra>',
    ))
    _theme(fig, height=320,
           showlegend=False,
           xaxis=dict(title='Giá TB (nghìn VNĐ)', showgrid=True),
           yaxis=dict(title='Tổng lượt bán', showgrid=True),
           margin=dict(l=14, r=60, t=20, b=28))
    fig.update_yaxes(tickformat=',')
    return fig


# ══════════════════════════════════════════════════════════════
#  MT3 — QUỐC GIA NHẬP KHẨU
# ══════════════════════════════════════════════════════════════

def _country_data():
    cp = df_nn_full.groupby('origin_corrected').agg(
        products=('product_id', 'count'),
        revenue=('estimated_revenue', 'sum'),
        sold=('sold_count', 'sum'),
        brands=('brand_name', 'nunique'),
    ).sort_values('products', ascending=False)
    cp['rev_B']   = cp['revenue'] / 1e9
    cp['pct_sp']  = cp['products'] / cp['products'].sum() * 100
    cp['pct_rev'] = cp['revenue']  / cp['revenue'].sum()  * 100
    return cp


def make_country_donut():
    cp   = _country_data()
    top7 = cp.head(7)
    others = cp.iloc[7:]['products'].sum()
    labels = list(top7.index) + ['Các nước khác']
    values = list(top7['products']) + [int(others)]
    colors = COUNTRY_COLS[:7] + [MUTED]

    total = int(cp['products'].sum())

    fig = go.Figure(go.Pie(
        values=values, labels=labels,
        marker=dict(colors=colors, line=dict(color=CARD, width=2.5)),
        hole=0.62,
        textinfo='percent',
        textfont=dict(size=11, weight=700),
        hovertemplate='<b>%{label}</b><br>%{value:,} SP (%{percent})<extra></extra>',
        pull=[0.05, 0, 0, 0, 0, 0, 0, 0],
        sort=False,
        direction='clockwise',
        textposition='inside',
    ))
    fig.add_annotation(
        text=f'<b style="font-size:20px">{total:,}</b><br><span style="font-size:10px;color:{SUBTXT}">SP nhập khẩu</span>',
        x=0.5, y=0.5, showarrow=False,
        font=dict(color=TXT),
        xref='paper', yref='paper',
    )
    _theme(fig, height=340,
           showlegend=True,
           legend=dict(orientation='h', yanchor='top', y=-0.08,
                       xanchor='center', x=0.5, font=dict(size=10, color=SUBTXT),
                       bgcolor='rgba(0,0,0,0)'),
           margin=dict(l=14, r=14, t=20, b=60))
    return fig


def make_country_compare():
    cp  = _country_data()
    TOP3 = cp.sort_values('revenue', ascending=False).head(3).index.tolist()
    compare = TOP3 + ['Việt Nam']
    bar_colors = COUNTRY_COLS[:3] + [C_DOM]

    rev_vals  = []
    sold_vals = []
    for c in compare:
        if c == 'Việt Nam':
            rev_vals.append(df_vn_full['estimated_revenue'].sum() / 1e9)
            sold_vals.append(df_vn_full['sold_count'].sum() / 1e3)
        else:
            sub = df_nn_full[df_nn_full['origin_corrected'] == c]
            rev_vals.append(sub['estimated_revenue'].sum() / 1e9)
            sold_vals.append(sub['sold_count'].sum() / 1e3)

    vn_rev  = rev_vals[-1]
    vn_sold = sold_vals[-1]

    fig = go.Figure()
    # Revenue bars
    fig.add_trace(go.Bar(
        name='Doanh thu (tỉ VNĐ)',
        x=compare, y=rev_vals,
        marker=dict(color=bar_colors, line=dict(color='rgba(0,0,0,0)')),
        text=[f'{v:.0f}' for v in rev_vals],
        textposition='outside',
        textfont=dict(size=11, weight=700, color=TXT),
        hovertemplate='<b>%{x}</b><br>Doanh thu: %{y:.1f} tỉ<extra></extra>',
        yaxis='y',
    ))
    # Sold bars on secondary axis
    fig.add_trace(go.Bar(
        name='Lượt bán (nghìn)',
        x=compare, y=sold_vals,
        marker=dict(
        color=[hex_to_rgba(c, 0.38) for c in bar_colors],
        line=dict(color='rgba(0,0,0,0)', width=0)
    ),
        text=[f'{v:,.0f}k' for v in sold_vals],
        textposition='outside',
        textfont=dict(size=10, color=SUBTXT),
        hovertemplate='<b>%{x}</b><br>Lượt bán: %{y:,.0f}k<extra></extra>',
        yaxis='y2',
    ))
    # VN reference lines
    fig.add_hline(y=vn_rev, line_dash='dot', line_color=C_DOM,
                  line_width=1.5, opacity=0.6, yref='y')

    _theme(fig, height=330,
           barmode='group',
           xaxis=dict(showgrid=False),
           yaxis=dict(title='Doanh thu (tỉ VNĐ)', side='left', showgrid=True),
           yaxis2=dict(title='Lượt bán (nghìn)', side='right',
                       overlaying='y', showgrid=False),
           legend=_leg(),
           showlegend=True)
    return fig


# ══════════════════════════════════════════════════════════════
#  UI COMPONENTS
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


def card(title, icon_dot, children, flex='1', min_w='280px',
         glow='g-cyan', extra_style=None):
    s = {'flex': flex, 'minWidth': min_w}
    if extra_style:
        s.update(extra_style)
    return html.Div([
        html.P([html.Span(className=f'dot'), f' {title}'],
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
#  LAYOUT
# ══════════════════════════════════════════════════════════════

def layout():
    # Global KPIs
    n_brands    = df_full['brand_name'].nunique()
    n_countries = df_nn_full['origin_corrected'].nunique()
    pct_ver     = df_full['tiki_verified'].mean() * 100
    top_brand   = (_brand_profile(df_full, 1).index[0]
                   if len(df_full) else 'N/A')
    top_country = (df_nn_full.groupby('origin_corrected')['estimated_revenue']
                   .sum().sort_values(ascending=False).index[0])

    # Verified stats
    vs = _verified_data()

    # Config for all graphs
    cfg = {'displayModeBar': False}

    return html.Div([

        # ── PAGE HERO ────────────────────────────────────────
        html.Div([
            # Decorative circles
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

            # Title block
            html.Div([
                html.H1('Thương hiệu & Hệ sinh thái · T3/2026', style={
                    'margin': '0 0 6px 0',
                    'fontSize': '30px',
                    'fontWeight': '700',
                    'color': '#FFFFFF',
                    'letterSpacing': '-0.02em',
                    'lineHeight': '1.15',
                }),
                html.P('Phân tích thương hiệu và định vị chiến lược trên Tiki (T3/2026)', style={
                    'margin': '0',
                    'fontSize': '13px',
                    'color': 'rgba(255,255,255,0.55)',
                    'fontWeight': '400',
                }),
            ], style={'marginBottom': '20px'}),

            # Badge strip
            html.Div([
                html.Span('🏷️  Thương hiệu', className='p3-hero-badge'),
                html.Span('✅  Tiki Verified', className='p3-hero-badge'),
                html.Span('🌏  Quốc gia nhập khẩu', className='p3-hero-badge'),
                html.Span('📊  Doanh thu ước tính', className='p3-hero-badge'),
            ], style={
                'display': 'flex', 'gap': '8px', 'flexWrap': 'wrap',
                'paddingTop': '18px',
                'borderTop': '1px solid rgba(255,255,255,0.12)',
            }),
        ], className='p3-hero'),

        # ── KPI CARDS ───────────────────────────────────────
        html.Div([
            kpi('🏷️', f'{n_brands:,}',       'Thương hiệu',    CYAN,    'rgba(34,211,238,.15)'),
            kpi('✅', f'{pct_ver:.1f}%',      'Tiki Verified',   EMERALD, 'rgba(52,211,153,.12)'),
            kpi('🌏', f'{n_countries}',        'Quốc gia nhập khẩu',     VIOLET,  'rgba(167,139,250,.12)'),
            kpi('🥇', top_brand[:12],          'Top brand nội',   C_DOM,   'rgba(56,189,248,.12)'),
            kpi('🌟', top_country,             'Top quốc gia nhập khẩu', AMBER,   'rgba(251,191,36,.12)'),
        ], className='p3-kpi-row', style={'marginBottom': '28px'}),

        # ─── inner content wrapper ───────────────────────────
        html.Div([

            # ══════════════════════════════════════════════════
            #  MT1 — TIKI VERIFIED
            # ══════════════════════════════════════════════════
            html.Div([
                card('Tỉ lệ Tiki Verified theo nguồn gốc', 'cyan',
                     dcc.Graph(figure=make_verified_stacked(), config=cfg),
                     flex='1', min_w='280px', glow='g-cyan'),
                card('Impact Tiki Verified lên lượt bán', 'cyan',
                     dcc.Graph(figure=make_verified_impact(), config=cfg),
                     flex='1', min_w='280px', glow='g-cyan'),
            ], className='p3-row', style={'marginBottom': '24px', 'gap': '16px'}),

            # ══════════════════════════════════════════════════
            #  MT2 — TOP 10 BRANDS
            # ══════════════════════════════════════════════════
            # Row A: Two hbar charts
            html.Div([
                card('Top 10 thương hiệu nội địa', 'amber',
                     dcc.Graph(figure=make_top10_bar(True), config=cfg),
                     flex='1', min_w='300px', glow='g-amber'),
                card('Top 10 thương hiệu quốc tế', 'amber',
                     dcc.Graph(figure=make_top10_bar(False), config=cfg),
                     flex='1', min_w='300px', glow='g-amber'),
            ], className='p3-row', style={'marginBottom': '24px', 'gap': '16px'}),

            # Row B: Two bubble charts
            html.Div([
                card('Định vị chiến lược · Nội địa', 'amber',
                     dcc.Graph(figure=make_bubble(True), config=cfg),
                     flex='1', min_w='300px', glow='g-amber'),
                card('Định vị chiến lược · Quốc tế', 'amber',
                     dcc.Graph(figure=make_bubble(False), config=cfg),
                     flex='1', min_w='300px', glow='g-amber'),
            ], className='p3-row', style={'marginBottom': '24px', 'gap': '16px'}),

            # ══════════════════════════════════════════════════
            #  MT3 — QUỐC GIA NHẬP KHẨU
            # ══════════════════════════════════════════════════
            html.Div([
                card('Tỷ trọng sản phẩm nhập khẩu', 'emerald',
                     dcc.Graph(figure=make_country_donut(), config=cfg),
                     flex='1', min_w='300px', glow='g-emerald'),
                card('Top 3 quốc gia vs Việt Nam', 'emerald',
                     dcc.Graph(figure=make_country_compare(), config=cfg),
                     flex='1.4', min_w='360px', glow='g-emerald'),
            ], className='p3-row', style={'marginBottom': '24px', 'gap': '16px'}),

        ], style={'padding': '0'}),

        # ── FOOTER ──────────────────────────────────────────
        html.Div([
            html.Span('⚠️ Dữ liệu crawl từ Tiki tháng 3/2026'),
            html.Span('Doanh thu = sold_count × price (ước tính)'),
            html.Span('Nhóm 05 · FIT-HCMUS'),
        ], className='p3-footer', style={'margin': '24px 0 0 0'}),

    ], className='p3-page', style={'padding': '0'})