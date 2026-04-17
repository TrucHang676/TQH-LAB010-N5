# pages/page0_overview.py
# ╔══════════════════════════════════════════════════════════════╗
# ║  Trang 0 — Overview: Tổng quan thị trường mỹ phẩm Tiki       ║
# ╚══════════════════════════════════════════════════════════════╝

import dash
from dash import html, dcc, callback, Input, Output, State
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from data_loader import load_data
from theme import (
    C_DOMESTIC, C_IMPORT, C_NEUTRAL, C_TEXT, C_SUBTEXT,
    C_BG, C_BG_PAGE, C_BORDER, C_SUCCESS, C_WARNING,
    PRODUCT_TYPE_COLORS, PRODUCT_TYPE_LIST, PRODUCT_TYPE_PALETTE,
    COUNTRY_PALETTE, PAGE_ACCENT, LAYOUT_BASE,
    donut_chart, hbar_chart, vbar_grouped,
    kpi_card, section_header, chart_card, apply_theme, make_layout,
)

dash.register_page(
    __name__,
    path='/',
    name='Overview',
    title='Overview | Mỹ phẩm Tiki',
    order=0,
)

df_full, df_vn_full, df_nn_full = load_data()
ALL_PRODUCT_TYPES = sorted(
    [t for t in df_full['product_type'].unique() if t != 'Khác']
) + ['Khác']

PRICE_ORDER = ['Dưới 100k', '100k – 300k', '300k – 700k', '700k – 2tr', 'Trên 2tr']

ACCENT = PAGE_ACCENT['overview']

# ── Design tokens riêng cho trang này ───────────────────────
HDR_BG        = '#102347'
HDR_ACCENT    = '#F6C453'
C_BLUE_MID    = '#2A5CAA'

PAGE_BG       = '#14233F'
C_CARD_BG     = '#1C2D55'
C_CARD_BORDER = 'rgba(255,255,255,0.08)'
C_CARD_SHADOW = '0 14px 34px rgba(3,10,25,0.22)'
NAV_COLORS  = {
    'Overview'   : ('#2563EB', '#EFF6FF'),
    'Thị phần'   : ('#7C3AED', '#F5F3FF'),
    'Uy tín'     : '#C026D3',
    'Chất lượng' : '#16A34A',
    'Giá cả'     : '#D97706',
}

# KPI icon mapping
KPI_META = [
    {'icon': '🛍️', 'key': 'total_products',  'label': 'Tổng sản phẩm',        'fmt': lambda v: f"{v:,}",             'color': '#2563EB'},
    {'icon': '💰', 'key': 'total_revenue',   'label': 'Doanh thu ước tính',    'fmt': lambda v: f"{v/1e9:.1f} tỉ",   'color': '#F59E0B'},
    {'icon': '🌍', 'key': 'pct_import',      'label': 'Thị phần ngoại nhập',   'fmt': lambda v: f"{v:.1f}%",         'color': '#DC2626'},
    {'icon': '🏷️', 'key': 'total_brands',    'label': 'Thương hiệu',           'fmt': lambda v: f"{v:,}",             'color': '#7C3AED'},
    {'icon': '✈️', 'key': 'total_countries', 'label': 'Quốc gia xuất xứ',      'fmt': lambda v: f"{v}",              'color': '#0891B2'},
    {'icon': '⭐', 'key': 'avg_rating',      'label': 'Rating trung bình',     'fmt': lambda v: f"{v:.2f}",          'color': '#F59E0B'},
    {'icon': '💵', 'key': 'avg_price',       'label': 'Giá trung bình',        'fmt': lambda v: f"{v/1e3:.0f}K đ",  'color': '#16A34A'},
    {'icon': '✅', 'key': 'pct_verified',    'label': 'Tiki Verified',         'fmt': lambda v: f"{v:.1f}%",         'color': '#16A34A'},
]


def compute_stats(df):
    total_products  = len(df)
    total_revenue   = df['estimated_revenue'].sum()
    total_brands    = df['brand_name'].nunique()
    total_countries = df['origin_corrected'].nunique()
    pct_import      = (df['origin_class_corrected'] == 'Ngoài nước').sum() / total_products * 100 if total_products > 0 else 0
    pct_verified    = df['tiki_verified'].sum() / total_products * 100 if total_products > 0 else 0
    avg_rating      = df[df['rating'] > 0]['rating'].mean() if len(df) > 0 else 0
    avg_price       = df['price'].mean() if len(df) > 0 else 0
    return dict(
        total_products=total_products, total_revenue=total_revenue,
        total_brands=total_brands, total_countries=total_countries,
        pct_import=pct_import, pct_verified=pct_verified,
        avg_rating=avg_rating, avg_price=avg_price,
    )


def _filter_df(origin_filter, product_type_filter):
    df = df_full.copy()
    if origin_filter == 'domestic':
        df = df[df['origin_class_corrected'] == 'Trong nước']
    elif origin_filter == 'import':
        df = df[df['origin_class_corrected'] == 'Ngoài nước']
    if product_type_filter != 'all':
        df = df[df['product_type'] == product_type_filter]
    return df


# ── Nav button tới các trang khác ───────────────────────────
def quick_nav_btn(label, href, icon, bg_color, text_color='white'):
    return dcc.Link(
        html.Div([
            html.Span(icon, style={'fontSize': '22px', 'marginBottom': '4px', 'display': 'block'}),
            html.Span(label, style={
                'fontSize': '12px', 'fontWeight': '700',
                'letterSpacing': '0.04em', 'textTransform': 'uppercase',
            }),
        ], style={
            'display': 'flex', 'flexDirection': 'column',
            'alignItems': 'center', 'justifyContent': 'center',
            'padding': '14px 20px',
            'background': bg_color,
            'borderRadius': '12px',
            'color': text_color,
            'minWidth': '110px',
            'textAlign': 'center',
            'boxShadow': '0 4px 12px rgba(0,0,0,0.15)',
            'transition': 'transform 0.15s ease, box-shadow 0.15s ease',
            'cursor': 'pointer',
        }),
        href=href,
        style={'textDecoration': 'none'},
    )


def bold_kpi_card(icon, value, label, color):
    """KPI card style bold như dashboard mẫu."""
    return html.Div([
        html.Div([
            html.Span(icon, style={'fontSize': '28px', 'lineHeight': '1'}),
        ], style={
            'width': '52px', 'height': '52px',
            'background': f'{color}18',
            'borderRadius': '12px',
            'display': 'flex', 'alignItems': 'center',
            'justifyContent': 'center',
            'flexShrink': '0',
        }),
        html.Div([
            html.P(value, style={
                'margin': '0', 'fontSize': '26px',
                'fontWeight': '800', 'color': color,
                'lineHeight': '1.1', 'letterSpacing': '-0.02em',
            }),
            html.P(label, style={
                'margin': '3px 0 0 0', 'fontSize': '11px',
                'color': C_SUBTEXT, 'fontWeight': '600',
                'textTransform': 'uppercase', 'letterSpacing': '0.06em',
            }),
        ]),
    ], style={
        'display': 'flex',
        'alignItems': 'center',
        'gap': '14px',
        'background': C_CARD_BG,
        'border': f'1px solid {C_CARD_BORDER}',
        'borderTop': f'3px solid {color}',
        'borderRadius': '14px',
        'padding': '16px 18px',
        'flex': '1',
        'minWidth': '155px',
        'boxShadow': C_CARD_SHADOW,
})


def section_hdr(title, subtitle='', color='#2563EB'):
    return html.Div([
        html.Div(style={
            'width': '5px', 'height': '22px',
            'background': f'linear-gradient(180deg, {color}, {color}88)',
            'borderRadius': '3px', 'flexShrink': '0',
        }),
        html.Div([
            html.H3(title, style={
                'margin': '0', 'fontSize': '14px',
                'fontWeight': '700', 'color': C_TEXT,
                'letterSpacing': '0.01em',
            }),
            html.P(subtitle, style={
                'margin': '2px 0 0 0', 'fontSize': '11px', 'color': C_SUBTEXT,
            }) if subtitle else None,
        ]),
    ], style={'display': 'flex', 'alignItems': 'center',
              'gap': '10px', 'marginBottom': '14px'})


def chart_wrap(children, flex='1', min_width='300px', extra_style=None):
    s = {
        'background': C_CARD_BG,
        'border': f'1px solid {C_CARD_BORDER}',
        'borderRadius': '16px',
        'padding': '20px',
        'flex': flex,
        'minWidth': min_width,
        'boxShadow': C_CARD_SHADOW,
    }
    if extra_style:
        s.update(extra_style)
    return html.Div(children, style=s)


# ╔══════════════════════════════════════════════════════════════╗
# ║  LAYOUT                                                      ║
# ╚══════════════════════════════════════════════════════════════╝

def layout():
    return html.Div([

        # ── HERO HEADER ──────────────────────────────────────
        html.Div([
            html.Div([
                # Left: title block
                html.Div([
                    html.Div([
                        html.Span('📊', style={'fontSize': '16px', 'marginRight': '6px'}),
                        html.Span('TỔNG QUAN THỊ TRƯỜNG', style={
                            'fontSize': '11px', 'fontWeight': '700',
                            'color': HDR_ACCENT, 'letterSpacing': '0.1em',
                        }),
                    ], style={'display': 'flex', 'alignItems': 'center', 'marginBottom': '8px'}),
                    html.H1('Mỹ phẩm Tiki · T3/2026', style={
                        'margin': '0 0 6px 0', 'fontSize': '28px',
                        'fontWeight': '900', 'color': 'white',
                        'letterSpacing': '-0.02em', 'lineHeight': '1.2',
                    }),
                    html.P('Phân tích 7,179 sản phẩm · Nội địa vs Ngoại nhập · Nhóm 05 FIT-HCMUS', style={
                        'margin': '0', 'fontSize': '13px',
                        'color': 'rgba(255,255,255,0.65)', 'fontWeight': '400',
                    }),
                ], style={'flex': '1'}),

                
            ], style={'display': 'flex', 'alignItems': 'center',
                      'gap': '32px', 'flexWrap': 'wrap', 'marginBottom': '20px'}),

            # ── Filter bar ───────────────────────────────────
            html.Div([
                html.Div([
                    html.Span('Lọc:', style={
                        'fontSize': '12px', 'color': 'rgba(255,255,255,0.6)',
                        'fontWeight': '600', 'marginRight': '4px',
                        'textTransform': 'uppercase', 'letterSpacing': '0.06em',
                    }),
                    html.Button('Tất cả', id='pill-all', n_clicks=0, className='filter-pill active'),
                    html.Button('Trong nước', id='pill-domestic', n_clicks=0, className='filter-pill'),
                    html.Button('Ngoài nước', id='pill-import', n_clicks=0, className='filter-pill'),
                ], style={'display': 'flex', 'alignItems': 'center', 'gap': '6px'}),

                dcc.Dropdown(
                    id='dropdown-product-type',
                    options=[{'label': 'Tất cả ngành hàng', 'value': 'all'}] +
                            [{'label': t, 'value': t} for t in ALL_PRODUCT_TYPES],
                    value='all',
                    clearable=False,
                    style={
                        'width': '210px', 'fontSize': '13px',
                        'borderRadius': '20px',
                        'background': 'rgba(255,255,255,0.12)',
                        'color': 'rgba( 0 , 0 , 0 , 1 )',
                    },
                ),
            ], style={
                'display': 'flex', 'alignItems': 'center',
                'gap': '12px', 'flexWrap': 'wrap',
                'paddingTop': '16px',
                'borderTop': '1px solid rgba(255,255,255,0.12)',
            }),

        ], style={
            'background': f'linear-gradient(135deg, {HDR_BG} 0%, #173463 58%, #24508E 100%)',
            'borderRadius': '16px',
            'padding': '28px 32px',
            'marginBottom': '20px',
            'position': 'relative',
            'overflow': 'hidden',
        }),

        dcc.Store(id='store-filter', data={'origin': 'all', 'product_type': 'all'}),

        # ── KPI Row ───────────────────────────────────────────
        html.Div(id='kpi-row', style={
            'display': 'flex', 'gap': '12px',
            'marginBottom': '20px', 'flexWrap': 'wrap',
        }),

        # ── Row 2: Donut + Bar ngành ─────────────────────────
        html.Div([
            chart_wrap([
                section_hdr('Thị phần doanh thu', 'Nội địa vs Ngoại nhập', '#2563EB'),
                dcc.Graph(id='chart-donut', config={'displayModeBar': False}),
            ], flex='1', min_width='300px'),

            chart_wrap([
                section_hdr('Số sản phẩm theo ngành hàng', 'Phân tầng nội / ngoại', '#7C3AED'),
                dcc.Graph(id='chart-bar-product-type', config={'displayModeBar': False}),
            ], flex='1.4', min_width='380px'),
        ], style={'display': 'flex', 'gap': '16px', 'marginBottom': '16px', 'flexWrap': 'wrap'}),

        # ── Row 3: Top quốc gia + Revenue by price ───────────
        html.Div([
            chart_wrap([
                section_hdr('Top quốc gia theo doanh thu', 'VN vs các nước dẫn đầu', '#DC2626'),
                dcc.Graph(id='chart-bar-country', config={'displayModeBar': False}),
            ], flex='1', min_width='340px'),

            chart_wrap([
                section_hdr('Doanh thu theo phân khúc giá', 'Nội địa vs Ngoại nhập', '#D97706'),
                dcc.Graph(id='chart-bar-price-seg', config={'displayModeBar': False}),
            ], flex='1', min_width='340px'),
        ], style={'display': 'flex', 'gap': '16px', 'marginBottom': '16px', 'flexWrap': 'wrap'}),

        # ── Row 4: Sold by type + Top brands ─────────────────
        html.Div([
            chart_wrap([
                section_hdr('Tổng lượt bán theo ngành hàng', 'Cơ cấu nội / ngoại', '#16A34A'),
                dcc.Graph(id='chart-sold-type', config={'displayModeBar': False}),
            ], flex='1', min_width='340px'),

            chart_wrap([
                section_hdr('Top 10 thương hiệu theo doanh thu', 'Xanh = nội địa · Đỏ = ngoại nhập', '#0891B2'),
                dcc.Graph(id='chart-top-brands', config={'displayModeBar': False}),
            ], flex='1', min_width='340px'),
        ], style={'display': 'flex', 'gap': '16px', 'marginBottom': '20px', 'flexWrap': 'wrap'}),

        # ── Footer ────────────────────────────────────────────
        html.Div([
            html.Span('⚠️ Dữ liệu crawl từ Tiki tháng 3/2026', style={'marginRight': '16px'}),
            html.Span('Doanh thu = sold_count × price (ước tính)', style={'marginRight': '16px'}),
            html.Span('Nhóm 05 · FIT-HCMUS'),
        ], style={
            'fontSize': '11px', 'color': C_SUBTEXT,
            'textAlign': 'center', 'padding': '14px 0',
            'borderTop': f'1px solid {C_BORDER}',
        }),

    ], style={'padding': '18px', 'background': PAGE_BG})


# ╔══════════════════════════════════════════════════════════════╗
# ║  CALLBACKS                                                   ║
# ╚══════════════════════════════════════════════════════════════╝

@callback(
    Output('store-filter', 'data'),
    Output('pill-all', 'className'),
    Output('pill-domestic', 'className'),
    Output('pill-import', 'className'),
    Input('pill-all', 'n_clicks'),
    Input('pill-domestic', 'n_clicks'),
    Input('pill-import', 'n_clicks'),
    Input('dropdown-product-type', 'value'),
    State('store-filter', 'data'),
    prevent_initial_call=False,
)
def sync_filter(n_all, n_dom, n_imp, pt_val, store):
    ctx = dash.callback_context
    origin = store.get('origin', 'all') if store else 'all'
    if ctx.triggered:
        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
        if trigger_id == 'pill-all':       origin = 'all'
        elif trigger_id == 'pill-domestic': origin = 'domestic'
        elif trigger_id == 'pill-import':   origin = 'import'
    cls_all = 'filter-pill active'     if origin == 'all'      else 'filter-pill'
    cls_dom = 'filter-pill active'     if origin == 'domestic' else 'filter-pill'
    cls_imp = 'filter-pill active-red' if origin == 'import'   else 'filter-pill'
    return {'origin': origin, 'product_type': pt_val or 'all'}, cls_all, cls_dom, cls_imp


@callback(Output('kpi-row', 'children'), Input('store-filter', 'data'))
def update_kpi(store):
    origin = store.get('origin', 'all')
    pt     = store.get('product_type', 'all')
    df     = _filter_df(origin, pt)
    s      = compute_stats(df)
    return [
        bold_kpi_card(m['icon'], m['fmt'](s[m['key']]), m['label'], m['color'])
        for m in KPI_META
    ]


@callback(Output('chart-donut', 'figure'), Input('store-filter', 'data'))
def update_donut(store):
    origin = store.get('origin', 'all')
    pt     = store.get('product_type', 'all')
    df     = _filter_df(origin, pt)
    rev_grp = df.groupby('origin_class_corrected')['estimated_revenue'].sum().sort_values(ascending=False)
    if len(rev_grp) == 0:
        return go.Figure()
    labels = rev_grp.index.tolist()
    values = rev_grp.values.tolist()
    colors = [C_DOMESTIC if l == 'Trong nước' else C_IMPORT for l in labels]
    fig = donut_chart(values, labels, colors, height=300)
    total = sum(values)
    fig.add_annotation(
        text=f'<b>{total/1e9:.1f}</b><br><span style="font-size:11px">tỉ VNĐ</span>',
        x=0.5, y=0.5, showarrow=False,
        font=dict(size=16, color=C_TEXT),
        xref='paper', yref='paper',
    )
    return fig


@callback(Output('chart-bar-product-type', 'figure'), Input('store-filter', 'data'))
def update_bar_type(store):
    origin = store.get('origin', 'all')
    pt     = store.get('product_type', 'all')
    df     = _filter_df(origin, pt)
    grp = df.groupby(['product_type', 'origin_class_corrected']).size().unstack(fill_value=0)
    types    = [t for t in PRODUCT_TYPE_LIST if t in grp.index]
    dom_vals = [grp.loc[t, 'Trong nước'] if 'Trong nước' in grp.columns else 0 for t in types]
    imp_vals = [grp.loc[t, 'Ngoài nước'] if 'Ngoài nước' in grp.columns else 0 for t in types]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name='Trong nước', x=types, y=dom_vals,
        marker_color=C_DOMESTIC, marker_line=dict(color='white', width=0.8),
        hovertemplate='<b>%{x}</b><br>Trong nước: %{y:,}<extra></extra>',
    ))
    fig.add_trace(go.Bar(
        name='Ngoài nước', x=types, y=imp_vals,
        marker_color=C_IMPORT, marker_line=dict(color='white', width=0.8),
        hovertemplate='<b>%{x}</b><br>Ngoài nước: %{y:,}<extra></extra>',
    ))
    apply_theme(fig, height=300,
                barmode='group',
                xaxis={**LAYOUT_BASE['xaxis'], 'showgrid': False},
                yaxis={**LAYOUT_BASE['yaxis'], 'title': 'Số sản phẩm', 'showgrid': True},
                legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1, font=dict(size=11)))
    return fig


@callback(Output('chart-bar-country', 'figure'), Input('store-filter', 'data'))
def update_bar_country(store):
    origin = store.get('origin', 'all')
    pt     = store.get('product_type', 'all')
    df     = _filter_df(origin, pt)
    rev_country = df.groupby('origin_corrected')['estimated_revenue'].sum().sort_values(ascending=False)
    top_n  = 8
    top    = rev_country.head(top_n).copy()
    others = rev_country.iloc[top_n:].sum()
    if others > 0:
        top['Khác'] = others
    countries  = top.index.tolist()[::-1]
    values     = (top.values / 1e9).tolist()[::-1]
    bar_colors = []
    for c in countries:
        if c == 'Việt Nam':   bar_colors.append(C_DOMESTIC)
        elif c == 'Khác':     bar_colors.append(C_NEUTRAL)
        else:                 bar_colors.append(C_IMPORT)
    fig = go.Figure(go.Bar(
        x=values, y=countries, orientation='h',
        marker=dict(color=bar_colors, line=dict(color='white', width=0.6)),
        text=[f'{v:.1f} tỉ' for v in values],
        textposition='outside', textfont=dict(size=10, color=C_TEXT),
        hovertemplate='<b>%{y}</b><br>%{x:.1f} tỉ VNĐ<extra></extra>',
        cliponaxis=False,
    ))
    apply_theme(fig, height=330,
                xaxis={**LAYOUT_BASE['xaxis'], 'title': 'Doanh thu (tỉ VNĐ)', 'showgrid': True},
                yaxis={**LAYOUT_BASE['yaxis'], 'showgrid': False},
                showlegend=False)
    fig.update_layout(margin=dict(l=16, r=60, t=16, b=30))
    return fig


@callback(Output('chart-bar-price-seg', 'figure'), Input('store-filter', 'data'))
def update_bar_price(store):
    origin = store.get('origin', 'all')
    pt     = store.get('product_type', 'all')
    df     = _filter_df(origin, pt)
    grp = (df.groupby(['price_segment', 'origin_class_corrected'], observed=True)['estimated_revenue']
           .sum().unstack(fill_value=0))
    segs    = [s for s in PRICE_ORDER if s in grp.index]
    dom_rev = [(grp.loc[s, 'Trong nước'] / 1e9 if 'Trong nước' in grp.columns else 0) for s in segs]
    imp_rev = [(grp.loc[s, 'Ngoài nước'] / 1e9 if 'Ngoài nước' in grp.columns else 0) for s in segs]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name='Trong nước', x=segs, y=dom_rev,
        marker_color=C_DOMESTIC, marker_line=dict(color='white', width=0.8),
        hovertemplate='<b>%{x}</b><br>Trong nước: %{y:.1f} tỉ<extra></extra>',
    ))
    fig.add_trace(go.Bar(
        name='Ngoài nước', x=segs, y=imp_rev,
        marker_color=C_IMPORT, marker_line=dict(color='white', width=0.8),
        hovertemplate='<b>%{x}</b><br>Ngoài nước: %{y:.1f} tỉ<extra></extra>',
    ))
    apply_theme(fig, height=330,
                barmode='group',
                xaxis={**LAYOUT_BASE['xaxis'], 'showgrid': False, 'tickangle': -15},
                yaxis={**LAYOUT_BASE['yaxis'], 'title': 'Doanh thu (tỉ VNĐ)', 'showgrid': True},
                legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1, font=dict(size=11)))
    return fig


@callback(Output('chart-sold-type', 'figure'), Input('store-filter', 'data'))
def update_sold_type(store):
    origin = store.get('origin', 'all')
    pt     = store.get('product_type', 'all')
    df     = _filter_df(origin, pt)
    grp = df.groupby(['product_type', 'origin_class_corrected'])['sold_count'].sum().unstack(fill_value=0)
    types    = [t for t in PRODUCT_TYPE_LIST if t in grp.index]
    dom_sold = [(grp.loc[t, 'Trong nước'] / 1e3 if 'Trong nước' in grp.columns else 0) for t in types]
    imp_sold = [(grp.loc[t, 'Ngoài nước'] / 1e3 if 'Ngoài nước' in grp.columns else 0) for t in types]
    totals   = [d + i for d, i in zip(dom_sold, imp_sold)]
    order    = sorted(range(len(totals)), key=lambda i: totals[i])
    types    = [types[i]    for i in order]
    dom_sold = [dom_sold[i] for i in order]
    imp_sold = [imp_sold[i] for i in order]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name='Trong nước', y=types, x=dom_sold, orientation='h',
        marker_color=C_DOMESTIC, marker_line=dict(color='white', width=0.6),
        hovertemplate='<b>%{y}</b><br>Trong nước: %{x:.1f}K lượt<extra></extra>',
    ))
    fig.add_trace(go.Bar(
        name='Ngoài nước', y=types, x=imp_sold, orientation='h',
        marker_color=C_IMPORT, marker_line=dict(color='white', width=0.6),
        hovertemplate='<b>%{y}</b><br>Ngoài nước: %{x:.1f}K lượt<extra></extra>',
    ))
    apply_theme(fig, height=310,
                barmode='stack',
                xaxis={**LAYOUT_BASE['xaxis'], 'title': 'Tổng lượt bán (nghìn)', 'showgrid': True},
                yaxis={**LAYOUT_BASE['yaxis'], 'showgrid': False},
                legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1, font=dict(size=11)))
    return fig


@callback(Output('chart-top-brands', 'figure'), Input('store-filter', 'data'))
def update_top_brands(store):
    origin = store.get('origin', 'all')
    pt     = store.get('product_type', 'all')
    df     = _filter_df(origin, pt)
    top_brands = (df.groupby('brand_name')
                  .agg(revenue=('estimated_revenue', 'sum'),
                       origin=('origin_class_corrected',
                               lambda x: x.mode()[0] if len(x) > 0 else 'Ngoài nước'))
                  .sort_values('revenue', ascending=False)
                  .head(10))
    brands = top_brands.index.tolist()[::-1]
    revs   = (top_brands['revenue'].values / 1e9).tolist()[::-1]
    origs  = top_brands['origin'].tolist()[::-1]
    colors = [C_DOMESTIC if o == 'Trong nước' else C_IMPORT for o in origs]
    fig = go.Figure(go.Bar(
        x=revs, y=brands, orientation='h',
        marker=dict(color=colors, line=dict(color='white', width=0.6)),
        text=[f'{v:.1f} tỉ' for v in revs],
        textposition='outside', textfont=dict(size=10, color=C_TEXT),
        hovertemplate='<b>%{y}</b><br>%{x:.1f} tỉ VNĐ<extra></extra>',
        cliponaxis=False,
    ))
    apply_theme(fig, height=310,
                xaxis={**LAYOUT_BASE['xaxis'], 'title': 'Doanh thu (tỉ VNĐ)', 'showgrid': True},
                yaxis={**LAYOUT_BASE['yaxis'], 'showgrid': False},
                showlegend=False)
    fig.update_layout(margin=dict(l=16, r=60, t=16, b=30))
    return fig