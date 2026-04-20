# pages/page0_overview.py
# ╔══════════════════════════════════════════════════════════════╗
# ║  Trang 0 — Overview: Tổng quan thị trường mỹ phẩm Tiki       ║
# ╚══════════════════════════════════════════════════════════════╝

import dash
from dash import html, dcc, callback, Input, Output, State
import plotly.graph_objects as go
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from data_loader import load_data
from theme import (
    C_DOMESTIC, C_IMPORT, C_TEXT, C_SUBTEXT,
    C_BORDER, PRODUCT_TYPE_LIST,
    donut_chart, apply_theme, LAYOUT_BASE,
)

dash.register_page(
    __name__,
    path='/',
    name='Overview',
    title='Overview | Mỹ phẩm Tiki',
    order=0,
)

# ── Load data ────────────────────────────────────────────────
df_full, df_vn_full, df_nn_full = load_data()

ALL_PRODUCT_TYPES = sorted(
    [t for t in df_full['product_type'].dropna().unique() if t != 'Khác']
) + (['Khác'] if 'Khác' in df_full['product_type'].dropna().unique() else [])

# ── Design tokens riêng cho trang này ───────────────────────
HDR_BG        = '#102347'
HDR_ACCENT    = '#F6C453'

PAGE_BG       = '#14233F'
C_CARD_BG     = '#1C2D55'
C_CARD_BORDER = 'rgba(255,255,255,0.08)'
C_CARD_SHADOW = '0 14px 34px rgba(3,10,25,0.22)'

# KPI icon mapping
KPI_META = [
    {'icon': '🛍️', 'key': 'total_products',  'label': 'Tổng sản phẩm',      'fmt': lambda v: f"{v:,}",           'color': '#2563EB'},
    {'icon': '💰', 'key': 'total_revenue',   'label': 'Doanh thu ước tính',  'fmt': lambda v: f"{v/1e9:.1f} tỉ", 'color': '#F59E0B'},
    {'icon': '🌍', 'key': 'pct_import',      'label': 'Thị phần ngoại nhập', 'fmt': lambda v: f"{v:.1f}%",       'color': '#DC2626'},
    {'icon': '🏷️', 'key': 'total_brands',    'label': 'Thương hiệu',         'fmt': lambda v: f"{v:,}",           'color': '#7C3AED'},
    {'icon': '✈️', 'key': 'total_countries', 'label': 'Quốc gia xuất xứ',    'fmt': lambda v: f"{v}",            'color': '#0891B2'},
    {'icon': '⭐', 'key': 'avg_rating',      'label': 'Rating trung bình',   'fmt': lambda v: f"{v:.2f}",        'color': '#F59E0B'},
    {'icon': '✅', 'key': 'pct_verified',    'label': 'Tiki Verified',       'fmt': lambda v: f"{v:.1f}%",       'color': '#16A34A'},
]


# ══════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════

def compute_stats(df):
    total_products = len(df)
    total_revenue = df['estimated_revenue'].sum()
    total_brands = df['brand_name'].nunique()
    total_countries = df['origin_corrected'].nunique()
    pct_import = (
        (df['origin_class_corrected'] == 'Ngoài nước').sum() / total_products * 100
        if total_products > 0 else 0
    )
    pct_verified = (
        df['tiki_verified'].sum() / total_products * 100
        if total_products > 0 else 0
    )
    avg_rating = df[df['rating'] > 0]['rating'].mean() if total_products > 0 else 0
    avg_price = df['price'].mean() if total_products > 0 else 0

    return dict(
        total_products=total_products,
        total_revenue=total_revenue,
        total_brands=total_brands,
        total_countries=total_countries,
        pct_import=pct_import,
        pct_verified=pct_verified,
        avg_rating=avg_rating if avg_rating == avg_rating else 0,
        avg_price=avg_price if avg_price == avg_price else 0,
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


def bold_kpi_card(icon, value, label, color):
    return html.Div([
        html.Div([
            html.Span(icon, style={'fontSize': '28px', 'lineHeight': '1'}),
        ], style={
            'width': '52px',
            'height': '52px',
            'background': f'{color}18',
            'borderRadius': '12px',
            'display': 'flex',
            'alignItems': 'center',
            'justifyContent': 'center',
            'flexShrink': '0',
        }),
        html.Div([
            html.P(value, style={
                'margin': '0',
                'fontSize': '26px',
                'fontWeight': '800',
                'color': color,
                'lineHeight': '1.1',
                'letterSpacing': '-0.02em',
            }),
            html.P(label, style={
                'margin': '3px 0 0 0',
                'fontSize': '11px',
                'color': C_SUBTEXT,
                'fontWeight': '600',
                'textTransform': 'uppercase',
                'letterSpacing': '0.06em',
            }),
        ]),
    ], className='kpi-card-dynamic', style={
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
            'width': '5px',
            'height': '22px',
            'background': f'linear-gradient(180deg, {color}, {color}88)',
            'borderRadius': '3px',
            'flexShrink': '0',
        }),
        html.Div([
            html.H3(title, style={
                'margin': '0',
                'fontSize': '14px',
                'fontWeight': '700',
                'color': C_TEXT,
                'letterSpacing': '0.01em',
            }),
            html.P(subtitle, style={
                'margin': '2px 0 0 0',
                'fontSize': '11px',
                'color': C_SUBTEXT,
            }) if subtitle else None,
        ]),
    ], style={
        'display': 'flex',
        'alignItems': 'center',
        'gap': '10px',
        'marginBottom': '14px'
    })


def chart_wrap(children, flex='1', min_width='300px', extra_style=None):
    style = {
        'background': C_CARD_BG,
        'border': f'1px solid {C_CARD_BORDER}',
        'borderRadius': '16px',
        'padding': '20px',
        'flex': flex,
        'minWidth': min_width,
        'boxShadow': C_CARD_SHADOW,
    }
    if extra_style:
        style.update(extra_style)
    return html.Div(children, style=style)


def _empty_fig(height=320):
    fig = go.Figure()
    fig.update_layout(
        height=height,
        paper_bgcolor=C_CARD_BG,
        plot_bgcolor=C_CARD_BG,
        font=dict(color=C_TEXT),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        annotations=[
            dict(
                text='Không có dữ liệu',
                x=0.5, y=0.5,
                xref='paper', yref='paper',
                showarrow=False,
                font=dict(size=14, color=C_SUBTEXT)
            )
        ],
        margin=dict(l=10, r=10, t=10, b=10),
    )
    return fig


# ══════════════════════════════════════════════════════════════
#  LAYOUT
# ══════════════════════════════════════════════════════════════

def layout():
    return html.Div([

        # ── HERO HEADER ──────────────────────────────────────
        html.Div([
            html.Div([
                html.Div([
                    html.H1('Tổng quan thị trường Mỹ phẩm Tiki · T3/2026', style={
                        'margin': '0 0 6px 0',
                        'fontSize': '28px',
                        'fontWeight': '900',
                        'color': 'white',
                        'letterSpacing': '-0.02em',
                        'lineHeight': '1.2',
                    }),
                ], style={'flex': '1'}),
            ], style={
                'display': 'flex',
                'alignItems': 'center',
                'gap': '32px',
                'flexWrap': 'wrap',
                'marginBottom': '20px'
            }),

            # ── Filter bar ───────────────────────────────────
            html.Div([
                html.Div([
                    html.Span('Lọc:', style={
                        'fontSize': '12px',
                        'color': 'rgba(255,255,255,0.6)',
                        'fontWeight': '600',
                        'marginRight': '4px',
                        'textTransform': 'uppercase',
                        'letterSpacing': '0.06em',
                    }),
                    html.Button('Tất cả', id='pill-all', n_clicks=0, className='filter-pill active'),
                    html.Button('Trong nước', id='pill-domestic', n_clicks=0, className='filter-pill'),
                    html.Button('Ngoài nước', id='pill-import', n_clicks=0, className='filter-pill'),
                ], style={
                    'display': 'flex',
                    'alignItems': 'center',
                    'gap': '6px'
                }),
            ], style={
                'display': 'flex',
                'alignItems': 'center',
                'gap': '12px',
                'flexWrap': 'wrap',
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
            'display': 'flex',
            'gap': '12px',
            'marginBottom': '20px',
            'flexWrap': 'wrap',
        }),

        # ── Main charts: only 2 charts kept ──────────────────
        html.Div([
            chart_wrap([
                section_hdr('Thị phần doanh thu', 'Ai đang chiếm phần lớn doanh thu của thị trường', '#2563EB'),
                dcc.Graph(
                    id='chart-donut',
                    config={'displayModeBar': False},
                    style={'height': '330px'}
                ),
            ], flex='0.95', min_width='320px'),

            chart_wrap([
                section_hdr('Số sản phẩm theo ngành hàng', 'Cơ cấu thị trường tổng quát theo ngành · Nội vs Ngoại', '#7C3AED'),
                dcc.Graph(
                    id='chart-bar-product-type',
                    config={'displayModeBar': False},
                    style={'height': '330px'}
                ),
            ], flex='1.45', min_width='420px'),
        ], style={
            'display': 'flex',
            'gap': '16px',
            'marginBottom': '20px',
            'flexWrap': 'wrap',
            'alignItems': 'stretch',
        }),

        # ── Footer ────────────────────────────────────────────
        html.Div([
            html.Span('⚠️ Dữ liệu crawl từ Tiki tháng 3/2026', style={'marginRight': '16px'}),
            html.Span('Nhóm 05 · FIT-HCMUS'),
        ], style={
            'fontSize': '11px',
            'color': C_SUBTEXT,
            'textAlign': 'center',
            'padding': '14px 0',
            'borderTop': f'1px solid {C_BORDER}',
        }),

    ], style={
        'padding': '18px',
        'background': PAGE_BG,
        'minHeight': '100vh',
    })


# ══════════════════════════════════════════════════════════════
#  CALLBACKS
# ══════════════════════════════════════════════════════════════

@callback(
    Output('store-filter', 'data'),
    Output('pill-all', 'className'),
    Output('pill-domestic', 'className'),
    Output('pill-import', 'className'),
    Input('pill-all', 'n_clicks'),
    Input('pill-domestic', 'n_clicks'),
    Input('pill-import', 'n_clicks'),
    State('store-filter', 'data'),
    prevent_initial_call=False,
)
def sync_filter(n_all, n_dom, n_imp, store):
    ctx = dash.callback_context
    origin = store.get('origin', 'all') if store else 'all'

    if ctx.triggered:
        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
        if trigger_id == 'pill-all':
            origin = 'all'
        elif trigger_id == 'pill-domestic':
            origin = 'domestic'
        elif trigger_id == 'pill-import':
            origin = 'import'

    cls_all = 'filter-pill active' if origin == 'all' else 'filter-pill'
    cls_dom = 'filter-pill active' if origin == 'domestic' else 'filter-pill'
    cls_imp = 'filter-pill active-red' if origin == 'import' else 'filter-pill'

    return {
        'origin': origin,
        'product_type': 'all'
    }, cls_all, cls_dom, cls_imp


@callback(
    Output('kpi-row', 'children'),
    Input('store-filter', 'data')
)
def update_kpi(store):
    origin = store.get('origin', 'all')
    pt = store.get('product_type', 'all')
    df = _filter_df(origin, pt)
    stats = compute_stats(df)

    return [
        bold_kpi_card(meta['icon'], meta['fmt'](stats[meta['key']]), meta['label'], meta['color'])
        for meta in KPI_META
    ]


@callback(
    Output('chart-donut', 'figure'),
    Input('store-filter', 'data')
)
def update_donut(store):
    origin = store.get('origin', 'all')
    pt = store.get('product_type', 'all')
    df = _filter_df(origin, pt)

    if df.empty:
        return _empty_fig(height=330)

    rev_grp = (
        df.groupby('origin_class_corrected')['estimated_revenue']
        .sum()
        .sort_values(ascending=False)
    )

    if rev_grp.empty:
        return _empty_fig(height=330)

    labels = rev_grp.index.tolist()
    values = rev_grp.values.tolist()
    colors = [C_DOMESTIC if label == 'Trong nước' else C_IMPORT for label in labels]

    fig = donut_chart(values, labels, colors, height=330)
    total = sum(values)

    fig.add_annotation(
        text=f'<b>{total/1e9:.1f}</b><br><span style="font-size:11px">tỉ VNĐ</span>',
        x=0.5,
        y=0.5,
        showarrow=False,
        font=dict(size=16, color=C_TEXT),
        xref='paper',
        yref='paper',
    )

    fig.update_layout(
        margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(
            orientation='h',
            yanchor='top',
            y=-0.08,
            xanchor='center',
            x=0.5,
            font=dict(size=11)
        )
    )

    return fig


@callback(
    Output('chart-bar-product-type', 'figure'),
    Input('store-filter', 'data')
)
def update_bar_type(store):
    origin = store.get('origin', 'all')
    pt = store.get('product_type', 'all')
    df = _filter_df(origin, pt)

    if df.empty:
        return _empty_fig(height=330)

    grp = (
        df.groupby(['product_type', 'origin_class_corrected'])
        .size()
        .unstack(fill_value=0)
    )

    if grp.empty:
        return _empty_fig(height=330)

    valid_types = [t for t in PRODUCT_TYPE_LIST if t in grp.index]
    if not valid_types:
        valid_types = grp.index.tolist()

    totals = grp.reindex(valid_types).sum(axis=1).sort_values(ascending=False)
    types = totals.index.tolist()

    dom_vals = [grp.loc[t, 'Trong nước'] if 'Trong nước' in grp.columns else 0 for t in types]
    imp_vals = [grp.loc[t, 'Ngoài nước'] if 'Ngoài nước' in grp.columns else 0 for t in types]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        name='Trong nước',
        x=types,
        y=dom_vals,
        marker_color=C_DOMESTIC,
        marker_line=dict(color='white', width=0.8),
        hovertemplate='<b>%{x}</b><br>Trong nước: %{y:,} sản phẩm<extra></extra>',
    ))

    fig.add_trace(go.Bar(
        name='Ngoài nước',
        x=types,
        y=imp_vals,
        marker_color=C_IMPORT,
        marker_line=dict(color='white', width=0.8),
        hovertemplate='<b>%{x}</b><br>Ngoài nước: %{y:,} sản phẩm<extra></extra>',
    ))

    apply_theme(
        fig,
        height=330,
        barmode='group',
        xaxis={
            **LAYOUT_BASE['xaxis'],
            'showgrid': False,
            'tickangle': -12,
        },
        yaxis={
            **LAYOUT_BASE['yaxis'],
            'title': 'Số sản phẩm',
            'showgrid': True,
        },
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1,
            font=dict(size=11)
        ),
    )

    fig.update_layout(
        margin=dict(l=16, r=16, t=16, b=40),
        bargap=0.22
    )

    return fig