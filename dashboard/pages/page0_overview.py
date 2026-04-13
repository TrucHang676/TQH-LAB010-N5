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

# ── Đăng ký trang với Dash Pages ────────────────────────────
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
    [t for t in df_full['product_type'].unique() if t != 'Khác']
) + ['Khác']

PRICE_ORDER = ['Dưới 100k', '100k – 300k', '300k – 700k', '700k – 2tr', 'Trên 2tr']

ACCENT = PAGE_ACCENT['overview']   # #2563EB

# ╔══════════════════════════════════════════════════════════════╗
# ║  HELPER: Tính toán KPI & data cho từng filter state         ║
# ╚══════════════════════════════════════════════════════════════╝

def compute_stats(df):
    """Tính tất cả số liệu từ 1 dataframe đã lọc."""
    total_products  = len(df)
    total_revenue   = df['estimated_revenue'].sum()
    total_brands    = df['brand_name'].nunique()
    total_countries = df['origin_corrected'].nunique()

    pct_import = (
        (df['origin_class_corrected'] == 'Ngoài nước').sum() / total_products * 100
        if total_products > 0 else 0
    )
    pct_verified = (
        df['tiki_verified'].sum() / total_products * 100
        if total_products > 0 else 0
    )
    avg_rating = df[df['rating'] > 0]['rating'].mean() if len(df) > 0 else 0
    avg_price  = df['price'].mean() if len(df) > 0 else 0

    return dict(
        total_products  = total_products,
        total_revenue   = total_revenue,
        total_brands    = total_brands,
        total_countries = total_countries,
        pct_import      = pct_import,
        pct_verified    = pct_verified,
        avg_rating      = avg_rating,
        avg_price       = avg_price,
    )


# ╔══════════════════════════════════════════════════════════════╗
# ║  LAYOUT                                                      ║
# ╚══════════════════════════════════════════════════════════════╝

def layout():
    return html.Div([

        # ── Page Header ──────────────────────────────────────
        html.Div([
            html.Div([
                html.Span('TRANG 0 · OVERVIEW', style={
                    'fontSize': '11px', 'fontWeight': '700',
                    'color': ACCENT, 'letterSpacing': '0.08em',
                    'textTransform': 'uppercase',
                }),
                html.H1('Tổng quan thị trường mỹ phẩm Tiki · T3/2026', style={
                    'margin': '6px 0 16px 0', 'fontSize': '22px',
                    'fontWeight': '700', 'color': C_TEXT, 'lineHeight': '1.3',
                }),
            ]),

            # ── Filter Bar ───────────────────────────────────
            html.Div([
                # Origin filter pills
                html.Div([
                    html.Button('Tất cả', id='pill-all', n_clicks=0,
                                className='filter-pill active'),
                    html.Button('Trong nước', id='pill-domestic', n_clicks=0,
                                className='filter-pill'),
                    html.Button('Ngoài nước', id='pill-import', n_clicks=0,
                                className='filter-pill'),
                ], style={'display': 'flex', 'gap': '6px'}),

                # Product type dropdown
                dcc.Dropdown(
                    id='dropdown-product-type',
                    options=[{'label': 'Tất cả ngành hàng', 'value': 'all'}] +
                            [{'label': t, 'value': t} for t in ALL_PRODUCT_TYPES],
                    value='all',
                    clearable=False,
                    style={
                        'width': '210px', 'fontSize': '13px',
                        'border': f'1.5px solid {C_BORDER}',
                        'borderRadius': '20px',
                    },
                ),
            ], style={
                'display': 'flex', 'alignItems': 'center',
                'gap': '12px', 'flexWrap': 'wrap',
                'marginBottom': '20px',
            }),

        ], style={
            'background': C_BG,
            'border': f'1px solid {C_BORDER}',
            'borderRadius': '16px',
            'padding': '24px 28px 0 28px',
            'marginBottom': '20px',
        }),

        # ── Store: trạng thái filter hiện tại ────────────────
        dcc.Store(id='store-filter', data={'origin': 'all', 'product_type': 'all'}),

        # ── Row 1: KPI Cards ─────────────────────────────────
        html.Div(id='kpi-row', style={
            'display': 'flex', 'gap': '12px',
            'marginBottom': '20px', 'flexWrap': 'wrap',
        }),

        # ── Row 2: Donut + Bar phân bổ ngành ─────────────────
        html.Div([
            # Donut thị phần
            html.Div([
                section_header('Thị phần doanh thu', 'Nội vs Ngoại', ACCENT),
                dcc.Graph(id='chart-donut', config={'displayModeBar': False}),
            ], style={**_card_style(), 'flex': '1', 'minWidth': '300px'}),

            # Bar phân bổ sản phẩm theo ngành
            html.Div([
                section_header('Số sản phẩm theo ngành hàng',
                               'Phân tầng nội địa / ngoại nhập', ACCENT),
                dcc.Graph(id='chart-bar-product-type',
                          config={'displayModeBar': False}),
            ], style={**_card_style(), 'flex': '1.4', 'minWidth': '380px'}),

        ], style={'display': 'flex', 'gap': '16px',
                  'marginBottom': '20px', 'flexWrap': 'wrap'}),

        # ── Row 3: Top quốc gia + Revenue by price segment ───
        html.Div([
            # Top 8 quốc gia doanh thu
            html.Div([
                section_header('Top quốc gia theo doanh thu ước tính',
                               'Highlight: VN vs nước dẫn đầu', ACCENT),
                dcc.Graph(id='chart-bar-country',
                          config={'displayModeBar': False}),
            ], style={**_card_style(), 'flex': '1', 'minWidth': '340px'}),

            # Revenue by price segment
            html.Div([
                section_header('Doanh thu theo phân khúc giá',
                               'So sánh nội địa vs ngoại nhập', ACCENT),
                dcc.Graph(id='chart-bar-price-seg',
                          config={'displayModeBar': False}),
            ], style={**_card_style(), 'flex': '1', 'minWidth': '340px'}),

        ], style={'display': 'flex', 'gap': '16px',
                  'marginBottom': '20px', 'flexWrap': 'wrap'}),

        # ── Row 4: Sold by product type + Top brands ─────────
        html.Div([
            # Lượt bán theo ngành
            html.Div([
                section_header('Tổng lượt bán theo ngành hàng',
                               'Cơ cấu nội / ngoại', ACCENT),
                dcc.Graph(id='chart-sold-type',
                          config={'displayModeBar': False}),
            ], style={**_card_style(), 'flex': '1', 'minWidth': '340px'}),

            # Top 10 thương hiệu
            html.Div([
                section_header('Top 10 thương hiệu theo doanh thu',
                               'Màu = nội địa (xanh) / ngoại (đỏ)', ACCENT),
                dcc.Graph(id='chart-top-brands',
                          config={'displayModeBar': False}),
            ], style={**_card_style(), 'flex': '1', 'minWidth': '340px'}),

        ], style={'display': 'flex', 'gap': '16px',
                  'marginBottom': '20px', 'flexWrap': 'wrap'}),

        # ── Footer note ───────────────────────────────────────
        html.P(
            '⚠️ Dữ liệu crawl từ Tiki tháng 3/2026 · '
            'Doanh thu = sold_count × price (ước tính) · '
            'Nhóm 05 - FIT-HCMUS',
            style={
                'fontSize': '11px', 'color': C_SUBTEXT,
                'textAlign': 'center', 'padding': '12px 0',
                'borderTop': f'1px solid {C_BORDER}',
            }
        )

    ], style={'padding': '0', 'background': C_BG_PAGE})


def _card_style():
    return {
        'background': C_BG,
        'border': f'1px solid {C_BORDER}',
        'borderRadius': '12px',
        'padding': '20px',
    }


# ╔══════════════════════════════════════════════════════════════╗
# ║  CALLBACKS                                                   ║
# ╚══════════════════════════════════════════════════════════════╝

def _filter_df(origin_filter, product_type_filter):
    """Lọc dataframe theo filter state hiện tại."""
    df = df_full.copy()
    if origin_filter == 'domestic':
        df = df[df['origin_class_corrected'] == 'Trong nước']
    elif origin_filter == 'import':
        df = df[df['origin_class_corrected'] == 'Ngoài nước']
    if product_type_filter != 'all':
        df = df[df['product_type'] == product_type_filter]
    return df


# ── 1. Sync filter pills → store ────────────────────────────
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
        if trigger_id == 'pill-all':
            origin = 'all'
        elif trigger_id == 'pill-domestic':
            origin = 'domestic'
        elif trigger_id == 'pill-import':
            origin = 'import'

    cls_all = 'filter-pill active'      if origin == 'all'      else 'filter-pill'
    cls_dom = 'filter-pill active'      if origin == 'domestic' else 'filter-pill'
    cls_imp = 'filter-pill active-red'  if origin == 'import'   else 'filter-pill'

    return {'origin': origin, 'product_type': pt_val or 'all'}, cls_all, cls_dom, cls_imp


# ── 2. KPI Cards ────────────────────────────────────────────
@callback(
    Output('kpi-row', 'children'),
    Input('store-filter', 'data'),
)
def update_kpi(store):
    origin = store.get('origin', 'all')
    pt     = store.get('product_type', 'all')
    df     = _filter_df(origin, pt)
    s      = compute_stats(df)

    origin_color = (C_DOMESTIC if origin == 'domestic'
                    else C_IMPORT if origin == 'import'
                    else ACCENT)

    cards = [
        kpi_card(f"{s['total_products']:,}",
                 'Tổng sản phẩm', C_TEXT),
        kpi_card(f"{s['total_revenue']/1e9:.1f} tỉ",
                 'Doanh thu ước tính', ACCENT,
                 border_accent=ACCENT),
        kpi_card(f"{s['pct_import']:.1f}%",
                 'Thị phần ngoại nhập',
                 C_IMPORT, border_accent=C_IMPORT),
        kpi_card(f"{s['total_brands']:,}",
                 'Số thương hiệu', C_TEXT),
        kpi_card(f"{s['total_countries']}",
                 'Quốc gia xuất xứ', C_TEXT),
        kpi_card(f"{s['avg_rating']:.2f} ★",
                 'Rating trung bình', C_SUCCESS),
        kpi_card(f"{s['avg_price']/1e3:.0f}K đ",
                 'Giá trung bình', C_WARNING),
        kpi_card(f"{s['pct_verified']:.1f}%",
                 'Tiki Verified', C_TEXT),
    ]
    return cards


# ── 3. Donut chart ──────────────────────────────────────────
@callback(
    Output('chart-donut', 'figure'),
    Input('store-filter', 'data'),
)
def update_donut(store):
    origin = store.get('origin', 'all')
    pt     = store.get('product_type', 'all')
    df     = _filter_df(origin, pt)

    rev_grp = (df.groupby('origin_class_corrected')['estimated_revenue']
               .sum().sort_values(ascending=False))

    if len(rev_grp) == 0:
        return go.Figure()

    labels = rev_grp.index.tolist()
    values = rev_grp.values.tolist()
    colors = [C_DOMESTIC if l == 'Trong nước' else C_IMPORT for l in labels]

    fig = donut_chart(values, labels, colors, height=300)

    # Annotation ở giữa donut
    total = sum(values)
    fig.add_annotation(
        text=f'<b>{total/1e9:.1f}</b><br><span style="font-size:11px">tỉ VNĐ</span>',
        x=0.5, y=0.5, showarrow=False,
        font=dict(size=16, color=C_TEXT),
        xref='paper', yref='paper',
    )
    return fig


# ── 4. Bar chart - product type count ───────────────────────
@callback(
    Output('chart-bar-product-type', 'figure'),
    Input('store-filter', 'data'),
)
def update_bar_type(store):
    origin = store.get('origin', 'all')
    pt     = store.get('product_type', 'all')
    df     = _filter_df(origin, pt)

    grp = (df.groupby(['product_type', 'origin_class_corrected'])
           .size().unstack(fill_value=0))

    types = [t for t in PRODUCT_TYPE_LIST if t in grp.index]
    dom_vals = [grp.loc[t, 'Trong nước'] if 'Trong nước' in grp.columns else 0
                for t in types]
    imp_vals = [grp.loc[t, 'Ngoài nước'] if 'Ngoài nước' in grp.columns else 0
                for t in types]

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
                yaxis={**LAYOUT_BASE['yaxis'],
                           'title': 'Số sản phẩm', 'showgrid': True},
                legend=dict(orientation='h', yanchor='bottom', y=1.02,
                            xanchor='right', x=1, font=dict(size=11)))
    return fig


# ── 5. Bar chart - top countries ────────────────────────────
@callback(
    Output('chart-bar-country', 'figure'),
    Input('store-filter', 'data'),
)
def update_bar_country(store):
    origin = store.get('origin', 'all')
    pt     = store.get('product_type', 'all')
    df     = _filter_df(origin, pt)

    rev_country = (df.groupby('origin_corrected')['estimated_revenue']
                   .sum().sort_values(ascending=False))

    top_n   = 8
    top     = rev_country.head(top_n)
    others  = rev_country.iloc[top_n:].sum()
    if others > 0:
        top['Khác'] = others

    countries = top.index.tolist()[::-1]
    values    = (top.values / 1e9).tolist()[::-1]

    bar_colors = []
    for c in countries:
        if c == 'Việt Nam':
            bar_colors.append(C_DOMESTIC)
        elif c == 'Khác':
            bar_colors.append(C_NEUTRAL)
        else:
            bar_colors.append(C_IMPORT)

    fig = go.Figure(go.Bar(
        x=values, y=countries,
        orientation='h',
        marker=dict(color=bar_colors, line=dict(color='white', width=0.6)),
        text=[f'{v:.1f} tỉ' for v in values],
        textposition='outside',
        textfont=dict(size=10, color=C_TEXT, weight=600),
        hovertemplate='<b>%{y}</b><br>%{x:.1f} tỉ VNĐ<extra></extra>',
        cliponaxis=False,
    ))
    apply_theme(fig, height=330,
                xaxis={**LAYOUT_BASE['xaxis'], 'title': 'Doanh thu (tỉ VNĐ)', 'showgrid': True},
                yaxis={**LAYOUT_BASE['yaxis'], 'showgrid': False},
                showlegend=False)
    fig.update_layout(margin=dict(l=16, r=60, t=16, b=30))
    return fig


# ── 6. Revenue by price segment ─────────────────────────────
@callback(
    Output('chart-bar-price-seg', 'figure'),
    Input('store-filter', 'data'),
)
def update_bar_price(store):
    origin = store.get('origin', 'all')
    pt     = store.get('product_type', 'all')
    df     = _filter_df(origin, pt)

    grp = (df.groupby(['price_segment', 'origin_class_corrected'],
                      observed=True)['estimated_revenue']
           .sum().unstack(fill_value=0))

    segs    = [s for s in PRICE_ORDER if s in grp.index]
    dom_rev = [(grp.loc[s, 'Trong nước'] / 1e9
                if 'Trong nước' in grp.columns else 0)
               for s in segs]
    imp_rev = [(grp.loc[s, 'Ngoài nước'] / 1e9
                if 'Ngoài nước' in grp.columns else 0)
               for s in segs]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name='Trong nước', x=segs, y=dom_rev,
        marker_color=C_DOMESTIC,
        marker_line=dict(color='white', width=0.8),
        hovertemplate='<b>%{x}</b><br>Trong nước: %{y:.1f} tỉ<extra></extra>',
    ))
    fig.add_trace(go.Bar(
        name='Ngoài nước', x=segs, y=imp_rev,
        marker_color=C_IMPORT,
        marker_line=dict(color='white', width=0.8),
        hovertemplate='<b>%{x}</b><br>Ngoài nước: %{y:.1f} tỉ<extra></extra>',
    ))
    apply_theme(fig, height=330,
                barmode='group',
                xaxis={**LAYOUT_BASE['xaxis'], 'showgrid':False,
                           'tickangle':-15},
                yaxis={**LAYOUT_BASE['yaxis'],
                           'title':'Doanh thu (tỉ VNĐ)', 'showgrid':True},
                legend=dict(orientation='h', yanchor='bottom', y=1.02,
                            xanchor='right', x=1, font=dict(size=11)))
    return fig


# ── 7. Sold by product type (stacked) ───────────────────────
@callback(
    Output('chart-sold-type', 'figure'),
    Input('store-filter', 'data'),
)
def update_sold_type(store):
    origin = store.get('origin', 'all')
    pt     = store.get('product_type', 'all')
    df     = _filter_df(origin, pt)

    grp = (df.groupby(['product_type', 'origin_class_corrected'])
           ['sold_count'].sum().unstack(fill_value=0))

    types    = [t for t in PRODUCT_TYPE_LIST if t in grp.index]
    dom_sold = [(grp.loc[t, 'Trong nước'] / 1e3
                 if 'Trong nước' in grp.columns else 0) for t in types]
    imp_sold = [(grp.loc[t, 'Ngoài nước'] / 1e3
                 if 'Ngoài nước' in grp.columns else 0) for t in types]

    # Sắp xếp theo tổng lượt bán
    totals  = [d + i for d, i in zip(dom_sold, imp_sold)]
    order   = sorted(range(len(totals)), key=lambda i: totals[i])
    types   = [types[i]    for i in order]
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
                xaxis={**LAYOUT_BASE['xaxis'],
                           'title':'Tổng lượt bán (nghìn)', 'showgrid':True},
                yaxis={**LAYOUT_BASE['yaxis'], 'showgrid': False},
                legend=dict(orientation='h', yanchor='bottom', y=1.02,
                            xanchor='right', x=1, font=dict(size=11)))
    return fig


# ── 8. Top 10 brands ────────────────────────────────────────
@callback(
    Output('chart-top-brands', 'figure'),
    Input('store-filter', 'data'),
)
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
        x=revs, y=brands,
        orientation='h',
        marker=dict(color=colors, line=dict(color='white', width=0.6)),
        text=[f'{v:.1f} tỉ' for v in revs],
        textposition='outside',
        textfont=dict(size=10, color=C_TEXT, weight=600),
        hovertemplate='<b>%{y}</b><br>%{x:.1f} tỉ VNĐ<extra></extra>',
        cliponaxis=False,
    ))
    apply_theme(fig, height=310,
                xaxis={**LAYOUT_BASE['xaxis'],
                           'title':'Doanh thu (tỉ VNĐ)', 'showgrid':True},
                yaxis={**LAYOUT_BASE['yaxis'], 'showgrid':False},
                showlegend=False)
    fig.update_layout(margin=dict(l=16, r=60, t=16, b=30))
    return fig
