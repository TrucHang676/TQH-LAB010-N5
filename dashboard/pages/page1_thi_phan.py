# pages/page1_thi_phan.py
# ╔══════════════════════════════════════════════════════════════╗
# ║  Trang 1 — Thị Phần: Phân tích thị trường Mỹ phẩm Tiki       ║
# ║  Focus: Biểu đồ chính - Clean Dashboard Layout               ║
# ╚══════════════════════════════════════════════════════════════╝

import dash
from dash import html, dcc, callback, Input, Output
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from data_loader import load_data
from theme import (
    C_DOMESTIC, C_IMPORT, C_NEUTRAL, C_TEXT, C_SUBTEXT,
    C_BG, C_BORDER, LAYOUT_BASE, apply_theme,
    PRODUCT_TYPE_COLORS, PRODUCT_TYPE_LIST,
)

dash.register_page(
    __name__,
    path='/thi-phan',
    name='Thị phần',
    title='Thị phần | Mỹ phẩm Tiki',
    order=1,
)

# ── Load data ────────────────────────────────────────────────
df_full, df_vn_full, df_nn_full = load_data()

PRICE_ORDER   = ['Dưới 100k', '100k – 300k', '300k – 700k', '700k – 2tr', 'Trên 2tr']
TIER_ORDER    = ['Bestseller', 'Bán chạy', 'Phổ biến', 'Mới', 'Chưa có lượt bán']
TIER_COLORS   = ['#4C1D95', '#6D28D9', '#8B5CF6', '#A78BFA', '#C4B5FD']
TYPE_COLORS   = PRODUCT_TYPE_COLORS

# ── Filter options ───────────────────────────────────────────
PRODUCT_TYPES_ALL = sorted(df_full['product_type'].unique().tolist())
PRICE_SEGMENTS_ALL = [p for p in PRICE_ORDER if p in df_full['price_segment'].unique()]
ORIGIN_OPTIONS = [
    {'label': 'Tất cả', 'value': 'all'},
    {'label': 'Trong nước', 'value': 'domestic'},
    {'label': 'Ngoài nước', 'value': 'import'},
]

# ── Design tokens ────────────────────────────────────────────
V_500   = '#8B5CF6'
V_700   = '#6D28D9'
V_300   = '#C4B5FD'
V_50    = 'rgba(139,92,246,0.10)'

GOLD    = '#F59E0B'
GREEN   = '#22C55E'

TEXT       = C_TEXT
SUBTEXT    = C_SUBTEXT
SURFACE    = '#1C2D55'
SURFACE_ALT = '#223763'
BORDER     = 'rgba(255,255,255,0.08)'
GRID       = 'rgba(255,255,255,0.06)'
HOVER_BG   = '#111827'
CARD_SHADOW = '0 14px 34px rgba(3,10,25,0.22)'
PAGE_BG    = '#14233F'


# ══════════════════════════════════════════════════════════════
#  FILTER & COMPUTE FUNCTIONS
# ══════════════════════════════════════════════════════════════

def apply_filters(df, product_types=None, price_segments=None, origin='all'):
    """
    Lọc dữ liệu theo các filter được chọn
    
    Args:
        df: Dataframe gốc
        product_types: List ngành hàng (None = tất cả)
        price_segments: List phân khúc giá (None = tất cả)
        origin: 'all' | 'domestic' | 'import'
    
    Returns:
        Filtered dataframe
    """
    result = df.copy()
    
    # Filter by product type
    if product_types and len(product_types) > 0:
        result = result[result['product_type'].isin(product_types)]
    
    # Filter by price segment
    if price_segments and len(price_segments) > 0:
        result = result[result['price_segment'].isin(price_segments)]
    
    # Filter by origin
    if origin == 'domestic':
        result = result[result['origin_class_corrected'] == 'Trong nước']
    elif origin == 'import':
        result = result[result['origin_class_corrected'] == 'Ngoài nước']
    
    return result


def compute_dynamic_kpi(df):
    """Tính KPI động từ filtered data"""
    if len(df) == 0:
        return {
            'total_products': 0,
            'total_revenue': 0,
            'import_pct': 0,
            'best_dsi_type': '-',
            'best_dsi_value': 0,
        }
    
    total_rev = df['estimated_revenue'].sum()
    imp_rev = df[df['origin_class_corrected'] == 'Ngoài nước']['estimated_revenue'].sum()
    imp_pct = (imp_rev / total_rev * 100) if total_rev > 0 else 0
    
    # Compute DSI for each type
    dsi_data = []
    for pt in df['product_type'].unique():
        df_type = df[df['product_type'] == pt]
        total_sold = df_type['sold_count'].sum()
        total_prod = len(df_type)
        dom_sold = df_type[df_type['origin_class_corrected'] == 'Trong nước']['sold_count'].sum()
        dom_prod = len(df_type[df_type['origin_class_corrected'] == 'Trong nước'])
        
        sold_share = (dom_sold / total_sold * 100) if total_sold > 0 else 0
        prod_share = (dom_prod / total_prod * 100) if total_prod > 0 else 0
        dsi = (sold_share + prod_share) / 2
        
        dsi_data.append({'type': pt, 'dsi': dsi})
    
    best_dsi = max(dsi_data, key=lambda x: x['dsi']) if dsi_data else None
    
    return {
        'total_products': len(df),
        'total_revenue': total_rev,
        'import_pct': imp_pct,
        'best_dsi_type': best_dsi['type'] if best_dsi else '-',
        'best_dsi_value': best_dsi['dsi'] if best_dsi else 0,
    }


# ══════════════════════════════════════════════════════════════
#  CHART BUILDERS (with data parameter)
# ══════════════════════════════════════════════════════════════

def section_header(num, title, desc, variant='muc1'):
    icons = {'muc1': '📊', 'muc2': '🏪', 'muc3': '🇻🇳'}
    return html.Div([
        html.Div([
            html.Span(icons[variant], style={'fontSize': '17px'}),
        ], className='p1-section-num'),
        html.Div([
            html.P(f'Mục tiêu {num}', style={
                'margin': '0 0 2px 0',
                'fontSize': '10px', 'fontWeight': '700',
                'color': {'muc1': V_500, 'muc2': GOLD, 'muc3': GREEN}[variant],
                'textTransform': 'uppercase', 'letterSpacing': '0.1em',
            }),
            html.H3(title, className='p1-section-title'),
            html.P(desc, className='p1-section-desc'),
        ]),
    ], className=f'p1-section-header {variant}')


def chart_card(icon, title, subtitle, children, flex='1', min_width='300px', icon_bg='#EDE9FE'):
    return html.Div([
        html.Div([
            html.Div(icon, className='p1-card-icon',
                     style={'background': icon_bg}),
            html.Div([
                html.P(title, className='p1-card-title'),
                html.P(subtitle, className='p1-card-subtitle'),
            ]),
        ], className='p1-card-header'),
        children,
    ], className='p1-card', style={'flex': flex, 'minWidth': min_width})


def insight_box(text, variant='violet'):
    icons = {'violet': '💡', 'gold': '📌', 'green': '✅'}
    return html.Div([
        html.Span(icons.get(variant, '💡'), style={'fontSize': '16px', 'flexShrink': '0'}),
        html.Span(text, style={'fontSize': '12px'}),
    ], className=f'p1-insight {variant}')


def kpi_card(icon, value, label, color, bg, tooltip=None):
    tooltip_node = html.Span(tooltip, className='p1-kpi-tip') if tooltip else None
    return html.Div([
        html.Div(icon, className='p1-kpi-dot',
                 style={'background': bg}),
        html.Div([
            html.P(value, className='p1-kpi-val', style={'color': color}),
            html.P(label, className='p1-kpi-label'),
        ]),
        tooltip_node,
    ], className='p1-kpi', style={'cursor': 'help'})


def _insight_style(n_clicks):
    return {'display': 'flex'} if n_clicks and n_clicks % 2 == 1 else {'display': 'none'}


def chart_panel(graph_id, figure, subtitle, insight_text, button_id, insight_id,
                insight_variant='violet', icon='i', icon_bg='#EDE9FE', flex='1', min_width='300px'):
    return html.Div([
        html.Div([
            html.Div([
                html.Div(icon, className='p1-card-icon', style={'background': icon_bg}),
                html.Div([
                    html.P(subtitle, className='p1-card-subtitle'),
                ], className='p1-card-copy'),
            ], className='p1-card-header-main'),
            html.Button('i', id=button_id, n_clicks=0, title='Xem insight',
                        className='p1-info-btn', **{'aria-label': f'Xem insight cho {subtitle}'}),
        ], className='p1-card-header'),
        html.Div(insight_text, id=insight_id, className=f'p1-insight {insight_variant}',
                 style={'display': 'none'}),
        dcc.Graph(id=graph_id, figure=figure, config={'displayModeBar': False}),
    ], className='p1-card', style={'flex': flex, 'minWidth': min_width})


# ══════════════════════════════════════════════════════════════
#  CHART BUILDERS (called once, not callbacks — static data)
# ══════════════════════════════════════════════════════════════

def _theme(fig, height=320, **kw):
    margin = kw.pop('margin', dict(l=16, r=16, t=50, b=16))
    fig.update_layout(
        height=height,
        font=dict(
            family="'DM Sans','Segoe UI',sans-serif",
            size=13,
            color=TEXT
        ),
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        margin=margin,
        hoverlabel=dict(
            bgcolor=HOVER_BG,
            bordercolor=BORDER,
            font=dict(size=13, color=TEXT)
        ),
        **kw,
    )

    fig.update_xaxes(
        gridcolor=GRID,
        gridwidth=1,
        linecolor='rgba(255,255,255,0.08)',
        linewidth=1,
        tickfont=dict(size=12, color=SUBTEXT),
        title_font=dict(size=13, color=SUBTEXT),
        zeroline=False,
    )
    fig.update_yaxes(
        gridcolor=GRID,
        gridwidth=1,
        linecolor='rgba(0,0,0,0)',
        tickfont=dict(size=12, color=SUBTEXT),
        title_font=dict(size=13, color=SUBTEXT),
        zeroline=False,
    )
    return fig


# ── MT1 Charts ───────────────────────────────────────────────

def make_donut():
    rev = df_full.groupby('origin_class_corrected')['estimated_revenue'].sum()
    labels = rev.index.tolist()
    values = rev.values.tolist()
    colors = [C_DOMESTIC if l == 'Trong nước' else C_IMPORT for l in labels]
    total  = sum(values)
    pct_import = rev.get('Ngoài nước', 0) / total * 100

    fig = go.Figure(go.Pie(
        values=values, labels=labels,
        marker=dict(colors=colors, line=dict(color='white', width=3)),
        hole=0.64,
        textinfo='percent',
        textfont=dict(size=16, color='white', weight=700),
        hovertemplate='<b>%{label}</b><br>%{value:,.0f} VNĐ<br><b>%{percent}</b><extra></extra>',
        pull=[0.05 if l == 'Ngoài nước' else 0 for l in labels],
        sort=False,
        direction='clockwise',
    ))
    fig.add_annotation(
        text=f'<b>{pct_import:.1f}%</b><br><span style="font-size:12px;color:{SUBTEXT}">Ngoại nhập</span>',
        x=0.5, y=0.5, showarrow=False,
        font=dict(size=20, color=C_IMPORT),
        xref='paper', yref='paper',
    )
    _theme(fig, height=300,
           showlegend=True,
           legend=dict(orientation='v', yanchor='middle', y=0.5,
                       xanchor='left', x=1.02,
                       font=dict(size=12)))
    fig.update_layout(margin=dict(l=16, r=80, t=50, b=16))
    return fig


def make_bar_country():
    rev_c = df_full.groupby('origin_corrected')['estimated_revenue'].sum().sort_values(ascending=False)
    top   = rev_c.head(8).copy()
    rest  = rev_c.iloc[8:].sum()
    if rest > 0:
        top['Khác'] = rest

    countries  = top.index.tolist()[::-1]
    values     = (top.values / 1e9).tolist()[::-1]
    bar_colors = []
    for c in countries:
        if c == 'Việt Nam':  bar_colors.append(C_DOMESTIC)
        elif c == 'Nhật Bản': bar_colors.append('#DC2626')
        elif c == 'Khác':    bar_colors.append(C_NEUTRAL)
        else:                 bar_colors.append('#F87171')

    fig = go.Figure(go.Bar(
        x=values, y=countries, orientation='h',
        marker=dict(color=bar_colors,
                    line=dict(color='white', width=0.8)),
        text=[f'{v:.1f} tỉ' for v in values],
        textposition='outside',
        textfont=dict(size=14, color=TEXT),
        hovertemplate='<b>%{y}</b><br>%{x:.1f} tỉ VNĐ<extra></extra>',
        cliponaxis=False,
    ))
    _theme(fig, height=330,
           showlegend=False,
           title=dict(text='<b>Doanh thu theo quốc gia xuất xứ</b>', x=0.5, xanchor='center', font=dict(size=18, color=TEXT)),
           xaxis=dict(title='Doanh thu (tỉ VNĐ)', showgrid=True,
                      gridcolor=GRID, tickfont=dict(size=12), title_font=dict(size=13, color=SUBTEXT)),
           yaxis=dict(showgrid=False, tickfont=dict(size=12)))
    fig.update_layout(margin=dict(l=16, r=70, t=70, b=30))
    return fig


def make_bar_price():
    grp = (df_full
           .groupby(['price_segment', 'origin_class_corrected'], observed=True)['estimated_revenue']
           .sum().unstack(fill_value=0))
    # Sắp xếp theo tổng doanh thu giảm dần
    grp['total'] = grp.sum(axis=1)
    segs = grp.sort_values('total', ascending=False).index.tolist()
    dom_rev = [(grp.loc[s, 'Trong nước'] / 1e9 if 'Trong nước' in grp.columns else 0) for s in segs]
    imp_rev = [(grp.loc[s, 'Ngoài nước'] / 1e9 if 'Ngoài nước' in grp.columns else 0) for s in segs]

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
    _theme(fig, height=300,
           barmode='group',
           title=dict(text='<b>Doanh thu theo phân khúc giá</b>', x=0.5, xanchor='center', font=dict(size=18, color=TEXT)),
           xaxis=dict(showgrid=False, tickfont=dict(size=12), tickangle=-10),
           yaxis=dict(title='Doanh thu (tỉ VNĐ)', showgrid=True,
                      gridcolor=GRID, tickfont=dict(size=12), title_font=dict(size=13, color=SUBTEXT)),
           legend=dict(orientation='h', yanchor='bottom', y=1.02,
                       xanchor='right', x=1, font=dict(size=12))
    )
    fig.update_layout(margin=dict(l=16, r=16, t=70, b=16))
    return fig


# ── MT2 Charts ───────────────────────────────────────────────

def make_dual_bar():
    """Tỉ trọng lượt bán vs doanh thu, nhóm theo ngành."""
    grp = (df_full.groupby('product_type')
           .agg(total_sold=('sold_count', 'sum'),
                total_rev=('estimated_revenue', 'sum'))
           .reset_index())
    grp['sold_share'] = grp['total_sold'] / grp['total_sold'].sum() * 100
    grp['rev_share']  = grp['total_rev']  / grp['total_rev'].sum()  * 100
    
    # Sắp xếp theo doanh thu giảm dần
    grp = grp.sort_values('total_rev', ascending=False)

    types = grp['product_type'].tolist()
    sold_vals = grp['sold_share'].tolist()
    rev_vals  = grp['rev_share'].tolist()
    bar_col   = [TYPE_COLORS.get(t, '#94A3B8') for t in types]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name='Tỉ trọng Lượt bán (%)',
        x=types, y=sold_vals,
        marker_color=bar_col,
        marker_line=dict(color='white', width=1),
        opacity=0.95,
        hovertemplate='<b>%{x}</b><br>Lượt bán: %{y:.1f}%<extra></extra>',
        text=[f'{v:.1f}%' for v in sold_vals],
        textposition='outside',
        textfont=dict(size=13, weight=700, color=TEXT),
    ))
    fig.add_trace(go.Bar(
        name='Tỉ trọng Doanh thu (%)',
        x=types, y=rev_vals,
        marker_color=bar_col,
        marker_line=dict(color='white', width=1),
        opacity=0.45,
        hovertemplate='<b>%{x}</b><br>Doanh thu: %{y:.1f}%<extra></extra>',
        text=[f'{v:.1f}%' for v in rev_vals],
        textposition='outside',
        textfont=dict(size=13, color=TEXT),
    ))
    # Delta annotations
    for i, (sold, rev, t) in enumerate(zip(sold_vals, rev_vals, types)):
        delta = rev - sold
        if abs(delta) > 1:
            color = C_IMPORT if delta > 0 else GREEN
            fig.add_annotation(
                x=t,
                y=max(sold, rev) + 14,
                text=f'Δ {delta:+.1f}%',
                showarrow=False,
                font=dict(size=12, color=color, weight=700),
                bgcolor='rgba(17,24,39,0.88)',
                bordercolor=color,
                borderwidth=1,
                borderpad=3,
            )

    _theme(fig, height=340,
           barmode='group',
           title=dict(text='<b>Tỉ trọng lượt bán & doanh thu theo ngành hàng</b>', x=0.5, xanchor='center', font=dict(size=18, color=TEXT)),
           xaxis=dict(showgrid=False, tickfont=dict(size=12, weight=700)),
           yaxis=dict(title='Tỉ trọng (%)', showgrid=True,
                      gridcolor=GRID, tickfont=dict(size=12),
                      range=[0, max(max(sold_vals), max(rev_vals)) * 1.45]),
           legend=dict(orientation='h', yanchor='bottom', y=1.02,
                       xanchor='right', x=1, font=dict(size=12))
    )
    fig.update_layout(margin=dict(l=16, r=16, t=70, b=16))
    return fig


def make_top_categories():
    """Top 3 danh mục doanh thu trong mỗi ngành."""
    top_cats = (df_full.groupby(['product_type', 'category'])
                .agg(cat_rev=('estimated_revenue', 'sum'))
                .reset_index()
                .sort_values(['product_type', 'cat_rev'], ascending=[True, False])
                .groupby('product_type')
                .head(3))

    y_pos, y_label, x_vals, colors, customdata = [], [], [], [], []
    offset = 0
    type_order = (df_full.groupby('product_type')['estimated_revenue']
                  .sum().sort_values(ascending=False).index.tolist())

    for pt in type_order:
        cats = top_cats[top_cats['product_type'] == pt].sort_values('cat_rev', ascending=False)
        col  = TYPE_COLORS.get(pt, '#94A3B8')
        for _, row in cats.iterrows():
            y_pos.append(offset)
            y_label.append(f"  {row['category']}")
            x_vals.append(row['cat_rev'] / 1e9)
            colors.append(col)
            customdata.append(pt)
            offset += 1
        offset += 0.6

    fig = go.Figure(go.Bar(
        x=x_vals, y=y_pos, orientation='h',
        marker=dict(color=colors, line=dict(color='white', width=0.6)),
        text=[f'{v:.0f} tỉ' for v in x_vals],
        textposition='outside',
        textfont=dict(size=12, color=TEXT),
        hovertemplate='<b>%{customdata}</b><br>%{y}<br>%{x:.1f} tỉ VNĐ<extra></extra>',
        customdata=customdata,
        cliponaxis=False,
    ))
    _theme(fig, height=360,
           showlegend=False,
           xaxis=dict(title='Doanh thu (tỉ VNĐ)', showgrid=True,
                      gridcolor=GRID, tickfont=dict(size=11)),
           yaxis=dict(showgrid=False, tickvals=y_pos, ticktext=y_label,
                      tickfont=dict(size=11.5))
    )
    fig.update_layout(margin=dict(l=16, r=70, t=50, b=16))
    return fig


def make_popularity_tier():
    """Stacked bar: popularity tier theo ngành."""
    type_order = (df_full.groupby('product_type')['estimated_revenue']
                  .sum().sort_values(ascending=False).index.tolist())

    avail_tiers = [t for t in TIER_ORDER if t in df_full['popularity_tier'].unique()]
    pop_pct = (pd.crosstab(df_full['product_type'], df_full['popularity_tier'], normalize='index') * 100)
    pop_pct = pop_pct.reindex(columns=avail_tiers, fill_value=0).loc[type_order[::-1]]

    fig = go.Figure()
    tier_clrs = TIER_COLORS
    for tier, color in zip(avail_tiers, tier_clrs):
        if tier not in pop_pct.columns:
            continue
        vals = pop_pct[tier].values
        fig.add_trace(go.Bar(
            name=tier,
            x=vals,
            y=pop_pct.index.tolist(),
            orientation='h',
            marker=dict(color=color, line=dict(color='white', width=0.8)),
            hovertemplate=f'<b>%{{y}}</b><br>{tier}: %{{x:.1f}}%<extra></extra>',
            text=[f'{v:.0f}%' if v > 7 else '' for v in vals],
            textposition='inside',
            textfont=dict(size=11, color='white'),
            insidetextanchor='middle',
        ))

    _theme(fig, height=300,
           barmode='stack',
           xaxis=dict(showgrid=False, tickfont=dict(size=11),
                      range=[0, 100]),
           yaxis=dict(showgrid=False, tickfont=dict(size=12)),
           legend=dict(orientation='h', yanchor='top', y=-0.15,
                       xanchor='center', x=0.5, font=dict(size=11))
    )
    fig.update_layout(margin=dict(l=16, r=16, t=50, b=30))
    return fig


# ── MT3 Charts ───────────────────────────────────────────────

def compute_dsi():
    grp = df_full.groupby('product_type').agg(
        total_sold=('sold_count', 'sum'),
        total_rev=('estimated_revenue', 'sum'),
        total_products=('product_id', 'count'),
    ).reset_index()
    dom = (df_full[df_full['origin_class_corrected'] == 'Trong nước']
           .groupby('product_type').agg(
               dom_sold=('sold_count', 'sum'),
               dom_rev=('estimated_revenue', 'sum'),
               dom_products=('product_id', 'count'),
           ).reset_index())
    dsi = grp.merge(dom, on='product_type', how='left').fillna(0)
    dsi['sold_share']    = dsi['dom_sold']     / dsi['total_sold'].replace(0, np.nan) * 100
    dsi['rev_share']     = dsi['dom_rev']      / dsi['total_rev'].replace(0, np.nan) * 100
    dsi['product_share'] = dsi['dom_products'] / dsi['total_products'].replace(0, np.nan) * 100
    dsi['DSI']           = (dsi['sold_share'] + dsi['rev_share'] + dsi['product_share']) / 3
    return dsi.fillna(0).sort_values('DSI', ascending=False)


def make_radar():
    dsi = compute_dsi()
    cats = dsi['product_type'].tolist()

    fig = go.Figure()
    metrics = {
        'Lượt bán nội (%)' : (dsi['sold_share'].tolist(),    C_DOMESTIC),
        'Doanh thu nội (%)': (dsi['rev_share'].tolist(),     C_IMPORT),
        'Sản phẩm nội (%)' : (dsi['product_share'].tolist(), GOLD),
    }
    for name, (vals, color) in metrics.items():
        fig.add_trace(go.Scatterpolar(
            r=vals + [vals[0]],
            theta=cats + [cats[0]],
            name=name,
            line=dict(color=color, width=2.2),
            fill='toself',
            fillcolor=color,
            opacity=0.13,
            marker=dict(size=7, color=color),
            hovertemplate=f'<b>%{{theta}}</b><br>{name}: %{{r:.1f}}%<extra></extra>',
        ))

    fig.update_layout(
        height=350,
        polar=dict(
            bgcolor=SURFACE_ALT,
            angularaxis=dict(
                tickfont=dict(size=12, weight=700, color=TEXT),
                linecolor=BORDER,
            ),
            radialaxis=dict(
                visible=True, range=[0, 100],
                tickfont=dict(size=11, color=SUBTEXT),
                gridcolor='rgba(255,255,255,0.10)',
                tickvals=[20, 40, 60, 80, 100],
                ticktext=['20%', '40%', '60%', '80%', '100%'],
            ),
        ),
        paper_bgcolor=SURFACE,
        font=dict(family="'DM Sans','Segoe UI',sans-serif", size=13, color=TEXT),
        legend=dict(orientation='h', yanchor='bottom', y=-0.18,
                    xanchor='center', x=0.5, font=dict(size=12)),
        margin=dict(l=40, r=40, t=50, b=50),
        hoverlabel=dict(
            bgcolor=HOVER_BG,
            bordercolor=BORDER,
            font=dict(size=13, color=TEXT)
        )
    )
    return fig


def make_dsi_stacked():
    dsi = compute_dsi().sort_values('total_sold', ascending=True)
    dom_vals  = dsi['dom_sold'].values
    total_vals = dsi['total_sold'].values
    imp_vals  = total_vals - dom_vals
    types     = dsi['product_type'].tolist()

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name='Trong nước', y=types, x=dom_vals / 1e3,
        orientation='h',
        marker=dict(color=C_DOMESTIC, line=dict(color='white', width=0.6)),
        hovertemplate='<b>%{y}</b><br>Trong nước: %{x:.1f}K lượt<extra></extra>',
    ))
    fig.add_trace(go.Bar(
        name='Ngoài nước', y=types, x=imp_vals / 1e3,
        orientation='h',
        marker=dict(color=C_IMPORT, line=dict(color='white', width=0.6)),
        hovertemplate='<b>%{y}</b><br>Ngoài nước: %{x:.1f}K lượt<extra></extra>',
    ))

    # Pct annotations
    for i, (d, tot, tp) in enumerate(zip(dom_vals, total_vals, types)):
        pct = d / tot * 100 if tot > 0 else 0
        fig.add_annotation(
            x=tot / 1e3 * 1.04, y=tp,
            text=f'Nội {pct:.1f}%',
            showarrow=False,
            xanchor='left',
            font=dict(size=13, color=TEXT, weight=600),
            bgcolor='rgba(17,24,39,0.88)',
            bordercolor=BORDER,
            borderwidth=1,
            borderpad=3,
        )

    _theme(fig, height=300,
           barmode='stack',
           title=dict(text='<b>Cơ cấu lượt bán nội/ngoại theo ngành</b>', x=0.5, xanchor='center', font=dict(size=18, color=TEXT)),
           xaxis=dict(title='Lượt bán (nghìn)', showgrid=True,
                      gridcolor=GRID, tickfont=dict(size=12), title_font=dict(size=13, color=SUBTEXT)),
           yaxis=dict(showgrid=False, tickfont=dict(size=12)),
           legend=dict(orientation='h', yanchor='bottom', y=1.02,
                       xanchor='right', x=1, font=dict(size=12))
    )
    fig.update_layout(margin=dict(l=16, r=16, t=70, b=16))
    return fig


def make_dsi_bar():
    """DSI score bar chart theo ngành."""
    dsi = compute_dsi().sort_values('DSI', ascending=True)

    types = dsi['product_type'].tolist()
    dsi_vals = dsi['DSI'].tolist()
    colors = []
    for v in dsi_vals:
        if v >= 50:  colors.append('#16A34A')
        elif v >= 25: colors.append('#F59E0B')
        else:         colors.append('#DC2626')

    fig = go.Figure(go.Bar(
        x=dsi_vals, y=types, orientation='h',
        marker=dict(color=colors,
                    line=dict(color='white', width=0.8)),
        text=[f'{v:.1f}' for v in dsi_vals],
        textposition='outside',
        textfont=dict(size=13, weight=700, color=TEXT),
        hovertemplate='<b>%{y}</b><br>DSI: %{x:.1f}<extra></extra>',
        cliponaxis=False,
    ))
    # Benchmark line at 33.3 (neutral)
    fig.add_vline(x=33.3, line_dash='dash',
                  line_color='#94A3B8', line_width=1.5)
    fig.add_annotation(x=33.3, y=len(types) - 0.5,
                       text='Cân bằng (33%)', showarrow=False,
                       font=dict(size=12, color=SUBTEXT),
                       xanchor='center', yanchor='bottom')

    _theme(fig, height=300,
           showlegend=False,
           title=dict(text='<b>Chỉ số Sức mạnh nội địa (DSI) theo ngành</b>', x=0.5, xanchor='center', font=dict(size=18, color=TEXT)),
           xaxis=dict(title='Chỉ số DSI (trung bình 3 chiều)',
                      showgrid=True, gridcolor=GRID,
                      range=[0, 100], tickfont=dict(size=12), title_font=dict(size=13, color=SUBTEXT)),
           yaxis=dict(showgrid=False, tickfont=dict(size=12))
    )
    fig.update_layout(margin=dict(l=16, r=16, t=70, b=16))
    return fig


# ══════════════════════════════════════════════════════════════
#  LAYOUT
# ══════════════════════════════════════════════════════════════

def layout():
    dsi_df   = compute_dsi()
    best_type = dsi_df.iloc[0]['product_type']
    best_dsi  = dsi_df.iloc[0]['DSI']

    # Top KPIs
    total_rev   = df_full['estimated_revenue'].sum()
    imp_rev     = df_nn_full['estimated_revenue'].sum()
    imp_pct_rev = imp_rev / total_rev * 100
    n_types     = df_full['product_type'].nunique()

    country_insight = (
        'Biểu đồ này trả lời câu hỏi doanh thu đang nghiêng về hàng trong nước hay ngoài nước. '
        'Hãy nhìn thêm phần “Khác” để nhận ra mức độ tập trung theo quốc gia xuất xứ.'
    )
    price_insight = (
        'Biểu đồ này cho biết phân khúc giá nào tạo doanh thu tốt nhất ở từng nhóm xuất xứ. '
        'Nếu một phân khúc cao nhưng doanh thu thấp, đó thường là tín hiệu nhu cầu chưa đủ rộng.'
    )
    dual_insight = (
        'Hai cột cùng màu giúp so sánh tỉ trọng lượt bán với tỉ trọng doanh thu. '
        'Ngành nào có doanh thu cao hơn lượt bán là ngành có giá trị đơn hàng tốt hơn mặt bằng chung.'
    )
    dsi_bar_insight = (
        'DSI đo mức độ “nội địa hóa” của từng ngành trên 3 chiều: lượt bán, doanh thu và số sản phẩm. '
        'Chỉ số càng cao thì vai trò của hàng nội địa trong ngành đó càng mạnh.'
    )
    dsi_stack_insight = (
        'Biểu đồ chồng này cho biết tổng lượt bán của từng ngành đang đến từ hàng nội hay hàng ngoại. '
        'Phần màu trong nước càng lớn thì ngành đó càng ít phụ thuộc vào nguồn cung nhập khẩu.'
    )
    kpi_tooltips = {
        'products': 'Tổng số sản phẩm có trong bộ dữ liệu đang phân tích.',
        'revenue': 'Tổng doanh thu ước tính từ các sản phẩm trong tập dữ liệu.',
        'import': 'Tỷ trọng doanh thu đến từ hàng ngoài nước so với tổng doanh thu.',
        'dsi': 'Ngành có chỉ số sức mạnh nội địa DSI cao nhất trong tập dữ liệu.',
    }

    return html.Div([

        # ── HEADER + FILTER (Integrated) ─────────────────────────
        html.Div([
            html.Div([
                html.H1('Thị phần Mỹ phẩm Tiki · T3/2026',
                        className='p1-hero-title')
            ], style={'marginBottom': '20px'}),

            # ── FILTER BAR (Inside Hero) ─────────────────────────
            html.Div([
                html.Div([
                    html.Span('Lọc:', style={
                        'fontSize': '12px',
                        'color': 'rgba(255,255,255,0.6)',
                        'fontWeight': '600',
                        'marginRight': '8px',
                        'textTransform': 'uppercase',
                        'letterSpacing': '0.06em',
                    }),
                    html.Div([
                        dcc.Dropdown(
                            id='p1-filter-product-type',
                            options=[{'label': 'Tất cả ngành', 'value': 'all'}] + 
                                    [{'label': pt, 'value': pt} for pt in PRODUCT_TYPES_ALL],
                            value='all',
                            multi=False,
                            clearable=False,
                            className='p1-filter-pill-select'
                        ),
                    ], style={'minWidth': '160px', 'display': 'flex'}),
                    
                    html.Div([
                        dcc.Dropdown(
                            id='p1-filter-price-segment',
                            options=[{'label': 'Tất cả giá', 'value': 'all'}] + 
                                    [{'label': seg, 'value': seg} for seg in PRICE_SEGMENTS_ALL],
                            value='all',
                            multi=False,
                            clearable=False,
                            className='p1-filter-pill-select'
                        ),
                    ], style={'minWidth': '160px', 'display': 'flex'}),
                ], style={
                    'display': 'flex',
                    'alignItems': 'center',
                    'gap': '12px',
                    'flexWrap': 'wrap',
                }),

                html.Div([
                    html.Span('Xuất xứ:', style={
                        'fontSize': '13px',
                        'color': 'rgba(255,255,255,0.6)',
                        'fontWeight': '600',
                        'marginRight': '8px',
                        'textTransform': 'uppercase',
                        'letterSpacing': '0.06em',
                    }),
                    dcc.RadioItems(
                        id='p1-filter-origin',
                        options=ORIGIN_OPTIONS,
                        value='all',
                        inline=True,
                        style={'display': 'flex', 'gap': '14px'},
                        className='p1-filter-radio'
                    ),
                ], style={
                    'display': 'flex',
                    'alignItems': 'center',
                    'gap': '13px',
                    'flexWrap': 'wrap',
                }),
            ], style={
                'display': 'flex',
                'alignItems': 'center',
                'gap': '20px',
                'paddingTop': '16px',
                'borderTop': '1px solid rgba(255,255,255,0.12)',
                'flexWrap': 'wrap',
            }),
        ], className='p1-hero', style={'paddingBottom': '20px'}),

        # ── KPI Row (DYNAMIC) ─────────────────────────────────────────
        html.Div(id='p1-kpi-row', children=[
            kpi_card('📦', f"{len(df_full):,}", 'Tổng sản phẩm', '#38BDF8', 'rgba(56,189,248,0.14)', kpi_tooltips['products']),
            kpi_card('💰', f"{total_rev/1e9:.1f} tỉ", 'Doanh thu ước tính', '#A78BFA', 'rgba(139,92,246,0.14)', kpi_tooltips['revenue']),
            kpi_card('🌍', f"{imp_pct_rev:.1f}%", 'Thị phần ngoại nhập', '#F87171', 'rgba(248,113,113,0.14)', kpi_tooltips['import']),
            kpi_card('🇻🇳', f"{best_dsi:.1f} DSI", f'Mạnh nhất: {best_type}', '#34D399', 'rgba(52,211,153,0.14)', kpi_tooltips['dsi']),
        ], className='p1-kpi-row'),

        # ── MAIN CHARTS: 2x2+1 Layout ──────────────────────
        html.Div([
            chart_panel(
                'p1-chart-country', make_bar_country(),
                'Nguồn doanh thu đang đến từ đâu?',
                country_insight, 'p1-btn-country', 'p1-insight-country',
                insight_variant='gold', icon='🌍', icon_bg='rgba(245,158,11,0.16)',
                min_width='320px'
            ),

            chart_panel(
                'p1-chart-price', make_bar_price(),
                'Mức giá nào đang tạo giá trị?',
                price_insight, 'p1-btn-price', 'p1-insight-price',
                insight_variant='violet', icon='💳', icon_bg='rgba(139,92,246,0.16)',
                min_width='320px'
            ),
        ], className='p1-row'),

        # Middle row
        html.Div([
            chart_panel(
                'p1-chart-dual-bar', make_dual_bar(),
                'Lượt bán và doanh thu có đang lệch nhau?',
                dual_insight, 'p1-btn-dual-bar', 'p1-insight-dual-bar',
                insight_variant='green', icon='⚖️', icon_bg='rgba(52,211,153,0.16)',
                min_width='320px'
            ),

            chart_panel(
                'p1-chart-dsi-bar', make_dsi_bar(),
                'Ngành nào có lợi thế nội địa rõ nhất?',
                dsi_bar_insight, 'p1-btn-dsi-bar', 'p1-insight-dsi-bar',
                insight_variant='gold', icon='🏅', icon_bg='rgba(245,158,11,0.16)',
                min_width='320px'
            ),
        ], className='p1-row'),

        # Bottom row (full width)
        html.Div([
            chart_panel(
                'p1-chart-dsi-stacked', make_dsi_stacked(),
                'Ngành nào đang phụ thuộc nhiều hơn vào hàng ngoại?',
                dsi_stack_insight, 'p1-btn-dsi-stacked', 'p1-insight-dsi-stacked',
                insight_variant='violet', icon='📊', icon_bg='rgba(124,58,237,0.16)',
                min_width='400px'
            ),
        ], className='p1-row'),

        # ── Footer ──────────────────────────────────────────
        html.Div([
            html.Span('⚠️ Dữ liệu crawl từ Tiki tháng 3/2026'),
            html.Span('DSI = (sold_share + rev_share + product_share) / 3'),
            html.Span('Nhóm 05 · FIT-HCMUS'),
        ], className='p1-footer'),

    ], className='p1-page', style={'padding': '0', 'background': PAGE_BG})


# ══════════════════════════════════════════════════════════════
#  CALLBACK: Handle filter changes & update all charts + KPI
# ══════════════════════════════════════════════════════════════

@callback(
    [
        Output('p1-kpi-row', 'children'),
        Output('p1-chart-country', 'figure'),
        Output('p1-chart-price', 'figure'),
        Output('p1-chart-dual-bar', 'figure'),
        Output('p1-chart-dsi-bar', 'figure'),
        Output('p1-chart-dsi-stacked', 'figure'),
    ],
    [
        Input('p1-filter-product-type', 'value'),
        Input('p1-filter-price-segment', 'value'),
        Input('p1-filter-origin', 'value'),
    ]
)
def update_dashboard(selected_type, selected_price, selected_origin):
    """Update toàn bộ dashboard khi filter thay đổi"""
    
    # Xác định filter values
    product_types = None if selected_type == 'all' else [selected_type]
    price_segments = None if selected_price == 'all' else [selected_price]
    origin = selected_origin
    
    # Apply filters
    df_filtered = apply_filters(df_full, product_types, price_segments, origin)
    
    # Compute dynamic KPI
    kpi = compute_dynamic_kpi(df_filtered)
    
    # Update KPI cards
    kpi_row_children = [
        kpi_card('📦', f"{kpi['total_products']:,}", 'Tổng sản phẩm', '#38BDF8', 'rgba(56,189,248,0.14)', 'Tổng số sản phẩm đang hiển thị sau khi lọc.'),
        kpi_card('💰', f"{kpi['total_revenue']/1e9:.1f} tỉ", 'Doanh thu ước tính', '#A78BFA', 'rgba(139,92,246,0.14)', 'Tổng doanh thu ước tính của tập dữ liệu đang lọc.'),
        kpi_card('🌍', f"{kpi['import_pct']:.1f}%", 'Thị phần ngoại nhập', '#F87171', 'rgba(248,113,113,0.14)', 'Tỷ trọng doanh thu đến từ hàng ngoài nước trong tập đang lọc.'),
        kpi_card('🇻🇳', f"{kpi['best_dsi_value']:.1f} DSI", f'Mạnh nhất: {kpi["best_dsi_type"]}', '#34D399', 'rgba(52,211,153,0.14)', 'Ngành có chỉ số DSI cao nhất trong tập dữ liệu đang lọc.'),
    ]
    
    # Regenerate charts with filtered data
    fig_country = make_bar_country_filtered(df_filtered)
    fig_price = make_bar_price_filtered(df_filtered)
    fig_dual = make_dual_bar_filtered(df_filtered)
    fig_dsi_bar = make_dsi_bar_filtered(df_filtered)
    fig_dsi_stacked = make_dsi_stacked_filtered(df_filtered)
    
    return kpi_row_children, fig_country, fig_price, fig_dual, fig_dsi_bar, fig_dsi_stacked


@callback(
    [
        Output('p1-insight-country', 'style'),
        Output('p1-insight-price', 'style'),
        Output('p1-insight-dual-bar', 'style'),
        Output('p1-insight-dsi-bar', 'style'),
        Output('p1-insight-dsi-stacked', 'style'),
    ],
    [
        Input('p1-btn-country', 'n_clicks'),
        Input('p1-btn-price', 'n_clicks'),
        Input('p1-btn-dual-bar', 'n_clicks'),
        Input('p1-btn-dsi-bar', 'n_clicks'),
        Input('p1-btn-dsi-stacked', 'n_clicks'),
    ]
)
def toggle_chart_insights(country_clicks, price_clicks, dual_clicks, dsi_clicks, stacked_clicks):
    return [
        _insight_style(country_clicks),
        _insight_style(price_clicks),
        _insight_style(dual_clicks),
        _insight_style(dsi_clicks),
        _insight_style(stacked_clicks),
    ]


# ══════════════════════════════════════════════════════════════
#  FILTERED CHART FUNCTIONS
# ══════════════════════════════════════════════════════════════

def make_bar_country_filtered(df):
    """Bar chart: Doanh thu theo quốc gia (filtered)"""
    if len(df) == 0:
        return go.Figure().update_layout(
            title='Không có dữ liệu',
            paper_bgcolor=SURFACE,
            plot_bgcolor=SURFACE,
            font=dict(color=TEXT),
        )
    
    rev_c = df.groupby('origin_corrected')['estimated_revenue'].sum().sort_values(ascending=False)
    top   = rev_c.head(8).copy()
    rest  = rev_c.iloc[8:].sum()
    if rest > 0:
        top['Khác'] = rest

    countries  = top.index.tolist()[::-1]
    values     = (top.values / 1e9).tolist()[::-1]
    bar_colors = []
    for c in countries:
        if c == 'Việt Nam':  bar_colors.append(C_DOMESTIC)
        elif c == 'Nhật Bản': bar_colors.append('#DC2626')
        elif c == 'Khác':    bar_colors.append(C_NEUTRAL)
        else:                 bar_colors.append('#F87171')

    fig = go.Figure(go.Bar(
        x=values, y=countries, orientation='h',
        marker=dict(color=bar_colors,
                    line=dict(color='white', width=0.8)),
        text=[f'{v:.1f} tỉ' for v in values],
        textposition='outside',
        textfont=dict(size=14, color=TEXT),
        hovertemplate='<b>%{y}</b><br>%{x:.1f} tỉ VNĐ<extra></extra>',
        cliponaxis=False,
    ))
    _theme(fig, height=330,
           showlegend=False,
           title=dict(text='<b>Doanh thu theo quốc gia xuất xứ</b>', x=0.5, xanchor='center', font=dict(size=18, color=TEXT)),
           xaxis=dict(title='Doanh thu (tỉ VNĐ)', showgrid=True,
                      gridcolor=GRID, tickfont=dict(size=12), title_font=dict(size=13, color=SUBTEXT)),
           yaxis=dict(showgrid=False, tickfont=dict(size=12)))
    fig.update_layout(margin=dict(l=16, r=70, t=70, b=30))
    return fig


def make_bar_price_filtered(df):
    """Bar chart: Doanh thu theo phân khúc giá (filtered)"""
    if len(df) == 0:
        return go.Figure().update_layout(
            title='Không có dữ liệu',
            paper_bgcolor=SURFACE,
            plot_bgcolor=SURFACE,
            font=dict(color=TEXT),
        )
    
    grp = (df.groupby(['price_segment', 'origin_class_corrected'], observed=True)['estimated_revenue']
           .sum().unstack(fill_value=0))
    grp['total'] = grp.sum(axis=1)
    segs = grp.sort_values('total', ascending=False).index.tolist()
    dom_rev = [(grp.loc[s, 'Trong nước'] / 1e9 if 'Trong nước' in grp.columns else 0) for s in segs]
    imp_rev = [(grp.loc[s, 'Ngoài nước'] / 1e9 if 'Ngoài nước' in grp.columns else 0) for s in segs]

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
    _theme(fig, height=300,
           barmode='group',
           title=dict(text='<b>Doanh thu theo phân khúc giá</b>', x=0.5, xanchor='center', font=dict(size=18, color=TEXT)),
           xaxis=dict(showgrid=False, tickfont=dict(size=12), tickangle=-10),
           yaxis=dict(title='Doanh thu (tỉ VNĐ)', showgrid=True,
                      gridcolor=GRID, tickfont=dict(size=12), title_font=dict(size=13, color=SUBTEXT)),
           legend=dict(orientation='h', yanchor='bottom', y=1.02,
                       xanchor='right', x=1, font=dict(size=12))
    )
    fig.update_layout(margin=dict(l=16, r=16, t=70, b=16))
    return fig


def make_dual_bar_filtered(df):
    """Bar chart: Tỉ trọng lượt bán vs doanh thu theo ngành (filtered)"""
    if len(df) == 0:
        return go.Figure().update_layout(
            title='Không có dữ liệu',
            paper_bgcolor=SURFACE,
            plot_bgcolor=SURFACE,
            font=dict(color=TEXT),
        )
    
    grp = (df.groupby('product_type')
           .agg(total_sold=('sold_count', 'sum'),
                total_rev=('estimated_revenue', 'sum'))
           .reset_index())
    grp['sold_share'] = grp['total_sold'] / grp['total_sold'].sum() * 100
    grp['rev_share']  = grp['total_rev']  / grp['total_rev'].sum()  * 100
    
    grp = grp.sort_values('total_rev', ascending=False)

    types = grp['product_type'].tolist()
    sold_vals = grp['sold_share'].tolist()
    rev_vals  = grp['rev_share'].tolist()
    bar_col   = [TYPE_COLORS.get(t, '#94A3B8') for t in types]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name='Tỉ trọng Lượt bán (%)',
        x=types, y=sold_vals,
        marker_color=bar_col,
        marker_line=dict(color='white', width=1),
        opacity=0.95,
        hovertemplate='<b>%{x}</b><br>Lượt bán: %{y:.1f}%<extra></extra>',
        text=[f'{v:.1f}%' for v in sold_vals],
        textposition='outside',
        textfont=dict(size=13, weight=700, color=TEXT),
    ))
    fig.add_trace(go.Bar(
        name='Tỉ trọng Doanh thu (%)',
        x=types, y=rev_vals,
        marker_color=bar_col,
        marker_line=dict(color='white', width=1),
        opacity=0.45,
        hovertemplate='<b>%{x}</b><br>Doanh thu: %{y:.1f}%<extra></extra>',
        text=[f'{v:.1f}%' for v in rev_vals],
        textposition='outside',
        textfont=dict(size=13, color=TEXT),
    ))
    # Delta annotations
    for i, (sold, rev, t) in enumerate(zip(sold_vals, rev_vals, types)):
        delta = rev - sold
        if abs(delta) > 1:
            color = C_IMPORT if delta > 0 else GREEN
            fig.add_annotation(
                x=t,
                y=max(sold, rev) + 14,
                text=f'Δ {delta:+.1f}%',
                showarrow=False,
                font=dict(size=12, color=color, weight=700),
                bgcolor='rgba(17,24,39,0.88)',
                bordercolor=color,
                borderwidth=1,
                borderpad=3,
            )

    _theme(fig, height=340,
           barmode='group',
           title=dict(text='<b>Tỉ trọng lượt bán & doanh thu theo ngành hàng</b>', x=0.5, xanchor='center', font=dict(size=18, color=TEXT)),
           xaxis=dict(showgrid=False, tickfont=dict(size=12, weight=700)),
           yaxis=dict(title='Tỉ trọng (%)', showgrid=True,
                      gridcolor=GRID, tickfont=dict(size=12),
                      range=[0, max(max(sold_vals), max(rev_vals)) * 1.45] if sold_vals and rev_vals else [0, 100]),
           legend=dict(orientation='h', yanchor='bottom', y=1.02,
                       xanchor='right', x=1, font=dict(size=12))
    )
    fig.update_layout(margin=dict(l=16, r=16, t=70, b=16))
    return fig


def make_dsi_bar_filtered(df):
    """DSI score bar chart theo ngành (filtered)"""
    if len(df) == 0:
        return go.Figure().update_layout(
            title='Không có dữ liệu',
            paper_bgcolor=SURFACE,
            plot_bgcolor=SURFACE,
            font=dict(color=TEXT),
        )
    
    grp = df.groupby('product_type').agg(
        total_sold=('sold_count', 'sum'),
        total_rev=('estimated_revenue', 'sum'),
        total_products=('product_id', 'count'),
    ).reset_index()
    dom = (df[df['origin_class_corrected'] == 'Trong nước']
           .groupby('product_type').agg(
               dom_sold=('sold_count', 'sum'),
               dom_rev=('estimated_revenue', 'sum'),
               dom_products=('product_id', 'count'),
           ).reset_index())
    dsi = grp.merge(dom, on='product_type', how='left').fillna(0)
    dsi['sold_share']    = dsi['dom_sold']     / dsi['total_sold'].replace(0, np.nan) * 100
    dsi['rev_share']     = dsi['dom_rev']      / dsi['total_rev'].replace(0, np.nan) * 100
    dsi['product_share'] = dsi['dom_products'] / dsi['total_products'].replace(0, np.nan) * 100
    dsi['DSI']           = (dsi['sold_share'] + dsi['rev_share'] + dsi['product_share']) / 3
    dsi = dsi.fillna(0).sort_values('DSI', ascending=True)

    types = dsi['product_type'].tolist()
    dsi_vals = dsi['DSI'].tolist()
    colors = []
    for v in dsi_vals:
        if v >= 50:  colors.append('#16A34A')
        elif v >= 25: colors.append('#F59E0B')
        else:         colors.append('#DC2626')

    fig = go.Figure(go.Bar(
        x=dsi_vals, y=types, orientation='h',
        marker=dict(color=colors,
                    line=dict(color='white', width=0.8)),
        text=[f'{v:.1f}' for v in dsi_vals],
        textposition='outside',
        textfont=dict(size=13, weight=700, color=TEXT),
        hovertemplate='<b>%{y}</b><br>DSI: %{x:.1f}<extra></extra>',
        cliponaxis=False,
    ))
    fig.add_vline(x=33.3, line_dash='dash',
                  line_color='#94A3B8', line_width=1.5)
    fig.add_annotation(x=33.3, y=len(types) - 0.5,
                       text='Cân bằng (33%)', showarrow=False,
                       font=dict(size=12, color=SUBTEXT),
                       xanchor='center', yanchor='bottom')

    _theme(fig, height=300,
           showlegend=False,
           title=dict(text='<b>Chỉ số Sức mạnh nội địa (DSI) theo ngành</b>', x=0.5, xanchor='center', font=dict(size=18, color=TEXT)),
           xaxis=dict(title='Chỉ số DSI (trung bình 3 chiều)',
                      showgrid=True, gridcolor=GRID,
                      range=[0, 100], tickfont=dict(size=12), title_font=dict(size=13, color=SUBTEXT)),
           yaxis=dict(showgrid=False, tickfont=dict(size=12))
    )
    fig.update_layout(margin=dict(l=16, r=16, t=70, b=16))
    return fig


def make_dsi_stacked_filtered(df):
    """Stacked bar: Cơ cấu lượt bán nội/ngoại theo ngành (filtered)"""
    if len(df) == 0:
        return go.Figure().update_layout(
            title='Không có dữ liệu',
            paper_bgcolor=SURFACE,
            plot_bgcolor=SURFACE,
            font=dict(color=TEXT),
        )
    
    grp = df.groupby('product_type').agg(
        total_sold=('sold_count', 'sum'),
        total_rev=('estimated_revenue', 'sum'),
        total_products=('product_id', 'count'),
    ).reset_index()
    dom = (df[df['origin_class_corrected'] == 'Trong nước']
           .groupby('product_type').agg(
               dom_sold=('sold_count', 'sum'),
               dom_rev=('estimated_revenue', 'sum'),
               dom_products=('product_id', 'count'),
           ).reset_index())
    dsi = grp.merge(dom, on='product_type', how='left').fillna(0)
    dsi = dsi.sort_values('total_sold', ascending=True)
    
    dom_vals  = dsi['dom_sold'].values
    total_vals = dsi['total_sold'].values
    imp_vals  = total_vals - dom_vals
    types     = dsi['product_type'].tolist()

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name='Trong nước', y=types, x=dom_vals / 1e3,
        orientation='h',
        marker=dict(color=C_DOMESTIC, line=dict(color='white', width=0.6)),
        hovertemplate='<b>%{y}</b><br>Trong nước: %{x:.1f}K lượt<extra></extra>',
    ))
    fig.add_trace(go.Bar(
        name='Ngoài nước', y=types, x=imp_vals / 1e3,
        orientation='h',
        marker=dict(color=C_IMPORT, line=dict(color='white', width=0.6)),
        hovertemplate='<b>%{y}</b><br>Ngoài nước: %{x:.1f}K lượt<extra></extra>',
    ))

    # Pct annotations
    for i, (d, tot, tp) in enumerate(zip(dom_vals, total_vals, types)):
        pct = d / tot * 100 if tot > 0 else 0
        fig.add_annotation(
            x=tot / 1e3 * 1.04, y=tp,
            text=f'Nội {pct:.1f}%',
            showarrow=False,
            xanchor='left',
            font=dict(size=13, color=TEXT, weight=600),
            bgcolor='rgba(17,24,39,0.88)',
            bordercolor=BORDER,
            borderwidth=1,
            borderpad=3,
        )

    _theme(fig, height=300,
           barmode='stack',
           title=dict(text='<b>Cơ cấu lượt bán nội/ngoại theo ngành</b>', x=0.5, xanchor='center', font=dict(size=18, color=TEXT)),
           xaxis=dict(title='Lượt bán (nghìn)', showgrid=True,
                      gridcolor=GRID, tickfont=dict(size=12), title_font=dict(size=13, color=SUBTEXT)),
           yaxis=dict(showgrid=False, tickfont=dict(size=12)),
           legend=dict(orientation='h', yanchor='bottom', y=1.02,
                       xanchor='right', x=1, font=dict(size=12))
    )
    fig.update_layout(margin=dict(l=16, r=16, t=70, b=16))
    return fig