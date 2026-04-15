# pages/page1_thi_phan.py
# ╔══════════════════════════════════════════════════════════════╗
# ║  Trang 1 — Thị Phần: Phân tích 3 Mục tiêu                   ║
# ║  MT1: Thị phần doanh thu nội vs ngoại                        ║
# ║  MT2: Cơ cấu lượt bán & doanh thu theo ngành hàng            ║
# ║  MT3: Sức mạnh nội địa (DSI) theo Product Type               ║
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
TIER_COLORS   = ['#4C1D95', '#6D28D9', '#A78BFA', '#DDD6FE', '#EDE9FE']
TYPE_COLORS   = PRODUCT_TYPE_COLORS

# ── Design tokens ────────────────────────────────────────────
V_500   = '#7C3AED'
V_700   = '#4C1D95'
V_300   = '#A78BFA'
V_50    = '#F5F3FF'
GOLD    = '#F59E0B'
GREEN   = '#16A34A'
TEXT    = '#0F172A'
SUBTEXT = '#64748B'
SURFACE = '#FFFFFF'
BORDER  = '#E4E4F0'


# ══════════════════════════════════════════════════════════════
#  HELPER COMPONENTS
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


def kpi_card(icon, value, label, color, bg):
    return html.Div([
        html.Div(icon, className='p1-kpi-dot',
                 style={'background': bg}),
        html.Div([
            html.P(value, className='p1-kpi-val', style={'color': color}),
            html.P(label, className='p1-kpi-label'),
        ]),
    ], className='p1-kpi')


# ══════════════════════════════════════════════════════════════
#  CHART BUILDERS (called once, not callbacks — static data)
# ══════════════════════════════════════════════════════════════

def _theme(fig, height=320, **kw):
    margin = kw.pop('margin', dict(l=16, r=16, t=30, b=16))
    fig.update_layout(
        height=height,
        font=dict(family="'DM Sans','Segoe UI',sans-serif", size=12, color=TEXT),
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        margin=margin,
        hoverlabel=dict(bgcolor='white', bordercolor=BORDER,
                        font=dict(size=12, color=TEXT)),
        **kw,
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
        textfont=dict(size=14, color='white', weight=700),
        hovertemplate='<b>%{label}</b><br>%{value:,.0f} VNĐ<br><b>%{percent}</b><extra></extra>',
        pull=[0.05 if l == 'Ngoài nước' else 0 for l in labels],
        sort=False,
        direction='clockwise',
    ))
    fig.add_annotation(
        text=f'<b>{pct_import:.1f}%</b><br><span style="font-size:10px;color:{SUBTEXT}">Ngoại nhập</span>',
        x=0.5, y=0.5, showarrow=False,
        font=dict(size=18, color=C_IMPORT),
        xref='paper', yref='paper',
    )
    _theme(fig, height=300,
           showlegend=True,
           legend=dict(orientation='v', yanchor='middle', y=0.5,
                       xanchor='left', x=1.02,
                       font=dict(size=12)))
    fig.update_layout(margin=dict(l=16, r=80, t=20, b=16))
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
        textfont=dict(size=10, color=TEXT),
        hovertemplate='<b>%{y}</b><br>%{x:.1f} tỉ VNĐ<extra></extra>',
        cliponaxis=False,
    ))
    _theme(fig, height=330,
           showlegend=False,
           xaxis=dict(title='Doanh thu (tỉ VNĐ)', showgrid=True,
                      gridcolor='#F1F0FA', tickfont=dict(size=10)),
           yaxis=dict(showgrid=False, tickfont=dict(size=11)))
    fig.update_layout(margin=dict(l=16, r=70, t=20, b=30))
    return fig


def make_bar_price():
    grp = (df_full
           .groupby(['price_segment', 'origin_class_corrected'], observed=True)['estimated_revenue']
           .sum().unstack(fill_value=0))
    segs    = [s for s in PRICE_ORDER if s in grp.index]
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
           xaxis=dict(showgrid=False, tickfont=dict(size=10), tickangle=-10),
           yaxis=dict(title='Doanh thu (tỉ VNĐ)', showgrid=True,
                      gridcolor='#F1F0FA', tickfont=dict(size=10)),
           legend=dict(orientation='h', yanchor='bottom', y=1.02,
                       xanchor='right', x=1, font=dict(size=11))
    )
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
        textfont=dict(size=10, weight=700, color=TEXT),
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
        textfont=dict(size=10, color=TEXT),
    ))
    # Delta annotations
    for i, (sold, rev, t) in enumerate(zip(sold_vals, rev_vals, types)):
        delta = rev - sold
        if abs(delta) > 1:
            color = C_IMPORT if delta > 0 else GREEN
            fig.add_annotation(
                x=t,
                y=max(sold, rev) + 6,
                text=f'Δ {delta:+.1f}%',
                showarrow=False,
                font=dict(size=9, color=color, weight=700),
                bgcolor='rgba(255,255,255,0.85)',
                bordercolor=color,
                borderwidth=1,
                borderpad=3,
            )

    _theme(fig, height=340,
           barmode='group',
           xaxis=dict(showgrid=False, tickfont=dict(size=11, weight=700)),
           yaxis=dict(title='Tỉ trọng (%)', showgrid=True,
                      gridcolor='#F1F0FA', tickfont=dict(size=10),
                      range=[0, max(max(sold_vals), max(rev_vals)) * 1.35]),
           legend=dict(orientation='h', yanchor='bottom', y=1.02,
                       xanchor='right', x=1, font=dict(size=11))
    )
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
        textfont=dict(size=10, color=TEXT),
        hovertemplate='<b>%{customdata}</b><br>%{y}<br>%{x:.1f} tỉ VNĐ<extra></extra>',
        customdata=customdata,
        cliponaxis=False,
    ))
    _theme(fig, height=360,
           showlegend=False,
           xaxis=dict(title='Doanh thu (tỉ VNĐ)', showgrid=True,
                      gridcolor='#F1F0FA', tickfont=dict(size=10)),
           yaxis=dict(showgrid=False, tickvals=y_pos, ticktext=y_label,
                      tickfont=dict(size=10.5))
    )
    return fig


def make_popularity_tier():
    """Stacked bar: popularity tier theo ngành."""
    type_order = (df_full.groupby('product_type')['estimated_revenue']
                  .sum().sort_values(ascending=False).index.tolist())

    avail_tiers = [t for t in TIER_ORDER if t in df_full['popularity_tier'].unique()]
    pop_pct = (pd.crosstab(df_full['product_type'], df_full['popularity_tier'], normalize='index') * 100)
    pop_pct = pop_pct.reindex(columns=avail_tiers, fill_value=0).loc[type_order[::-1]]

    fig = go.Figure()
    tier_clrs = ['#4C1D95', '#7C3AED', '#A78BFA', '#DDD6FE', '#EDE9FE']
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
            textfont=dict(size=10, color='white'),
            insidetextanchor='middle',
        ))

    _theme(fig, height=300,
           barmode='stack',
           xaxis=dict(showgrid=False, tickfont=dict(size=10),
                      range=[0, 100]),
           yaxis=dict(showgrid=False, tickfont=dict(size=11)),
           legend=dict(orientation='h', yanchor='top', y=-0.15,
                       xanchor='center', x=0.5, font=dict(size=10))
    )
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
            marker=dict(size=6, color=color),
            hovertemplate=f'<b>%{{theta}}</b><br>{name}: %{{r:.1f}}%<extra></extra>',
        ))

    fig.update_layout(
        height=350,
        polar=dict(
            bgcolor='#FAFAFF',
            angularaxis=dict(
                tickfont=dict(size=11, weight=700, color=TEXT),
                linecolor=BORDER,
            ),
            radialaxis=dict(
                visible=True, range=[0, 100],
                tickfont=dict(size=9, color=SUBTEXT),
                gridcolor='#E8E6F5',
                tickvals=[20, 40, 60, 80, 100],
                ticktext=['20%', '40%', '60%', '80%', '100%'],
            ),
        ),
        paper_bgcolor=SURFACE,
        font=dict(family="'DM Sans','Segoe UI',sans-serif", size=12, color=TEXT),
        legend=dict(orientation='h', yanchor='bottom', y=-0.18,
                    xanchor='center', x=0.5, font=dict(size=11)),
        margin=dict(l=40, r=40, t=30, b=50),
        hoverlabel=dict(bgcolor='white', bordercolor=BORDER,
                        font=dict(size=12, color=TEXT))
    )
    return fig


def make_dsi_stacked():
    dsi = compute_dsi().sort_values('DSI', ascending=True)
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
            font=dict(size=10, color=TEXT, weight=600),
            bgcolor='rgba(255,255,255,0.85)',
            bordercolor=BORDER,
            borderwidth=1,
            borderpad=3,
        )

    _theme(fig, height=300,
           barmode='stack',
           xaxis=dict(title='Lượt bán (nghìn)', showgrid=True,
                      gridcolor='#F1F0FA', tickfont=dict(size=10)),
           yaxis=dict(showgrid=False, tickfont=dict(size=11)),
           legend=dict(orientation='h', yanchor='bottom', y=1.02,
                       xanchor='right', x=1, font=dict(size=11))
    )
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
        textfont=dict(size=11, weight=700, color=TEXT),
        hovertemplate='<b>%{y}</b><br>DSI: %{x:.1f}<extra></extra>',
        cliponaxis=False,
    ))
    # Benchmark line at 33.3 (neutral)
    fig.add_vline(x=33.3, line_dash='dash',
                  line_color='#94A3B8', line_width=1.5)
    fig.add_annotation(x=33.3, y=len(types) - 0.5,
                       text='Cân bằng (33%)', showarrow=False,
                       font=dict(size=9, color=SUBTEXT),
                       xanchor='center', yanchor='bottom')

    _theme(fig, height=300,
           showlegend=False,
           xaxis=dict(title='Chỉ số DSI (trung bình 3 chiều)',
                      showgrid=True, gridcolor='#F1F0FA',
                      range=[0, 100], tickfont=dict(size=10)),
           yaxis=dict(showgrid=False, tickfont=dict(size=11))
    )
    return fig


# ══════════════════════════════════════════════════════════════
#  LAYOUT
# ══════════════════════════════════════════════════════════════

def layout():
    dsi_df   = compute_dsi()
    best_type = dsi_df.iloc[0]['product_type']
    best_dsi  = dsi_df.iloc[0]['DSI']
    worst_dsi = dsi_df.iloc[-1]['DSI']

    # Top KPIs
    total_rev   = df_full['estimated_revenue'].sum()
    imp_rev     = df_nn_full['estimated_revenue'].sum()
    imp_pct_rev = imp_rev / total_rev * 100
    top_country = (df_full.groupby('origin_corrected')['estimated_revenue']
                   .sum().sort_values(ascending=False).index[0])
    n_types     = df_full['product_type'].nunique()

    return html.Div([

        # ── HERO ────────────────────────────────────────────
        html.Div([
            html.Div([
                html.Span('🔍', style={'fontSize': '14px'}),
                html.Span('PHÂN TÍCH THỊ PHẦN', style={'fontSize': '11px'}),
            ], className='p1-hero-eyebrow'),

            html.H1('Thị Phần Mỹ Phẩm Tiki · T3/2026',
                    className='p1-hero-title'),

            html.P('Phân tích 3 mục tiêu · Thị phần doanh thu · Cơ cấu ngành hàng · Sức mạnh nội địa (DSI)',
                   className='p1-hero-sub'),

            html.Div([
                html.Span('📊 7,179 sản phẩm', className='p1-badge'),
                html.Span(f'🌍 Ngoại nhập: {imp_pct_rev:.1f}%', className='p1-badge highlight'),
                html.Span(f'🥇 {top_country} dẫn đầu nhập khẩu', className='p1-badge'),
                html.Span(f'🏪 {n_types} ngành hàng', className='p1-badge'),
                html.Span('📅 Nhóm 05 · FIT-HCMUS', className='p1-badge'),
            ], className='p1-hero-badges'),
        ], className='p1-hero'),

        # ── KPI Row ─────────────────────────────────────────
        html.Div([
            kpi_card('📦', f"{len(df_full):,}", 'Tổng sản phẩm', '#2563EB', '#EFF6FF'),
            kpi_card('💰', f"{total_rev/1e9:.1f} tỉ", 'Doanh thu ước tính', '#7C3AED', '#EDE9FE'),
            kpi_card('🌍', f"{imp_pct_rev:.1f}%", 'Thị phần ngoại nhập', '#DC2626', '#FEE2E2'),
            kpi_card('🏆', top_country, 'Nước dẫn đầu NK', '#DC2626', '#FEE2E2'),
            kpi_card('🇻🇳', f"{best_dsi:.1f} DSI", f'Mạnh nhất: {best_type}', '#16A34A', '#DCFCE7'),
        ], className='p1-kpi-row'),

        # ═══════════════════════════════════════════════════
        #  MỤC TIÊU 1
        # ═══════════════════════════════════════════════════
        section_header(
            1,
            'Xác định thị phần doanh thu giữa hàng nội và hàng ngoại',
            'Người tiêu dùng đang chi tiêu cho ai? Nội địa hay quốc tế? '
            'Phân tích theo tổng thị phần, quốc gia xuất xứ và phân khúc giá.',
            'muc1',
        ),

        # Row 1-A: Donut + Country bar
        html.Div([
            chart_card('🍩', 'Tổng thị phần doanh thu',
                       'Nội địa vs Ngoại nhập · Toàn danh mục',
                       dcc.Graph(figure=make_donut(),
                                 config={'displayModeBar': False}),
                       flex='1', min_width='280px', icon_bg='#EDE9FE'),

            chart_card('🌐', 'Doanh thu theo quốc gia xuất xứ',
                       'Top 8 quốc gia + Khác · Nhật Bản dẫn đầu',
                       dcc.Graph(figure=make_bar_country(),
                                 config={'displayModeBar': False}),
                       flex='1.3', min_width='340px', icon_bg='#FEE2E2'),
        ], className='p1-row'),

        # Row 1-B: Price segment
        html.Div([
            chart_card('💵', 'Doanh thu theo phân khúc giá',
                       'So sánh nội / ngoại theo từng khoảng giá',
                       dcc.Graph(figure=make_bar_price(),
                                 config={'displayModeBar': False}),
                       flex='1', min_width='340px', icon_bg='#FEF3C7'),

            html.Div([
                html.H4('📝 Kết luận Mục tiêu 1', style={
                    'margin': '0 0 14px 0', 'fontSize': '14px',
                    'fontWeight': '700', 'color': V_700,
                }),
                insight_box(
                    html.Span([
                        html.Strong('Hàng ngoại chiếm ~89.8% doanh thu.'),
                        ' Dù hàng nội có số lượng sản phẩm đáng kể (~26.7%), '
                        'doanh thu ước tính chỉ ~10.2% — phản ánh giá trị đơn hàng '
                        'và lượt bán của hàng ngoại vượt trội hơn đáng kể.',
                    ]),
                    'violet',
                ),
                html.Div(style={'height': '10px'}),
                insight_box(
                    html.Span([
                        html.Strong('Nhật Bản dẫn đầu tuyệt đối'),
                        ' với ~190 tỉ VNĐ, gần gấp đôi vị trí thứ 2 (Đức). '
                        'Các thương hiệu Nhật (Hada Labo, Senka, Sunplay) '
                        'có sức hút rất mạnh tại thị trường Việt Nam.',
                    ]),
                    'violet',
                ),
                html.Div(style={'height': '10px'}),
                insight_box(
                    html.Span([
                        html.Strong('Phân khúc dưới 100k và 100k–300k'),
                        ' là hai phân khúc tập trung doanh thu cao nhất '
                        'cho cả hai nhóm, phản ánh hành vi tiêu dùng mỹ phẩm '
                        'đại trà tại Việt Nam. Hàng ngoại áp đảo ngay '
                        'ở phân khúc giá rẻ — điều khá bất ngờ.',
                    ]),
                    'gold',
                ),
            ], style={
                'flex': '0.7', 'minWidth': '280px',
                'background': SURFACE,
                'border': f'1px solid {BORDER}',
                'borderLeft': f'3px solid {V_500}',
                'borderRadius': '14px',
                'padding': '22px',
                'boxShadow': '0 1px 6px rgba(109,40,217,.07)',
            }),
        ], className='p1-row'),

        html.Div(className='p1-divider'),

        # ═══════════════════════════════════════════════════
        #  MỤC TIÊU 2
        # ═══════════════════════════════════════════════════
        section_header(
            2,
            'Cơ cấu lượt bán & doanh thu theo từng ngành hàng',
            'Ngành hàng nào đang dẫn dắt tăng trưởng? '
            'Phân tích tỉ trọng lượt bán vs doanh thu, danh mục đầu tàu và mức độ chín của thị trường.',
            'muc2',
        ),

        # Row 2-A: Dual bar (full width)
        html.Div([
            chart_card('⚖️', 'Tương quan Lượt bán & Doanh thu theo ngành hàng',
                       'Cột đậm = Lượt bán · Cột mờ = Doanh thu · Δ = chênh lệch',
                       dcc.Graph(figure=make_dual_bar(),
                                 config={'displayModeBar': False}),
                       flex='1', min_width='400px', icon_bg='#FEF3C7'),
        ], className='p1-row'),

        # Row 2-B: Top cats + Tier
        html.Div([
            chart_card('🏷️', 'Top 3 danh mục "kiếm tiền" giỏi nhất mỗi ngành',
                       'Màu = ngành hàng · Doanh thu ước tính (tỉ VNĐ)',
                       dcc.Graph(figure=make_top_categories(),
                                 config={'displayModeBar': False}),
                       flex='1', min_width='340px', icon_bg='#FEF3C7'),

            chart_card('📈', 'Mức độ "chín" của thị trường theo ngành hàng',
                       'Bestseller → Chưa bán · Tỉ lệ % trong mỗi ngành',
                       dcc.Graph(figure=make_popularity_tier(),
                                 config={'displayModeBar': False}),
                       flex='1', min_width='320px', icon_bg='#EDE9FE'),
        ], className='p1-row'),

        # Insight box MT2
        html.Div([
            insight_box(
                html.Span([
                    html.Strong('Skincare dẫn đầu trên cả 3 chiều'),
                    ' (sản phẩm, lượt bán, doanh thu) — là "động cơ chính" của thị trường. '
                    'Fragrance có tỉ lệ doanh thu/lượt bán cao nhất vì giá đơn hàng trung bình rất cao.',
                ]),
                'gold',
            ),
            html.Div(style={'height': '8px'}),
            insight_box(
                html.Span([
                    html.Strong('Body Care & Hair Care'),
                    ' có lượt bán lớn nhưng giá trị đơn hàng thấp hơn — phản ánh '
                    'thị trường tiêu dùng đại trà. Body Care có tỉ lệ Bestseller cao nhất, '
                    'cho thấy thị trường đã định hình rõ ràng.',
                ]),
                'gold',
            ),
        ], style={
            'background': '#FFFBEB',
            'border': f'1px solid #FDE68A',
            'borderLeft': f'3px solid {GOLD}',
            'borderRadius': '14px',
            'padding': '18px 22px',
            'marginBottom': '18px',
        }),

        html.Div(className='p1-divider'),

        # ═══════════════════════════════════════════════════
        #  MỤC TIÊU 3
        # ═══════════════════════════════════════════════════
        section_header(
            3,
            'Chỉ số Sức mạnh nội địa (DSI) theo từng ngành hàng',
            'Thương hiệu Việt thực sự mạnh ở đâu? '
            'DSI = trung bình 3 chiều: tỉ lệ lượt bán · doanh thu · số sản phẩm của hàng nội.',
            'muc3',
        ),

        # DSI legend
        html.Div([
            html.Span('Đọc chỉ số DSI: ', style={
                'fontSize': '12px', 'fontWeight': '600', 'color': SUBTEXT,
            }),
            html.Span('≥50 = Nội địa ưu thế', style={
                'padding': '3px 10px', 'borderRadius': '20px',
                'background': '#D1FAE5', 'color': '#065F46',
                'fontSize': '11px', 'fontWeight': '700',
                'marginLeft': '8px',
            }),
            html.Span('25–50 = Cân bằng / tranh giành', style={
                'padding': '3px 10px', 'borderRadius': '20px',
                'background': '#FEF3C7', 'color': '#78350F',
                'fontSize': '11px', 'fontWeight': '700',
                'marginLeft': '6px',
            }),
            html.Span('<25 = Ngoại nhập áp đảo', style={
                'padding': '3px 10px', 'borderRadius': '20px',
                'background': '#FEE2E2', 'color': '#7F1D1D',
                'fontSize': '11px', 'fontWeight': '700',
                'marginLeft': '6px',
            }),
        ], style={
            'display': 'flex', 'alignItems': 'center',
            'flexWrap': 'wrap', 'gap': '4px',
            'marginBottom': '16px',
            'padding': '12px 18px',
            'background': SURFACE,
            'border': f'1px solid {BORDER}',
            'borderRadius': '10px',
        }),

        # Row 3-A: Radar + DSI bar
        html.Div([
            chart_card('🕸️', 'Radar: 3 chiều DSI theo ngành hàng',
                       'Lượt bán · Doanh thu · Sản phẩm nội địa (%) · Càng rộng = càng mạnh',
                       dcc.Graph(figure=make_radar(),
                                 config={'displayModeBar': False}),
                       flex='1', min_width='310px', icon_bg='#DCFCE7'),

            chart_card('📊', 'Chỉ số DSI theo ngành hàng',
                       '🟢 ≥50 Ưu thế · 🟡 25–50 Cạnh tranh · 🔴 <25 Yếu',
                       dcc.Graph(figure=make_dsi_bar(),
                                 config={'displayModeBar': False}),
                       flex='1', min_width='310px', icon_bg='#DCFCE7'),
        ], className='p1-row'),

        # Row 3-B: Stacked sold
        html.Div([
            chart_card('📦', 'Cơ cấu lượt bán nội / ngoại theo ngành hàng',
                       'Sắp xếp theo DSI tăng dần · Nhãn = % hàng nội',
                       dcc.Graph(figure=make_dsi_stacked(),
                                 config={'displayModeBar': False}),
                       flex='1', min_width='400px', icon_bg='#DCFCE7'),

            # Kết luận MT3
            html.Div([
                html.H4('🇻🇳 Kết luận Mục tiêu 3', style={
                    'margin': '0 0 14px 0', 'fontSize': '14px',
                    'fontWeight': '700', 'color': GREEN,
                }),
                insight_box(
                    html.Span([
                        html.Strong('Makeup — ngành duy nhất thương hiệu Việt chiếm ưu thế'),
                        f' với DSI ~{dsi_df[dsi_df["product_type"]=="Makeup"]["DSI"].values[0]:.1f} '
                        'và ~87.8% thị phần lượt bán. '
                        'Cocoon, Emmié by Happy Skin đang hoạt động cực kỳ hiệu quả.',
                    ]),
                    'green',
                ),
                html.Div(style={'height': '10px'}),
                insight_box(
                    html.Span([
                        html.Strong('Skincare là "đại dương" quan trọng nhất'),
                        ' (~48.7% doanh thu toàn thị trường) nhưng hàng Việt chỉ chiếm '
                        '~11% lượt bán. Để tăng tổng thị phần, thương hiệu Việt cần '
                        'thâm nhập vào chính ngành này.',
                    ]),
                    'green',
                ),
                html.Div(style={'height': '10px'}),
                insight_box(
                    html.Span([
                        html.Strong('Body Care & Hair Care'),
                        ' bị thống lĩnh bởi một vài thương hiệu ngoại '
                        '(Selsun, Romano, Dove, Clear) tạo rào cản cực kỳ cao. '
                        'Fragrance có DSI cao nhờ đặc thù cấu trúc ngành hơn '
                        'là năng lực cạnh tranh thực sự.',
                    ]),
                    'violet',
                ),
            ], style={
                'flex': '0.7', 'minWidth': '280px',
                'background': SURFACE,
                'border': f'1px solid {BORDER}',
                'borderLeft': f'3px solid {GREEN}',
                'borderRadius': '14px',
                'padding': '22px',
                'boxShadow': '0 1px 6px rgba(22,163,74,.07)',
            }),
        ], className='p1-row'),

        # ── Footer ──────────────────────────────────────────
        html.Div([
            html.Span('⚠️ Dữ liệu crawl từ Tiki tháng 3/2026'),
            html.Span('DSI = (sold_share + rev_share + product_share) / 3'),
            html.Span('Doanh thu = sold_count × price (ước tính)'),
            html.Span('Nhóm 05 · FIT-HCMUS'),
        ], className='p1-footer'),

    ], className='p1-page', style={'padding': '0'})