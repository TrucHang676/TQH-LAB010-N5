# pages/page1_thi_phan.py  — Dark theme · Bold · Minimal text
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

df_full, df_vn_full, df_nn_full = load_data()

PRICE_ORDER = ['Dưới 100k', '100k – 300k', '300k – 700k', '700k – 2tr', 'Trên 2tr']
TIER_ORDER  = ['Bestseller', 'Bán chạy', 'Phổ biến', 'Mới', 'Chưa có lượt bán']
TYPE_COLORS = PRODUCT_TYPE_COLORS

# ── Dark theme palette ────────────────────────────────────────
D_BG        = '#0D1117'   # nền chính — dark navy
D_CARD      = '#161B22'   # card bg
D_CARD2     = '#1C2333'   # card alt
D_BORDER    = '#30363D'   # border
D_TEXT      = '#E6EDF3'   # text chính
D_SUBTEXT   = '#8B949E'   # text phụ
D_ACCENT    = '#7C3AED'   # tím — accent chính
D_GOLD      = '#F59E0B'
D_GREEN     = '#22C55E'
D_RED       = '#EF4444'
D_BLUE      = '#3B82F6'
D_PURPLE_LT = '#A78BFA'

CHART_BG    = '#161B22'
GRID_COLOR  = '#21262D'


# ══════════════════════════════════════════════════════════════
#  CHART THEME HELPER
# ══════════════════════════════════════════════════════════════

def _dark_theme(fig, height=320, **kw):
    margin = kw.pop('margin', dict(l=16, r=16, t=28, b=16))
    xaxis  = kw.pop('xaxis', {})
    yaxis  = kw.pop('yaxis', {})
    legend = kw.pop('legend', {})

    base_axis = dict(
        gridcolor=GRID_COLOR, gridwidth=1,
        linecolor=D_BORDER,
        tickfont=dict(size=11, color=D_SUBTEXT),
        title=dict(font=dict(size=11, color=D_SUBTEXT)),
        showgrid=True,
        zeroline=False,
    )

    fig.update_layout(
        height=height,
        font=dict(family="'DM Sans','Segoe UI',sans-serif", size=12, color=D_TEXT),
        paper_bgcolor=CHART_BG,
        plot_bgcolor=CHART_BG,
        margin=margin,
        hoverlabel=dict(
            bgcolor='#21262D', bordercolor=D_BORDER,
            font=dict(size=12, color=D_TEXT),
        ),
        legend=dict(
            bgcolor='rgba(0,0,0,0)', bordercolor=D_BORDER, borderwidth=0,
            font=dict(size=11, color=D_SUBTEXT),
            orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1,
            **legend,
        ),
        xaxis={**base_axis, **xaxis},
        yaxis={**base_axis, **yaxis},
        **kw,
    )
    return fig


# ══════════════════════════════════════════════════════════════
#  CHART BUILDERS
# ══════════════════════════════════════════════════════════════

def make_donut():
    rev    = df_full.groupby('origin_class_corrected')['estimated_revenue'].sum()
    labels = rev.index.tolist()
    values = rev.values.tolist()
    total  = sum(values)
    pct_import = rev.get('Ngoài nước', 0) / total * 100

    colors = ['#3B82F6' if l == 'Trong nước' else '#EF4444' for l in labels]

    fig = go.Figure(go.Pie(
        values=values, labels=labels,
        marker=dict(colors=colors, line=dict(color=D_BG, width=3)),
        hole=0.66,
        textinfo='percent',
        textfont=dict(size=13, color='white'),
        hovertemplate='<b>%{label}</b><br>%{value:,.0f} VNĐ · <b>%{percent}</b><extra></extra>',
        pull=[0.05 if l == 'Ngoài nước' else 0 for l in labels],
        sort=False,
        direction='clockwise',
    ))
    fig.add_annotation(
        text=f'<b style="font-size:20px;color:#EF4444">{pct_import:.1f}%</b>'
             f'<br><span style="font-size:10px;color:{D_SUBTEXT}">Ngoại nhập</span>',
        x=0.5, y=0.5, showarrow=False,
        font=dict(size=14, color=D_TEXT),
        xref='paper', yref='paper',
    )
    _dark_theme(fig, height=290,
                showlegend=True,
                legend=dict(orientation='v', yanchor='middle', y=0.5,
                            xanchor='left', x=1.02, font=dict(size=12, color=D_SUBTEXT)))
    fig.update_layout(margin=dict(l=16, r=90, t=20, b=16))
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
        if c == 'Việt Nam':   bar_colors.append('#3B82F6')
        elif c == 'Nhật Bản': bar_colors.append('#EF4444')
        elif c == 'Khác':     bar_colors.append('#374151')
        else:                 bar_colors.append('#F87171')

    fig = go.Figure(go.Bar(
        x=values, y=countries, orientation='h',
        marker=dict(color=bar_colors, line=dict(color=D_BG, width=0.8)),
        text=[f'{v:.1f} tỉ' for v in values],
        textposition='outside',
        textfont=dict(size=10, color=D_SUBTEXT),
        hovertemplate='<b>%{y}</b><br>%{x:.1f} tỉ VNĐ<extra></extra>',
        cliponaxis=False,
    ))
    _dark_theme(fig, height=320,
                showlegend=False,
                xaxis=dict(title='Doanh thu (tỉ VNĐ)', showgrid=True,
                           gridcolor=GRID_COLOR, tickfont=dict(size=10, color=D_SUBTEXT)),
                yaxis=dict(showgrid=False, tickfont=dict(size=11, color=D_TEXT)))
    fig.update_layout(margin=dict(l=16, r=70, t=20, b=20))
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
        marker_color='#3B82F6', marker_line=dict(color=D_BG, width=0.8),
        hovertemplate='<b>%{x}</b><br>Trong nước: %{y:.1f} tỉ<extra></extra>',
    ))
    fig.add_trace(go.Bar(
        name='Ngoài nước', x=segs, y=imp_rev,
        marker_color='#EF4444', marker_line=dict(color=D_BG, width=0.8),
        hovertemplate='<b>%{x}</b><br>Ngoài nước: %{y:.1f} tỉ<extra></extra>',
    ))
    _dark_theme(fig, height=290,
                barmode='group',
                xaxis=dict(showgrid=False, tickfont=dict(size=10, color=D_SUBTEXT), tickangle=-10),
                yaxis=dict(title='Doanh thu (tỉ VNĐ)', showgrid=True,
                           gridcolor=GRID_COLOR, tickfont=dict(size=10, color=D_SUBTEXT)),
                legend=dict(orientation='h', yanchor='bottom', y=1.02,
                            xanchor='right', x=1, font=dict(size=11, color=D_SUBTEXT)))
    return fig


def make_dual_bar():
    grp = (df_full.groupby('product_type')
           .agg(total_sold=('sold_count', 'sum'),
                total_rev=('estimated_revenue', 'sum'))
           .reset_index())
    grp['sold_share'] = grp['total_sold'] / grp['total_sold'].sum() * 100
    grp['rev_share']  = grp['total_rev']  / grp['total_rev'].sum()  * 100

    types     = grp['product_type'].tolist()
    sold_vals = grp['sold_share'].tolist()
    rev_vals  = grp['rev_share'].tolist()
    bar_col   = [TYPE_COLORS.get(t, '#94A3B8') for t in types]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name='Tỉ trọng Lượt bán (%)',
        x=types, y=sold_vals,
        marker_color=bar_col, marker_line=dict(color=D_BG, width=1),
        opacity=1.0,
        hovertemplate='<b>%{x}</b><br>Lượt bán: %{y:.1f}%<extra></extra>',
        text=[f'{v:.1f}%' for v in sold_vals],
        textposition='outside',
        textfont=dict(size=10, color=D_TEXT),
    ))
    fig.add_trace(go.Bar(
        name='Tỉ trọng Doanh thu (%)',
        x=types, y=rev_vals,
        marker_color=bar_col, marker_line=dict(color=D_BG, width=1),
        opacity=0.4,
        hovertemplate='<b>%{x}</b><br>Doanh thu: %{y:.1f}%<extra></extra>',
        text=[f'{v:.1f}%' for v in rev_vals],
        textposition='outside',
        textfont=dict(size=10, color=D_SUBTEXT),
    ))
    for i, (sold, rev, t) in enumerate(zip(sold_vals, rev_vals, types)):
        delta = rev - sold
        if abs(delta) > 1:
            color = D_RED if delta > 0 else D_GREEN
            fig.add_annotation(
                x=t, y=max(sold, rev) + 6,
                text=f'Δ {delta:+.1f}%',
                showarrow=False,
                font=dict(size=9, color=color),
                bgcolor='rgba(0,0,0,0.6)',
                bordercolor=color, borderwidth=1, borderpad=3,
            )
    _dark_theme(fig, height=330,
                barmode='group',
                xaxis=dict(showgrid=False, tickfont=dict(size=11, color=D_TEXT)),
                yaxis=dict(title='Tỉ trọng (%)', showgrid=True, gridcolor=GRID_COLOR,
                           tickfont=dict(size=10, color=D_SUBTEXT),
                           range=[0, max(max(sold_vals), max(rev_vals)) * 1.35]),
                legend=dict(orientation='h', yanchor='bottom', y=1.02,
                            xanchor='right', x=1, font=dict(size=11, color=D_SUBTEXT)))
    return fig


def make_top_categories():
    top_cats = (df_full.groupby(['product_type', 'category'])
                .agg(cat_rev=('estimated_revenue', 'sum'))
                .reset_index()
                .sort_values(['product_type', 'cat_rev'], ascending=[True, False])
                .groupby('product_type').head(3))

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
        marker=dict(color=colors, line=dict(color=D_BG, width=0.6)),
        text=[f'{v:.0f} tỉ' for v in x_vals],
        textposition='outside',
        textfont=dict(size=10, color=D_SUBTEXT),
        hovertemplate='<b>%{customdata}</b><br>%{y}<br>%{x:.1f} tỉ VNĐ<extra></extra>',
        customdata=customdata,
        cliponaxis=False,
    ))
    _dark_theme(fig, height=360,
                showlegend=False,
                xaxis=dict(title='Doanh thu (tỉ VNĐ)', showgrid=True,
                           gridcolor=GRID_COLOR, tickfont=dict(size=10, color=D_SUBTEXT)),
                yaxis=dict(showgrid=False, tickvals=y_pos, ticktext=y_label,
                           tickfont=dict(size=10.5, color=D_TEXT)))
    return fig


def make_popularity_tier():
    type_order = (df_full.groupby('product_type')['estimated_revenue']
                  .sum().sort_values(ascending=False).index.tolist())
    avail_tiers = [t for t in TIER_ORDER if t in df_full['popularity_tier'].unique()]
    pop_pct = (pd.crosstab(df_full['product_type'], df_full['popularity_tier'], normalize='index') * 100)
    pop_pct = pop_pct.reindex(columns=avail_tiers, fill_value=0).loc[type_order[::-1]]

    tier_clrs = ['#7C3AED', '#8B5CF6', '#A78BFA', '#C4B5FD', '#2D2040']
    fig = go.Figure()
    for tier, color in zip(avail_tiers, tier_clrs):
        if tier not in pop_pct.columns:
            continue
        vals = pop_pct[tier].values
        fig.add_trace(go.Bar(
            name=tier, x=vals, y=pop_pct.index.tolist(), orientation='h',
            marker=dict(color=color, line=dict(color=D_BG, width=0.8)),
            hovertemplate=f'<b>%{{y}}</b><br>{tier}: %{{x:.1f}}%<extra></extra>',
            text=[f'{v:.0f}%' if v > 7 else '' for v in vals],
            textposition='inside', textfont=dict(size=10, color='white'),
            insidetextanchor='middle',
        ))
    _dark_theme(fig, height=290,
                barmode='stack',
                xaxis=dict(showgrid=False, tickfont=dict(size=10, color=D_SUBTEXT), range=[0, 100]),
                yaxis=dict(showgrid=False, tickfont=dict(size=11, color=D_TEXT)),
                legend=dict(orientation='h', yanchor='top', y=-0.18,
                            xanchor='center', x=0.5, font=dict(size=10, color=D_SUBTEXT)))
    return fig


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
    dsi['sold_share']    = dsi['dom_sold']    / dsi['total_sold'].replace(0, 1) * 100
    dsi['rev_share']     = dsi['dom_rev']     / dsi['total_rev'].replace(0, 1)  * 100
    dsi['product_share'] = dsi['dom_products']/ dsi['total_products'].replace(0,1)* 100
    dsi['DSI'] = (dsi['sold_share'] + dsi['rev_share'] + dsi['product_share']) / 3
    return dsi.sort_values('DSI', ascending=False).reset_index(drop=True)


def make_radar():
    dsi_df = compute_dsi()
    types  = dsi_df['product_type'].tolist()
    cats   = ['Lượt bán', 'Doanh thu', 'Sản phẩm', 'Lượt bán']  # đóng vòng
    fig    = go.Figure()
    colors_radar = ['#3B82F6', '#EC4899', '#F97316', '#22C55E', '#8B5CF6', '#94A3B8']

    for i, row in dsi_df.iterrows():
        vals = [row['sold_share'], row['rev_share'], row['product_share'], row['sold_share']]
        fig.add_trace(go.Scatterpolar(
            r=vals, theta=cats, fill='toself',
            name=row['product_type'],
            line=dict(color=colors_radar[i % len(colors_radar)], width=2),
            fillcolor=f'{colors_radar[i % len(colors_radar)]}22',
            hovertemplate=f"<b>{row['product_type']}</b><br>%{{theta}}: %{{r:.1f}}%<extra></extra>",
        ))
    fig.update_layout(
        height=310,
        font=dict(family="'DM Sans','Segoe UI',sans-serif", size=11, color=D_TEXT),
        paper_bgcolor=CHART_BG, plot_bgcolor=CHART_BG,
        polar=dict(
            bgcolor=CHART_BG,
            radialaxis=dict(visible=True, range=[0, 100],
                            gridcolor=GRID_COLOR, tickfont=dict(size=9, color=D_SUBTEXT),
                            linecolor=D_BORDER),
            angularaxis=dict(gridcolor=GRID_COLOR,
                             tickfont=dict(size=11, color=D_TEXT),
                             linecolor=D_BORDER),
        ),
        showlegend=True,
        legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(size=10, color=D_SUBTEXT),
                    orientation='h', yanchor='bottom', y=-0.2, xanchor='center', x=0.5),
        margin=dict(l=40, r=40, t=28, b=60),
        hoverlabel=dict(bgcolor='#21262D', bordercolor=D_BORDER,
                        font=dict(size=12, color=D_TEXT)),
    )
    return fig


def make_dsi_bar():
    dsi = compute_dsi().sort_values('DSI', ascending=True)
    types    = dsi['product_type'].tolist()
    dsi_vals = dsi['DSI'].tolist()
    colors   = [D_GREEN if v >= 50 else D_GOLD if v >= 25 else D_RED for v in dsi_vals]

    fig = go.Figure(go.Bar(
        x=dsi_vals, y=types, orientation='h',
        marker=dict(color=colors, line=dict(color=D_BG, width=0.8)),
        text=[f'{v:.1f}' for v in dsi_vals],
        textposition='outside',
        textfont=dict(size=11, color=D_TEXT),
        hovertemplate='<b>%{y}</b><br>DSI: %{x:.1f}<extra></extra>',
        cliponaxis=False,
    ))
    fig.add_vline(x=33.3, line_dash='dash', line_color=D_SUBTEXT, line_width=1.5)
    fig.add_annotation(x=33.3, y=len(types) - 0.5,
                       text='Cân bằng', showarrow=False,
                       font=dict(size=9, color=D_SUBTEXT),
                       xanchor='center', yanchor='bottom')
    _dark_theme(fig, height=290,
                showlegend=False,
                xaxis=dict(title='Chỉ số DSI', showgrid=True,
                           gridcolor=GRID_COLOR, range=[0, 100],
                           tickfont=dict(size=10, color=D_SUBTEXT)),
                yaxis=dict(showgrid=False, tickfont=dict(size=11, color=D_TEXT)))
    return fig


def make_dsi_stacked():
    dsi_df = compute_dsi()
    types_sorted = dsi_df.sort_values('DSI')['product_type'].tolist()
    grp = df_full.groupby(['product_type', 'origin_class_corrected'])['sold_count'].sum().unstack(fill_value=0)
    dom_vals   = [(grp.loc[t, 'Trong nước'] / 1e3 if 'Trong nước' in grp.columns else 0) for t in types_sorted]
    imp_vals   = [(grp.loc[t, 'Ngoài nước'] / 1e3 if 'Ngoài nước' in grp.columns else 0) for t in types_sorted]
    total_vals = [d + i for d, i in zip(dom_vals, imp_vals)]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name='Trong nước', y=types_sorted, x=dom_vals, orientation='h',
        marker_color='#3B82F6', marker_line=dict(color=D_BG, width=0.6),
        hovertemplate='<b>%{y}</b><br>Trong nước: %{x:.1f}K lượt<extra></extra>',
    ))
    fig.add_trace(go.Bar(
        name='Ngoài nước', y=types_sorted, x=imp_vals, orientation='h',
        marker_color='#EF4444', marker_line=dict(color=D_BG, width=0.6),
        hovertemplate='<b>%{y}</b><br>Ngoài nước: %{x:.1f}K lượt<extra></extra>',
    ))
    for i, (d, tot, tp) in enumerate(zip(dom_vals, total_vals, types_sorted)):
        pct = d / tot * 100 if tot > 0 else 0
        fig.add_annotation(
            x=tot / 1e3 * 1.04 if tot > 1000 else tot * 1.04, y=tp,
            text=f'Nội {pct:.0f}%',
            showarrow=False, xanchor='left',
            font=dict(size=10, color=D_TEXT),
            bgcolor='rgba(0,0,0,0.6)',
            bordercolor=D_BORDER, borderwidth=1, borderpad=3,
        )
    _dark_theme(fig, height=290,
                barmode='stack',
                xaxis=dict(title='Lượt bán (nghìn)', showgrid=True,
                           gridcolor=GRID_COLOR, tickfont=dict(size=10, color=D_SUBTEXT)),
                yaxis=dict(showgrid=False, tickfont=dict(size=11, color=D_TEXT)),
                legend=dict(orientation='h', yanchor='bottom', y=1.02,
                            xanchor='right', x=1, font=dict(size=11, color=D_SUBTEXT)))
    return fig


# ══════════════════════════════════════════════════════════════
#  COMPONENT HELPERS
# ══════════════════════════════════════════════════════════════

def dark_kpi(icon, value, label, color):
    return html.Div([
        html.Div(icon, style={
            'fontSize': '26px', 'width': '48px', 'height': '48px',
            'borderRadius': '12px', 'display': 'flex',
            'alignItems': 'center', 'justifyContent': 'center',
            'background': f'{color}22', 'flexShrink': '0',
        }),
        html.Div([
            html.P(value, style={
                'margin': '0', 'fontSize': '22px', 'fontWeight': '800',
                'color': color, 'lineHeight': '1.1', 'letterSpacing': '-0.02em',
            }),
            html.P(label, style={
                'margin': '3px 0 0 0', 'fontSize': '10px', 'fontWeight': '600',
                'color': D_SUBTEXT, 'textTransform': 'uppercase', 'letterSpacing': '0.06em',
            }),
        ]),
    ], style={
        'display': 'flex', 'alignItems': 'center', 'gap': '12px',
        'background': D_CARD, 'border': f'1px solid {D_BORDER}',
        'borderTop': f'2px solid {color}',
        'borderRadius': '12px', 'padding': '14px 16px',
        'flex': '1', 'minWidth': '150px',
        'boxShadow': f'0 4px 16px rgba(0,0,0,0.3)',
    })


def dark_card(title, subtitle, children, flex='1', min_width='300px', accent=D_ACCENT):
    return html.Div([
        html.Div([
            html.Div(style={
                'width': '4px', 'height': '18px', 'borderRadius': '2px',
                'background': accent, 'flexShrink': '0',
            }),
            html.Div([
                html.P(title, style={
                    'margin': '0', 'fontSize': '13px', 'fontWeight': '700',
                    'color': D_TEXT, 'letterSpacing': '0.01em',
                }),
                html.P(subtitle, style={
                    'margin': '2px 0 0 0', 'fontSize': '11px', 'color': D_SUBTEXT,
                }) if subtitle else None,
            ]),
        ], style={'display': 'flex', 'alignItems': 'center', 'gap': '9px', 'marginBottom': '12px'}),
        children,
    ], style={
        'background': D_CARD, 'border': f'1px solid {D_BORDER}',
        'borderRadius': '14px', 'padding': '18px',
        'flex': flex, 'minWidth': min_width,
        'boxShadow': '0 4px 20px rgba(0,0,0,0.3)',
    })


def section_label(number, title, color):
    return html.Div([
        html.Div([
            html.Span(number, style={
                'fontSize': '11px', 'fontWeight': '800',
                'color': color, 'fontFamily': 'monospace',
                'letterSpacing': '0.08em',
            }),
        ], style={
            'width': '30px', 'height': '30px', 'borderRadius': '8px',
            'background': f'{color}22', 'border': f'1px solid {color}44',
            'display': 'flex', 'alignItems': 'center', 'justifyContent': 'center',
        }),
        html.Span(title, style={
            'fontSize': '15px', 'fontWeight': '700', 'color': D_TEXT,
        }),
    ], style={
        'display': 'flex', 'alignItems': 'center', 'gap': '12px',
        'padding': '14px 20px', 'marginBottom': '16px',
        'background': D_CARD2, 'borderRadius': '10px',
        'borderLeft': f'3px solid {color}',
    })


def dsi_legend_badge(text, bg, fg):
    return html.Span(text, style={
        'padding': '4px 12px', 'borderRadius': '20px',
        'background': bg, 'color': fg,
        'fontSize': '11px', 'fontWeight': '700',
        'border': f'1px solid {fg}44',
    })


# ══════════════════════════════════════════════════════════════
#  LAYOUT
# ══════════════════════════════════════════════════════════════

def layout():
    dsi_df   = compute_dsi()
    best_type = dsi_df.iloc[0]['product_type']
    best_dsi  = dsi_df.iloc[0]['DSI']

    total_rev   = df_full['estimated_revenue'].sum()
    imp_rev     = df_nn_full['estimated_revenue'].sum()
    imp_pct_rev = imp_rev / total_rev * 100
    top_country = (df_full.groupby('origin_corrected')['estimated_revenue']
                   .sum().sort_values(ascending=False).index[0])
    n_types     = df_full['product_type'].nunique()

    return html.Div([

        # ── HERO ──────────────────────────────────────────────
        html.Div([
            html.Div([
                html.Div([
                    html.Span('📈', style={'fontSize': '13px', 'marginRight': '6px'}),
                    html.Span('PHÂN TÍCH THỊ PHẦN', style={
                        'fontSize': '10px', 'fontWeight': '800',
                        'color': D_PURPLE_LT, 'letterSpacing': '0.12em',
                    }),
                ], style={'display': 'flex', 'alignItems': 'center', 'marginBottom': '8px'}),
                html.H1('Thị Phần Mỹ Phẩm Tiki', style={
                    'margin': '0 0 6px 0', 'fontSize': '28px', 'fontWeight': '900',
                    'color': D_TEXT, 'letterSpacing': '-0.02em', 'lineHeight': '1.2',
                }),
                html.P('T3/2026 · 7,179 sản phẩm · Nhóm 05 FIT-HCMUS', style={
                    'margin': '0', 'fontSize': '13px', 'color': D_SUBTEXT,
                }),
            ], style={'flex': '1'}),

            # Summary badges
            html.Div([
                html.Div([
                    html.Span(f'{imp_pct_rev:.1f}%', style={
                        'fontSize': '36px', 'fontWeight': '900',
                        'color': D_RED, 'lineHeight': '1', 'letterSpacing': '-0.03em',
                    }),
                    html.Span(' ngoại nhập', style={'fontSize': '14px', 'color': D_SUBTEXT}),
                ], style={'textAlign': 'right'}),
                html.Div([
                    html.Span(f'🥇 {top_country}', style={
                        'fontSize': '13px', 'color': D_GOLD, 'fontWeight': '700',
                    }),
                    html.Span(' · ', style={'color': D_SUBTEXT}),
                    html.Span(f'{n_types} ngành hàng', style={
                        'fontSize': '13px', 'color': D_SUBTEXT,
                    }),
                ]),
            ], style={'textAlign': 'right', 'flexShrink': '0'}),
        ], style={
            'display': 'flex', 'alignItems': 'center',
            'justifyContent': 'space-between', 'flexWrap': 'wrap', 'gap': '20px',
        }),

        # Navigation back
        html.Div([
            dcc.Link('← Overview', href='/', style={
                'color': D_PURPLE_LT, 'fontSize': '12px',
                'fontWeight': '600', 'textDecoration': 'none',
                'padding': '5px 12px', 'borderRadius': '20px',
                'border': f'1px solid {D_BORDER}',
                'background': D_CARD2,
            }),
        ], style={'marginTop': '16px', 'paddingTop': '14px', 'borderTop': f'1px solid {D_BORDER}'}),

    ], style={
        'background': f'linear-gradient(135deg, #0D1117 0%, #161B22 50%, #1A1040 100%)',
        'border': f'1px solid {D_BORDER}',
        'borderRadius': '16px', 'padding': '28px 32px',
        'marginBottom': '20px',
    }) ,

    html.Div([

        # ── KPI Row ───────────────────────────────────────────
        html.Div([
            dark_kpi('📦', f"{len(df_full):,}",        'Tổng sản phẩm',      '#3B82F6'),
            dark_kpi('💰', f"{total_rev/1e9:.1f} tỉ",  'Doanh thu ước tính', D_PURPLE_LT),
            dark_kpi('🌍', f"{imp_pct_rev:.1f}%",       'Thị phần ngoại',     D_RED),
            dark_kpi('🥇', top_country,                 'Nước dẫn đầu NK',    D_GOLD),
            dark_kpi('🇻🇳', f"{best_dsi:.1f}",          f'DSI · {best_type}', D_GREEN),
        ], style={
            'display': 'flex', 'gap': '12px',
            'flexWrap': 'wrap', 'marginBottom': '24px',
        }),

        # ═══════ MỤC TIÊU 1 ══════════════════════════════════
        section_label('MT 1', 'Thị phần doanh thu: Nội vs Ngoại', D_PURPLE_LT),

        html.Div([
            dark_card('Tổng thị phần doanh thu', 'Nội địa vs Ngoại nhập',
                      dcc.Graph(figure=make_donut(), config={'displayModeBar': False}),
                      flex='1', min_width='280px', accent='#3B82F6'),
            dark_card('Doanh thu theo quốc gia', 'Top 8 quốc gia xuất xứ',
                      dcc.Graph(figure=make_bar_country(), config={'displayModeBar': False}),
                      flex='1.3', min_width='340px', accent=D_RED),
        ], style={'display': 'flex', 'gap': '14px', 'marginBottom': '14px', 'flexWrap': 'wrap'}),

        html.Div([
            dark_card('Doanh thu theo phân khúc giá', 'Nội / ngoại theo khoảng giá',
                      dcc.Graph(figure=make_bar_price(), config={'displayModeBar': False}),
                      flex='1', min_width='340px', accent=D_GOLD),
            # Key findings MT1
            html.Div([
                html.P('Phát hiện chính', style={
                    'margin': '0 0 14px 0', 'fontSize': '12px', 'fontWeight': '800',
                    'color': D_PURPLE_LT, 'textTransform': 'uppercase', 'letterSpacing': '0.08em',
                }),
                _finding(f'Hàng ngoại chiếm ~{imp_pct_rev:.1f}% doanh thu', 'Dù hàng nội ~26.7% sản phẩm', D_RED),
                _finding(f'{top_country} dẫn đầu tuyệt đối ~190 tỉ VNĐ', 'Hada Labo, Senka, Sunplay', D_GOLD),
                _finding('Phân khúc dưới 300k tập trung doanh thu cao nhất', 'Cả nội lẫn ngoại', D_BLUE),
            ], style={
                'flex': '0.7', 'minWidth': '260px',
                'background': D_CARD, 'border': f'1px solid {D_BORDER}',
                'borderLeft': f'3px solid {D_PURPLE_LT}',
                'borderRadius': '14px', 'padding': '20px',
            }),
        ], style={'display': 'flex', 'gap': '14px', 'marginBottom': '24px', 'flexWrap': 'wrap'}),

        # ═══════ MỤC TIÊU 2 ══════════════════════════════════
        section_label('MT 2', 'Cơ cấu lượt bán & doanh thu theo ngành hàng', D_GOLD),

        html.Div([
            dark_card('Tỉ trọng Lượt bán vs Doanh thu', 'Δ = chênh lệch giữa 2 chỉ số',
                      dcc.Graph(figure=make_dual_bar(), config={'displayModeBar': False}),
                      flex='1', min_width='400px', accent=D_GOLD),
        ], style={'display': 'flex', 'gap': '14px', 'marginBottom': '14px', 'flexWrap': 'wrap'}),

        html.Div([
            dark_card('Top danh mục doanh thu theo ngành', 'Top 3 mỗi ngành · màu = ngành',
                      dcc.Graph(figure=make_top_categories(), config={'displayModeBar': False}),
                      flex='1', min_width='340px', accent=D_GOLD),
            dark_card('Mức độ "chín" của thị trường', 'Bestseller → Chưa bán · % trong mỗi ngành',
                      dcc.Graph(figure=make_popularity_tier(), config={'displayModeBar': False}),
                      flex='1', min_width='320px', accent=D_PURPLE_LT),
        ], style={'display': 'flex', 'gap': '14px', 'marginBottom': '24px', 'flexWrap': 'wrap'}),

        # ═══════ MỤC TIÊU 3 ══════════════════════════════════
        section_label('MT 3', 'Chỉ số Sức mạnh nội địa (DSI) theo ngành hàng', D_GREEN),

        # DSI legend strip
        html.Div([
            html.Span('Đọc DSI:', style={'fontSize': '11px', 'color': D_SUBTEXT, 'fontWeight': '600'}),
            dsi_legend_badge('≥50 · Ưu thế nội', '#052E16', D_GREEN),
            dsi_legend_badge('25–50 · Cạnh tranh', '#451A03', D_GOLD),
            dsi_legend_badge('<25 · Ngoại áp đảo', '#450A0A', D_RED),
        ], style={
            'display': 'flex', 'alignItems': 'center', 'gap': '8px',
            'flexWrap': 'wrap', 'marginBottom': '14px',
            'padding': '10px 16px', 'background': D_CARD2,
            'border': f'1px solid {D_BORDER}', 'borderRadius': '10px',
        }),

        html.Div([
            dark_card('Radar: 3 chiều DSI', 'Lượt bán · Doanh thu · Sản phẩm · % nội địa',
                      dcc.Graph(figure=make_radar(), config={'displayModeBar': False}),
                      flex='1', min_width='310px', accent=D_GREEN),
            dark_card('Chỉ số DSI theo ngành hàng', '🟢 ≥50 · 🟡 25–50 · 🔴 <25',
                      dcc.Graph(figure=make_dsi_bar(), config={'displayModeBar': False}),
                      flex='1', min_width='310px', accent=D_GREEN),
        ], style={'display': 'flex', 'gap': '14px', 'marginBottom': '14px', 'flexWrap': 'wrap'}),

        html.Div([
            dark_card('Cơ cấu lượt bán nội / ngoại theo ngành', 'Sắp xếp theo DSI tăng dần',
                      dcc.Graph(figure=make_dsi_stacked(), config={'displayModeBar': False}),
                      flex='1', min_width='400px', accent=D_GREEN),
            html.Div([
                html.P('Phát hiện chính', style={
                    'margin': '0 0 14px 0', 'fontSize': '12px', 'fontWeight': '800',
                    'color': D_GREEN, 'textTransform': 'uppercase', 'letterSpacing': '0.08em',
                }),
                _finding('Makeup — ngành duy nhất hàng Việt ưu thế', 'Cocoon, Emmié by Happy Skin dẫn đầu', D_GREEN),
                _finding('Skincare chiếm ~48.7% doanh thu', 'Nhưng hàng Việt chỉ ~11% lượt bán', D_GOLD),
                _finding('Body Care & Hair Care: ngoại thống lĩnh', 'Selsun, Dove, Clear tạo rào cản cao', D_RED),
            ], style={
                'flex': '0.7', 'minWidth': '260px',
                'background': D_CARD, 'border': f'1px solid {D_BORDER}',
                'borderLeft': f'3px solid {D_GREEN}',
                'borderRadius': '14px', 'padding': '20px',
            }),
        ], style={'display': 'flex', 'gap': '14px', 'marginBottom': '20px', 'flexWrap': 'wrap'}),

        # ── Footer ────────────────────────────────────────────
        html.Div([
            html.Span('⚠️ Dữ liệu crawl Tiki T3/2026', style={'marginRight': '16px'}),
            html.Span('DSI = (sold_share + rev_share + product_share) / 3', style={'marginRight': '16px'}),
            html.Span('Nhóm 05 · FIT-HCMUS'),
        ], style={
            'fontSize': '11px', 'color': D_SUBTEXT,
            'textAlign': 'center', 'padding': '14px 0',
            'borderTop': f'1px solid {D_BORDER}',
        }),

    ], style={'background': D_BG, 'padding': '0 2px'})


def _finding(title, subtitle, color):
    """Mini finding row."""
    return html.Div([
        html.Div(style={
            'width': '8px', 'height': '8px', 'borderRadius': '50%',
            'background': color, 'flexShrink': '0', 'marginTop': '4px',
        }),
        html.Div([
            html.P(title, style={
                'margin': '0', 'fontSize': '12px',
                'fontWeight': '700', 'color': D_TEXT,
            }),
            html.P(subtitle, style={
                'margin': '2px 0 8px 0', 'fontSize': '11px', 'color': D_SUBTEXT,
            }),
        ]),
    ], style={'display': 'flex', 'gap': '10px', 'alignItems': 'flex-start'})
