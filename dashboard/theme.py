# theme.py — Design tokens & helpers cho Plotly/Dash dashboard
# from theme import *

import plotly.graph_objects as go
import plotly.express as px

# ──────────────────────────────────────────────────────────────
# 1. MÀUSẮC
# ──────────────────────────────────────────────────────────────
C_DOMESTIC  = '#2563EB'   # xanh dương — hàng trong nước
C_IMPORT    = '#DC2626'   # đỏ         — hàng ngoài nước
C_NEUTRAL   = '#94A3B8'
C_TEXT      = '#1E293B'
C_SUBTEXT   = '#64748B'
C_GRID      = '#E2E8F0'
C_BG        = '#FFFFFF'
C_BG_PAGE   = '#F8FAFC'
C_BORDER    = '#E2E8F0'
C_SUCCESS   = '#16A34A'
C_WARNING   = '#D97706'

PALETTE_ORIGIN = [C_DOMESTIC, C_IMPORT]

# Màu 5 ngành hàng
PRODUCT_TYPE_COLORS = {
    'Skincare'  : '#3B82F6',
    'Makeup'    : '#EC4899',
    'Hair Care' : '#F97316',
    'Body Care' : '#22C55E',
    'Fragrance' : '#8B5CF6',
    'Khác'      : '#94A3B8',
}
PRODUCT_TYPE_LIST   = list(PRODUCT_TYPE_COLORS.keys())
PRODUCT_TYPE_PALETTE = list(PRODUCT_TYPE_COLORS.values())

# Màu top quốc gia
COUNTRY_PALETTE = [
    '#1D4ED8','#DC2626','#D97706','#16A34A','#9333EA',
    '#0891B2','#EA580C','#4F46E5','#DB2777','#65A30D',
]

# Accent màu theo page (cho header card, border)
PAGE_ACCENT = {
    'overview'    : '#2563EB',
    'thi_phan'    : '#2563EB',
    'uy_tin'      : '#C026D3',
    'chat_luong'  : '#16A34A',
    'gia_ca'      : '#D97706',
}

# ──────────────────────────────────────────────────────────────
# 2. PLOTLY LAYOUT TEMPLATE
# ──────────────────────────────────────────────────────────────
LAYOUT_BASE = dict(
    font=dict(family="'Inter', 'Segoe UI', sans-serif",
              size=12, color=C_TEXT),
    paper_bgcolor=C_BG,
    plot_bgcolor=C_BG,
    margin=dict(l=16, r=16, t=40, b=16),
    legend=dict(
        orientation='h',
        yanchor='bottom', y=1.02,
        xanchor='right',  x=1,
        bgcolor='rgba(255,255,255,0.85)',
        bordercolor=C_BORDER, borderwidth=1,
        font=dict(size=11),
    ),
    hoverlabel=dict(
        bgcolor='white', bordercolor=C_BORDER,
        font=dict(size=12, color=C_TEXT),
    ),
    xaxis=dict(
    gridcolor=C_GRID, gridwidth=1,
    linecolor=C_BORDER, linewidth=1,
    tickfont=dict(size=11, color=C_SUBTEXT),
    title=dict(font=dict(size=12, color=C_SUBTEXT)),  
    showgrid=False,
),
yaxis=dict(
    gridcolor=C_GRID, gridwidth=1,
    linecolor='rgba(0,0,0,0)',
    tickfont=dict(size=11, color=C_SUBTEXT),
    title=dict(font=dict(size=12, color=C_SUBTEXT)), 
    showgrid=True,
),
    colorway=PALETTE_ORIGIN,
    hovermode='closest',
    transition=dict(duration=400, easing='cubic-in-out'),
)


def make_layout(title='', height=320, **overrides):
    """
    Tạo plotly layout chuẩn.
    make_layout('Tiêu đề', height=360, showlegend=False)
    """
    layout = dict(**LAYOUT_BASE)
    layout.update(dict(
        height=height,
        title=dict(text=title, font=dict(size=14, color=C_TEXT, weight=700),
                   x=0, pad=dict(l=4)),
    ))
    layout.update(overrides)
    return layout


def apply_theme(fig, title='', height=320, **overrides):
    """Apply theme lên 1 figure có sẵn."""
    fig.update_layout(**make_layout(title, height, **overrides))
    return fig


# ──────────────────────────────────────────────────────────────
# 3. REUSABLE CHART HELPERS
# ──────────────────────────────────────────────────────────────

def donut_chart(values, labels, colors, title='', height=300,
                hole=0.62, textinfo='percent'):
    """Donut chart chuẩn."""
    fig = go.Figure(go.Pie(
        values=values, labels=labels,
        marker=dict(colors=colors, line=dict(color='white', width=2.5)),
        hole=hole,
        textinfo=textinfo,
        textfont=dict(size=13, color='white', weight=700),
        hovertemplate='<b>%{label}</b><br>%{value:,.0f}<br>%{percent}<extra></extra>',
        pull=[0.04 if i == 0 else 0 for i in range(len(values))],
        sort=False,
    ))
    apply_theme(fig, title, height,
                showlegend=True,
                legend=dict(orientation='v', yanchor='middle', y=0.5,
                            xanchor='left', x=1.02,
                            font=dict(size=12)))
    fig.update_layout(margin=dict(l=16, r=80, t=40, b=16))
    return fig


def hbar_chart(y_vals, x_vals, colors, title='', height=320,
               x_label='', text_fmt='.0f'):
    """Horizontal bar chart chuẩn."""
    fig = go.Figure(go.Bar(
        x=x_vals, y=y_vals,
        orientation='h',
        marker=dict(color=colors, line=dict(color='white', width=0.8)),
        text=[f'{v:,.{text_fmt[-1]}}' for v in x_vals],
        textposition='outside',
        textfont=dict(size=11, color=C_TEXT, weight=600),
        hovertemplate='<b>%{y}</b><br>' + x_label + ': %{x:,.0f}<extra></extra>',
        cliponaxis=False,
    ))
    apply_theme(fig, title, height,
                xaxis=dict(**LAYOUT_BASE['xaxis'],
                           title=x_label, showgrid=True),
                yaxis=dict(**LAYOUT_BASE['yaxis'],
                           showgrid=False, autorange='reversed'),
                showlegend=False)
    return fig


def vbar_grouped(categories, series_dict, title='', height=340,
                 y_label='', barmode='group'):
    """
    Grouped / stacked vertical bar.
    series_dict = {'Trong nước': [v1,v2,...], 'Ngoài nước': [v1,v2,...]}
    """
    colors_map = {'Trong nước': C_DOMESTIC, 'Ngoài nước': C_IMPORT}
    fig = go.Figure()
    for name, vals in series_dict.items():
        fig.add_trace(go.Bar(
            name=name, x=categories, y=vals,
            marker_color=colors_map.get(name, C_NEUTRAL),
            marker_line=dict(color='white', width=0.8),
            hovertemplate=f'<b>%{{x}}</b><br>{name}: %{{y:,.0f}}<extra></extra>',
        ))
    apply_theme(fig, title, height,
                barmode=barmode,
                xaxis=dict(**LAYOUT_BASE['xaxis'], showgrid=False),
                yaxis=dict(**LAYOUT_BASE['yaxis'],
                           title=y_label, showgrid=True))
    return fig


# ──────────────────────────────────────────────────────────────
# 4. COMPONENT HELPERS — Dash HTML/dcc shortcuts
# ──────────────────────────────────────────────────────────────

from dash import html, dcc


def kpi_card(value, label, value_color=C_TEXT, bg='#FFFFFF',
             border_accent=None):
    """KPI card component."""
    border_style = (f'2px solid {border_accent}' if border_accent
                    else f'1px solid {C_BORDER}')
    return html.Div([
        html.P(label, style={
            'margin': '0 0 6px 0', 'fontSize': '12px',
            'color': C_SUBTEXT, 'fontWeight': '500',
            'textTransform': 'uppercase', 'letterSpacing': '0.05em',
        }),
        html.P(value, style={
            'margin': '0', 'fontSize': '28px',
            'fontWeight': '700', 'color': value_color,
            'lineHeight': '1',
        }),
    ], style={
        'background': bg,
        'border': border_style,
        'borderRadius': '12px',
        'padding': '18px 20px',
        'flex': '1',
        'minWidth': '150px',
    })


def section_header(title, subtitle='', accent_color=C_DOMESTIC):
    """Tiêu đề section nhỏ."""
    return html.Div([
        html.Div(style={
            'width': '4px', 'height': '20px',
            'background': accent_color,
            'borderRadius': '2px', 'flexShrink': '0',
        }),
        html.Div([
            html.H3(title, style={
                'margin': '0', 'fontSize': '15px',
                'fontWeight': '600', 'color': C_TEXT,
            }),
            html.P(subtitle, style={
                'margin': '2px 0 0 0', 'fontSize': '12px',
                'color': C_SUBTEXT,
            }) if subtitle else None,
        ]),
    ], style={'display': 'flex', 'alignItems': 'center',
              'gap': '10px', 'marginBottom': '12px'})


def chart_card(children, padding='16px'):
    """Card wrapper cho 1 chart."""
    return html.Div(children, style={
        'background': C_BG,
        'border': f'1px solid {C_BORDER}',
        'borderRadius': '12px',
        'padding': padding,
        'height': '100%',
    })


def filter_pill(label, value, active=False, color=C_DOMESTIC):
    """Pill button cho filter."""
    return html.Button(label, id={'type': 'filter-pill', 'value': value},
                       n_clicks=0, style={
                           'padding': '6px 14px',
                           'borderRadius': '20px',
                           'border': f'1.5px solid {color if active else C_BORDER}',
                           'background': color if active else 'white',
                           'color': 'white' if active else C_SUBTEXT,
                           'fontSize': '13px', 'fontWeight': '500',
                           'cursor': 'pointer',
                           'transition': 'all 0.2s ease',
                       })
