# pages/page2_uy_tin.py
# ╔══════════════════════════════════════════════════════════════╗
# ║  Trang 2 — Uy Tín: Phân tích tương tác & tin cậy khách hàng  ║
# ║  Focus: Review Ratio & Engagement - Trust Indicators         ║
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
    path='/uy-tin',
    name='Uy tín',
    title='Uy tín | Mỹ phẩm Tiki',
    order=2,
)

# ── Load data ────────────────────────────────────────────────
df_full, df_vn_full, df_nn_full = load_data()

# Filter để chỉ lấy sản phẩm có review
df_review = df_full[df_full['review_ratio'] > 0].copy()
df_review_vn = df_vn_full[df_vn_full['review_ratio'] > 0].copy()
df_review_nn = df_nn_full[df_nn_full['review_ratio'] > 0].copy()

# Tạo cột origin_group từ origin_class_corrected
df_review['origin_group'] = df_review['origin_class_corrected'].apply(
    lambda x: 'Nội' if x == 'Trong nước' else 'Ngoại'
)

PRICE_ORDER   = ['Dưới 100k', '100k – 300k', '300k – 700k', '700k – 2tr', 'Trên 2tr']
TYPE_COLORS   = PRODUCT_TYPE_COLORS

# ── Filter options ───────────────────────────────────────────
PRODUCT_TYPES_ALL = sorted(df_review['product_type'].unique().tolist())
PRICE_SEGMENTS_ALL = [p for p in PRICE_ORDER if p in df_review['price_segment'].unique()]
ORIGIN_OPTIONS = [
    {'label': 'Tất cả', 'value': 'all'},
    {'label': 'Trong nước', 'value': 'domestic'},
    {'label': 'Ngoài nước', 'value': 'import'},
]

# ── Design tokens ────────────────────────────────────────────
A_500   = '#F59E0B'
A_700   = '#B45309'
A_300   = '#FCD34D'
A_50    = 'rgba(245,158,11,0.10)'

CYAN    = '#06B6D4'
EMERALD = '#10B981'
ROSE    = '#F43F5E'

TEXT       = C_TEXT
SUBTEXT    = C_SUBTEXT
SURFACE    = '#1C2D55'
SURFACE_ALT = '#223763'
BORDER     = 'rgba(255,255,255,0.08)'
GRID       = 'rgba(255,255,255,0.06)'
HOVER_BG   = '#111827'
CARD_SHADOW = '0 14px 34px rgba(3,10,25,0.22)'
PAGE_BG    = '#14233F'

my_palette = {'Nội': C_DOMESTIC, 'Ngoại': C_IMPORT}
order_list = ['Nội', 'Ngoại']


# ══════════════════════════════════════════════════════════════
#  FILTER & COMPUTE FUNCTIONS
# ══════════════════════════════════════════════════════════════

def apply_filters(df, product_types=None, price_segments=None, origin='all'):
    """Lọc dữ liệu theo các filter được chọn"""
    result = df[df['review_ratio'] > 0].copy()
    
    if product_types and len(product_types) > 0:
        result = result[result['product_type'].isin(product_types)]
    
    if price_segments and len(price_segments) > 0:
        result = result[result['price_segment'].isin(price_segments)]
    
    if origin == 'domestic':
        result = result[result['origin_class_corrected'] == 'Trong nước']
    elif origin == 'import':
        result = result[result['origin_class_corrected'] == 'Ngoài nước']
    
    return result


def compute_engagement_level(df):
    """Phân loại mức độ engagement"""
    df_copy = df.copy()
    df_copy['engagement_level'] = pd.cut(
        df_copy['review_ratio'],
        bins=[0, 0.05, 0.15, 0.3, float('inf')],
        labels=['Thấp (0-5%)', 'Trung bình (5-15%)', 'Cao (15-30%)', 'Rất cao (>30%)']
    )
    return df_copy


def compute_dynamic_kpi(df):
    """Tính KPI động từ filtered data"""
    if len(df) == 0:
        return {
            'avg_review_ratio': 0,
            'high_engagement_pct': 0,
            'total_reviews': 0,
            'best_category': '-',
        }
    
    avg_ratio = df['review_ratio'].mean()
    total_reviews = df['review_count'].sum()
    
    df_eng = compute_engagement_level(df)
    high_eng = len(df_eng[df_eng['engagement_level'].isin(['Cao (15-30%)', 'Rất cao (>30%)'])])
    high_eng_pct = (high_eng / len(df) * 100) if len(df) > 0 else 0
    
    best_cat = df.groupby('primary_category')['review_ratio'].mean().idxmax() if len(df) > 0 else '-'
    
    return {
        'avg_review_ratio': avg_ratio,
        'high_engagement_pct': high_eng_pct,
        'total_reviews': total_reviews,
        'best_category': best_cat,
    }


# ══════════════════════════════════════════════════════════════
#  UI COMPONENTS
# ══════════════════════════════════════════════════════════════

def section_header(num, title, desc, variant='muc1'):
    icons = {'muc1': '⭐', 'muc2': '💬', 'muc3': '📈'}
    return html.Div([
        html.Div([
            html.Span(icons[variant], style={'fontSize': '17px'}),
        ], className='p2-section-num'),
        html.Div([
            html.P(f'Mục tiêu {num}', style={
                'margin': '0 0 2px 0',
                'fontSize': '10px', 'fontWeight': '700',
                'color': {'muc1': A_500, 'muc2': CYAN, 'muc3': EMERALD}[variant],
                'textTransform': 'uppercase', 'letterSpacing': '0.1em',
            }),
            html.H3(title, className='p2-section-title'),
            html.P(desc, className='p2-section-desc'),
        ]),
    ], className=f'p2-section-header {variant}')


def chart_card(icon, title, subtitle, children, flex='1', min_width='300px', icon_bg='#FEF3C7'):
    return html.Div([
        html.Div([
            html.Div(icon, className='p2-card-icon',
                     style={'background': icon_bg}),
            html.Div([
                html.P(title, className='p2-card-title'),
                html.P(subtitle, className='p2-card-subtitle'),
            ]),
        ], className='p2-card-header'),
        children,
    ], className='p2-card', style={'flex': flex, 'minWidth': min_width})


def insight_box(text, variant='amber'):
    icons = {'amber': '💡', 'cyan': '🔍', 'emerald': '✅'}
    return html.Div([
        html.Span(icons.get(variant, '💡'), style={'fontSize': '16px', 'flexShrink': '0'}),
        html.Span(text, style={'fontSize': '12px'}),
    ], className=f'p2-insight {variant}')


def kpi_card(icon, value, label, color, bg):
    return html.Div([
        html.Div(icon, className='p2-kpi-dot',
                 style={'background': bg}),
        html.Div([
            html.P(value, className='p2-kpi-val', style={'color': color}),
            html.P(label, className='p2-kpi-label'),
        ]),
    ], className='p2-kpi')


# ══════════════════════════════════════════════════════════════
#  CHART BUILDERS
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


def make_review_ratio_box(df=None):
    if df is None:
        df = df_review

    df_lim = df[df['review_ratio'] <= df['review_ratio'].quantile(0.95)].copy()

    fig = go.Figure()

    for origin, color in zip(order_list, [C_DOMESTIC, C_IMPORT]):
        data = df_lim[df_lim['origin_group'] == origin]['review_ratio']
        if len(data) > 0:
            fig.add_trace(go.Box(
                y=data,
                name=origin,
                marker=dict(color=color),
                boxmean=True,   # sửa ở đây
                hovertemplate='<b>%{fullData.name}</b><br>Review Ratio: %{y:.3f}<extra></extra>',
            ))

    _theme(
        fig,
        height=340,
        title=dict(
            text='<b>Phân bố Review Ratio (Nội vs Ngoại)</b>',
            x=0.5, xanchor='center',
            font=dict(size=16, color=TEXT)
        ),
        yaxis=dict(title='Review Ratio', showgrid=True, gridcolor=GRID),
        showlegend=True,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )
    return fig


def make_engagement_stacked(df=None):
    """Stacked bar: Phân bố mức độ Engagement"""
    if df is None:
        df = df_review
    
    df_eng = compute_engagement_level(df)
    eng_by_origin = pd.crosstab(df_eng['origin_group'], df_eng['engagement_level'])
    eng_pct = eng_by_origin.div(eng_by_origin.sum(axis=1), axis=0) * 100
    eng_pct = eng_pct.reindex(order_list, fill_value=0)
    
    fig = go.Figure()
    colors_eng = ['#FEE2E2', '#FCA5A5', '#F59E0B', '#78350F']
    labels_eng = eng_pct.columns.tolist()
    
    for i, label in enumerate(labels_eng):
        fig.add_trace(go.Bar(
            name=label,
            x=eng_pct.index,
            y=eng_pct[label],
            marker=dict(color=colors_eng[i] if i < len(colors_eng) else '#999'),
            hovertemplate=f'<b>%{{x}}</b><br>{label}: %{{y:.1f}}%<extra></extra>',
            text=[f'{v:.0f}%' if v >= 8 else '' for v in eng_pct[label]],
            textposition='inside',
            textfont=dict(size=11, color='white'),
        ))
    
    _theme(fig, height=320,
           barmode='stack',
           title=dict(text='<b>Phân bố Mức độ Engagement (%)</b>', x=0.5, xanchor='center', font=dict(size=16, color=TEXT)),
           xaxis=dict(showgrid=False, tickfont=dict(size=12)),
           yaxis=dict(showgrid=True, gridcolor=GRID, range=[0, 100]),
           legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )
    return fig


def make_category_bar(df=None):
    """Horizontal bar: Top categories theo review ratio"""
    if df is None:
        df = df_review
    
    if len(df) == 0:
        # Return empty figure if no data
        return go.Figure()
    
    top_n = 10
    cat_mean = df.groupby('primary_category')['review_ratio'].mean().sort_values(ascending=True).tail(top_n)
    avg_all = df['review_ratio'].mean()
    
    colors = [A_500 if v >= avg_all else '#7C3AED' for v in cat_mean.values]
    
    fig = go.Figure(go.Bar(
        x=cat_mean.values,
        y=cat_mean.index,
        orientation='h',
        marker=dict(color=colors, line=dict(color='white', width=0.5)),
        text=[f'{v:.3f}' for v in cat_mean.values],
        textposition='outside',
        textfont=dict(size=11, color=TEXT),
        hovertemplate='<b>%{y}</b><br>Review Ratio: %{x:.3f}<extra></extra>',
        cliponaxis=False,
    ))
    
    fig.add_vline(
    x=avg_all,
    line_dash='dash',
    line_color=CYAN,
    line_width=2,
    annotation_text=f'Trung bình ({avg_all:.3f})',
    annotation_position='top right'
    )
    
    _theme(fig, height=340,
           showlegend=False,
           title=dict(text='<b>Review Ratio theo Dòng Sản Phẩm (Top 10)</b>', x=0.5, xanchor='center', font=dict(size=16, color=TEXT)),
           xaxis=dict(title='Review Ratio', showgrid=True, gridcolor=GRID, tickfont=dict(size=11)),
           yaxis=dict(showgrid=False, tickfont=dict(size=11)),
           margin=dict(l=16, r=80, t=70, b=16)
    )
    return fig


def make_noi_vs_ngoai_comparison(df=None):
    """Grouped bar: So sánh Review Ratio Nội vs Ngoại theo top categories"""
    if df is None:
        df = df_review
    
    if len(df) == 0:
        return go.Figure()
    
    top_n = 8
    top_cats = df['primary_category'].value_counts().head(top_n).index.tolist()
    df_top = df[df['primary_category'].isin(top_cats)]
    
    comparison = df_top.groupby(['primary_category', 'origin_group'])['review_ratio'].mean().unstack(fill_value=0)
    comparison = comparison.reindex(top_cats)
    
    fig = go.Figure()
    
    for origin, color in zip(order_list, [C_DOMESTIC, C_IMPORT]):
        if origin in comparison.columns:
            fig.add_trace(go.Bar(
                x=comparison.index,
                y=comparison[origin],
                name=origin,
                marker=dict(color=color),
                text=[f'{v:.3f}' for v in comparison[origin]],
                textposition='outside',
                textfont=dict(size=10, color=TEXT),
                hovertemplate=f'<b>%{{x}}</b><br>{origin}: %{{y:.3f}}<extra></extra>',
            ))
    
    _theme(fig, height=340,
           barmode='group',
           title=dict(text='<b>So Sánh Review Ratio Nội vs Ngoại</b>', x=0.5, xanchor='center', font=dict(size=16, color=TEXT)),
           xaxis=dict(showgrid=False, tickfont=dict(size=11), tickangle=-15),
           yaxis=dict(title='Review Ratio', showgrid=True, gridcolor=GRID),
           legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )
    return fig


# ══════════════════════════════════════════════════════════════
#  LAYOUT
# ══════════════════════════════════════════════════════════════

def layout():
    kpi_data = compute_dynamic_kpi(df_review)
    best_cat = kpi_data['best_category']
    avg_ratio = kpi_data['avg_review_ratio']
    high_eng = kpi_data['high_engagement_pct']
    total_rev = kpi_data['total_reviews']
    
    return html.Div([

        # ── HEADER + FILTER ─────────────────────────
        html.Div([
            html.Div([
                html.Span('🔐', style={'fontSize': '18px', 'marginRight': '10px'}),
                html.H1('Uy tín & Tương tác Khách hàng · T3/2026',
                        className='p2-hero-title', style={'display': 'inline'})
            ], style={'marginBottom': '20px', 'display': 'flex', 'alignItems': 'center'}),

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
                            id='p2-filter-product-type',
                            options=[{'label': 'Tất cả ngành', 'value': 'all'}] + 
                                    [{'label': pt, 'value': pt} for pt in PRODUCT_TYPES_ALL],
                            value='all',
                            multi=False,
                            clearable=False,
                            className='p2-filter-pill-select'
                        ),
                    ], style={'minWidth': '160px'}),
                    
                    html.Div([
                        dcc.Dropdown(
                            id='p2-filter-price-segment',
                            options=[{'label': 'Tất cả giá', 'value': 'all'}] + 
                                    [{'label': seg, 'value': seg} for seg in PRICE_SEGMENTS_ALL],
                            value='all',
                            multi=False,
                            clearable=False,
                            className='p2-filter-pill-select'
                        ),
                    ], style={'minWidth': '160px'}),
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
                        id='p2-filter-origin',
                        options=ORIGIN_OPTIONS,
                        value='all',
                        inline=True,
                        style={'display': 'flex', 'gap': '14px'},
                        className='p2-filter-radio'
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
        ], className='p2-hero', style={'paddingBottom': '20px'}),

        # ── KPI Row ─────────────────────────────
        html.Div(id='p2-kpi-row', children=[
            kpi_card('⭐', f"{avg_ratio:.3f}", 'Review Ratio TB', '#F59E0B', 'rgba(245, 158, 11, 0.14)'),
            kpi_card('🔥', f"{high_eng:.1f}%", 'Engagement cao', '#06B6D4', 'rgba(6, 182, 212, 0.14)'),
            kpi_card('💬', f"{int(total_rev):,}", 'Tổng reviews', '#10B981', 'rgba(16, 185, 129, 0.14)'),
            kpi_card('🏆', best_cat[:15] + ('...' if len(best_cat) > 15 else ''), f'Dòng nổi bật', '#F43F5E', 'rgba(244, 63, 94, 0.14)'),
        ], className='p2-kpi-row'),

        # ── MAIN CHARTS: 2x2 Layout ──────────────────
        html.Div([
            html.Div(dcc.Graph(id='p2-chart-box',
                               config={'displayModeBar': False}),
                     className='p2-card', style={'flex': '1', 'minWidth': '320px'}),

            html.Div(dcc.Graph(id='p2-chart-stacked',
                               config={'displayModeBar': False}),
                     className='p2-card', style={'flex': '1', 'minWidth': '320px'}),
        ], className='p2-row'),

        html.Div([
            html.Div(dcc.Graph(id='p2-chart-category',
                               config={'displayModeBar': False}),
                     className='p2-card', style={'flex': '1', 'minWidth': '320px'}),

            html.Div(dcc.Graph(id='p2-chart-comparison',
                               config={'displayModeBar': False}),
                     className='p2-card', style={'flex': '1', 'minWidth': '320px'}),
        ], className='p2-row')

    ], className='p2-page', style={'padding': '20px 24px', 'maxWidth': '1600px', 'margin': '0 auto'})


# ══════════════════════════════════════════════════════════════
#  CALLBACKS (Interactive updates)
# ══════════════════════════════════════════════════════════════

@callback(
    [Output('p2-chart-box', 'figure'),
     Output('p2-chart-stacked', 'figure'),
     Output('p2-chart-category', 'figure'),
     Output('p2-chart-comparison', 'figure'),
     Output('p2-kpi-row', 'children')],
    [Input('p2-filter-product-type', 'value'),
     Input('p2-filter-price-segment', 'value'),
     Input('p2-filter-origin', 'value')],
    prevent_initial_call=False
)
def update_charts(product_type, price_segment, origin):
    # Apply filters
    filtered_df = apply_filters(
        df_review,
        product_types=[product_type] if product_type != 'all' else None,
        price_segments=[price_segment] if price_segment != 'all' else None,
        origin=origin
    )
    
    # Update KPI
    kpi_data = compute_dynamic_kpi(filtered_df)
    kpi_children = [
        kpi_card('⭐', f"{kpi_data['avg_review_ratio']:.3f}", 'Review Ratio TB', '#F59E0B', 'rgba(245, 158, 11, 0.14)'),
        kpi_card('🔥', f"{kpi_data['high_engagement_pct']:.1f}%", 'Engagement cao', '#06B6D4', 'rgba(6, 182, 212, 0.14)'),
        kpi_card('💬', f"{int(kpi_data['total_reviews']):,}", 'Tổng reviews', '#10B981', 'rgba(16, 185, 129, 0.14)'),
        kpi_card('🏆', kpi_data['best_category'][:15] + ('...' if len(kpi_data['best_category']) > 15 else ''), f'Dòng nổi bật', '#F43F5E', 'rgba(244, 63, 94, 0.14)'),
    ]
    
    # Rebuild charts with filtered data
    return (
        make_review_ratio_box(filtered_df),
        make_engagement_stacked(filtered_df),
        make_category_bar(filtered_df),
        make_noi_vs_ngoai_comparison(filtered_df),
        kpi_children
    )