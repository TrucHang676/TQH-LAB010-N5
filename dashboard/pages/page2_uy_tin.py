
# pages/page2_uy_tin.py
# ╔══════════════════════════════════════════════════════════════╗
# ║  Trang 2 — Uy Tín: Phân tích tương tác & tin cậy khách hàng  ║
# ║  Focus: Review Ratio & Engagement - Trust Indicators         ║
# ╚══════════════════════════════════════════════════════════════╝

import dash
from dash import html, dcc, callback, Input, Output
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from data_loader import load_data
from theme import (
    C_DOMESTIC, C_IMPORT, C_TEXT, C_SUBTEXT,
    PRODUCT_TYPE_COLORS,
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

PRICE_ORDER = ['Dưới 100k', '100k – 300k', '300k – 700k', '700k – 2tr', 'Trên 2tr']
TYPE_COLORS = PRODUCT_TYPE_COLORS
ENGAGEMENT_ORDER = ['Thấp (0-5%)', 'Trung bình (5-15%)', 'Cao (15-30%)', 'Rất cao (>30%)']
TOP_CATEGORY_N = 12
MIN_PRODUCTS_FOR_BEST_CATEGORY = 5

# ── Filter options ───────────────────────────────────────────
PRODUCT_TYPES_ALL = sorted(df_review['product_type'].dropna().unique().tolist())
PRICE_SEGMENTS_ALL = [p for p in PRICE_ORDER if p in df_review['price_segment'].dropna().unique()]
ORIGIN_OPTIONS = [
    {'label': 'Tất cả', 'value': 'all'},
    {'label': 'Trong nước', 'value': 'domestic'},
    {'label': 'Ngoài nước', 'value': 'import'},
]

# ── Design tokens ────────────────────────────────────────────
A_500 = '#F59E0B'
CYAN = '#06B6D4'
EMERALD = '#10B981'
ROSE = '#F43F5E'
PURPLE = '#7C3AED'

TEXT = C_TEXT
SUBTEXT = C_SUBTEXT
SURFACE = '#1C2D55'
BORDER = 'rgba(255,255,255,0.08)'
GRID = 'rgba(255,255,255,0.06)'
HOVER_BG = '#111827'

my_palette = {'Nội': C_DOMESTIC, 'Ngoại': C_IMPORT}
order_list = ['Nội', 'Ngoại']

BASE_FONT = 15
TICK_FONT = 14
AXIS_TITLE_FONT = 15
TITLE_FONT = 18
BAR_TEXT_FONT = 13
LEGEND_FONT = 13
HOVER_FONT = 14

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
        labels=ENGAGEMENT_ORDER
    )
    return df_copy


def get_category_stats(df):
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=['avg_review_ratio', 'product_count', 'total_reviews'])

    stats = (
        df.groupby('primary_category')
        .agg(
            avg_review_ratio=('review_ratio', 'mean'),
            product_count=('primary_category', 'size'),
            total_reviews=('review_count', 'sum'),
        )
        .sort_values(['product_count', 'total_reviews', 'avg_review_ratio'], ascending=[False, False, False])
    )
    return stats


def get_shared_top_category_stats(df, top_n=TOP_CATEGORY_N):
    stats = get_category_stats(df)
    if stats.empty:
        return stats
    return stats.head(top_n).copy()


def get_best_category(df):
    stats = get_category_stats(df)
    if stats.empty:
        return '-'

    eligible = stats[stats['product_count'] >= MIN_PRODUCTS_FOR_BEST_CATEGORY].copy()
    if eligible.empty:
        eligible = stats.head(min(5, len(stats))).copy()

    best_cat = eligible.sort_values(
        ['avg_review_ratio', 'product_count', 'total_reviews'],
        ascending=[False, False, False]
    ).index[0]
    return best_cat


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

    return {
        'avg_review_ratio': avg_ratio,
        'high_engagement_pct': high_eng_pct,
        'total_reviews': total_reviews,
        'best_category': get_best_category(df),
    }


def format_best_category_label(label, max_len=15):
    if not label or label == '-':
        return '-'
    return label[:max_len] + ('...' if len(label) > max_len else '')


# ══════════════════════════════════════════════════════════════
#  UI COMPONENTS
# ══════════════════════════════════════════════════════════════

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
#  CHART HELPERS
# ══════════════════════════════════════════════════════════════

def _chart_title(text):
    return dict(
        text=f'<b>{text}</b>',
        x=0.5,
        xanchor='center',
        y=0.98,
        yanchor='top',
        pad=dict(b=12),
        font=dict(size=TITLE_FONT, color=TEXT)
    )


def _theme(fig, height=320, **kw):
    margin = kw.pop('margin', dict(l=16, r=16, t=72, b=16))
    fig.update_layout(
        height=height,
        font=dict(
            family="'DM Sans','Segoe UI',sans-serif",
            size=BASE_FONT,
            color=TEXT
        ),
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        margin=margin,
        hoverlabel=dict(
            bgcolor=HOVER_BG,
            bordercolor=BORDER,
            font=dict(size=HOVER_FONT, color=TEXT)
        ),
        **kw,
    )

    fig.update_xaxes(
        gridcolor=GRID,
        gridwidth=1,
        linecolor='rgba(255,255,255,0.08)',
        linewidth=1,
        tickfont=dict(size=TICK_FONT, color=SUBTEXT),
        title_font=dict(size=AXIS_TITLE_FONT, color=SUBTEXT),
        zeroline=False,
    )
    fig.update_yaxes(
        gridcolor=GRID,
        gridwidth=1,
        linecolor='rgba(0,0,0,0)',
        tickfont=dict(size=TICK_FONT, color=SUBTEXT),
        title_font=dict(size=AXIS_TITLE_FONT, color=SUBTEXT),
        zeroline=False,
    )
    return fig


def make_empty_figure(message, title='Không có dữ liệu'):
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        x=0.5,
        y=0.5,
        xref='paper',
        yref='paper',
        showarrow=False,
        align='center',
        font=dict(size=14, color=SUBTEXT)
    )
    _theme(
        fig,
        height=340,
        title=_chart_title(title),
        showlegend=False,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        margin=dict(l=16, r=16, t=72, b=16)
    )
    return fig


# ══════════════════════════════════════════════════════════════
#  CHART BUILDERS
# ══════════════════════════════════════════════════════════════

def make_review_ratio_box(df=None):
    if df is None:
        df = df_review

    if len(df) == 0:
        return make_empty_figure('Không có dữ liệu review phù hợp với bộ lọc hiện tại.', 'Phân bố Review Ratio theo Xuất xứ')

    q95 = df['review_ratio'].quantile(0.95)
    df_lim = df[df['review_ratio'] <= q95].copy()
    if len(df_lim) == 0:
        df_lim = df.copy()

    fig = go.Figure()

    for origin, color in zip(order_list, [C_DOMESTIC, C_IMPORT]):
        data = df_lim[df_lim['origin_group'] == origin]['review_ratio']
        if len(data) > 0:
            group_size = len(df[df['origin_group'] == origin])
            fig.add_trace(go.Box(
                y=data,
                name=origin,
                marker=dict(color=color),
                line=dict(color=color),
                fillcolor=color.replace('rgb', 'rgba').replace(')', ',0.25)') if isinstance(color, str) and color.startswith('rgb(') else None,
                boxmean=True,
                customdata=np.column_stack([
                    np.repeat(group_size, len(data))
                ]),
                hovertemplate=(
                    '<b>%{fullData.name}</b><br>'
                    'Review Ratio: %{y:.3f}<br>'
                    'Số sản phẩm trong nhóm: %{customdata[0]:.0f}<extra></extra>'
                ),
            ))

    _theme(
        fig,
        height=350,
        title=_chart_title('Phân bố Review Ratio theo Xuất xứ'),
        yaxis=dict(title='Review Ratio', showgrid=True, gridcolor=GRID),
        showlegend=False,
        margin=dict(l=16, r=16, t=78, b=16)
    )
    return fig


def make_engagement_stacked(df=None):
    if df is None:
        df = df_review

    if len(df) == 0:
        return make_empty_figure('Không có dữ liệu engagement phù hợp với bộ lọc hiện tại.', 'Phân bố Mức độ Engagement (%)')

    df_eng = compute_engagement_level(df)
    eng_by_origin = pd.crosstab(df_eng['origin_group'], df_eng['engagement_level'])
    eng_by_origin = eng_by_origin.reindex(index=order_list, columns=ENGAGEMENT_ORDER, fill_value=0)

    row_totals = eng_by_origin.sum(axis=1).replace(0, np.nan)
    eng_pct = eng_by_origin.div(row_totals, axis=0).fillna(0) * 100

    fig = go.Figure()
    colors_eng = ['#FEE2E2', '#FCA5A5', '#F59E0B', '#78350F']

    for i, label in enumerate(ENGAGEMENT_ORDER):
        counts = eng_by_origin[label].reindex(order_list).fillna(0)
        percents = eng_pct[label].reindex(order_list).fillna(0)
        fig.add_trace(go.Bar(
            name=label,
            x=order_list,
            y=percents,
            marker=dict(color=colors_eng[i]),
            customdata=np.column_stack([counts.values]),
            hovertemplate=(
                '<b>%{x}</b><br>'
                f'{label}: %{{y:.1f}}%<br>'
                'Số sản phẩm: %{customdata[0]:.0f}<extra></extra>'
            ),
            text=[f'{v:.0f}%' if v >= 8 else '' for v in percents],
            textposition='inside',
            textfont=dict(size=11, color='white'),
        ))

    _theme(
        fig,
        height=350,
        barmode='stack',
        title=_chart_title('Phân bố Mức độ Engagement (%)'),
        xaxis=dict(showgrid=False, tickfont=dict(size=12)),
        yaxis=dict(title='Tỷ trọng (%)', showgrid=True, gridcolor=GRID, range=[0, 100]),
        legend=dict(
            orientation='h',
            yanchor='top',
            y=1.35,
            xanchor='center',
            x=0.5,
            font=dict(size=11),
            entrywidthmode='fraction',
            entrywidth=0.24,
            traceorder='normal'
        ),
        margin=dict(l=16, r=16, t=125, b=16)
    )
    return fig


def make_category_bar(df=None):
    if df is None:
        df = df_review

    shared_stats = get_shared_top_category_stats(df)
    if shared_stats.empty:
        return make_empty_figure('Không có category phù hợp với bộ lọc hiện tại.', 'Review Ratio theo Dòng Sản Phẩm')

    cat_overall = shared_stats.sort_values('avg_review_ratio', ascending=True)
    avg_all = df['review_ratio'].mean()

    colors = [A_500 if v >= avg_all else PURPLE for v in cat_overall['avg_review_ratio'].values]

    customdata = np.column_stack([
        cat_overall['product_count'].values,
        cat_overall['total_reviews'].values
    ])

    fig = go.Figure(go.Bar(
        x=cat_overall['avg_review_ratio'].values,
        y=cat_overall.index,
        orientation='h',
        marker=dict(color=colors, line=dict(color='white', width=0.5)),
        text=[f'{v:.3f}' for v in cat_overall['avg_review_ratio'].values],
        textposition='outside',
        textfont=dict(size=11, color=TEXT),
        customdata=customdata,
        hovertemplate=(
            '<b>%{y}</b><br>'
            'Review Ratio TB: %{x:.3f}<br>'
            'Số sản phẩm: %{customdata[0]:.0f}<br>'
            'Tổng reviews: %{customdata[1]:.0f}<extra></extra>'
        ),
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

    _theme(
        fig,
        height=460,
        showlegend=False,
        title=_chart_title(f'Review Ratio theo Dòng Sản Phẩm Phổ Biến (Top {len(cat_overall)})'),
        xaxis=dict(title='Review Ratio trung bình', showgrid=True, gridcolor=GRID, tickfont=dict(size=11)),
        yaxis=dict(showgrid=False, tickfont=dict(size=11)),
        margin=dict(l=150, r=90, t=78, b=16)
    )
    return fig


def make_noi_vs_ngoai_comparison(df=None):
    """Horizontal grouped bar: So sánh Review Ratio Nội vs Ngoại theo cùng tập top categories"""
    if df is None:
        df = df_review

    shared_stats = get_shared_top_category_stats(df)
    if shared_stats.empty:
        return make_empty_figure('Không có category để so sánh theo xuất xứ.', 'So Sánh Review Ratio Nội vs Ngoại')

    top_cats = shared_stats.index.tolist()
    df_top = df[df['primary_category'].isin(top_cats)].copy()
    comparison = (
        df_top.groupby(['primary_category', 'origin_group'])['review_ratio']
        .mean()
        .unstack(fill_value=0)
        .reindex(top_cats)
    )

    sort_series = comparison['Ngoại'] if 'Ngoại' in comparison.columns else comparison.mean(axis=1)
    ordered_cats = sort_series.sort_values(ascending=True).index.tolist()
    comparison = comparison.reindex(ordered_cats)
    ordered_stats = shared_stats.reindex(ordered_cats)

    fig = go.Figure()

    for origin, color in zip(order_list, [C_DOMESTIC, C_IMPORT]):
        if origin in comparison.columns:
            fig.add_trace(go.Bar(
                x=comparison[origin].fillna(0).values,
                y=ordered_cats,
                name=origin,
                orientation='h',
                marker=dict(color=color),
                text=[f'{v:.3f}' if v > 0 else '' for v in comparison[origin].fillna(0).values],
                textposition='outside',
                textfont=dict(size=13, color=TEXT),
                customdata=np.column_stack([
                    ordered_stats['product_count'].values,
                    ordered_stats['total_reviews'].values
                ]),
                hovertemplate=(
                    '<b>%{y}</b><br>'
                    f'{origin}: %{{x:.3f}}<br>'
                    'Số sản phẩm: %{customdata[0]:.0f}<br>'
                    'Tổng reviews: %{customdata[1]:.0f}<extra></extra>'
                ),
                cliponaxis=False,
            ))

    _theme(
        fig,
        height=520,
        barmode='group',
        title=_chart_title(f'So Sánh Review Ratio Nội vs Ngoại (Top {len(ordered_cats)})'),
        xaxis=dict(title='Review Ratio trung bình', showgrid=True, gridcolor=GRID),
        yaxis=dict(showgrid=False, tickfont=dict(size=13)),
        legend=dict(
            orientation='h',
            yanchor='top',
            y=1.3,
            xanchor='right',
            x=1
        ),
        margin=dict(l=180, r=100, t=120, b=20)
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
            kpi_card('🏆', format_best_category_label(best_cat), 'Dòng nổi bật', '#F43F5E', 'rgba(244, 63, 94, 0.14)'),
        ], className='p2-kpi-row'),

        # ── MAIN CHARTS: 2x2 Layout ──────────────────
        html.Div([
            html.Div(dcc.Graph(id='p2-chart-box',
                               config={'displayModeBar': False}),
                     className='p2-card', style={'flex': '1', 'minWidth': '340px'}),

            html.Div(dcc.Graph(id='p2-chart-stacked',
                               config={'displayModeBar': False}),
                     className='p2-card', style={'flex': '1', 'minWidth': '340px'}),
        ], className='p2-row'),

        html.Div([
            html.Div(
                dcc.Graph(
                    id='p2-chart-category',
                    config={'displayModeBar': False}
                ),
                className='p2-card',
                style={'width': '100%'}
            ),
        ], className='p2-row'),

        html.Div([
            html.Div(
                dcc.Graph(
                    id='p2-chart-comparison',
                    config={'displayModeBar': False}
                ),
                className='p2-card',
                style={'width': '100%'}
            ),
        ], className='p2-row'),

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
    product_types = [product_type] if product_type != 'all' else None
    price_segments = [price_segment] if price_segment != 'all' else None

    # Bộ biểu đồ dùng logic chung theo notebook: lọc theo ngành + giá, không triệt tiêu so sánh xuất xứ
    chart_df = apply_filters(
        df_review,
        product_types=product_types,
        price_segments=price_segments,
        origin='all'
    )

    # KPI vẫn cho phép focus theo xuất xứ được chọn
    kpi_df = apply_filters(
        df_review,
        product_types=product_types,
        price_segments=price_segments,
        origin=origin
    )

    kpi_data = compute_dynamic_kpi(kpi_df)
    kpi_children = [
        kpi_card('⭐', f"{kpi_data['avg_review_ratio']:.3f}", 'Review Ratio TB', '#F59E0B', 'rgba(245, 158, 11, 0.14)'),
        kpi_card('🔥', f"{kpi_data['high_engagement_pct']:.1f}%", 'Engagement cao', '#06B6D4', 'rgba(6, 182, 212, 0.14)'),
        kpi_card('💬', f"{int(kpi_data['total_reviews']):,}", 'Tổng reviews', '#10B981', 'rgba(16, 185, 129, 0.14)'),
        kpi_card('🏆', format_best_category_label(kpi_data['best_category']), 'Dòng nổi bật', '#F43F5E', 'rgba(244, 63, 94, 0.14)'),
    ]

    return (
        make_review_ratio_box(chart_df),
        make_engagement_stacked(chart_df),
        make_category_bar(chart_df),
        make_noi_vs_ngoai_comparison(chart_df),
        kpi_children
    )
