# pages/page4_gia_ca.py
# ╔══════════════════════════════════════════════════════════════╗
# ║  Trang 4 — Phân khúc giá & Chiến lược khuyến mãi            ║
# ║  Focus: Mật độ lượt bán + Hiệu quả giảm giá                 ║
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
    path='/gia-ca',
    name='Giá cả',
    title='Giá cả | Mỹ phẩm Tiki',
    order=4,
)

# ── Load data ────────────────────────────────────────────────
df_full, df_vn_full, df_nn_full = load_data()

PRICE_ORDER   = ['Dưới 100k', '100k – 300k', '300k – 700k', '700k – 2tr', 'Trên 2tr']
PRICE_LABELS  = ['< 100k', '100k–300k', '300k–700k', '700k–2tr', '> 2tr']

# ── Design tokens ────────────────────────────────────────────
TEAL_900    = '#0D4F52'   # Dark teal for hero
TEAL_700    = '#0F766E'
TEAL_500    = '#14B8A6'
TEAL_300    = '#7EE8D6'
TEAL_50     = 'rgba(20,184,166,0.10)'

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
#  DATA PROCESSING
# ══════════════════════════════════════════════════════════════

def prepare_price_analysis():
    """Tính toán dữ liệu cho phân tích phân khúc giá"""
    # IQR capping để loại outlier
    def iqr_cap(s):
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        return s.clip(upper=q3 + 1.5 * (q3 - q1))
    
    df_full['sold_capped'] = (df_full.groupby(['price_segment', 'origin_class_corrected'],
                                                observed=True)['sold_count']
                              .transform(iqr_cap))
    
    # Median sold count theo phân khúc & nguồn gốc
    med = (df_full.groupby(['price_segment', 'origin_class_corrected'], observed=True)
           ['sold_count'].median().reset_index())
    med.columns = ['segment', 'origin', 'median_sold']
    
    med_vn = med[med['origin'] == 'Trong nước'].set_index('segment').reindex(PRICE_ORDER)['median_sold'].fillna(0).values
    med_nn = med[med['origin'] == 'Ngoài nước'].set_index('segment').reindex(PRICE_ORDER)['median_sold'].fillna(0).values
    
    # Hit rate (% sản phẩm có sold_count > 0)
    def hit_rate(s):
        return (s > 0).mean() * 100
    
    hr = (df_full.groupby(['price_segment', 'origin_class_corrected'], observed=True)
          ['sold_count'].agg(hit_rate).reset_index())
    hr.columns = ['segment', 'origin', 'hit_rate']
    
    hr_vn = hr[hr['origin'] == 'Trong nước'].set_index('segment').reindex(PRICE_ORDER)['hit_rate'].fillna(0).values
    hr_nn = hr[hr['origin'] == 'Ngoài nước'].set_index('segment').reindex(PRICE_ORDER)['hit_rate'].fillna(0).values
    
    return {
        'med_vn': med_vn, 'med_nn': med_nn,
        'hr_vn': hr_vn, 'hr_nn': hr_nn,
    }


def prepare_discount_analysis():
    """Tính toán dữ liệu cho phân tích giảm giá"""
    df_full['disc_bin'] = df_full['discount_rate'].apply(
        lambda x: 'Giảm > 30%' if x > 30 else 'Giảm ≤ 30%'
    )
    
    # Median sold theo segment × origin × discount bin
    med = (df_full.groupby(['price_segment', 'origin_class_corrected', 'disc_bin'], observed=True)
           ['sold_count'].median().reset_index())
    med.columns = ['segment', 'origin', 'disc_bin', 'median_sold']
    
    def get_med(origin, disc):
        return (med[(med['origin'] == origin) & (med['disc_bin'] == disc)]
                .set_index('segment').reindex(PRICE_ORDER)['median_sold']
                .fillna(0).values)
    
    vn_lo = get_med('Trong nước', 'Giảm ≤ 30%')
    vn_hi = get_med('Trong nước', 'Giảm > 30%')
    nn_lo = get_med('Ngoài nước', 'Giảm ≤ 30%')
    nn_hi = get_med('Ngoài nước', 'Giảm > 30%')
    
    # % sản phẩm với discount > 30%
    disc_pct = (df_full.groupby(['price_segment', 'origin_class_corrected'], observed=True)
                .apply(lambda g: (g['discount_rate'] > 30).mean() * 100)
                .reset_index())
    disc_pct.columns = ['segment', 'origin', 'pct_deep_disc']
    
    dp_vn = disc_pct[disc_pct['origin'] == 'Trong nước'].set_index('segment').reindex(PRICE_ORDER)['pct_deep_disc'].fillna(0).values
    dp_nn = disc_pct[disc_pct['origin'] == 'Ngoài nước'].set_index('segment').reindex(PRICE_ORDER)['pct_deep_disc'].fillna(0).values
    
    return {
        'vn_lo': vn_lo, 'vn_hi': vn_hi,
        'nn_lo': nn_lo, 'nn_hi': nn_hi,
        'dp_vn': dp_vn, 'dp_nn': dp_nn,
    }


# ══════════════════════════════════════════════════════════════
#  CHART BUILDERS
# ══════════════════════════════════════════════════════════════

def _theme(fig, height=320, **kw):
    """Apply consistent theme to all charts"""
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


def make_median_price_chart():
    """Biểu đồ trung vị lượt bán theo phân khúc giá"""
    data = prepare_price_analysis()
    med_vn, med_nn = data['med_vn'], data['med_nn']
    
    x = PRICE_LABELS
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='Trong nước', x=x, y=med_vn,
        marker_color=C_DOMESTIC,
        marker_line=dict(color='white', width=0.8),
        hovertemplate='<b>%{x}</b><br>Trong nước: %{y:.0f} sản phẩm<extra></extra>',
    ))
    fig.add_trace(go.Bar(
        name='Ngoài nước', x=x, y=med_nn,
        marker_color=C_IMPORT,
        marker_line=dict(color='white', width=0.8),
        hovertemplate='<b>%{x}</b><br>Ngoài nước: %{y:.0f} sản phẩm<extra></extra>',
    ))
    
    _theme(fig, height=320,
           barmode='group',
           title=dict(text='<b>Trung vị lượt bán theo phân khúc giá</b>', 
                      x=0.5, xanchor='center', font=dict(size=14, color=TEXT)),
           xaxis=dict(showgrid=False, tickfont=dict(size=11)),
           yaxis=dict(title='Trung vị lượt bán', showgrid=True,
                      gridcolor=GRID, tickfont=dict(size=11), 
                      title_font=dict(size=12, color=SUBTEXT)),
           legend=dict(orientation='h', yanchor='bottom', y=1.02,
                       xanchor='center', x=0.5, font=dict(size=11))
    )
    return fig


def make_hitrate_chart():
    """Biểu đồ hit rate (% sản phẩm có lượt bán > 0)"""
    data = prepare_price_analysis()
    hr_vn, hr_nn = data['hr_vn'], data['hr_nn']
    
    x = PRICE_LABELS
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='Trong nước', x=x, y=hr_vn,
        marker_color=C_DOMESTIC,
        marker_line=dict(color='white', width=0.8),
        hovertemplate='<b>%{x}</b><br>Trong nước: %{y:.1f}%<extra></extra>',
    ))
    fig.add_trace(go.Bar(
        name='Ngoài nước', x=x, y=hr_nn,
        marker_color=C_IMPORT,
        marker_line=dict(color='white', width=0.8),
        hovertemplate='<b>%{x}</b><br>Ngoài nước: %{y:.1f}%<extra></extra>',
    ))
    
    _theme(fig, height=320,
           barmode='group',
           title=dict(text='<b>Hit rate: Tỉ lệ sản phẩm có lượt bán > 0</b>', 
                      x=0.5, xanchor='center', font=dict(size=14, color=TEXT)),
           xaxis=dict(showgrid=False, tickfont=dict(size=11)),
           yaxis=dict(title='Tỉ lệ (%)', showgrid=True,
                      gridcolor=GRID, tickfont=dict(size=11), 
                      title_font=dict(size=12, color=SUBTEXT),
                      range=[0, 100]),
           legend=dict(orientation='h', yanchor='bottom', y=1.02,
                       xanchor='center', x=0.5, font=dict(size=11))
    )
    return fig


def make_discount_comparison_chart():
    """Biểu đồ so sánh hiệu quả giảm giá"""
    data = prepare_discount_analysis()
    vn_lo, vn_hi = data['vn_lo'], data['vn_hi']
    nn_lo, nn_hi = data['nn_lo'], data['nn_hi']
    
    x = PRICE_LABELS
    
    fig = go.Figure()
    
    # 4 bars per position
    fig.add_trace(go.Bar(
        name='Nội ≤ 30%', x=x, y=vn_lo,
        marker_color='#93C5FD',
        marker_line=dict(color='white', width=0.6),
        hovertemplate='<b>%{x}</b><br>Nội ≤ 30%: %{y:.0f}<extra></extra>',
    ))
    fig.add_trace(go.Bar(
        name='Nội > 30%', x=x, y=vn_hi,
        marker_color=C_DOMESTIC,
        marker_line=dict(color='white', width=0.6),
        hovertemplate='<b>%{x}</b><br>Nội > 30%: %{y:.0f}<extra></extra>',
    ))
    fig.add_trace(go.Bar(
        name='Ngoài ≤ 30%', x=x, y=nn_lo,
        marker_color='#FCA5A5',
        marker_line=dict(color='white', width=0.6),
        hovertemplate='<b>%{x}</b><br>Ngoài ≤ 30%: %{y:.0f}<extra></extra>',
    ))
    fig.add_trace(go.Bar(
        name='Ngoài > 30%', x=x, y=nn_hi,
        marker_color=C_IMPORT,
        marker_line=dict(color='white', width=0.6),
        hovertemplate='<b>%{x}</b><br>Ngoài > 30%: %{y:.0f}<extra></extra>',
    ))
    
    _theme(fig, height=340,
           barmode='group',
           title=dict(text='<b>Lượt bán theo mức giảm giá & nguồn gốc</b>', 
                      x=0.5, xanchor='center', font=dict(size=14, color=TEXT)),
           xaxis=dict(showgrid=False, tickfont=dict(size=11)),
           yaxis=dict(title='Trung vị lượt bán', showgrid=True,
                      gridcolor=GRID, tickfont=dict(size=11), 
                      title_font=dict(size=12, color=SUBTEXT)),
           legend=dict(orientation='h', yanchor='bottom', y=1.02,
                       xanchor='center', x=0.5, font=dict(size=10))
    )
    return fig


def make_discount_penetration_chart():
    """Biểu đồ tỉ lệ sản phẩm áp dụng giảm giá sâu"""
    data = prepare_discount_analysis()
    dp_vn, dp_nn = data['dp_vn'], data['dp_nn']
    
    x = PRICE_LABELS
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='Trong nước', x=x, y=dp_vn,
        marker_color=C_DOMESTIC,
        marker_line=dict(color='white', width=0.8),
        hovertemplate='<b>%{x}</b><br>Trong nước: %{y:.1f}%<extra></extra>',
    ))
    fig.add_trace(go.Bar(
        name='Ngoài nước', x=x, y=dp_nn,
        marker_color=C_IMPORT,
        marker_line=dict(color='white', width=0.8),
        hovertemplate='<b>%{x}</b><br>Ngoài nước: %{y:.1f}%<extra></extra>',
    ))
    
    _theme(fig, height=320,
           barmode='group',
           title=dict(text='<b>Tỉ lệ sản phẩm áp dụng giảm giá sâu (> 30%)</b>', 
                      x=0.5, xanchor='center', font=dict(size=14, color=TEXT)),
           xaxis=dict(showgrid=False, tickfont=dict(size=11)),
           yaxis=dict(title='Tỉ lệ (%)', showgrid=True,
                      gridcolor=GRID, tickfont=dict(size=11), 
                      title_font=dict(size=12, color=SUBTEXT)),
           legend=dict(orientation='h', yanchor='bottom', y=1.02,
                       xanchor='center', x=0.5, font=dict(size=11))
    )
    return fig


# ══════════════════════════════════════════════════════════════
#  LAYOUT COMPONENTS
# ══════════════════════════════════════════════════════════════

def section_header(num, title, desc, variant='muc1'):
    icons = {'muc1': '📊', 'muc2': '🏪'}
    return html.Div([
        html.Div([
            html.Span(icons.get(variant, '📊'), style={'fontSize': '17px'}),
        ], className='p4-section-num'),
        html.Div([
            html.P(f'Mục tiêu {num}', style={
                'margin': '0 0 2px 0',
                'fontSize': '10px', 'fontWeight': '700',
                'color': TEAL_500 if variant == 'muc1' else GOLD,
                'textTransform': 'uppercase', 'letterSpacing': '0.1em',
            }),
            html.H3(title, className='p4-section-title'),
            html.P(desc, className='p4-section-desc'),
        ]),
    ], className=f'p4-section-header {variant}')


def chart_card(icon, subtitle, children, icon_bg=None):
    if icon_bg is None:
        icon_bg = TEAL_50
    return html.Div([
        html.Div([
            html.Div(icon, className='p4-card-icon', style={'background': icon_bg}),
            html.Div([
                html.P(subtitle, className='p4-card-subtitle'),
            ]),
        ], className='p4-card-header'),
        children,
    ], className='p4-card')


def insight_box(text, variant='teal'):
    icons = {'teal': '💡', 'gold': '📌'}
    return html.Div([
        html.Span(icons.get(variant, '💡'), style={'fontSize': '16px', 'flexShrink': '0'}),
        html.Span(text, style={'fontSize': '12px', 'lineHeight': '1.5'}),
    ], className=f'p4-insight {variant}')


# Build static charts
chart_median = make_median_price_chart()
chart_hitrate = make_hitrate_chart()
chart_discount_comp = make_discount_comparison_chart()
chart_discount_pen = make_discount_penetration_chart()


# ══════════════════════════════════════════════════════════════
#  PAGE LAYOUT
# ══════════════════════════════════════════════════════════════

layout = html.Div([
    html.Div(className='p4-page', children=[
        # HERO HEADER
        html.Div([
            html.Span('🎯', style={'fontSize': '20px'}),
            html.Div([
                html.P('PHÂN KHÚC GIÁ & CHIẾN LƯỢC KHUYẾN MÃI', className='p4-hero-title'),
                html.P('Phân tích lượt bán, mật độ tiêu thụ và hiệu quả chiến dịch giảm giá', 
                       className='p4-hero-sub'),
            ]),
        ], className='p4-hero'),
        
        # MỤC TIÊU 1: Phân tích phân khúc giá
        section_header(
            1,
            'Xác định Sweet Spot theo phân khúc giá',
            'Phân tích trung vị lượt bán và tỉ lệ sản phẩm có đơn hàng (hit rate) để xác định phân khúc giá tối ưu cho mỹ phẩm nội địa so với ngoại nhập',
            variant='muc1'
        ),
        
        html.Div([
            chart_card('📈', 'Trung vị lượt bán',
                      dcc.Graph(figure=chart_median, config={'displayModeBar': False}),
                      icon_bg=TEAL_50),
            chart_card('🎯', 'Hit rate - Tỉ lệ sản phẩm bán được',
                      dcc.Graph(figure=chart_hitrate, config={'displayModeBar': False}),
                      icon_bg='rgba(52,211,153,0.10)'),
        ], className='p4-card-row'),
        
        html.Div([
            insight_box(
                '💡 Sweet Spot tại phân khúc "Dưới 100k" với trung vị lượt bán 6-8 sản phẩm. '
                'Phân khúc 700k+ có trung vị bằng 0 cho cả hàng nội lẫn ngoại.',
                variant='teal'
            ),
            insight_box(
                '📌 Hàng ngoại duy trì hit rate > 50% đến phân khúc 300k-700k, '
                'trong khi hàng nội địa sụt giảm mạnh khi vượt qua 100k.',
                variant='gold'
            ),
        ], className='p4-insights-row'),
        
        # MỤC TIÊU 2: Hiệu quả giảm giá
        section_header(
            2,
            'Đánh giá hiệu quả chiến lược giảm giá sâu (> 30%)',
            'So sánh lượt bán và tỉ lệ áp dụng chiến dịch giảm giá sâu giữa hàng nội và ngoại để xác định ROI của khoản đầu tư khuyến mãi',
            variant='muc2'
        ),
        
        html.Div([
            chart_card('💰', 'Lượt bán theo mức giảm giá',
                      dcc.Graph(figure=chart_discount_comp, config={'displayModeBar': False}),
                      icon_bg=TEAL_50),
            chart_card('📊', 'Tỉ lệ sản phẩm áp dụng giảm giá sâu',
                      dcc.Graph(figure=chart_discount_pen, config={'displayModeBar': False}),
                      icon_bg='rgba(245,158,11,0.10)'),
        ], className='p4-card-row'),
        
        html.Div([
            insight_box(
                '💡 Tại phân khúc "Dưới 100k", hàng nội địa giảm lượt bán khi discount tăng '
                '(từ 7 xuống 4 sản phẩm), trong khi hàng ngoại tăng trưởng (7 lên 9). '
                'Giảm giá sâu không hiệu quả cho hàng nội ở phân khúc này.',
                variant='teal'
            ),
            insight_box(
                '📌 Tỉ lệ áp dụng giảm giá sâu cao nhất ở phân khúc rẻ nhất (Dưới 100k). '
                'Hàng ngoại có chiến lược khuyến mãi ổn định hơn khi vươn lên các phân khúc cao hơn.',
                variant='gold'
            ),
        ], className='p4-insights-row'),
        
        # KẾT LUẬN
        html.Div([
            html.H3('🎓 Kết luận & Khuyến nghị', className='p4-conclusion-title'),
            html.Div([
                html.P(
                    '✓ Sweet Spot thị trường tập trung tại "Dưới 100k" cho cả hai nhóm nguồn gốc. '
                    'Thương hiệu nội địa nên tập trung định vị sản phẩm ở mức giá này để tối đa hóa lượt bán và doanh số.',
                    style={'marginBottom': '10px'}
                ),
                html.P(
                    '✓ Chiến lược giảm giá sâu (> 30%) không hiệu quả cho hàng nội địa, nhất là tại phân khúc đầu vào. '
                    'Nên xem xét các công cụ marketing khác (xây dựng thương hiệu, review, influencer) thay vì chỉ rely vào giảm giá.',
                    style={'marginBottom': '10px'}
                ),
                html.P(
                    '✓ Phân khúc 300k+ là rào cản lớn cho hàng nội, độc lập với chiến lược giảm giá. '
                    'Cần xây dựng tín cậy thương hiệu trước khi định vị cao cấp.',
                    style={'marginBottom': '0px'}
                ),
            ], className='p4-conclusion-items'),
        ], className='p4-conclusion'),
        
    ]),
], style={'backgroundColor': PAGE_BG, 'minHeight': '100vh', 'paddingBottom': '30px'})
