# pages/page2_uy_tin.py
# ╔══════════════════════════════════════════════════════════════╗
# ║  Trang 2 — Uy Tín & Trải nghiệm Khách hàng                   ║
# ║  MT1: So sánh Rating (sold > 50) — Box, Violin, Hist, Bar    ║
# ║  MT2: Review Ratio — Box, Stacked, HBar, Grouped             ║
# ║  MT3: Phân khúc "Thất vọng" (Rating < 3.5)                  ║
# ╚══════════════════════════════════════════════════════════════╝

import dash
from dash import html, dcc
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from data_loader import load_data
from theme import C_DOMESTIC, C_IMPORT, C_NEUTRAL, C_TEXT, C_SUBTEXT, C_BG, C_BORDER

dash.register_page(
    __name__,
    path='/uy-tin',
    name='Uy tín',
    title='Uy tín | Mỹ phẩm Tiki',
    order=2,
)

# ── Load & prepare ───────────────────────────────────────────
df_full, df_vn_full, df_nn_full = load_data()

PRICE_ORDER = ['Dưới 100k', '100k – 300k', '300k – 700k', '700k – 2tr', 'Trên 2tr']

# origin_group: Nội / Ngoại mapping
df_full['origin_group'] = df_full['origin_class_corrected'].map({
    'Trong nước': 'Nội',
    'Ngoài nước': 'Ngoại',
})

# ── Design tokens ────────────────────────────────────────────
R700    = '#BE185D'
R600    = '#DB2777'
R400    = '#F472B6'
R100    = '#FCE7F3'
R50     = '#FDF2F8'
AMBER   = '#F59E0B'
TEAL    = '#0D9488'
DANGER  = '#DC2626'
GREEN   = '#16A34A'
SURFACE = '#FFFFFF'
BORDER  = '#F0D6E4'
TEXT    = '#1A0A12'
SUBTEXT = '#78373E'
GRID    = '#FBE8F0'


# ══════════════════════════════════════════════════════════════
#  SHARED THEME HELPER
# ══════════════════════════════════════════════════════════════

def _t(fig, height=300, **kw):
    margin = kw.pop('margin', dict(l=16, r=16, t=36, b=16))
    xaxis = kw.pop('xaxis', {})
    yaxis = kw.pop('yaxis', {})
    legend = kw.pop('legend', {})

    fig.update_layout(
        height=height,
        font=dict(family="'Sora','Segoe UI',sans-serif", size=12, color=TEXT),
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        margin=margin,
        hoverlabel=dict(
            bgcolor='white',
            bordercolor=BORDER,
            font=dict(size=12, color=TEXT)
        ),
        xaxis=_xax(**xaxis),
        yaxis=_yax(**yaxis),
        legend=_legend(**legend),
        **kw,
    )
    return fig


def _xax(**kw):
    title = kw.pop('title', None)

    axis = dict(
        gridcolor=GRID,
        gridwidth=1,
        linecolor=BORDER,
        linewidth=1,
        tickfont=dict(size=10, color=SUBTEXT),
        title=dict(font=dict(size=11, color=SUBTEXT)),
        zeroline=False,
        showgrid=True,
    )

    if title is not None:
        axis['title'] = dict(text=title, font=dict(size=11, color=SUBTEXT))

    axis.update(kw)
    return axis


def _yax(**kw):
    title = kw.pop('title', None)

    axis = dict(
        gridcolor=GRID,
        gridwidth=1,
        linecolor='rgba(0,0,0,0)',
        tickfont=dict(size=10, color=SUBTEXT),
        title=dict(font=dict(size=11, color=SUBTEXT)),
        zeroline=False,
        showgrid=True,
    )

    if title is not None:
        axis['title'] = dict(text=title, font=dict(size=11, color=SUBTEXT))

    axis.update(kw)
    return axis


def _legend(**kw):
    axis = dict(
        orientation='h',
        yanchor='bottom',
        y=1.02,
        xanchor='right',
        x=1,
        font=dict(size=11),
        bgcolor='rgba(255,255,255,0)',
    )
    axis.update(kw)
    return axis


# ══════════════════════════════════════════════════════════════
#  MT1  —  Rating Comparison (sold > 50)
# ══════════════════════════════════════════════════════════════

def _df_mt1():
    return df_full[(df_full['sold_count'] > 50) & (df_full['rating'] > 0)].copy()


def _hex_to_rgba(hex_color, alpha=0.12):
    hex_color = hex_color.lstrip('#')
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return f'rgba({r}, {g}, {b}, {alpha})'


def make_mt1_box():
    df = _df_mt1()
    fig = go.Figure()

    for grp, color, name in [('Nội', C_DOMESTIC, 'Trong nước'),
                             ('Ngoại', C_IMPORT, 'Ngoài nước')]:
        sub = df[df['origin_group'] == grp]['rating'].dropna()

        fig.add_trace(go.Box(
            y=sub,
            name=name,
            marker_color=color,
            line_color=color,
            fillcolor=_hex_to_rgba(color, 0.12),
            boxpoints='outliers',
            jitter=0.25,
            pointpos=0,
            marker=dict(size=4, opacity=0.5, color=color),
            hovertemplate=f'<b>{name}</b><br>Rating: %{{y:.2f}}<extra></extra>',
        ))

        mean_val = sub.mean()
        fig.add_annotation(
            x=name,
            y=mean_val + 0.12,
            text=f'TB: {mean_val:.2f}',
            showarrow=False,
            font=dict(size=9, color='#1A0A12'),
            bgcolor='#FEF08A',
            bordercolor='#D4A017',
            borderwidth=1,
            borderpad=3,
        )

    _t(
        fig,
        height=320,
        showlegend=False,
        yaxis={'title': 'Rating (điểm)', 'range': [0.8, 5.5]},
        xaxis={'showgrid': False}
    )
    return fig


def make_mt1_violin():
    """Violin plot: mật độ phân bố rating."""
    df = _df_mt1()
    fig = go.Figure()

    for grp, color, name in [
        ('Nội', C_DOMESTIC, 'Trong nước'),
        ('Ngoại', C_IMPORT, 'Ngoài nước')
    ]:
        sub = df[df['origin_group'] == grp]['rating'].dropna()

        fig.add_trace(go.Violin(
            y=sub,
            name=name,
            line_color=color,
            fillcolor=_hex_to_rgba(color, 0.20),
            box_visible=True,
            meanline_visible=True,
            points=False,
            hovertemplate=f'<b>{name}</b><br>Rating: %{{y:.2f}}<extra></extra>',
        ))

    _t(
        fig,
        height=320,
        showlegend=False,
        violinmode='group',
        yaxis={'title': 'Rating (điểm)'},
        xaxis={'showgrid': False}
    )
    return fig


def make_mt1_hist():
    """Histogram + KDE overlay."""
    df = _df_mt1()
    fig = go.Figure()

    for grp, color, name in [('Nội', C_DOMESTIC, 'Trong nước'),
                             ('Ngoại', C_IMPORT, 'Ngoài nước')]:
        sub = df[df['origin_group'] == grp]['rating'].dropna()

        if len(sub) == 0:
            continue

        fig.add_trace(go.Histogram(
            x=sub,
            name=name,
            marker_color=_hex_to_rgba(color, 0.75),
            marker_line=dict(color='white', width=0.7),
            nbinsx=22,
            opacity=0.82,
            hovertemplate=f'<b>{name}</b><br>Rating: %{{x:.1f}}<br>Số SP: %{{y}}<extra></extra>',
        ))

        # KDE approximation using scatter
        if len(sub) > 1 and sub.nunique() > 1:
            from scipy.stats import gaussian_kde

            kde = gaussian_kde(sub)
            x_kde = np.linspace(sub.min(), sub.max(), 200)
            y_kde = kde(x_kde)

            hist_max = np.histogram(sub, bins=22)[0].max()
            if y_kde.max() > 0:
                scale = hist_max / y_kde.max()

                fig.add_trace(go.Scatter(
                    x=x_kde,
                    y=y_kde * scale,
                    mode='lines',
                    name=f'KDE {name}',
                    line=dict(color=color, width=2.5),
                    showlegend=False,
                    hoverinfo='skip',
                ))

    _t(
        fig,
        height=320,
        barmode='overlay',
        xaxis={'title': 'Rating (điểm)'},
        yaxis={'title': 'Số lượng sản phẩm'},
        legend={}
    )
    return fig


def make_mt1_stats_bar():
    """Grouped bar: so sánh Mean, Median, Q1, Q3."""
    df = _df_mt1()
    metrics  = ['Trung bình', 'Trung vị', 'Q1 (25%)', 'Q3 (75%)']
    noi_vals  = [
        df[df['origin_group'] == 'Nội']['rating'].mean(),
        df[df['origin_group'] == 'Nội']['rating'].median(),
        df[df['origin_group'] == 'Nội']['rating'].quantile(0.25),
        df[df['origin_group'] == 'Nội']['rating'].quantile(0.75),
    ]
    ngoai_vals = [
        df[df['origin_group'] == 'Ngoại']['rating'].mean(),
        df[df['origin_group'] == 'Ngoại']['rating'].median(),
        df[df['origin_group'] == 'Ngoại']['rating'].quantile(0.25),
        df[df['origin_group'] == 'Ngoại']['rating'].quantile(0.75),
    ]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name='Trong nước', x=metrics, y=noi_vals,
        marker_color=C_DOMESTIC, marker_line=dict(color='white', width=0.8),
        text=[f'{v:.2f}' for v in noi_vals], textposition='outside',
        textfont=dict(size=10, weight=700, color=TEXT),
        hovertemplate='<b>%{x}</b><br>Trong nước: %{y:.2f}<extra></extra>',
    ))
    fig.add_trace(go.Bar(
        name='Ngoài nước', x=metrics, y=ngoai_vals,
        marker_color=C_IMPORT, marker_line=dict(color='white', width=0.8),
        text=[f'{v:.2f}' for v in ngoai_vals], textposition='outside',
        textfont=dict(size=10, weight=700, color=TEXT),
        hovertemplate='<b>%{x}</b><br>Ngoài nước: %{y:.2f}<extra></extra>',
    ))
    _t(
        fig,
        height=320,
        barmode='group',
        xaxis={'showgrid': False},
        yaxis={'title': 'Rating (điểm)', 'range': [4.0, 5.3]},
        legend={}
    )
    return fig


# ── MT1 KPI stats ────────────────────────────────────────────
def _mt1_kpis():
    df = _df_mt1()
    n_noi   = (df['origin_group'] == 'Nội').sum()
    n_ngoai = (df['origin_group'] == 'Ngoại').sum()
    avg_noi   = df[df['origin_group'] == 'Nội']['rating'].mean()
    avg_ngoai = df[df['origin_group'] == 'Ngoại']['rating'].mean()
    return n_noi, n_ngoai, avg_noi, avg_ngoai


# ══════════════════════════════════════════════════════════════
#  MT2  —  Review Ratio
# ══════════════════════════════════════════════════════════════

def _df_mt2():
    df = df_full[df_full['review_ratio'] > 0].copy()
    df['engagement'] = pd.cut(
        df['review_ratio'],
        bins=[0, 0.05, 0.15, 0.30, float('inf')],
        labels=['Thấp (0–5%)', 'Trung bình (5–15%)', 'Cao (15–30%)', 'Rất cao (>30%)'],
    )
    return df


def make_mt2_box():
    """Box plot: review_ratio Nội vs Ngoại (cap 95th pct)."""
    df = _df_mt2().copy()

    cap = df['review_ratio'].dropna().quantile(0.95)
    df_lim = df[df['review_ratio'] <= cap].copy()

    fig = go.Figure()

    for grp, color, name in [
        ('Nội', C_DOMESTIC, 'Trong nước'),
        ('Ngoại', C_IMPORT, 'Ngoài nước')
    ]:
        sub = df_lim[df_lim['origin_group'] == grp]['review_ratio'].dropna()

        fig.add_trace(go.Box(
            y=sub,
            name=name,
            marker_color=color,
            line_color=color,
            fillcolor=_hex_to_rgba(color, 0.12),
            boxpoints='outliers',
            jitter=0.3,
            pointpos=0,
            marker=dict(size=3.5, opacity=0.45, color=color),
            hovertemplate=f'<b>{name}</b><br>Ratio: %{{y:.3f}}<extra></extra>',
        ))

        mean_val = df[df['origin_group'] == grp]['review_ratio'].dropna().mean()
        fig.add_annotation(
            x=name,
            y=cap * 0.87,
            text=f'TB: {mean_val:.3f}',
            showarrow=False,
            font=dict(size=9, color=TEXT),
            bgcolor='#FEF08A',
            bordercolor='#D4A017',
            borderwidth=1,
            borderpad=3,
        )

    _t(
    fig,
    height=310,
    showlegend=False,
    yaxis={'title': 'Review Ratio (≤95th pct)'},
    xaxis={'showgrid': False}
    )   
    return fig


def make_mt2_engagement():
    """Stacked bar: engagement level Nội vs Ngoại."""
    df = _df_mt2().copy()

    levels  = ['Thấp (0–5%)', 'Trung bình (5–15%)', 'Cao (15–30%)', 'Rất cao (>30%)']
    clrs    = ['#FCE7F3', '#F9A8D4', '#DB2777', '#881337']
    origins = ['Nội', 'Ngoại']
    names   = ['Trong nước', 'Ngoài nước']

    if df.empty or 'origin_group' not in df.columns or 'engagement' not in df.columns:
        return go.Figure()

    ct = pd.crosstab(df['origin_group'], df['engagement'])
    ct = ct.reindex(index=origins, columns=levels, fill_value=0)

    row_sum = ct.sum(axis=1).replace(0, np.nan)
    pct = ct.div(row_sum, axis=0).fillna(0) * 100

    fig = go.Figure()

    for lvl, clr in zip(levels, clrs):
        vals = [pct.loc[o, lvl] if o in pct.index else 0 for o in origins]
        txt_color = '#1A0A12' if clr in ('#FCE7F3', '#F9A8D4') else 'white'

        fig.add_trace(go.Bar(
            name=lvl,
            x=names,
            y=vals,
            marker_color=clr,
            marker_line=dict(color='white', width=0.8),
            text=[f'{v:.1f}%' if v >= 5 else '' for v in vals],
            textposition='inside',
            textfont=dict(size=10, color=txt_color),
            insidetextanchor='middle',
            hovertemplate=f'<b>%{{x}}</b><br>{lvl}: %{{y:.1f}}%<extra></extra>',
        ))

    _t(
        fig,
        height=310,
        barmode='stack',
        xaxis={'showgrid': False},
        yaxis={'title': 'Phần trăm (%)', 'range': [0, 105]},
        legend={
            'orientation': 'h',
            'yanchor': 'top',
            'y': -0.15,
            'xanchor': 'center',
            'x': 0.5,
            'font': {'size': 10}
        },
        margin={'l': 16, 'r': 16, 't': 36, 'b': 60}
    )
    return fig


def make_mt2_hbar():
    """Horizontal bar: review_ratio trung bình theo top 12 categories."""
    df = _df_mt2().copy()

    cat_col = 'primary_category' if 'primary_category' in df.columns else 'category'
    if cat_col not in df.columns:
        return go.Figure()

    df = df[[cat_col, 'review_ratio']].dropna()
    if df.empty:
        return go.Figure()

    top_cats = (
        df.groupby(cat_col)['review_ratio']
        .count()
        .sort_values(ascending=False)
        .head(12)
        .index
        .tolist()
    )

    cat_mean = (
        df[df[cat_col].isin(top_cats)]
        .groupby(cat_col)['review_ratio']
        .mean()
        .sort_values(ascending=True)
    )

    if cat_mean.empty:
        return go.Figure()

    overall_mean = df['review_ratio'].mean()
    bar_colors = [TEAL if v >= overall_mean else '#CBD5E1' for v in cat_mean.values]

    fig = go.Figure(go.Bar(
        x=cat_mean.values,
        y=cat_mean.index.tolist(),
        orientation='h',
        marker=dict(color=bar_colors, line=dict(color='white', width=0.6)),
        text=[f'{v:.3f}' for v in cat_mean.values],
        textposition='outside',
        textfont=dict(size=10, color=TEXT),
        hovertemplate='<b>%{y}</b><br>Review Ratio: %{x:.3f}<extra></extra>',
        cliponaxis=False,
    ))

    fig.add_vline(
        x=overall_mean,
        line_dash='dash',
        line_color=DANGER,
        line_width=1.8
    )

    fig.add_annotation(
        x=overall_mean,
        y=cat_mean.index[-1],
        text=f'TB chung: {overall_mean:.3f}',
        showarrow=False,
        xanchor='left',
        font=dict(size=9, color=DANGER),
        bgcolor='rgba(255,255,255,0.85)',
        bordercolor=DANGER,
        borderwidth=1,
        borderpad=3,
    )

    _t(
    fig,
    height=360,
    showlegend=False,
    xaxis={'title': 'Review Ratio (trung bình)'},
    yaxis={'showgrid': False, 'tickfont': dict(size=10.5)},
    margin=dict(l=16, r=60, t=20, b=30)
    )
    return fig


def make_mt2_grouped():
    """Grouped bar: Nội vs Ngoại per category."""
    df = _df_mt2()
    cat_col = 'primary_category' if 'primary_category' in df.columns else 'category'
    if cat_col not in df.columns:
        return go.Figure()

    top_cats = (df.groupby(cat_col)['review_ratio']
                .count().sort_values(ascending=False).head(12).index.tolist())
    pivot = (df[df[cat_col].isin(top_cats)]
             .groupby([cat_col, 'origin_group'])['review_ratio']
             .mean().unstack(fill_value=0))

    cat_order = (pivot.get('Ngoại', pd.Series(dtype=float))
                 .sort_values(ascending=False).index.tolist())
    pivot = pivot.reindex(cat_order)

    fig = go.Figure()
    for grp, color, name in [('Nội', C_DOMESTIC, 'Trong nước'), ('Ngoại', C_IMPORT, 'Ngoài nước')]:
        vals = [pivot.loc[c, grp] if grp in pivot.columns else 0 for c in cat_order]
        fig.add_trace(go.Bar(
            name=name, x=cat_order, y=vals,
            marker_color=color, marker_line=dict(color='white', width=0.7),
            hovertemplate=f'<b>%{{x}}</b><br>{name}: %{{y:.3f}}<extra></extra>',
        ))
    _t(
    fig,
    height=340,
    barmode='group',
    xaxis={
        'showgrid': False,
        'tickangle': -40,
        'tickfont': dict(size=9.5)
    },
    yaxis={'title': 'Review Ratio (trung bình)'},
    legend={},
    margin=dict(l=16, r=16, t=40, b=80)
    )
    return fig


# ── MT2 KPI ─────────────────────────────────────────────────
def _mt2_kpis():
    df = _df_mt2()
    avg_noi   = df[df['origin_group'] == 'Nội']['review_ratio'].mean()
    avg_ngoai = df[df['origin_group'] == 'Ngoại']['review_ratio'].mean()
    eng_noi   = (df[df['origin_group'] == 'Nội']['review_ratio'] > 0.15).mean() * 100
    eng_ngoai = (df[df['origin_group'] == 'Ngoại']['review_ratio'] > 0.15).mean() * 100
    return avg_noi, avg_ngoai, eng_noi, eng_ngoai


# ══════════════════════════════════════════════════════════════
#  MT3  —  Phân khúc "Thất vọng" (Rating < 3.5)
# ══════════════════════════════════════════════════════════════

def _df_mt3():
    df_all  = df_full[df_full['rating'] > 0].copy()
    df_bad  = df_full[(df_full['rating'] > 0) & (df_full['rating'] < 3.5)].copy()
    return df_all, df_bad


def make_mt3_donut(group='Nội'):
    """Donut chart tỉ lệ thất vọng cho 1 nhóm."""
    df_all, df_bad = _df_mt3()
    orig_map = {'Nội': 'Nội', 'Ngoại': 'Ngoại'}
    total   = (df_all['origin_group'] == orig_map[group]).sum()
    bad_cnt = (df_bad['origin_group'] == orig_map[group]).sum()
    ok_cnt  = total - bad_cnt
    pct     = bad_cnt / total * 100 if total > 0 else 0

    color  = C_DOMESTIC if group == 'Nội' else C_IMPORT
    name   = 'Trong nước' if group == 'Nội' else 'Ngoài nước'

    fig = go.Figure(go.Pie(
        values=[ok_cnt, bad_cnt],
        labels=['Hài lòng', 'Thất vọng'],
        marker=dict(
            colors=['#E5E7EB', color],
            line=dict(color='white', width=2.5),
        ),
        hole=0.65,
        textinfo='percent',
        textfont=dict(size=13, weight=700),
        pull=[0, 0.10],
        sort=False,
        hovertemplate='<b>%{label}</b><br>%{value:,} SP (%{percent})<extra></extra>',
    ))
    fig.add_annotation(
        text=f'<b>{pct:.1f}%</b><br><span style="font-size:10px">Thất vọng</span>',
        x=0.5, y=0.5, showarrow=False,
        font=dict(size=16, color=color),
        xref='paper', yref='paper',
    )
    _t(fig, height=280,
       showlegend=True,
       title=dict(text=f'{name} (n={total:,})',
                  font=dict(size=13, color=TEXT, weight=700), x=0.5),
       legend=dict(orientation='v', yanchor='middle', y=0.5,
                   xanchor='left', x=1.02, font=dict(size=11)),
       margin=dict(l=16, r=80, t=40, b=16))
    return fig


def make_mt3_hist():
    """Histogram: phân bố rating trong phân khúc thất vọng."""
    _, df_bad = _df_mt3()
    fig = go.Figure()

    for grp, color, name in [('Nội', C_DOMESTIC, 'Trong nước'),
                             ('Ngoại', C_IMPORT, 'Ngoài nước')]:
        sub = df_bad[df_bad['origin_group'] == grp]['rating'].dropna()

        fig.add_trace(go.Histogram(
            x=sub,
            name=name,
            marker_color=_hex_to_rgba(color, 0.8),
            marker_line=dict(color='white', width=0.7),
            nbinsx=12,
            opacity=0.85,
            hovertemplate=f'<b>{name}</b><br>Rating: %{{x:.1f}}<br>Số SP: %{{y}}<extra></extra>',
        ))

    fig.add_vline(x=3.5, line_dash='dash', line_color=TEXT, line_width=2)
    fig.add_annotation(
        x=3.5, y=1,
        text='Ngưỡng 3.5',
        showarrow=False,
        xanchor='right',
        font=dict(size=9, color=TEXT)
    )

    _t(
        fig,
        height=290,
        barmode='overlay',
        xaxis={'title': 'Rating (điểm)', 'range': [0.5, 3.6]},
        yaxis={'title': 'Số lượng sản phẩm'},
        legend={}
    )
    return fig


def make_mt3_price_bar():
    """Grouped bar: sản phẩm thất vọng theo khoảng giá."""
    _, df_bad = _df_mt3()
    ct = pd.crosstab(df_bad['origin_group'], df_bad['price_segment'])
    ct = ct.reindex(index=['Nội', 'Ngoại'], fill_value=0)
    ct = ct.reindex(columns=[p for p in PRICE_ORDER if p in ct.columns], fill_value=0)

    price_clrs = ['#FCE7F3', '#F9A8D4', '#EC4899', '#BE185D', '#881337']

    fig = go.Figure()
    for i, seg in enumerate(ct.columns):
        vals = [
            ct.loc['Nội', seg] if 'Nội' in ct.index else 0,
            ct.loc['Ngoại', seg] if 'Ngoại' in ct.index else 0
        ]
        clr = price_clrs[i % len(price_clrs)]

        fig.add_trace(go.Bar(
            name=seg,
            x=['Trong nước', 'Ngoài nước'],
            y=vals,
            marker_color=clr,
            marker_line=dict(color='white', width=0.8),
            text=[str(v) if v > 0 else '' for v in vals],
            textposition='outside',
            textfont=dict(size=10, color=TEXT),
            hovertemplate=f'<b>%{{x}}</b><br>{seg}: %{{y}} SP<extra></extra>',
        ))

    _t(
        fig,
        height=290,
        barmode='group',
        xaxis={'showgrid': False},
        yaxis={'title': 'Số sản phẩm thất vọng'},
        legend={
            'orientation': 'h',
            'yanchor': 'top',
            'y': -0.18,
            'xanchor': 'center',
            'x': 0.5,
            'font': dict(size=10)
        },
        margin=dict(l=16, r=16, t=36, b=65)
    )
    return fig


# ── MT3 KPI ─────────────────────────────────────────────────
def _mt3_kpis():
    df_all, df_bad = _df_mt3()
    n_noi   = (df_all['origin_group'] == 'Nội').sum()
    n_ngoai = (df_all['origin_group'] == 'Ngoại').sum()
    bad_noi   = (df_bad['origin_group'] == 'Nội').sum()
    bad_ngoai = (df_bad['origin_group'] == 'Ngoại').sum()
    pct_noi   = bad_noi   / n_noi   * 100 if n_noi   else 0
    pct_ngoai = bad_ngoai / n_ngoai * 100 if n_ngoai else 0
    avg_sold_noi   = df_bad[df_bad['origin_group'] == 'Nội']['sold_count'].mean() if bad_noi   else 0
    avg_sold_ngoai = df_bad[df_bad['origin_group'] == 'Ngoại']['sold_count'].mean() if bad_ngoai else 0
    return pct_noi, pct_ngoai, avg_sold_noi, avg_sold_ngoai


# ══════════════════════════════════════════════════════════════
#  COMPONENT HELPERS
# ══════════════════════════════════════════════════════════════

def kpi(icon, val, lbl, color, bg):
    return html.Div([
        html.Div(icon, className='p2-kpi-icon', style={'background': bg}),
        html.Div([
            html.P(val, className='p2-kpi-val', style={'color': color}),
            html.P(lbl, className='p2-kpi-lbl'),
        ]),
    ], className='p2-kpi')


def sec_hdr(num, icon, eyebrow, title, desc, variant='s1'):
    return html.Div([
        html.Div(icon, className='p2-sec-num'),
        html.Div([
            html.P(f'Mục tiêu {num} · {eyebrow}',
                   className='p2-sec-eyebrow'),
            html.H3(title, className='p2-sec-title'),
            html.P(desc, className='p2-sec-desc'),
        ]),
    ], className=f'p2-section {variant}')


def card(icon, title, sub, children, flex='1', min_w='280px', ico_bg='#FCE7F3'):
    return html.Div([
        html.Div([
            html.Div(icon, className='p2-card-ico', style={'background': ico_bg}),
            html.Div([
                html.P(title, className='p2-card-title'),
                html.P(sub,   className='p2-card-sub'),
            ]),
        ], className='p2-card-hdr'),
        children,
    ], className='p2-card', style={'flex': flex, 'minWidth': min_w})


def callout(icon, text, variant='rose'):
    return html.Div([
        html.Span(icon, className='p2-callout-icon'),
        html.Span(text, style={'fontSize': '12px'}),
    ], className=f'p2-callout {variant}')


def stat(val, lbl, color):
    return html.Div([
        html.P(val, className='p2-stat-val', style={'color': color}),
        html.P(lbl, className='p2-stat-lbl'),
    ], className='p2-stat-item')


# ══════════════════════════════════════════════════════════════
#  LAYOUT
# ══════════════════════════════════════════════════════════════

def layout():
    # Pre-compute KPIs
    n_noi, n_ngoai, avg_r_noi, avg_r_ngoai = _mt1_kpis()
    rr_noi, rr_ngoai, eng_noi, eng_ngoai   = _mt2_kpis()
    pct_bad_noi, pct_bad_ngoai, sold_bad_noi, sold_bad_ngoai = _mt3_kpis()

    # Pre-build figures (static — no callbacks needed)
    fig_box1   = make_mt1_box()
    fig_vln    = make_mt1_violin()
    fig_hist1  = make_mt1_hist()
    fig_stats  = make_mt1_stats_bar()
    fig_box2   = make_mt2_box()
    fig_eng    = make_mt2_engagement()
    fig_hbar   = make_mt2_hbar()
    fig_grp2   = make_mt2_grouped()
    fig_donut_noi   = make_mt3_donut('Nội')
    fig_donut_ngoai = make_mt3_donut('Ngoại')
    fig_hist3  = make_mt3_hist()
    fig_price  = make_mt3_price_bar()

    cfg = {'displayModeBar': False}

    return html.Div([

        # ── HERO ────────────────────────────────────────────
        html.Div([
            html.Div('⭐ UY TÍN & TRẢI NGHIỆM KHÁCH HÀNG',
                     className='p2-hero-tag'),
            html.H1('Chất lượng & Uy tín Mỹ phẩm Tiki · T3/2026',
                    className='p2-hero-title'),
            html.P('3 Mục tiêu · Rating (Sold > 50) · Review Ratio · Phân khúc Thất vọng · Nội vs Ngoại',
                   className='p2-hero-sub'),
            html.Div([
                html.Span('⭐ Rating trung bình > 4.5', className='p2-chip'),
                html.Span(f'📝 Review Ratio Ngoại: {rr_ngoai:.3f}', className='p2-chip gold'),
                html.Span(f'⚠️ Thất vọng Nội: {pct_bad_noi:.1f}%', className='p2-chip warning'),
                html.Span('📅 Nhóm 05 · FIT-HCMUS', className='p2-chip'),
            ], className='p2-hero-chips'),
        ], className='p2-hero'),

        # ── KPI Row ─────────────────────────────────────────
        html.Div([
            kpi('⭐', f'{avg_r_noi:.2f}', 'Rating TB · Nội địa', C_DOMESTIC, '#EFF6FF'),
            kpi('⭐', f'{avg_r_ngoai:.2f}', 'Rating TB · Ngoại nhập', C_IMPORT, '#FEE2E2'),
            kpi('📝', f'{rr_ngoai:.3f}', 'Review Ratio · Ngoại', C_IMPORT, '#FEE2E2'),
            kpi('📝', f'{rr_noi:.3f}', 'Review Ratio · Nội', C_DOMESTIC, '#EFF6FF'),
            kpi('😟', f'{pct_bad_ngoai:.1f}%', 'Tỉ lệ thất vọng · Ngoại', '#D97706', '#FEF3C7'),
            kpi('😟', f'{pct_bad_noi:.1f}%', 'Tỉ lệ thất vọng · Nội', C_DOMESTIC, '#EFF6FF'),
        ], className='p2-kpi-row'),

        # ══════════════════════════════════════════════════
        #  MỤC TIÊU 1 — Rating
        # ══════════════════════════════════════════════════
        sec_hdr(1, '⭐', 'So sánh Rating',
                'So sánh điểm đánh giá trung bình trên sản phẩm bán chạy (Sold > 50)',
                'Kiểm chứng sự ổn định chất lượng giữa hàng Nội và Ngoại thông qua 4 góc nhìn: '
                'phân bố, mật độ, tần số và thống kê cốt lõi.',
                's1'),

        # Row 1-A: Box + Violin
        html.Div([
            card('📦', 'Phân bố Rating (Box Plot)',
                 'Trung vị · IQR · Outliers · Nhãn trung bình vàng',
                 dcc.Graph(figure=fig_box1, config=cfg),
                 flex='1', min_w='290px', ico_bg=R100),
            card('🎻', 'Mật độ phân bố Rating (Violin)',
                 'Hình dạng mật độ KDE · Đường trung bình đỏ nét',
                 dcc.Graph(figure=fig_vln, config=cfg),
                 flex='1', min_w='290px', ico_bg=R100),
        ], className='p2-row'),

        # Row 1-B: Hist + Stats bar
        html.Div([
            card('📊', 'Phân bố tần số & Đường KDE',
                 'Histogram kết hợp KDE · Xu hướng hội tụ rating cao',
                 dcc.Graph(figure=fig_hist1, config=cfg),
                 flex='1.2', min_w='340px', ico_bg=R100),
            card('📐', 'So sánh chỉ số thống kê cốt lõi',
                 'Mean · Median · Q1 · Q3 — Nội vs Ngoại',
                 dcc.Graph(figure=fig_stats, config=cfg),
                 flex='1', min_w='300px', ico_bg=R100),
        ], className='p2-row'),

        # Insight MT1
        html.Div([
            html.H4('📌 Kết luận Mục tiêu 1', style={
                'margin': '0 0 12px 0', 'fontSize': '13.5px',
                'fontWeight': '700', 'color': R700,
                'fontFamily': "'Sora','Segoe UI',sans-serif",
            }),
            html.Div(className='p2-stat-grid', children=[
                stat(f'{avg_r_noi:.2f}', 'Rating TB · Nội (sold>50)', C_DOMESTIC),
                stat(f'{avg_r_ngoai:.2f}', 'Rating TB · Ngoại (sold>50)', C_IMPORT),
                stat(f'{n_noi:,}', 'SP Nội có sold>50', C_DOMESTIC),
                stat(f'{n_ngoai:,}', 'SP Ngoại có sold>50', C_IMPORT),
            ]),
            html.Div(style={'height': '10px'}),
            callout('⭐', html.Span([
                html.Strong('Cả hai nhóm đều đạt rating > 4.5'), ' — xác nhận sản phẩm bán chạy gắn liền '
                'với chất lượng tốt. Tuy nhiên hàng Nội có outliers '
                'kéo xuống mức 1–2 sao rõ rệt hơn, phản ánh sự thiếu đồng nhất trong kiểm soát chất lượng.',
            ]), 'rose'),
            html.Div(style={'height': '8px'}),
            callout('🎯', html.Span([
                html.Strong('Hàng Ngoại ổn định hơn'), ' — Q1 Ngoại (~4.6) nhỉnh hơn Q1 Nội (~4.5), '
                'rủi ro nhận hàng kém chất lượng thấp hơn. Hàng Nội bù lại có nhiều '
                'sản phẩm đạt 5.0 tuyệt đối — cơ hội làm "mũi nhọn" marketing.',
            ]), 'amber'),
        ], style={
            'background': SURFACE,
            'border': f'1px solid {BORDER}',
            'borderLeft': f'3px solid {R600}',
            'borderRadius': '14px',
            'padding': '20px 22px',
            'marginBottom': '18px',
            'boxShadow': '0 1px 6px rgba(190,24,93,.06)',
            'fontFamily': "'Sora','Segoe UI',sans-serif",
        }),

        html.Div(className='p2-divider'),

        # ══════════════════════════════════════════════════
        #  MỤC TIÊU 2 — Review Ratio
        # ══════════════════════════════════════════════════
        sec_hdr(2, '📝', 'Review Ratio',
                'Đo lường mức độ tương tác và nhiệt tình đóng góp ý kiến của khách hàng',
                'Phân tích Review Ratio (review_count / sold_count) theo nguồn gốc, '
                'mức độ engagement và từng dòng sản phẩm Top 12.',
                's2'),

        # Row 2-A: Box + Stacked engagement
        html.Div([
            card('📦', 'Box Plot: Phân bố Review Ratio',
                 'Giới hạn 95th percentile · Nhãn trung bình vàng',
                 dcc.Graph(figure=fig_box2, config=cfg),
                 flex='1', min_w='290px', ico_bg='#FEF3C7'),
            card('📊', 'Cơ cấu mức độ Engagement (%)',
                 'Thấp · TB · Cao · Rất cao — Nội vs Ngoại',
                 dcc.Graph(figure=fig_eng, config=cfg),
                 flex='1', min_w='290px', ico_bg='#FEF3C7'),
        ], className='p2-row'),

        # Row 2-B: HBar + Grouped
        html.Div([
            card('📈', 'Review Ratio TB theo Dòng Sản Phẩm (Top 12)',
                 '🟢 Trên TB · ⬜ Dưới TB · Đường đỏ = TB chung',
                 dcc.Graph(figure=fig_hbar, config=cfg),
                 flex='1', min_w='320px', ico_bg='#FEF3C7'),
            card('⚖️', 'So sánh Review Ratio Nội vs Ngoại',
                 'Theo từng dòng sản phẩm Top 12',
                 dcc.Graph(figure=fig_grp2, config=cfg),
                 flex='1.2', min_w='360px', ico_bg='#FEF3C7'),
        ], className='p2-row'),

        # Insight MT2
        html.Div([
            html.H4('📌 Kết luận Mục tiêu 2', style={
                'margin': '0 0 12px 0', 'fontSize': '13.5px',
                'fontWeight': '700', 'color': '#92400E',
                'fontFamily': "'Sora','Segoe UI',sans-serif",
            }),
            html.Div(className='p2-stat-grid', children=[
                stat(f'{rr_noi:.3f}',   'Review Ratio TB · Nội',    C_DOMESTIC),
                stat(f'{rr_ngoai:.3f}', 'Review Ratio TB · Ngoại',  C_IMPORT),
                stat(f'{eng_noi:.1f}%',   'Engagement > 15% · Nội',  C_DOMESTIC),
                stat(f'{eng_ngoai:.1f}%', 'Engagement > 15% · Ngoại', C_IMPORT),
            ]),
            html.Div(style={'height': '10px'}),
            callout('📝', html.Span([
                html.Strong(f'Hàng Ngoại có Review Ratio TB = {rr_ngoai:.3f}'), f' vs Nội = {rr_noi:.3f}. '
                'Cứ 100 lượt mua Ngoại thì ~30 khách để lại review, Nội chỉ ~22. '
                'Engagement Ngoại >15% đạt ~66.8% sản phẩm, vượt Nội (~48.7%).',
            ]), 'amber'),
            html.Div(style={'height': '8px'}),
            callout('🌺', html.Span([
                html.Strong('Kem dưỡng trắng là ngoại lệ:'),
                ' hàng Nội có Review Ratio cao hơn Ngoại — '
                '"lợi thế bản địa" khi thương hiệu Việt thấu hiểu insight dưỡng da của người tiêu dùng trong nước.',
            ]), 'teal'),
        ], style={
            'background': SURFACE,
            'border': f'1px solid #FDE68A',
            'borderLeft': f'3px solid {AMBER}',
            'borderRadius': '14px',
            'padding': '20px 22px',
            'marginBottom': '18px',
            'boxShadow': f'0 1px 6px rgba(245,158,11,.08)',
            'fontFamily': "'Sora','Segoe UI',sans-serif",
        }),

        html.Div(className='p2-divider'),

        # ══════════════════════════════════════════════════
        #  MỤC TIÊU 3 — Phân khúc Thất vọng
        # ══════════════════════════════════════════════════
        sec_hdr(3, '😟', 'Phân khúc Thất vọng',
                'Nhận diện sản phẩm Rating < 3.5 và phân tích yếu tố gây không hài lòng',
                'So sánh tỉ lệ thất vọng, mức độ nghiêm trọng và tác động '
                'thực tế (lượt bán) của sản phẩm kém chất lượng theo phân khúc giá.',
                's3'),

        # Row 3-A: 2 Donuts
        html.Div([
            card('🍩', 'Tỉ lệ thất vọng · Trong nước',
                 'Sản phẩm có Rating > 0 · Màu xanh = thất vọng nổi bật',
                 dcc.Graph(figure=fig_donut_noi, config=cfg),
                 flex='1', min_w='260px', ico_bg='#EFF6FF'),
            card('🍩', 'Tỉ lệ thất vọng · Ngoài nước',
                 'Sản phẩm có Rating > 0 · Màu đỏ = thất vọng nổi bật',
                 dcc.Graph(figure=fig_donut_ngoai, config=cfg),
                 flex='1', min_w='260px', ico_bg='#FEE2E2'),

            # Quick insight box
            html.Div([
                html.H4('⚠️ Nghịch lý thú vị', style={
                    'margin': '0 0 10px 0', 'fontSize': '13px',
                    'fontWeight': '700', 'color': DANGER,
                    'fontFamily': "'Sora','Segoe UI',sans-serif",
                }),
                html.Div(className='p2-stat-grid', children=[
                    stat(f'{pct_bad_noi:.1f}%',   'Tỉ lệ thất vọng · Nội',    C_DOMESTIC),
                    stat(f'{pct_bad_ngoai:.1f}%', 'Tỉ lệ thất vọng · Ngoại',  C_IMPORT),
                    stat(f'{int(sold_bad_noi):,}',  'Lượt bán TB · Nội tệ',     DANGER),
                    stat(f'{int(sold_bad_ngoai):,}', 'Lượt bán TB · Ngoại tệ',  '#D97706'),
                ]),
                html.Div(style={'height': '10px'}),
                callout('❗', html.Span([
                    html.Strong('Hàng Ngoại có tỉ lệ thất vọng cao hơn'),
                    f' ({pct_bad_ngoai:.1f}% vs {pct_bad_noi:.1f}%) do kỳ vọng khắt khe hơn từ giá cao. '
                    'Nhưng sản phẩm Nội tệ lại có lượt bán TB ~161 — '
                    'gây thiệt hại thực tế lớn hơn nhiều.',
                ]), 'danger'),
            ], style={
                'flex': '1', 'minWidth': '250px',
                'background': SURFACE,
                'border': f'1px solid #FECDD3',
                'borderLeft': f'3px solid {DANGER}',
                'borderRadius': '14px',
                'padding': '18px',
                'boxShadow': '0 1px 6px rgba(220,38,38,.08)',
                'fontFamily': "'Sora','Segoe UI',sans-serif",
            }),
        ], className='p2-row'),

        # Row 3-B: Hist + Price bar
        html.Div([
            card('📊', 'Phân bố Rating trong phân khúc thất vọng',
                 'Hàng Nội tập trung ở 1–2 sao (thất bại toàn diện) · Ngoại tập trung ở 3 sao',
                 dcc.Graph(figure=fig_hist3, config=cfg),
                 flex='1', min_w='300px', ico_bg='#FFF1F2'),
            card('💰', 'Phân bố sản phẩm thất vọng theo khoảng giá',
                 'Ngoại tệ: tập trung 300k–700k · Nội tệ: tập trung 100k–300k',
                 dcc.Graph(figure=fig_price, config=cfg),
                 flex='1', min_w='300px', ico_bg='#FFF1F2'),
        ], className='p2-row'),

        # Footer kết luận MT3
        html.Div([
            html.H4('📌 Kết luận Mục tiêu 3', style={
                'margin': '0 0 12px 0', 'fontSize': '13.5px',
                'fontWeight': '700', 'color': DANGER,
                'fontFamily': "'Sora','Segoe UI',sans-serif",
            }),
            callout('🔍', html.Span([
                html.Strong('Hàng Nội: thấp hơn về tỉ lệ thất vọng'),
                ' nhưng mật độ điểm 1–2 sao cao hơn, cho thấy thất bại toàn diện '
                '(hàng giả, kích ứng da). Sản phẩm tệ vẫn bán được 161 lượt TB — '
                'cần cơ chế cảnh báo sớm trên sàn.',
            ]), 'danger'),
            html.Div(style={'height': '8px'}),
            callout('📊', html.Span([
                html.Strong('Hàng Ngoại thất vọng tập trung ở 300k–700k'),
                ' — phân khúc kỳ vọng cao nhất. Hàng Nội không có sản phẩm thất vọng '
                'ở "Trên 2tr" vì vốn không hiện diện phân khúc cao cấp.',
            ]), 'rose'),
        ], style={
            'background': SURFACE,
            'border': f'1px solid #FECDD3',
            'borderLeft': f'3px solid {DANGER}',
            'borderRadius': '14px',
            'padding': '20px 22px',
            'marginBottom': '18px',
            'boxShadow': '0 1px 6px rgba(220,38,38,.06)',
            'fontFamily': "'Sora','Segoe UI',sans-serif",
        }),

        # ── Footer ──────────────────────────────────────────
        html.Div([
            html.Span('⚠️ Dữ liệu crawl từ Tiki tháng 3/2026'),
            html.Span('Review Ratio = review_count / sold_count'),
            html.Span('Rating chỉ tính SP có sold > 50 (MT1) hoặc rating > 0'),
            html.Span('Nhóm 05 · FIT-HCMUS'),
        ], className='p2-footer'),

    ], className='p2-page', style={'padding': '0'})
