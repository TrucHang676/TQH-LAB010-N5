# ╔══════════════════════════════════════════════════════════════╗
# ║              app.py — Dashboard Entry Point                  ║
# ║  Chạy: python app.py  →  mở http://127.0.0.1:8050            ║
# ╚══════════════════════════════════════════════════════════════╝

import dash
from dash import html, dcc

# ── App ───────────────────────────────────────────────────────
app = dash.Dash(
    __name__,
    use_pages=True,
    suppress_callback_exceptions=True,
    title='Mỹ phẩm Tiki',
    update_title=None,
)

# ── Global dark theme ────────────────────────────────────────
BG_DARK    = '#14233F'
NAV_BG     = '#102347'
NAV_BORDER = 'rgba(255,255,255,0.08)'
TXT        = '#F0F6FF'
SUBTXT     = '#94A3B8'
ACCENT     = '#A78BFA'

# ── Shared nav link style ────────────────────────────────────
_NAV_LINK = {
    'textDecoration': 'none',
    'padding': '7px 16px',
    'borderRadius': '8px',
    'fontSize': '13px',
    'fontWeight': '600',
    'color': SUBTXT,
    'transition': 'all 0.15s ease',
}

_NAV_EXTERNAL = {
    **_NAV_LINK,
    'color': 'rgba(148,163,184,0.55)',
}

# ── Cross-app pages (dashboardML on port 8051) ───────────────
ML_PAGES = [
    {'name': 'ML · Regression',       'href': 'http://127.0.0.1:8051/'},
    {'name': 'ML · Clustering', 'href': 'http://127.0.0.1:8051/market-segmentation'},
]

# ── Layout ───────────────────────────────────────────────────
app.layout = html.Div([

    # ── Navigation Bar ──────────────────────────────────────
    html.Nav(
        html.Div([
            # Logo / branding
            html.Div([
                html.Span('Mỹ phẩm Tiki', style={
                    'fontWeight': '700',
                    'fontSize': '15px',
                    'color': TXT,
                    'marginRight': '4px',
                }),
                html.Span('T3/2026', style={
                    'fontSize': '13px',
                    'color': SUBTXT,
                }),
            ], style={
                'display': 'flex',
                'alignItems': 'baseline',
                'gap': '4px'
            }),

            # Nav links: local pages + external ML links
            html.Div([
                # ── Local pages (dcc.Link — client-side routing) ──
                *[
                    dcc.Link(
                        page['name'],
                        href=page['path'],
                        id=f"nav-{page['module']}",
                        style=_NAV_LINK,
                    )
                    for page in sorted(
                        dash.page_registry.values(),
                        key=lambda p: 999 if p.get('order') is None else p.get('order')
                    )
                ],

                # ── Divider ──
                html.Span('│', style={
                    'color': 'rgba(255,255,255,0.15)',
                    'fontSize': '16px',
                    'margin': '0 4px',
                    'userSelect': 'none',
                }),

                # ── External ML pages (html.A — full page navigation) ──
                *[
                    html.A(
                        p['name'],
                        href=p['href'],
                        target='_self',
                        style=_NAV_EXTERNAL,
                        className='nav-ext-link',
                    )
                    for p in ML_PAGES
                ],
            ], id='nav-links', style={
                'display': 'flex',
                'gap': '4px',
                'flexWrap': 'wrap',
                'alignItems': 'center',
            }),
        ], style={
            'maxWidth': '1400px',
            'margin': '0 auto',
            'padding': '0 24px',
            'display': 'flex',
            'alignItems': 'center',
            'justifyContent': 'space-between',
            'height': '56px',
        }),
        style={
            'background': NAV_BG,
            'borderBottom': f'1px solid {NAV_BORDER}',
            'position': 'sticky',
            'top': '0',
            'zIndex': '100',
            'boxShadow': '0 4px 18px rgba(0,0,0,0.18)',
        }
    ),

    # ── Page Content ─────────────────────────────────────────
    html.Div(
        html.Main(
            dash.page_container,
            style={
                'maxWidth': '1400px',
                'margin': '0 auto',
                'padding': '24px',
                'minHeight': 'calc(100vh - 56px)',
            }
        ),
        style={
            'background': BG_DARK,
            'minHeight': 'calc(100vh - 56px)',
        }
    ),

], id='app-root',
   style={
       'fontFamily': "'Inter', 'Segoe UI', sans-serif",
       'background': BG_DARK,
       'minHeight': '100vh',
   })


if __name__ == '__main__':
    print('\n' + '='*55)
    print('  🚀  Dashboard khởi động tại http://127.0.0.1:8050')
    print('  Ctrl+C để dừng')
    print('='*55 + '\n')
    app.run(debug=True, port=8050)