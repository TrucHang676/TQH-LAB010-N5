# ╔══════════════════════════════════════════════════════════════╗
# ║              app.py — Dashboard Entry Point                  ║
# ║  Chạy: python app.py  →  mở http://127.0.0.1:8050            ║
# ╚══════════════════════════════════════════════════════════════╝

import dash
from dash import html, dcc, callback, Input, Output
import sys
import os


# ── App ───────────────────────────────────────────────────────
app = dash.Dash(
    __name__,
    use_pages=True,
    suppress_callback_exceptions=True,
    title='Thị trường mỹ phẩm Tiki',
    update_title=None,
)
server = app.server

# ── Global dark theme ────────────────────────────────────────
BG_DARK    = '#14233F'
NAV_BG     = '#102347'
NAV_BORDER = 'rgba(255,255,255,0.08)'
TXT        = '#F0F6FF'
SUBTXT     = '#94A3B8'

# ── Layout ───────────────────────────────────────────────────
app.layout = html.Div([

    # Store để lưu current path
    dcc.Location(id='url', refresh=False),

    # ── Navigation Bar ──────────────────────────────────────
    html.Nav(
        html.Div([
            # Logo / branding
            html.Div([
                html.Span('🛒', style={
                    'fontSize': '18px',
                    'marginRight': '6px',
                }),
                html.Span('Thị trường mỹ phẩm Tiki', style={
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
                'alignItems': 'center',
                'gap': '2px'
            }),

            # Nav links
            html.Div(id='nav-links-container', style={
                'display': 'flex',
                'gap': '4px',
                'flexWrap': 'wrap',
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


# ── Callback để update nav links với active highlight ────────
@callback(
    Output('nav-links-container', 'children'),
    Input('url', 'pathname')
)
def update_nav_links(pathname):
    pages = sorted(
        dash.page_registry.values(),
        key=lambda p: 999 if p.get('order') is None else p.get('order')
    )
    
    nav_links = []
    for page in pages:
        is_active = pathname == page['path'] or (pathname == '/' and page['path'] == '/')
        nav_links.append(
            dcc.Link(
                page['name'],
                href=page['path'],
                id=f"nav-{page['module']}",
                style={
                    'textDecoration': 'none',
                    'padding': '7px 16px',
                    'borderRadius': '8px',
                    'fontSize': '13px',
                    'fontWeight': '600',
                    'color': TXT if is_active else SUBTXT,
                    'background': 'rgba(99,102,241,0.15)' if is_active else 'transparent',
                    'border': f'1px solid {"rgba(99,102,241,0.4)" if is_active else "transparent"}',
                    'transition': 'all 0.15s ease',
                }
            )
        )
    
    return nav_links


if __name__ == '__main__':
    host = os.getenv('HOST', '127.0.0.1')
    port = int(os.getenv('PORT', '8050'))
    debug = os.getenv('DASH_DEBUG', 'true').lower() in ('1', 'true', 'yes')

    print('\n' + '='*55)
    print(f'  🚀  Dashboard khởi động tại http://{host}:{port}')
    print('  Ctrl+C để dừng')
    print('='*55 + '\n')
    app.run(host=host, port=port, debug=debug)