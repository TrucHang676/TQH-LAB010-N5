# ╔══════════════════════════════════════════════════════════════╗
# ║              app.py — Dashboard Entry Point                  ║
# ║  Chạy: python app.py  →  mở http://127.0.0.1:8050            ║
# ╚══════════════════════════════════════════════════════════════╝

import dash
from dash import html, dcc, Input, Output, callback
import dash_bootstrap_components as dbc

app = dash.Dash(
    __name__,
    use_pages=True,
    suppress_callback_exceptions=True,
    title='Mỹ phẩm Tiki',
    update_title=None,
)

# ── Mapping: path → màu nền ──────────────────────────────────
# Các trang dark theme dùng màu tối, trang light dùng màu nhạt
DARK_PATHS = {
    '/thi-phan',
    '/chat-luong',
    # thêm '/uy-tin', '/gia-ca' vào đây nếu làm dark theme
}

BG_LIGHT = '#F1F5F9'
BG_DARK  = '#172542'

# ── Layout ───────────────────────────────────────────────────
app.layout = html.Div([

    dcc.Location(id='url', refresh=False),

    # ── Navigation Bar ──────────────────────────────────────
    html.Nav(
        html.Div([
            # Logo / branding
            html.Div([
                html.Span('Mỹ phẩm Tiki', style={
                    'fontWeight': '700', 'fontSize': '15px',
                    'color': '#1E293B', 'marginRight': '4px',
                }),
                html.Span('T3/2026', style={
                    'fontSize': '13px', 'color': '#64748B',
                }),
            ], style={'display': 'flex', 'alignItems': 'baseline', 'gap': '4px'}),

            # Nav links
            html.Div([
                dcc.Link(page['name'], href=page['path'],
                         style={'textDecoration': 'none'},
                         id=f"nav-{page['module']}")
                for page in dash.page_registry.values()
            ], id='nav-links', style={'display': 'flex', 'gap': '4px'}),
        ], style={
            'maxWidth': '1400px', 'margin': '0 auto',
            'padding': '0 24px',
            'display': 'flex', 'alignItems': 'center',
            'justifyContent': 'space-between', 'height': '56px',
        }),
        style={
            'background': '#FFFFFF',
            'borderBottom': '1px solid #E2E8F0',
            'position': 'sticky', 'top': '0', 'zIndex': '100',
            'boxShadow': '0 1px 6px rgba(0,0,0,0.06)',
        }
    ),

    # ── Page Content wrapper ─────────────────────────────────
    # id='page-wrapper' để callback đổi màu nền động
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
        id='page-wrapper',
        style={
            'background': BG_LIGHT,   # default, sẽ được callback override
            'minHeight': 'calc(100vh - 56px)',
            'transition': 'background 0.3s ease',
        }
    ),

], id='app-root',
   style={
       'fontFamily': "'Inter', 'Segoe UI', sans-serif",
       'background': BG_LIGHT,
       'minHeight': '100vh',
       'transition': 'background 0.3s ease',
   })


# ── Callback: đổi màu nền theo URL ───────────────────────────
@callback(
    Output('page-wrapper', 'style'),
    Output('app-root', 'style'),
    Input('url', 'pathname'),
)
def update_bg(pathname):
    is_dark = pathname in DARK_PATHS

    bg = BG_DARK if is_dark else BG_LIGHT

    wrapper_style = {
        'background': bg,
        'minHeight': 'calc(100vh - 56px)',
        'transition': 'background 0.3s ease',
    }
    root_style = {
        'fontFamily': "'Inter', 'Segoe UI', sans-serif",
        'background': bg,
        'minHeight': '100vh',
        'transition': 'background 0.3s ease',
    }
    return wrapper_style, root_style


if __name__ == '__main__':
    print('\n' + '='*55)
    print('  🚀  Dashboard khởi động tại http://127.0.0.1:8050')
    print('  Ctrl+C để dừng')
    print('='*55 + '\n')
    app.run(debug=True, port=8050)
