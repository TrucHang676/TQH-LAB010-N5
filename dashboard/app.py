# ╔══════════════════════════════════════════════════════════════╗
# ║              app.py — Dashboard Entry Point                  ║
# ║  Chạy: python app.py  →  mở http://127.0.0.1:8050            ║
# ╚══════════════════════════════════════════════════════════════╝
#
# Cài đặt thư viện (chạy 1 lần):
#   pip install dash plotly pandas numpy
#
# Sau đó chạy dashboard:
#   cd dashboard/
#   python app.py

import dash
from dash import html, dcc
import dash_bootstrap_components as dbc  # optional, fallback nếu không có

app = dash.Dash(
    __name__,
    use_pages=True,                 # tự động load tất cả file trong pages/
    suppress_callback_exceptions=True,
    title='Mỹ phẩm Tiki',
    update_title=None,
)

# ── Layout khung chung (nav + page content) ──────────────────
app.layout = html.Div([

    # URL tracking (cần thiết cho routing)
    dcc.Location(id='url', refresh=False),

    # ── Navigation Bar ──────────────────────────────────────
    html.Nav(
        html.Div([
            html.Div([
                html.Span('Mỹ phẩm Tiki', style={
                    'fontWeight': '700', 'fontSize': '15px',
                    'color': '#1E293B', 'marginRight': '4px',
                }),
                html.Span('T3/2026', style={
                    'fontSize': '13px', 'color': '#64748B',
                }),
            ], style={'display': 'flex', 'alignItems': 'baseline', 'gap': '4px'}),

            html.Div([
                dcc.Link(page['name'], href=page['path'],
                         style={'textDecoration': 'none'},
                         id=f"nav-{page['module']}")
                for page in dash.page_registry.values()
            ], id='nav-links', style={
                'display': 'flex', 'gap': '4px',
            }),
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
        }
    ),

    # ── Page Content ─────────────────────────────────────────
    html.Main(
        dash.page_container,
        style={
            'maxWidth': '1400px',
            'margin': '0 auto',
            'padding': '24px',
            'minHeight': 'calc(100vh - 56px)',
            'background': '#F8FAFC',
        }
    ),

], style={'fontFamily': "'Inter', 'Segoe UI', sans-serif",
          'background': '#F8FAFC', 'minHeight': '100vh'})


if __name__ == '__main__':
    print('\n' + '='*55)
    print('  🚀  Dashboard khởi động tại http://127.0.0.1:8050')
    print('  Ctrl+C để dừng')
    print('='*55 + '\n')
    app.run(debug=True, port=8050)
