# ╔══════════════════════════════════════════════════════════════╗
# ║        app.py — ML Dashboard Entry Point                     ║
# ║  Chạy: python app.py  →  mở http://127.0.0.1:8051            ║
# ╚══════════════════════════════════════════════════════════════╝

import dash
from dash import html, dcc

app = dash.Dash(
    __name__,
    use_pages=True,
    suppress_callback_exceptions=True,
    title='ML Dashboard | Mỹ phẩm Tiki',
    update_title=None,
)

# ── Global dark theme (đồng bộ với dashboard/app.py) ─────────
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




app.layout = html.Div([

    # ── Navigation Bar (ML standalone) ───────────────────────
    html.Nav(
        html.Div([
            # Logo / branding
            html.Div([
                html.Span('🧠', style={'fontSize': '18px', 'marginRight': '6px'}),
                html.Span('Machine Learning', style={
                    'fontWeight': '700',
                    'fontSize': '15px',
                    'color': TXT,
                    'marginRight': '6px',
                }),
                html.Span('Mỹ phẩm Tiki · T3/2026', style={
                    'fontSize': '12px',
                    'color': SUBTXT,
                }),
            ], style={
                'display': 'flex',
                'alignItems': 'center',
            }),
        ], style={
            'maxWidth': '1400px',
            'margin': '0 auto',
            'padding': '0 24px',
            'display': 'flex',
            'alignItems': 'center',
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
        html.Main(dash.page_container, style={
            'maxWidth': '1400px',
            'margin': '0 auto',
            'padding': '24px',
            'minHeight': 'calc(100vh - 56px)',
        }),
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
    print('  🧠  ML Dashboard khởi động tại http://127.0.0.1:8051')
    print('  Ctrl+C để dừng')
    print('='*55 + '\n')
    app.run(debug=True, port=8051)