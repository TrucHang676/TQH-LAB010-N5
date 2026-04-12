import dash
from dash import html

dash.register_page(__name__, path='/chat-luong', name='Chất lượng', title='Chất lượng')

layout = html.Div([
    html.H2('Trang đang được phát triển')
])