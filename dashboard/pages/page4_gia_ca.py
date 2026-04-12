import dash
from dash import html

dash.register_page(__name__, path='/gia-ca', name='Giá cả', title='Giá cả')

layout = html.Div([
    html.H2('Trang đang được phát triển')
])