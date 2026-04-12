import dash
from dash import html

dash.register_page(__name__, path='/thi-phan', name='Thị phần', title='Thị phần')

layout = html.Div([
    html.H2('Trang đang được phát triển')
])