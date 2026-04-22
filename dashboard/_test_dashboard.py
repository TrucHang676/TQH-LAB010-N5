"""Temp test script — verify dashboard loads both ML pages."""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import dash

app = dash.Dash(__name__, use_pages=True, pages_folder='pages')

for mod_name in ['pages.page_ml_regression', 'pages.page_ml_clustering']:
    try:
        mod = __import__(mod_name, fromlist=['layout'])
        layout = mod.layout()
        print(f'  OK {mod_name}: layout built (type={type(layout).__name__})')
    except Exception as e:
        print(f'  FAILED {mod_name}: {e}')
        import traceback
        traceback.print_exc()
