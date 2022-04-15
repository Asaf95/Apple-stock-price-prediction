import base64
import datetime
import io
import dash
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
from dash import dash_table
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
from dash.dependencies import State
import data.dataframe_gantt

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

colors = {
    'background': '#f1f7fd',
    'text': '#061932'
}
df = data.dataframe_gantt.df
fig = ff.create_gantt(df)
# fig.layout.xaxis.tickvals = pd.date_range('2009-01-01', '2009-03-01', freq='d')
# fig.layout.xaxis.ticktext = list(range(len(fig.layout.xaxis.tickvals)))


app = dash.Dash(__name__, external_stylesheets=external_stylesheets, suppress_callback_exceptions=True)

app.layout = html.Div(style={'backgroundColor': colors['background']},
                      children=[
                          html.Br(),
                          html.H1("AAPL Predection project ", style={
                              'width': '100%',
                              'height': '60px',
                              'lineHeight': '60px',
                              'borderWidth': '1px',
                              'borderRadius': '5px',
                              'textAlign': 'center',
                              'margin': '10px'
                          }),
                          html.Br(),
                          dcc.Tabs([
                              dcc.Tab(label='Output', children=[
                                  dcc.Graph(
                                      figure=fig
                                  ),
                              ]),
                              dcc.Tab(label='How To Use', children=[
                                  html.I("How to use this app?"),
                                  html.Br()
                              ]),
                          ]),

                      ])



if __name__ == '__main__':
    app.run_server(debug=True)
