import dash
import dash.dependencies
import dash_core_components
import dash_html_components
from flask import Flask

from ... import KryptoniteDataClient
from ...Base import LOGGER, CONFIG

LOGGER = LOGGER.Monitor
CONFIG = CONFIG

REDIS_HOST = CONFIG.WebApp.Monitor.REDIS_HOST
REDIS_PORT = CONFIG.WebApp.Monitor.REDIS_PORT
REDIS_AUTH = CONFIG.WebApp.Monitor.REDIS_AUTH

FLASK_APP = Flask('Monitor')
DASH_APP = dash.Dash(name=__name__, server=FLASK_APP, requests_pathname_prefix='/Monitor/', update_title=None, title='Monitor')

from . import Market, Trade

DATA_CLIENT = KryptoniteDataClient.RelayClient(
    url=f'redis://{REDIS_HOST}:{REDIS_PORT}',
    password=REDIS_AUTH,
    on_bar=Market.on_bar,
    on_trade=Market.on_trade,
    on_orderbook=Market.on_orderbook
)


def init_layout() -> dash.Dash:
    """
    init a plotly app layout
    :keyword: external_stylesheets (List[str]): a list of additional css file string path
    :return: a Dash app
    """
    layout = dash_html_components.Div(
        children=[
            dash_html_components.Div(
                children=[
                    dash_html_components.Table(
                        children=[
                            dash_html_components.Tr(
                                children=[
                                    dash_html_components.Td(
                                        children=[
                                            dash_html_components.H2(children=['Kryptontie Monitor']),
                                        ],
                                        style={'width': "100%", 'border': 'medium solid'},
                                        colSpan=4
                                    )
                                ],
                                style={'height': '5%'}
                            ),
                            dash_html_components.Tr(
                                children=[
                                    dash_html_components.Td(
                                        children=[
                                            dash_html_components.Div(
                                                children=[
                                                    dash_html_components.Table(
                                                        children=[
                                                            dash_html_components.Tr(
                                                                children=[
                                                                    dash_html_components.Td(
                                                                        children=[
                                                                            dash_html_components.P(children=["Monitor Ticker: "]),
                                                                        ],
                                                                        style={'width': "50%", 'border': 'medium solid'},
                                                                    ),
                                                                    dash_html_components.Td(
                                                                        children=[
                                                                            dash_core_components.Dropdown(id='Kryptonite-Monitor-Ticker-Dropdown', options=[{'label': x.upper(), 'value': x} for x in DATA_CLIENT.tickers], style={'width': "100%", 'height': '100%'})
                                                                        ],
                                                                        style={'width': "50%", 'border': 'medium solid'},
                                                                    ),
                                                                ],
                                                                style={'height': "100%"}
                                                            )
                                                        ],
                                                        style={'width': "100%", 'border': 'medium solid'},
                                                    )
                                                ],
                                                style={'width': "50%"},
                                            ),
                                        ],
                                        style={'width': "100%", 'border': 'medium solid'},
                                        colSpan=4
                                    )
                                ],
                                style={'height': '5%'}
                            ),
                            dash_html_components.Tr(
                                children=[
                                    dash_html_components.Td(
                                        children=[
                                            dash_html_components.Div(id='Kryptonite-Monitor-Text', style={'width': "100%", 'height': '100%'}),
                                        ],
                                        style={'width': "25%", 'border': 'medium solid'}
                                    ),
                                    dash_html_components.Td(
                                        children=[
                                            dash_html_components.Div(
                                                children=[
                                                    dash_html_components.Table(
                                                        children=[
                                                            dash_html_components.Tr(
                                                                children=[
                                                                    dash_html_components.Td(
                                                                        children=[
                                                                            dash_html_components.Button(children=['Wind'], id='Wind-Button', n_clicks=0, style={'width': "100%", 'height': '100%'}),
                                                                        ],
                                                                        style={'width': "100%", 'border': 'medium solid'}
                                                                    )
                                                                ],
                                                                style={'height': '25%'}
                                                            ),
                                                            dash_html_components.Tr(
                                                                children=[
                                                                    dash_html_components.Td(
                                                                        children=[
                                                                            dash_html_components.Button(children=['Unwind'], id='Unwind-Button', n_clicks=0, style={'width': "100%", 'height': '100%'}),
                                                                        ],
                                                                        style={'width': "100%", 'border': 'medium solid'}
                                                                    )
                                                                ],
                                                                style={'height': '25%'}
                                                            ),
                                                            dash_html_components.Tr(
                                                                children=[
                                                                    dash_html_components.Td(
                                                                        children=[
                                                                            dash_html_components.Button(children=['Cancel'], id='Cancel-Button', n_clicks=0, style={'width': "100%", 'height': '100%'}),
                                                                        ],
                                                                        style={'width': "100%", 'border': 'medium solid'}
                                                                    )
                                                                ],
                                                                style={'height': '25%'}
                                                            ),
                                                            dash_html_components.Tr(
                                                                children=[
                                                                    dash_html_components.Td(
                                                                        children=[
                                                                            dash_html_components.Button(children=['Refresh'], id='Refresh-Button', n_clicks=0, style={'width': "100%", 'height': '100%'}),
                                                                        ],
                                                                        style={'width': "100%", 'border': 'medium solid'}
                                                                    )
                                                                ],
                                                                style={'height': '25%'}
                                                            )
                                                        ],
                                                        style={'width': "100%", 'height': '100%', 'border': 'medium solid'},
                                                    )
                                                ],
                                                style={'width': "100%", 'height': '100%'}
                                            )
                                        ],
                                        style={'width': "25%", 'border': 'medium solid'}
                                    ),
                                    dash_html_components.Td(
                                        children=[
                                            dash_html_components.Div(
                                                children=[
                                                    dash_html_components.Table(
                                                        children=[
                                                            dash_html_components.Tr(
                                                                children=[
                                                                    dash_html_components.Td(
                                                                        children=['Order Book'],
                                                                        style={'width': "100%", 'border': 'medium solid'}
                                                                    )
                                                                ],
                                                                style={'height': '10%', 'border': 'medium solid'}
                                                            ),
                                                            dash_html_components.Tr(
                                                                children=[
                                                                    dash_html_components.Td(
                                                                        children=[
                                                                            dash_core_components.Graph(id='Kryptonite-Monitor-OrderBook', style={'width': "100%", 'height': '100%'}, config={'displayModeBar': False},
                                                                                                       figure={'layout': {"xaxis": {"visible": False}, "yaxis": {"visible": False}, "annotations": [{"text": "No matching data found", "xref": "paper", "yref": "paper", "showarrow": False, "font": {"size": 28}}]}}),
                                                                        ],
                                                                        style={'width': "100%", 'border': 'medium solid'}
                                                                    )
                                                                ],
                                                                style={'height': '90%', 'border': 'medium solid'}
                                                            )
                                                        ],
                                                        style={'width': "100%", 'height': '100%', 'border': 'medium solid'}
                                                    )
                                                ],
                                                style={'width': "100%", 'height': '100%'}
                                            )
                                        ],
                                        style={'width': "25%", 'border': 'medium solid'}
                                    ),
                                    dash_html_components.Td(
                                        children=[
                                            dash_html_components.Div(
                                                children=[
                                                    dash_html_components.Table(
                                                        children=[
                                                            dash_html_components.Tr(
                                                                children=[
                                                                    dash_html_components.Td(
                                                                        children=['Balance'],
                                                                        style={'width': "100%", 'border': 'medium solid'}
                                                                    )
                                                                ],
                                                                style={'height': '10%', 'border': 'medium solid'}
                                                            ),
                                                            dash_html_components.Tr(
                                                                children=[
                                                                    dash_html_components.Td(
                                                                        children=[
                                                                            dash_core_components.Graph(id='Kryptonite-Monitor-Balance', style={'width': "100%", 'height': '100%'}, config={'displayModeBar': False},
                                                                                                       figure={'layout': {"xaxis": {"visible": False}, "yaxis": {"visible": False}, "annotations": [{"text": "No matching data found", "xref": "paper", "yref": "paper", "showarrow": False, "font": {"size": 28}}]}}),
                                                                        ],
                                                                        style={'width': "100%", 'border': 'medium solid'}
                                                                    )
                                                                ],
                                                                style={'height': '90%', 'border': 'medium solid'}
                                                            )
                                                        ],
                                                        style={'width': "100%", 'height': '100%', 'border': 'medium solid'}
                                                    )
                                                ],
                                                style={'width': "100%", 'height': '100%'}
                                            )
                                        ],
                                        style={'width': "25%", 'border': 'medium solid'}
                                    )
                                ],
                                style={'height': '10%'}
                            ),
                            dash_html_components.Tr(
                                children=[
                                    dash_html_components.Td(
                                        children=[
                                            dash_core_components.Graph(id='Kryptonite-Monitor-MarketView', style={'width': "100%", 'height': '100%'},
                                                                       figure={'layout': {"xaxis": {"visible": False}, "yaxis": {"visible": False}, "annotations": [{"text": "No matching data found", "xref": "paper", "yref": "paper", "showarrow": False, "font": {"size": 28}}]}}),
                                        ],
                                        style={'width': "100%", 'border': 'medium solid'},
                                        colSpan=4
                                    )
                                ],
                                style={'height': '80%'}
                            )
                        ],
                        style={'width': "100%", 'height': '100vh', 'border': 'medium solid'}
                    )
                ],
                style={'display': 'grid'}
            ),
            dash_core_components.Interval(
                id='interval-component',
                interval=1 * 1000,  # in milliseconds
                n_intervals=0
            )
        ]
    )

    return layout


def register_callbacks():
    # Register Timer Callback: render monitor text Div
    DASH_APP.callback(
        output=dash.dependencies.Output(component_id='Kryptonite-Monitor-Text', component_property='children'),
        inputs=[dash.dependencies.Input(component_id='interval-component', component_property='n_intervals')],
        state=[
            dash.dependencies.State(component_id='Kryptonite-Monitor-Ticker-Dropdown', component_property='value'),
            dash.dependencies.State(component_id='Kryptonite-Monitor-Text', component_property='children')
        ]
    )(lambda _, ticker, status: Market.render_monitor_text(ticker=ticker, status=status))

    # Register Timer Callback: render MarketView Graph
    DASH_APP.callback(
        output=dash.dependencies.Output(component_id='Kryptonite-Monitor-MarketView', component_property='figure'),
        inputs=[dash.dependencies.Input(component_id='interval-component', component_property='n_intervals')],
        state=[
            dash.dependencies.State(component_id='Kryptonite-Monitor-MarketView', component_property='figure'),
            dash.dependencies.State(component_id='Kryptonite-Monitor-Ticker-Dropdown', component_property='value')
        ]
    )(lambda _, fig, ticker: Market.render_market_view(fig=fig, ticker=ticker))

    # Register Button Callback: Wind function
    DASH_APP.callback(
        output=dash.dependencies.Output(component_id='Wind-Button', component_property='children'),
        inputs=[dash.dependencies.Input(component_id='Wind-Button', component_property='n_clicks')]
    )(lambda n_clicks: Trade.wind(clicked=n_clicks))

    # Register Button Callback: Wind function
    DASH_APP.callback(
        output=dash.dependencies.Output(component_id='Unwind-Button', component_property='children'),
        inputs=[dash.dependencies.Input(component_id='Unwind-Button', component_property='n_clicks')]
    )(lambda n_clicks: Trade.unwind(clicked=n_clicks))

    # Register Button Callback: Wind function
    DASH_APP.callback(
        output=dash.dependencies.Output(component_id='Cancel-Button', component_property='children'),
        inputs=[dash.dependencies.Input(component_id='Cancel-Button', component_property='n_clicks')]
    )(lambda n_clicks: Trade.cancel(clicked=n_clicks))

    # Register Button Callback: Wind function
    DASH_APP.callback(
        output=dash.dependencies.Output(component_id='Refresh-Button', component_property='children'),
        inputs=[dash.dependencies.Input(component_id='Refresh-Button', component_property='n_clicks')]
    )(lambda n_clicks: Trade.refresh(clicked=n_clicks))


DASH_APP.layout = init_layout()
register_callbacks()
