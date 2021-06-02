import datetime
from collections import defaultdict
from typing import Optional, Dict

import dash
import dash.dependencies
import dash_core_components
import dash_html_components
import numpy as np
import pandas as pd
import plotly.graph_objects
import plotly.subplots
import scipy.signal

from . import DATA_CLIENT
from ...Base import CONFIG, LOGGER
from ...Res.ToolKit import BarData, TradeData, OrderBook

LOGGER = LOGGER.getChild('WebApp.Monitor.Market')
CACHE_SIZE = CONFIG.WebApp.Monitor.CACHE_SIZE
VISIBLE_SIZE = CONFIG.WebApp.Monitor.VISIBLE_SIZE
DEFAULT_TICKER = CONFIG.WebApp.Monitor.DEFAULT_TICKER

data_storage: Dict[str, dict] = defaultdict(lambda: {'BarData': {}, 'OrderBook': None, 'ActiveBar': None, 'LastUpdate': None})


def subscribe(ticker):
    LOGGER.debug(f'Subscribe ticker {ticker}')
    if ticker not in data_storage:
        DATA_CLIENT.subscribe(ticker=ticker, dtype=BarData)
        DATA_CLIENT.subscribe(ticker=ticker, dtype=TradeData)
        DATA_CLIENT.subscribe(ticker=ticker, dtype=OrderBook)
        bar_data_list = DATA_CLIENT.load_remote(ticker=ticker, size=CACHE_SIZE, dtype=BarData)

        for bar_data in bar_data_list:
            data_storage[ticker]['BarData'][bar_data.market_time] = bar_data
            data_storage[ticker]['LastUpdate'] = bar_data.market_time


def on_bar(market_data: BarData):
    ticker = market_data.ticker
    data_storage[ticker][market_data.__class__.__name__][market_data.market_time] = market_data
    data_storage[ticker]['LastUpdate'] = market_data.market_time

    # chop storage cache
    stored_keys = sorted(list(data_storage[ticker][market_data.__class__.__name__].keys()))
    if len(data_storage[ticker]) > CACHE_SIZE:
        for pop_key in stored_keys[:-CACHE_SIZE]:
            data_storage[ticker][market_data.__class__.__name__].pop(pop_key)


def on_trade(market_data: TradeData):
    ticker = market_data.ticker

    # update active bar
    active_bar: Optional[BarData] = data_storage[ticker].get('ActiveBar')
    last_update: Optional[datetime.datetime] = data_storage[ticker].get('LastUpdate')

    if last_update is not None:
        active_bar_start_time = last_update + datetime.timedelta(seconds=60)
    else:
        active_bar_start_time = datetime.datetime(
            market_data.market_time.year,
            market_data.market_time.month,
            market_data.market_time.day,
            market_data.market_time.hour,
            market_data.market_time.minute,
        )

    if market_data.market_time >= active_bar_start_time:
        if active_bar is None or active_bar_start_time > active_bar.market_time:
            active_bar = BarData(
                ticker=market_data.ticker,
                high_price=market_data.price,
                low_price=market_data.price,
                open_price=market_data.price,
                close_price=market_data.price,
                bar_start_time=active_bar_start_time,
                bar_span=datetime.timedelta(seconds=60),
                volume=0.,
                notional=0.,
                trade_count=0
            )
        else:
            active_bar.high_price = np.nanmax([active_bar.high_price, market_data.price])
            active_bar.low_price = np.nanmin([active_bar.low_price, market_data.price])
            active_bar.close_price = market_data.price
            active_bar.volume += market_data.volume
            active_bar.notional += market_data.notional
            active_bar.trade_count += 1

    data_storage[ticker]['ActiveBar'] = active_bar


def on_orderbook(market_data: OrderBook):
    ticker = market_data.ticker
    data_storage[ticker][market_data.__class__.__name__] = market_data


def get_bar_df(ticker, size) -> Optional[pd.DataFrame]:
    active_bar: BarData = data_storage[ticker]['ActiveBar']
    result = pd.DataFrame()

    for bar_start_time in sorted(data_storage[ticker]['BarData'].keys())[-size:]:
        bar_data = data_storage[ticker]['BarData'][bar_start_time]
        result.at[bar_start_time, 'OPEN'] = bar_data.open_price
        result.at[bar_start_time, 'CLOSE'] = bar_data.close_price
        result.at[bar_start_time, 'HIGH'] = bar_data.high_price
        result.at[bar_start_time, 'LOW'] = bar_data.low_price
        result.at[bar_start_time, 'VOLUME'] = bar_data.volume
        result.at[bar_start_time, 'NOTIONAL'] = bar_data.notional

    if active_bar is not None:
        result.at[active_bar.market_time, 'OPEN'] = active_bar.open_price
        result.at[active_bar.market_time, 'CLOSE'] = active_bar.close_price
        result.at[active_bar.market_time, 'HIGH'] = active_bar.high_price
        result.at[active_bar.market_time, 'LOW'] = active_bar.low_price
        result.at[active_bar.market_time, 'VOLUME'] = active_bar.volume
        result.at[active_bar.market_time, 'NOTIONAL'] = active_bar.notional

    return result


def render_market_view(fig, ticker: str):
    if ticker is None:
        return dash.no_update

    if ticker not in data_storage:
        subscribe(ticker=ticker)

    active_bar: datetime.datetime = data_storage[ticker]['ActiveBar']
    last_update: datetime.datetime = data_storage[ticker]['LastUpdate']

    if fig and last_update and not active_bar:
        target = fig.get('layout', {}).get('title', {}).get('text')
        last_x = fig.get('data', {})[0].get('x', [None])[-1]

        # state = json.loads(args[0])
        # target = state['target']
        # last_x = state['LastUpdate']

        if target == ticker.upper() and last_x == last_update.strftime('%Y-%m-%dT%H:%M:%S'):
            return dash.no_update
        elif target != ticker.upper():
            reload = True
        else:
            reload = False
    else:
        reload = True

    bar_df = get_bar_df(ticker, CACHE_SIZE)

    if len(bar_df) < VISIBLE_SIZE:
        return dash.no_update

    volatility = bar_df.CLOSE.pct_change().rolling(20).std() * np.sqrt(365 * 1440)
    # noinspection PyTypeChecker
    cwt = scipy.signal.cwt(bar_df.CLOSE.pct_change().fillna(0.), scipy.signal.ricker, np.arange(1, 20))
    shared_x = bar_df.index
    fig = plotly.subplots.make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02
    )

    fig.add_trace(
        plotly.graph_objects.Candlestick(
            name=f"{ticker.upper()} Kline",
            x=shared_x,
            open=bar_df.OPEN,
            high=bar_df.HIGH,
            low=bar_df.LOW,
            close=bar_df.CLOSE
        ),
        row=1,
        col=1
    )

    fig.add_trace(
        plotly.graph_objects.Bar(
            name=f"{ticker.upper()} Volume",
            x=shared_x,
            y=bar_df.VOLUME
        ),
        row=2,
        col=1
    )

    fig.add_trace(
        plotly.graph_objects.Scatter(
            name=f"{ticker.upper()} Volatility",
            x=shared_x,
            y=volatility
        ),
        row=3,
        col=1
    )

    fig.add_trace(
        plotly.graph_objects.Heatmap(
            name=f"{ticker.upper()} CWT",
            x=shared_x,
            y=np.arange(1, 20),
            z=cwt,
            showscale=False
        ),
        row=4,
        col=1
    )

    visible_low = bar_df.LOW[-VISIBLE_SIZE:].min()
    visible_high = bar_df.HIGH[-VISIBLE_SIZE:].max()
    buff = (visible_high - visible_low) * 0.05

    fig.update_layout(
        title=ticker.upper(),
        xaxis_rangeslider_visible=False,
        showlegend=False,
        autosize=True,
        # xaxis_range=[shared_x[-60], shared_x[-1]],
        # xaxis2_range=[shared_x[-60], shared_x[-1]],
        # xaxis3_range=[shared_x[-60], shared_x[-1]],
        # xaxis4_range=[shared_x[-self.visible_size], shared_x[-1] + datetime.timedelta(seconds=300)],
        # xaxis={'zeroline': True, 'zerolinewidth': 2, 'zerolinecolor': 'black', 'ticks': 'outside', 'tickson': 'boundaries'},
        xaxis4={'zeroline': True, 'zerolinewidth': 2, 'zerolinecolor': 'black', 'ticks': 'outside', 'position': 0.5, 'range': [shared_x[-VISIBLE_SIZE], shared_x[-1] + datetime.timedelta(seconds=300)]},
        yaxis={'autorange': False, 'fixedrange': False, 'title': 'Price', 'range': [visible_low - buff, visible_high + buff]},
        yaxis2={'autorange': False, 'fixedrange': False, 'title': 'Volume', 'range': [0, bar_df.VOLUME[-VISIBLE_SIZE:].max() * 1.1]},
        yaxis3={'autorange': False, 'fixedrange': False, 'title': 'Volatility', 'tickformat': ',.2%', 'range': [0, volatility[-VISIBLE_SIZE:].max() * 1.1]},
        yaxis4={'title': 'CWT'}
    )

    if not reload:
        fig.update_layout(
            transition={'duration': 300, 'easing': 'cubic-in-out'}
        )

    LOGGER.debug('graph updated!')

    return fig


def render_monitor_text(ticker, status):
    if ticker is None:
        return dash.no_update

    last_update: datetime.datetime = data_storage[ticker]['LastUpdate']

    if not status:
        status = dash_html_components.Table(
            children=[
                dash_html_components.Tr(
                    children=[
                        dash_html_components.Td(
                            children=['Last Update Time:'],
                            style={'width': "50%", 'border': 'medium solid'}
                        ),
                        dash_html_components.Td(
                            children=[f'{datetime.datetime.utcnow():%Y-%m-%d %H:%M:%S}'],
                            style={'width': "50%", 'border': 'medium solid'}
                        )
                    ],
                    style={'width': "100%", 'height': '20%', 'border': 'medium solid'}
                ),
                dash_html_components.Tr(
                    children=[
                        dash_html_components.Td(
                            children=['Last Bar Time:'],
                            style={'width': "50%", 'border': 'medium solid'}
                        ),
                        dash_html_components.Td(
                            children=[f'{last_update + datetime.timedelta(seconds=60):%Y-%m-%d %H:%M:%S}' if last_update else 'NA'],
                            style={'width': "50%", 'border': 'medium solid'}
                        )
                    ],
                    style={'width': "100%", 'height': '20%', 'border': 'medium solid'}
                ),
                dash_html_components.Tr(
                    children=[
                        dash_html_components.Td(
                            children=['Monitoring Ticker:'],
                            style={'width': "50%", 'border': 'medium solid'}
                        ),
                        dash_html_components.Td(
                            children=[ticker.upper()],
                            style={'width': "50%", 'border': 'medium solid'}
                        )
                    ],
                    style={'width': "100%", 'height': '20%', 'border': 'medium solid'}
                ),
                dash_html_components.Tr(
                    children=[
                        dash_html_components.Td(
                            children=['Predicted Trajectory:'],
                            style={'width': "50%", 'border': 'medium solid'}
                        ),
                        dash_html_components.Td(
                            children=[dash_core_components.Graph(id='Kryptonite-Monitor-PredictionView', style={'width': "100%", 'height': '100%'}, config={'displayModeBar': False})],
                            style={'width': "50%", 'border': 'medium solid'}
                        )
                    ],
                    style={'width': "100%", 'height': '20%', 'border': 'medium solid'}
                ),
                dash_html_components.Tr(
                    children=[
                        dash_html_components.Td(
                            children=['Trade Status:'],
                            style={'width': "50%", 'border': 'medium solid'}
                        ),
                        dash_html_components.Td(
                            children=['Unknown'],
                            style={'width': "50%", 'border': 'medium solid'}
                        )
                    ],
                    style={'width': "100%", 'height': '20%', 'border': 'medium solid'}
                )
            ],
            style={'width': "100%", 'height': '100%', 'border': 'medium solid'}
        )
    else:
        # Last Update Time
        status['props']['children'][0]['props']['children'][1]['props']['children'][0] = f'{datetime.datetime.utcnow():%Y-%m-%d %H:%M:%S}'

        # Last Bar Time
        status['props']['children'][1]['props']['children'][1]['props']['children'][0] = f'{last_update + datetime.timedelta(seconds=60):%Y-%m-%d %H:%M:%S}' if last_update else 'NA'

        # Monitoring Ticker
        status['props']['children'][2]['props']['children'][1]['props']['children'][0] = ticker.upper()

        # Predicted Trajectory
        status['props']['children'][3]['props']['children'][1]['props']['children'][0]['props']['figure'] = plotly.graph_objects.Bar(x=['Up', 'Uncertain', 'Down'], y=[1 / 4, 1 / 2, 1 / 4])

        # Trade Status
        status['props']['children'][4]['props']['children'][1]['props']['children'][0] = 'Unknown'

    return status
