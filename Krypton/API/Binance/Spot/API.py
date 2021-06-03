import datetime
import time

from .. import WebsocketsClient
from ... import Template
from ....Res.ToolKit import BarData, TickData, TradeData, OrderBook, TradeSide

__all__ = ['Market']


class Market(Template.Market):
    def __init__(
            self,
            market_url,
            event_engine,
            logger,
            http_proxy: str = None,
            bar_span='1m'
    ):
        super(Market, self).__init__(
            event_engine=event_engine,
            logger=logger,
        )

        self.market_client = WebsocketsClient.BinanceWebsocketsClient(
            base_url=market_url,
            proxy=http_proxy
        )

        self.bar_span = bar_span

    def _parse_bar(self, message, **kwargs):
        topic: str = kwargs['topic']
        ticker = topic.split('@')[0]

        bar_start_time = datetime.datetime.utcfromtimestamp(float(message['k']['t']) / 1000)
        bar_span = datetime.timedelta(milliseconds=float(message['k']['T']) - float(message['k']['t']))

        bar_data = BarData(
            ticker=ticker,
            bar_start_time=bar_start_time,
            bar_span=bar_span,
            open_price=float(message['k']['o']),
            close_price=float(message['k']['c']),
            high_price=float(message['k']['h']),
            low_price=float(message['k']['l']),
            volume=float(message['k']['v']),
            notional=float(message['k']['q']),
            trade_count=int(message['k']['n']),
            buy_notional=float(message['k']['Q']),
            buy_volume=float(message['k']['V'])
        )

        self._on_market(market_data=bar_data)

    def _parse_tick(self, message, **kwargs):
        topic = kwargs['topic']
        ticker = topic.split('@')[0]
        timestamp = float(message['E']) / 1000
        market_time = datetime.datetime.utcfromtimestamp(timestamp)

        tick_data = TickData(
            ticker=ticker,
            last_price=float(message['c']),
            market_time=market_time,
            timestamp=timestamp,
            total_traded_volume=float(message['v']),
            total_traded_notional=float(message['q']),
            total_trade_count=int(message['n']),
            bid_price=float(message['b']),
            ask_price=float(message['B']),
            bid_volume=float(message['a']),
            ask_volume=float(message['A']),
        )

        self._on_market(market_data=tick_data)

    def _parse_trade(self, message, **kwargs):
        topic = kwargs['topic']
        ticker = topic.split('@')[0]
        timestamp = float(message['T']) / 1000
        price = float(message['p'])
        volume = float(message['q'])
        side = TradeSide.LongOpen if message['m'] else TradeSide.ShortOpen

        trade_data = TradeData(
            ticker=ticker,
            trade_time=datetime.datetime.utcfromtimestamp(timestamp),
            timestamp=timestamp,
            trade_price=price,
            trade_volume=volume,
            side=side
        )

        self._on_market(market_data=trade_data)

    def _parse_order_book(self, message, **kwargs):
        topic = kwargs['topic']
        ticker = topic.split('@')[0]
        timestamp = time.time()

        order_book = OrderBook(
            ticker=ticker,
            market_time=datetime.datetime.utcfromtimestamp(timestamp),
            timestamp=timestamp
        )

        for entry in message['bids']:
            order_book.bid.add(price=float(entry[0]), volume=float(entry[1]))

        for entry in message['asks']:
            order_book.ask.add(price=float(entry[0]), volume=float(entry[1]))

        self._on_market(market_data=order_book)

    def _bar_thread(self, ticker: str):
        self.market_client.subscribe(topic=f'{ticker}@kline_{self.bar_span}', callback=self._parse_bar)

    def _tick_thread(self, ticker: str):
        self.market_client.subscribe(topic=f'{ticker}@ticker', callback=self._parse_tick)

    def _trade_thread(self, ticker: str):
        self.market_client.subscribe(topic=f'{ticker}@trade', callback=self._parse_trade)

    def _order_book_thread(self, ticker: str):
        self.market_client.subscribe(topic=f'{ticker}@depth20', callback=self._parse_order_book)

    def _on_subscribe(self, ticker: str, dtype: str, **kwargs):
        if dtype == 'BarData':
            self._bar_thread(ticker=ticker)
        elif dtype == 'TickData':
            self._tick_thread(ticker=ticker)
        elif dtype == 'TradeData':
            self._trade_thread(ticker=ticker)
        elif dtype == 'OrderBook':
            self._order_book_thread(ticker=ticker)
        else:
            self.logger.warning(f'Invalid dtype {dtype}')
