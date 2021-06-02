import abc
import datetime
import json
import logging
import os
import threading
import time
import traceback
import warnings
from typing import Optional, Union, Dict, Callable, Type, List

import redis

from .API import Template
from .Res.ToolKit import count_ordinal, get_current_path, MarketData, BarData, TradeData, OrderBook, TickData


class DataClient(object, metaclass=abc.ABCMeta):
    """
    Abstract parent class of RelayClient and MarketClient
    """

    @abc.abstractmethod
    def subscribe(self, *args, **kwargs): ...

    @abc.abstractmethod
    def on_feed(self, market_data: MarketData): ...


class RelayClient(DataClient):
    def __init__(
            self,
            url: str,
            on_bar: Callable[[BarData], None] = None,
            on_tick: Callable[[TickData], None] = None,
            on_trade: Callable[[TradeData], None] = None,
            on_order_book: Callable[[OrderBook], None] = None,
            sequential_callback: bool = True,
            logger: Optional[logging.Logger] = None,
            **kwargs
    ):
        """
        init a relay client
        :param url: a redis client, like:
            redis://[[username]:[password]]@localhost:6379/0
            rediss://[[username]:[password]]@localhost:6379/0
            unix://[[username]:[password]]@/path/to/socket.sock?db=0
        :param queue_maxsize: max_length of queue
        :param on_bar: callback for TickData
        :param on_tick: callback for TickData
        :param on_trade: callback for TradeData 
        :param on_order_book: callback for OrderBook
        :param sequential_callback: callback function will run in temporal sequential mode if True. Otherwise in parallel mode
        :param kwargs: additional parameter for redis client
        """
        self._on_bar = on_bar
        self._on_tick = on_tick
        self._on_trade = on_trade
        self._on_order_book = on_order_book

        self.url = url
        self.logger = logger if logger else logging.getLogger('Kryptonite.RelayClient')
        self.addition_para = kwargs
        self.sequential_callback = sequential_callback
        self.subscribed = set()
        self.p_subscribed = set()

        # add default para for redis client here
        self.addition_para.update(
            dict(
                socket_keepalive=True
            )
        )

        self.redis_conn = redis.Redis.from_url(self.url, **self.addition_para)
        self.redis_conn.connection_pool.connection_kwargs.update(self.addition_para)
        self.subscription = self.redis_conn.pubsub()
        self._listen_thread = threading.Thread(target=self._listen, args=[], daemon=False, name='KryptoniteDataClient_Listen')

        self._listen_thread.start()

    def subscribe(
            self,
            ticker: str,
            dtype: Union[Type[TickData], Type[TradeData], Type[BarData], Type[OrderBook]] = BarData,
            source: Optional[str] = None
    ):
        retry_count = 0
        max_retry = 10
        sleep_time = 15

        while True:
            if retry_count > max_retry:
                raise KeyError(f'{source} {ticker} {dtype.__name__} subscription failed! Too many retries.')

            try:
                if source is None:
                    keys = self.redis_conn.keys(f'*{ticker}.{dtype.__name__}')
                else:
                    keys = self.redis_conn.keys(f'*{source}.{ticker}.{dtype.__name__}')

                if not keys:
                    self.logger.warning(f'{source} {ticker} {dtype.__name__} not found! {count_ordinal(retry_count)} retry...')
                    retry_count += 1
                    time.sleep(sleep_time)
                    continue
                elif len(keys) > 1:
                    raise ValueError(f'multiple entry of {source} {ticker} {dtype.__name__} found')

                key = keys[0]

                self.subscribed.add(key)
                self.subscription.subscribe(key)
                self.logger.info(f'{source} {ticker} {dtype.__name__} subscription successful!')
                break
            except redis.exceptions.ConnectionError:
                self.reconnect()

            time.sleep(sleep_time)

    def psubscribe(self, pattern: str):
        try:
            self.p_subscribed.add(pattern)
            try:
                self.subscription.psubscribe(pattern)
                self.logger.info(f'{pattern} subscription successful!')
            except redis.exceptions.ConnectionError:
                self.reconnect()
                time.sleep(0.1)
        except redis.exceptions.ConnectionError:
            self.reconnect()

    def load_remote(
            self,
            ticker: str,
            source: Optional[str] = None,
            size: int = 100,
            dtype: Union[Type[TickData], Type[TradeData], Type[BarData], Type[OrderBook]] = BarData,
            reverse: bool = False
    ) -> Union[List[BarData], List[TradeData], List[TickData], Type[OrderBook]]:
        data_list = []
        while True:
            try:
                if source is None:
                    keys = self.redis_conn.keys(f'*{ticker}.{dtype.__name__}')
                else:
                    keys = self.redis_conn.keys(f'*{source}.{ticker}.{dtype.__name__}')

                if not keys:
                    raise Exception(f'{source} {ticker} {dtype.__name__} not found')
                elif len(keys) > 1:
                    raise Exception(f'multiple entry of {source} {ticker} {dtype.__name__} found')

                redis_data = self.redis_conn.lrange(keys[0], -size, -1)
                for data_json in redis_data:
                    data = dtype.from_json(data_json)
                    data_list.append(data)

                data_list = sorted(data_list, reverse=reverse)
                break
            except redis.exceptions.ConnectionError:
                self.reconnect()
                time.sleep(0.1)

        return data_list

    def _listen(self):
        while True:
            try:
                # while not self.subscription.subscribed:
                #     time.sleep(0.1)

                for message in self.subscription.listen():
                    # noinspection PyBroadException
                    try:
                        # noinspection SpellCheckingInspection
                        if message['type'] == 'subscribe':
                            self.logger.debug(f'Redis successfully subscribe channel: {message["channel"]}')
                        elif message['type'] == 'psubscribe':
                            self.logger.debug(f'Redis successfully subscribe channel: {message["channel"]}')
                        elif message['type'] in ['message', 'pmessage']:
                            channel: str = message['channel'].decode()
                            dtype = channel.split('.')[-1]
                            # ticker = channel.split('.')[-2]
                            # source = channel.split('.')[-3]
                            if dtype == 'TickData':
                                data = TickData.from_json(message['data'])
                            elif dtype == 'TradeData':
                                data = TradeData.from_json(message['data'])
                            elif dtype == 'OrderBook':
                                data = OrderBook.from_json(message['data'])
                            else:
                                raise ValueError(f'Invalid dtype {dtype}')

                            self.on_feed(market_data=data)
                        else:
                            warnings.warn(f'Message not recognized: {message}')
                    except Exception as _:
                        warnings.warn(traceback.format_exc())
            except redis.exceptions.ConnectionError:
                self.reconnect()
                time.sleep(0.1)

    # noinspection DuplicatedCode
    def on_feed(self, market_data):
        if isinstance(market_data, BarData):
            callback = self._on_bar
        elif isinstance(market_data, TickData):
            callback = self._on_tick
        elif isinstance(market_data, TradeData):
            callback = self._on_trade
        elif isinstance(market_data, OrderBook):
            callback = self._on_order_book
        else:
            self.logger.error(f'Invalid market data: {market_data}')
            return

        if self.sequential_callback:
            callback(market_data)
        else:
            threading.Thread(target=callback, args=[market_data])

    def reconnect(self):
        self.logger.info('Reconnect RelayClient')
        try:
            self.redis_conn.close()
        finally:
            self.redis_conn = redis.Redis.from_url(self.url, **self.addition_para)
            self.redis_conn.connection_pool.connection_kwargs.update(self.addition_para)
            self.subscription = self.redis_conn.pubsub()

            for key in self.subscribed:
                self.subscription.subscribe(key)

            for pattern in self.p_subscribed:
                self.subscription.psubscribe(pattern)

    @property
    def tickers(self) -> set:
        _tickers = set()
        keys = self.redis_conn.keys()

        for key in keys:
            key = key.decode()
            if '.BarData' in key:
                ticker = key.split('.')[-2]
                _tickers.add(ticker)

        return _tickers

    @classmethod
    def load_local(
            cls,
            ticker: str,
            source: Optional[str] = None,
            start_date: Optional[datetime.date] = None,
            end_date: Optional[datetime.date] = None,
            data_dir: Optional[Union[os.PathLike, str]] = None,
            dtype: Union[Type[TickData], Type[TradeData], Type[BarData], Type[OrderBook]] = BarData,
    ) -> Dict[Union[datetime.date, datetime.datetime], BarData]:
        bar_data_dict = {}

        if data_dir is None or not os.path.isdir(data_dir):
            if source is None:
                data_dir = get_current_path().parent.joinpath('SavedBarData', ticker)
            else:
                data_dir = get_current_path().parent.joinpath('SavedBarData', source, ticker)

        file_list = [x for x in os.listdir(data_dir) if ticker in x]

        date_list = [datetime.datetime.strptime(x.replace(f'{ticker}_', '').replace('.json', ''), '%Y%m%d').date() for x in file_list]

        if start_date is not None:
            date_list = [x for x in date_list if x >= start_date]

        if end_date is not None:
            date_list = [x for x in date_list if x <= end_date]

        for market_date in sorted(date_list):
            file_path = os.path.join(data_dir, f'{ticker}_{market_date:%Y%m%d}.json')
            bar_data_json_list = json.load(open(file_path))
            for bar_data_json in bar_data_json_list:
                bar_data = dtype.from_json(bar_data_json)
                bar_data_dict[bar_data.bar_start_time] = bar_data

        if not bar_data_dict:
            warnings.warn(f'Empty {ticker} bar data!')

        return bar_data_dict


class MarketClient(DataClient):
    def __init__(
            self,
            api: Template.Market,
            on_bar: Callable[[BarData], None] = None,
            on_tick: Callable[[TickData], None] = None,
            on_trade: Callable[[TradeData], None] = None,
            on_order_book: Callable[[OrderBook], None] = None,
            sequential_callback: bool = True,
            logger: Optional[logging.Logger] = None,
            **kwargs
    ):
        self._on_bar = on_bar
        self._on_tick = on_tick
        self._on_trade = on_trade
        self._on_order_book = on_order_book

        self.api = api
        self.logger = logger if logger else logging.getLogger('Kryptonite.MarketClient')
        self.addition_para = kwargs
        self.sequential_callback = sequential_callback
        self.subscribed = set()

    def subscribe(self, ticker: str, dtype: Union[Type[TickData], Type[TradeData], Type[BarData], Type[OrderBook]] = BarData):
        subscribed_key = f'{ticker}.{dtype.__name__}'

        if subscribed_key not in self.subscribed:
            self.api.subscribe(ticker=ticker, dtype=dtype, callback=self.on_feed)
            self.subscribed.add(subscribed_key)
        else:
            warnings.warn(f'Duplicated subscription {ticker} of {dtype.__name__}! Ignored!')

    # noinspection DuplicatedCode
    def on_feed(self, market_data):
        if isinstance(market_data, BarData):
            callback = self._on_bar
        elif isinstance(market_data, TickData):
            callback = self._on_tick
        elif isinstance(market_data, TradeData):
            callback = self._on_trade
        elif isinstance(market_data, OrderBook):
            callback = self._on_order_book
        else:
            self.logger.error(f'Invalid market data: {market_data}')
            return

        if self.sequential_callback:
            callback(market_data)
        else:
            threading.Thread(target=callback, args=[market_data])


if __name__ == '__main__':
    pass
