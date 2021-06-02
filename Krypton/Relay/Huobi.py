__package__ = 'Krypton.Relay'

import json
import os
import pathlib
import traceback
from typing import List

import redis

from ..API.Huobi.Spot import API
from ..Base import CONFIG, EVENT_ENGINE, LOGGER, GlobalStatics
from ..Res.ToolKit import TradeData, OrderBook, MarketData

LOGGER = LOGGER.getChild('Relay')
CWD = pathlib.Path(GlobalStatics.WORKING_DIRECTORY.value)


class RelayServer(object):
    def __init__(self, name: str, api: API.Market, subscribed_tickers: List[str], redis_conn: redis.Redis, cache_size: int = 1440):
        self.name = name
        self.api = api

        self.logger = LOGGER.getChild(self.name)
        self.logger.info(f'{self.logger.name} service started')
        self.subscribed_ticker = subscribed_tickers
        self.redis_conn = redis_conn
        self.cache_size = cache_size

        self.enable_redis = True
        self.cache_dir = CWD.joinpath('DataCache').joinpath(f'{self.name}')

        os.makedirs(self.cache_dir, exist_ok=True)

    @classmethod
    def to_json(cls, market_data: MarketData):
        if isinstance(market_data, TradeData):
            return json.dumps([
                market_data.timestamp,
                market_data.volume,
                market_data.price,
                market_data.side.sign
            ])
        elif isinstance(market_data, OrderBook):
            return json.dumps([
                market_data.timestamp,
                [entry.price for entry in market_data.bid],
                [entry.volume for entry in market_data.bid],
                [entry.price for entry in market_data.ask],
                [entry.volume for entry in market_data.ask]
            ])
        else:
            raise NotImplementedError()

    def _publish(self, market_data: MarketData, **_):
        # noinspection PyBroadException
        try:
            channel = f'{self.name}.{market_data.ticker}.{market_data.__class__.__name__}'
            self.logger.debug(f'channel <{channel}> publish: {market_data}')
            message = market_data.to_json()
            self.redis_conn.publish(channel, message)
            self.redis_conn.rpush(channel, message)
            self.redis_conn.ltrim(channel, -self.cache_size, -1)
        except Exception as _:
            self.logger.warning(traceback.format_exc())

    def _dump_data(self, market_data: MarketData, **_):
        dtype = market_data.__class__.__name__
        ticker = market_data.ticker
        market_date = market_data.market_time
        cache_dir = self.cache_dir.joinpath(ticker).joinpath(f'{market_date:%Y-%m}').joinpath(f'{market_date:%Y-%m-%d}')

        if not os.path.isdir(cache_dir):
            os.makedirs(cache_dir, exist_ok=True)

        cache_path = cache_dir.joinpath(f'{dtype}.{ticker}.{market_date:%Y%m%d%H}.json')

        if not os.path.isfile(cache_path):
            with open(cache_path, mode='w') as file:
                file.write(f'[\n  {self.to_json(market_data)}\n]')
        else:
            with open(cache_path, mode="r+") as file:
                file.seek(os.stat(cache_path).st_size - 2)
                file.write(f",\n  {self.to_json(market_data)}\n]")

    def _purge_record(self):
        self.logger.info('Clearing all redis record...')
        for key in self.redis_conn.keys(pattern='*'):
            self.redis_conn.flushall()
            self.redis_conn.delete(key)

    def subscribe_all(self):
        for ticker in self.subscribed_ticker:
            self.logger.info(f'Starting {ticker} subscription...')
            # noinspection PyBroadException

            self.api.event_engine.put(topic=self.api.topic_set.subscribe(ticker=ticker, dtype='TradeData'))
            self.api.event_engine.put(topic=self.api.topic_set.subscribe(ticker=ticker, dtype='OrderBook'))

        self.api.event_engine.register_handler(topic=self.api.topic_set.realtime, handler=self._dump_data)

        if self.enable_redis:
            self.api.event_engine.register_handler(topic=self.api.topic_set.realtime, handler=self._publish)

    def start(self):
        self.api.register()

        try:
            self._purge_record()
        except redis.exceptions.ConnectionError as _:
            self.enable_redis = False
            self.logger.error('Redis connection failed! publish function disabled!')

        self.subscribe_all()

        self.logger.info('Initialization all good!')


def main():
    # subscribe config
    subscribed_tickers = CONFIG.Relay.Huobi.SUBSCRIBED.split(',')

    # redis config
    port = CONFIG.Relay.REDIS_PORT
    password = CONFIG.Relay.REDIS_AUTH
    cache_size = CONFIG.Relay.CACHE_SIZE

    # api config
    tick_span = CONFIG.API.Huobi.Spot.TICK_SPAN
    market_url = CONFIG.API.Huobi.Spot.MARKET_URL
    proxy = CONFIG.API.Huobi.Spot.PROXY

    spot_market = API.Market(
        market_url=market_url,
        event_engine=EVENT_ENGINE,
        http_proxy=proxy,
        logger=LOGGER,
        tick_span=tick_span
    )

    redis_conn = redis.Redis(
        host='127.0.0.1',
        port=port,
        password=password,
        db=0,
        socket_keepalive=True
    )

    relay_server = RelayServer(
        name='Huobi',
        api=spot_market,
        subscribed_tickers=subscribed_tickers,
        redis_conn=redis_conn,
        cache_size=cache_size,
    )

    relay_server.start()


if __name__ == '__main__':
    main()
