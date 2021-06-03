__package__ = 'Krypton.Relay'

import os
import sys

import redis

sys.path.append(os.getcwd())

from . import RelayServer, LOGGER
from ..API.Huobi.Spot import API
from ..Base import CONFIG, EVENT_ENGINE

LOGGER = LOGGER.getChild('Huobi')


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
