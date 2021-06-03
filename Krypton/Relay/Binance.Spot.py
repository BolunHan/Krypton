__package__ = 'Krypton.Relay'

import os
import sys

import redis

sys.path.append(os.getcwd())

from . import RelayServer, LOGGER
from ..API.Binance.Spot import API
from ..Base import CONFIG, EVENT_ENGINE

LOGGER = LOGGER.getChild('Binance')


def main():
    # subscribe config
    subscribed_tickers = CONFIG.Relay.Binance.SUBSCRIBED.split(',')

    # redis config
    port = CONFIG.Relay.REDIS_PORT
    password = CONFIG.Relay.REDIS_AUTH
    cache_size = CONFIG.Relay.CACHE_SIZE

    # api config
    market_url = CONFIG.API.Binance.Spot.MARKET_URL  # wss://stream.binance.com:9443
    proxy = CONFIG.API.Binance.Spot.PROXY

    spot_market = API.Market(
        market_url=market_url,
        event_engine=EVENT_ENGINE,
        http_proxy=proxy,
        logger=LOGGER,
    )

    redis_conn = redis.Redis(
        host='127.0.0.1',
        port=port,
        password=password,
        db=0,
        socket_keepalive=True
    )

    relay_server = RelayServer(
        name='Binance',
        api=spot_market,
        subscribed_tickers=subscribed_tickers,
        redis_conn=redis_conn,
        cache_size=cache_size,
    )

    relay_server.start()


if __name__ == '__main__':
    main()
