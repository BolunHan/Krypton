import abc
from collections import defaultdict
from typing import Dict

from ..Base import GlobalStatics, EVENT_ENGINE, LOGGER
from ..Res.ToolKit import TradeInstruction, MarketData, TradeReport

LOGGER = LOGGER.getChild('API')
__all__ = ['Market', 'Trade', 'Balance', 'Account']


def set_timer(interval, callback, event_engine=EVENT_ENGINE):
    topic = event_engine.get_timer(interval=interval)
    event_engine.register_handler(topic=topic, handler=callback)
    return topic


class Market(object, metaclass=abc.ABCMeta):
    def __init__(self, topic_set=GlobalStatics.TOPIC, event_engine=EVENT_ENGINE, logger=LOGGER):
        self.topic_set = topic_set
        self.event_engine = event_engine
        self.logger = logger

        self._subscribed = defaultdict(set)

    def register(self, topic_set=None, event_engine=None):
        if topic_set is not None:
            self.topic_set = topic_set

        if event_engine is not None:
            self.event_engine = event_engine

        self.event_engine.register_handler(topic=GlobalStatics.TOPIC.subscribe, handler=self.subscribe)

    def unregister(self):
        self.event_engine.unregister_handler(topic=GlobalStatics.TOPIC.subscribe, handler=self.subscribe)

    def _on_market(self, market_data: MarketData, **kwargs):
        self.event_engine.put(topic=self.topic_set.push(market_data=market_data), market_data=market_data, **kwargs)

    @abc.abstractmethod
    def _on_subscribe(self, ticker: str, dtype: str, **kwargs):
        ...

    def subscribe(self, **kwargs):
        topic = kwargs.pop('topic')
        ticker = topic['ticker']
        dtype = topic['dtype']
        if dtype not in self._subscribed[ticker]:
            self._on_subscribe(ticker=ticker, dtype=dtype, **kwargs)
            self.logger.debug(f'{self} subscribe {ticker} {dtype}')
            self._subscribed[ticker].add(dtype)
        else:
            self.logger.debug(f'{self} already subscribed {ticker} {dtype}')

    @property
    def subscription(self):
        return self._subscribed

    def set_timer(self, interval, callback):
        return set_timer(interval=interval, callback=callback, event_engine=self.event_engine)


class Trade(object, metaclass=abc.ABCMeta):
    def __init__(self, topic_set=GlobalStatics.TOPIC, event_engine=EVENT_ENGINE, logger=LOGGER):
        self.topic_set = topic_set
        self.event_engine = event_engine
        self.logger = logger

    def register(self, topic_set=None, event_engine=None):
        if topic_set is not None:
            self.topic_set = topic_set

        if event_engine is not None:
            self.event_engine = event_engine

        self.event_engine.register_handler(topic=self.topic_set.launch_order, handler=self.launch_order)
        self.event_engine.register_handler(topic=self.topic_set.cancel_order, handler=self.cancel_order)

    def unregister(self):
        self.event_engine.unregister_handler(topic=self.topic_set.launch_order, handler=self.launch_order)
        self.event_engine.unregister_handler(topic=self.topic_set.cancel_order, handler=self.cancel_order)

    def _on_order(self, order: TradeInstruction):
        self.event_engine.put(topic=self.topic_set.on_order, order=order)

    def _on_report(self, report: TradeReport):
        self.event_engine.put(topic=self.topic_set.on_report, report=report)

    def _on_error(self, **kwargs):
        self.event_engine.put(topic=self.topic_set.on_report, **kwargs)

    @abc.abstractmethod
    def launch_order(self, order: TradeInstruction, **kwargs) -> str:
        ...

    @abc.abstractmethod
    def cancel_order(self, order_id: TradeInstruction, **kwargs):
        ...

    @property
    def working_orders(self) -> Dict[str, TradeInstruction]:
        raise NotImplementedError

    @property
    def position(self) -> Dict[str, Dict[str, float]]:
        raise NotImplementedError

    @property
    def balance(self) -> Dict[str, float]:
        raise NotImplementedError


class Balance(object, metaclass=abc.ABCMeta):
    def __init__(self):
        raise NotImplementedError()


class Account(object, metaclass=abc.ABCMeta):
    def __init__(self):
        raise NotImplementedError()
