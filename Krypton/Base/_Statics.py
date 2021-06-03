import os
from types import SimpleNamespace

from ._Exceptions import Exceptions
from ..Res.ToolKit import Topic, PatternTopic, MarketData, get_current_path

__all__ = ['GlobalStatics']
FILE_PATH = get_current_path()


class _Constants(object):
    def __init__(self, value: str):
        self.value = value

    def __str__(self):
        return str(self.value)

    def __repr__(self):
        return f'<GlobalConstants>({self.value})'


class _TopicSet(object):
    on_order = Topic('on_order')
    on_report = Topic('on_report')
    on_error = Topic('on_error')

    launch_order = PatternTopic('launch_order.{ticker}')
    cancel_order = PatternTopic('cancel_order.{ticker}')
    query = PatternTopic('query.{ticker}.{dtype}')
    subscribe = PatternTopic('subscribe.{ticker}.{dtype}')
    realtime = PatternTopic('realtime.{ticker}.{dtype}')
    history = PatternTopic('history.{ticker}.{dtype}')

    @classmethod
    def push(cls, market_data: MarketData):
        return cls.realtime(ticker=market_data.ticker, dtype=market_data.__class__.__name__)

    @classmethod
    def parse(cls, topic: Topic) -> SimpleNamespace:
        try:
            _ = topic.value.split('.')

            action = _.pop(0)
            if action in ['open', 'close']:
                dtype = None
            else:
                dtype = _.pop(-1)
            ticker = '.'.join(_)

            p = SimpleNamespace(
                action=action,
                dtype=dtype,
                ticker=ticker
            )
            return p
        except Exception as _:
            raise Exceptions.TopicError(f'Invalid topic {topic}')


class _GlobalStatics(object):
    def __init__(self):
        self.CURRENCY = 'USDT'
        self.SHARE = 'x'
        self.TOPIC = _TopicSet

        if 'KRYPTON_CWD' in os.environ:
            self.WORKING_DIRECTORY = _Constants(os.path.realpath(os.environ['KRYPTON_CWD']))
        else:
            self.WORKING_DIRECTORY = _Constants(str(FILE_PATH.parent.parent))

        os.makedirs(self.WORKING_DIRECTORY.value, exist_ok=True)

    def add_static(self, name: str, value):
        setattr(self, name, _Constants(value))


GlobalStatics = _GlobalStatics()
