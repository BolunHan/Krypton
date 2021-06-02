import queue
import re
import threading
import traceback
from enum import Enum
from string import Formatter
from typing import Optional, Iterable, Dict, Union, List, Callable

from . import GLOBAL_LOGGER

LOGGER = GLOBAL_LOGGER.getChild('EventEngine')
__all__ = ['EventHook', 'EventEngine', 'Topic', 'RegularTopic', 'PatternTopic']


class Topic(dict):
    """
    topic for event hook. e.g. "TickData.002410.SZ.Realtime"
    """

    class Error(Exception):
        def __init__(self, msg):
            super().__init__(msg)

    def __init__(self, topic: str, *args, **kwargs):
        self._value = topic
        super().__init__(*args, **kwargs)

    def __repr__(self):
        return f'<{self.__class__.__name__}>({self._value}){super().__repr__()}'

    def __str__(self):
        return self.value

    def __bool__(self):
        return True

    def __hash__(self):
        return self.value.__hash__()

    def match(self, topic: str) -> Optional['Topic']:
        if self._value == topic:
            return self.__class__(topic=topic)
        else:
            return None

    @classmethod
    def cast(cls, topic: Union['Topic', str, Enum], dtype=None) -> 'Topic':
        if isinstance(topic, Enum):
            topic = topic.value
        elif isinstance(topic, Topic):
            return topic

        if dtype is None:
            if re.search(r'{(.+?)}', topic):
                t = PatternTopic(pattern=topic)
            elif '*' in topic or '+' in topic or '|' in topic:
                re.compile(pattern=topic)
                t = RegularTopic(pattern=topic)
            else:
                t = Topic(topic=topic)
        else:
            t = dtype(topic)

        return t

    @property
    def value(self) -> str:
        return self._value


class RegularTopic(Topic):
    """
    topic in regular expression. e.g. "TickData.(.+).((SZ)|(SH)).((Realtime)|(History))"
    """

    def __init__(self, pattern: str):
        super().__init__(topic=pattern)

    def match(self, topic: str) -> Optional[Topic]:
        if re.match(self._value, topic):
            match = Topic(topic=topic)
            match['pattern'] = self._value
            return match
        else:
            return None


class PatternTopic(Topic):
    """
    topic for event hook. e.g. "TickData.{symbol}.{market}.{flag}"
    """

    def __init__(self, pattern: str):
        super().__init__(topic=pattern)

    def __call__(self, **kwargs):
        return self.format_map(kwargs)

    def format_map(self, mapping: dict) -> Topic:
        for key in self.keys():
            if key not in mapping:
                mapping[key] = f'{{{key}}}'

        return Topic.cast(self._value.format_map(mapping))

    def string_to_dict(self, target: str, pattern: str):
        pattern = re.escape(pattern)
        regex = re.sub(r'\\{(.+?)\\}', r'(?P<_\1>.+)', pattern)
        match = re.match(regex, target)
        if match:
            values = list(match.groups())
            keys = re.findall(r'\\{(.+?)\\}', pattern)
            m = dict(zip(keys, values))
            return m
        else:
            raise self.Error(f'pattern {pattern} not in string {target} found!')

    def keys(self):
        keys = [i[1] for i in Formatter().parse(self._value) if i[1] is not None]
        return keys

    def match(self, topic: str) -> Optional[Topic]:
        try:
            keyword_dict = self.string_to_dict(target=topic, pattern=self._value)
            match = Topic(topic=topic)
            match.update(keyword_dict)
            return match
        except self.Error as _:
            return None

    @property
    def value(self) -> str:
        return self._value.format_map({_: '*' for _ in self.keys()})


class EventHook(object):
    """
    Event object with
    a string topic for event engine to distribute event,
    and a list of handler to process data
    """

    def __init__(self, topic: Topic, handler: Optional[Union[List[Callable], Callable]] = None):
        self.topic = topic

        if isinstance(handler, Callable):
            self.handlers = [handler]
        elif isinstance(handler, Iterable):
            self.handlers = []
            for hdl in handler:
                if isinstance(hdl, Callable):
                    self.handlers.append(hdl)
                else:
                    raise ValueError(f'invalid handler {hdl}')
        elif handler is None:
            self.handlers = []
        else:
            raise ValueError(f'Invalid handler {handler}')

    def __call__(self, *args, **kwargs):
        self.trigger(topic=self.topic, args=args, kwargs=kwargs)

    def __iadd__(self, handler):
        self.add_handler(handler)
        return self

    def __isub__(self, handler):
        self.remove_handler(handler)
        return self

    def trigger(self, topic: Topic, args: tuple = None, kwargs: dict = None):
        if args is None:
            args = ()

        if kwargs is None:
            kwargs = {}

        for handler in self.handlers:
            try:
                try:
                    handler(topic=topic, *args, **kwargs)
                except TypeError:
                    handler(*args, **kwargs)
            except Exception as _:
                LOGGER.error(traceback.format_exc())

    def add_handler(self, handler: Callable):
        self.handlers.append(handler)

    def remove_handler(self, handler: Callable):
        self.handlers.remove(handler)


class EventEngine(object):
    def __init__(self, max_size=0):
        self.lock = threading.Lock()

        self._max_size = max_size
        self._queue: queue.Queue = queue.Queue(maxsize=self._max_size)
        self._active: bool = False
        self._engine: threading.Thread = threading.Thread(target=self._run, name='EventEngine')
        self._event_hooks: Dict[Topic, EventHook] = {}

    def _run(self) -> None:
        """
        Get event from queue and then process it.
        """
        while self._active:
            try:
                event_dict = self._queue.get(block=True, timeout=1)
                topic = event_dict['topic']
                args = event_dict.get('args', ())
                kwargs = event_dict.get('kwargs', {})
                self._process(topic, *args, **kwargs)
            except queue.Empty:
                pass

    def _process(self, topic: str, *args, **kwargs) -> None:
        """
        distribute data to registered event hook in the order of registration
        """
        for _ in list(self._event_hooks):
            m = _.match(topic=topic)
            if m:
                event_hook = self._event_hooks.get(_)

                if event_hook is not None:
                    event_hook.trigger(topic=m, args=args, kwargs=kwargs)

    def start(self) -> None:
        """
        Start event engine to process events and generate timer events.
        """
        self._active = True
        self._engine.start()

    def stop(self) -> None:
        """
        Stop event engine.
        """
        self._active = False
        self._engine.join()

    def put(self, topic: Union[str, Topic], block: bool = True, timeout: float = None, *args, **kwargs):
        """
        fast way to put an event, kwargs MUST NOT contains "topic", "block" and "timeout" keywords
        :param topic: the topic to put into engine
        :param block: block if necessary until a free slot is available
        :param timeout: If 'timeout' is a non-negative number, it blocks at most 'timeout' seconds and raises the Full exception
        :param args: args for handlers
        :param kwargs: kwargs for handlers
        :return: nothing
        """
        self.publish(topic=topic, block=block, timeout=timeout, args=args, kwargs=kwargs)

    def publish(self, topic: Union[str, Topic], block: bool = True, timeout: float = None, args=None, kwargs=None):
        """
        safe way to publish an event
        :param topic: the topic to put into engine
        :param block: block if necessary until a free slot is available
        :param timeout: If 'timeout' is a non-negative number, it blocks at most 'timeout' seconds and raises the Full exception
        :param args: a list / tuple, args for handlers
        :param kwargs: a dict, kwargs for handlers
        :return: nothing
        """
        if isinstance(topic, Topic):
            topic = topic.value
        elif not isinstance(topic, str):
            raise ValueError(f'Invalid topic {topic}')

        event_dict = {'topic': topic}

        if args is not None:
            event_dict['args'] = args

        if kwargs is not None:
            event_dict['kwargs'] = kwargs

        self._queue.put(event_dict, block=block, timeout=timeout)

    def register_hook(self, hook: EventHook) -> None:
        """
        register a hook event
        """
        if hook.topic in self._event_hooks:
            for handler in hook.handlers:
                self._event_hooks[hook.topic].add_handler(handler)
        else:
            self._event_hooks[hook.topic] = hook

    def unregister_hook(self, topic: Topic) -> None:
        """
        Unregister an existing hook
        """
        if topic in self._event_hooks:
            self._event_hooks.pop(topic)

    def register_handler(self, topic: Topic, handler: Union[Callable, Iterable[Callable]]) -> None:
        """
        Register one or more handler for a specific topic
        """

        if not isinstance(topic, Topic):
            raise TypeError(f'Invalid topic {topic}')

        if topic not in self._event_hooks:
            self._event_hooks[topic] = EventHook(topic=topic, handler=handler)
        else:
            self._event_hooks[topic].add_handler(handler)

    def unregister_handler(self, topic: Topic, handler: Callable) -> None:
        """
        Unregister an existing handler function.
        """
        if topic in self._event_hooks:
            self._event_hooks[topic].remove_handler(handler=handler)

    @property
    def max_size(self):
        return self._max_size

    @max_size.setter
    def max_size(self, size: int):
        self._max_size = size
        self._queue.maxsize = size


if __name__ == '__main__':
    a = "TickData.002410.SZ.Realtime"
    b = "TickData.{symbol}.{market}.{flag}"
    c = "TickData.(.+).((SZ)|(SH)).((Realtime)|(History))"


    class TopicSet(object):
        t_a = Topic.cast(a)
        t_b = Topic.cast(b)
        t_c = Topic.cast(c)
        t_d = PatternTopic(b)(symbol='002410', market='((SZ)|(SH))', flag='Realtime')


    print(TopicSet.t_a)
    print(TopicSet.t_b)
    print(TopicSet.t_c)
    print(TopicSet.t_d)

    print(TopicSet.t_a.match(a))
    print(TopicSet.t_b.match(a))
    print(TopicSet.t_c.match(a))
