import os
import pickle
import queue
import re
import threading
import time
import traceback
import uuid
from collections import defaultdict, deque
from typing import Iterable, List, Callable, Optional, Union

import numpy as np

from . import GLOBAL_LOGGER
from .Exceptions import Exceptions

LOGGER = GLOBAL_LOGGER.getChild(os.path.basename(os.path.splitext(__file__)[0]))
__all__ = ['AttrDict', 'SizedQueue']


class AttrDict(defaultdict):
    """A dictionary whose value can be access by attribute, with default factory for quick initialization.
    Useful for storing, accessing and altering nested parameters
    Keep the code clean and readable
    e.g.
        parameter = AttrDict({'MovingAverage': AttrDict({'Window': 10, 'Target': 'ClosePrice'})})
        ...
        parameter.update({'MovingAverage.Window' = 20})
        ...
        factor = df[parameter.MovingAverage.Target].rolling(parameter.MovingAverage.Window).mean()
    """

    def __init__(
            self,
            init_dict: Optional[dict] = None,
            default_factory: Optional[callable] = None,
            alphanumeric: Optional[bool] = None,
            nested: Optional[bool] = None,
            **kwargs
    ):
        """
        Init a AttrDict Instance
        :param init_dict: positional only, init value in dict format
        :param default_factory: default factory for SuperClass
        :param alphanumeric: specify whether dict key must be alphanumeric or not
        :param nested: specify if it is a nested AttrDict
        :param kwargs: Additional key value pair as default value, will append the item after init_dict
        """
        # set default value for nested and alphanumeric
        if nested is None and alphanumeric is None:
            nested = alphanumeric = True
        elif nested is None:
            nested = alphanumeric
        elif alphanumeric is None:
            alphanumeric = nested
        else:
            if nested and (not alphanumeric):
                raise Exceptions.InvalidValue(f'Nested {self.__class__.__name__} must be alphanumeric, use alphanumeric=True')

        # set default value for default_factory
        if nested and default_factory is None:
            default_factory = self.__class__

        super().__init__(default_factory)
        # super().__setattr__('alphanumeric', alphanumeric)
        # super().__setattr__('nested', nested)
        self.alphanumeric = alphanumeric
        self.nested = nested
        if init_dict is not None:
            self.update(init_dict)

        self.update(kwargs)

    def __reduce__(self):
        return self.__class__, (dict(self), self.default_factory, self.alphanumeric, self.nested)

    def __setstate__(self, state):
        data, self.default_factory, self.alphanumeric, self.nested = state
        self.update(data)

    def __getstate__(self):
        return dict(self), self.default_factory, self.alphanumeric, self.nested

    def __str__(self):
        return f'{dict(self)}'

    def __repr__(self):
        return f'<{self.__class__.__name__}>({self.default_factory}, {dict(self)})'

    def __getitem__(self, item):
        return self._get_entry(key=item)

    def __setitem__(self, key, value):
        self._set_entry(key=key, value=value)

    def __getattr__(self, item):
        return self._get_entry(key=item)

    def __setattr__(self, key, value):
        if key in dir(self.__class__):
            super().__setattr__(key, value)
        else:
            return self._set_entry(key=key, value=value)

    def _set_mapping(self, key, value):
        if self.valid_attr(key=key, nested=False):
            super().__setattr__(key, value)
        super().__setitem__(key, value)

    def _get_mapping(self, key):
        return super().__getitem__(key)

    def pop(self, key, *args):

        if len(args) > 1:
            raise TypeError('Only one fallback value is allowed')
        elif len(args) == 1:
            value = super().pop(key, args[0])
        else:
            value = super().pop(key)

        if key in self.__dict__:
            delattr(self, key)
        return value

    def popitem(self):
        key, value = super().popitem()
        if key in self.__dict__:
            delattr(self, key)
        return key, value

    @classmethod
    def valid_attr(cls, key, nested: bool = False):
        """
        Check whether a key is a valid attribute name.
        A key may be used as an attribute if:
         * It is a string
         * It matches /^[A-Za-z][A-Za-z0-9_]*$/ (i.e., a public attribute)
         * The key doesn't overlap with any class attributes (for Attr,
            those would be 'get', 'items', 'keys', 'values', 'mro', and
            'register').
        """
        if nested:
            return (
                    isinstance(key, str) and
                    re.match('^([a-zA-Z_][a-zA-Z0-9_]*[.]?)*$', key) and not key.endswith('.') and
                    not hasattr(cls, key)
            )
        else:
            return (
                    isinstance(key, str) and
                    re.match('^[a-zA-Z_][a-zA-Z0-9_]*$', key) and
                    not hasattr(cls, key)
            )

    def _set_entry(self, key, value):
        if self.alphanumeric and not self.valid_attr(key=key, nested=self.__getattribute__('nested')):
            raise Exceptions.InvalidKey(f'key must be alphanumeric, invalid key [{key}]')

        if key in self:
            self._set_mapping(key, value)
        else:
            if self.nested:
                key_list = key.split('.')

                _value = self
                for _key in key_list[:-1]:
                    if _key in _value:
                        _new_value = _value._get_mapping(_key)
                    else:
                        _new_value = self.__class__(default_factory=self.__class__, nested=True, alphanumeric=True)
                        _value._set_mapping(_key, _new_value)

                    if not isinstance(_new_value, self.__class__):
                        raise Exceptions.InvalidKey(f'[{_key}] in [{key}] can not be resolved')

                    _value = _new_value

                _last_key = key_list[-1]
                _value._set_mapping(_last_key, value)
            else:
                self._set_mapping(key, value)

    def _get_entry(self, key):
        if self.nested:
            key_list = key.split('.')

            _value = self
            for _key in key_list[:-1]:
                _value = _value._get_mapping(_key)

                if not isinstance(_value, self.__class__):
                    raise Exceptions.InvalidKey(f'[{_key}] in [{key}] can not be resolved')

            _last_key = key_list[-1]
            value = _value._get_mapping(_last_key)
        else:
            value = self._get_mapping(key)

        return value

    def flattened(self):
        """
        Convert nested AttrDict to flattened AttrDict
        :return: a flattened AttrDict
        """

        def flatten(input_dict):
            flattened_dict = {}
            for key, value in input_dict.items():
                if isinstance(value, defaultdict) and hasattr(value, 'flattened'):
                    for sub_key, sub_value in flatten(value).items():
                        flattened_dict[f'{key}.{sub_key}'] = sub_value
                else:
                    flattened_dict[key] = value

            return self.__class__(flattened_dict, nested=False, alphanumeric=False)

        return flatten(self)

    def nest(self):
        """
        Convert flattened AttrDict to nested AttrDict
        :return: a nested AttrDict
        """
        raise NotImplementedError

    def update(self, other, *args) -> None:
        to_update = [other]
        to_update.extend(args)

        for source in to_update:
            if 'keys' in dir(source):
                for key in source.keys():
                    self._set_entry(key, source[key])
            else:
                for key, value in source:
                    self._set_entry(key, value)

    @property
    def alphanumeric(self):
        return self.__dict__.get('alphanumeric')

    @alphanumeric.setter
    def alphanumeric(self, value):
        self.__dict__['alphanumeric'] = bool(value)

    @property
    def nested(self):
        return self.__dict__.get('nested')

    @nested.setter
    def nested(self, value):
        self.__dict__['nested'] = bool(value)


class SizedQueue(queue.Queue):
    """
    A specialized queue with a given size.
    Once the queue reaches its max_size, the oldest (earliest) item will be popped out to append new ones in.
    """

    def __init__(
            self,
            init_entry: Optional[Iterable] = None,
            max_size: Optional[int] = None,
            order: str = 'fifo',
            in_queue: Optional[Union[List[Callable], Callable]] = None,
            out_queue: Optional[Union[List[Callable], Callable]] = None,
            drop_queue: Optional[Union[List[Callable], Callable]] = None,
    ):
        """
        Initialize SizedQueue with provided parameters
        :param init_entry: the init values given to the queue
        :param max_size: max size of the queue
        :param order: 'fifo' for first-in-first-out or 'lifo' for last-in-first-out
        """
        super(SizedQueue, self).__init__(max_size)

        self._order = order.upper()

        self.callback = AttrDict()

        self.callback.in_queue = [] if in_queue is None else in_queue if isinstance(in_queue, Iterable) else [in_queue]
        self.callback.out_queue = [] if out_queue is None else out_queue if isinstance(out_queue, Iterable) else [out_queue]
        self.callback.drop_queue = [] if drop_queue is None else drop_queue if isinstance(drop_queue, Iterable) else [drop_queue]

        # set init values
        if init_entry is not None:
            for v in init_entry:
                self.put(v, block=False)

    def _init(self, maxsize: int) -> None:
        self.queue = deque(maxlen=maxsize)

    def put(self, item, block=True, timeout=None):
        """Put an item into the queue.

        If optional args 'block' is true and 'timeout' is None (the default), block if necessary until a free slot is available.
        If 'timeout' is a non-negative number, it blocks at most 'timeout' seconds and drop the earliest item.
        Otherwise ('block' is false), put an item on the queue if a free slot is immediately available, else drop the earliest item and then put ('timeout' is ignored in that case).
        """
        with self.not_full:
            if self.maxsize > 0:
                if not block:
                    pass
                elif timeout is None:
                    while self._qsize() >= self.maxsize:
                        self.not_full.wait()
                elif timeout < 0:
                    raise ValueError("'timeout' must be a non-negative number")
                else:
                    end_time = time.monotonic() + timeout
                    while self._qsize() >= self.maxsize:
                        remaining = end_time - time.monotonic()
                        if remaining <= 0.0:
                            break
                        self.not_full.wait(remaining)
            self._put(item)
            self.unfinished_tasks += 1
            self.not_empty.notify()

    # Get an item from the queue
    def _get(self):
        if self.order == 'FIFO':
            item = self.queue.popleft()
        elif self.order == 'LIFO':
            item = self.queue.pop()
        else:
            raise ValueError(f'Invalid Order {self.order}')

        self._callback(flag='out_queue', item=item)
        return item

    # Put a new item in the queue
    def _put(self, item):
        while self._qsize() >= self.maxsize:
            _ = self.queue.popleft()
            self._callback(flag='drop_queue', item=_)

        self.queue.append(item)
        self._callback(flag='in_queue', item=item)

    @property
    def order(self):
        return self._order

    @order.setter
    def order(self, order):
        self._order = order.upper()

    def add_callback(self, callback_func: callable, flag: str = 'in_queue'):
        if flag in self.callback:
            self.callback[flag].append(callback_func)
        else:
            raise Exceptions.InvalidKey(f'Invalid callback flag {flag}')

    def _callback(self, flag: str, item):
        for callback in self.callback[flag]:
            # noinspection PyBroadException
            try:
                threading.Thread(target=callback, args=[item]).start()
            except Exception as _:
                LOGGER.warning(traceback.format_exc())


class RedisQueue(object):
    import redis

    class Empty(Exception):
        def __init__(self, error_message: str, error_code: int = 8011, *args, **kwargs):
            super(RedisQueue.Empty, self).__init__(error_message, *args)
            self.error_code = error_code

            for kwarg in kwargs:
                setattr(self, kwarg, kwargs[kwarg])

    class Full(Exception):
        def __init__(self, error_message: str, error_code: int = 8012, *args, **kwargs):
            super(RedisQueue.Full, self).__init__(error_message, *args)
            self.error_code = error_code

            for kwarg in kwargs:
                setattr(self, kwarg, kwargs[kwarg])

    class Invalid(Exception):
        def __init__(self, error_message: str, error_code: int = 8010, *args, **kwargs):
            super(RedisQueue.Invalid, self).__init__(error_message, *args)
            self.error_code = error_code

            for kwarg in kwargs:
                setattr(self, kwarg, kwargs[kwarg])

    class Disconnected(Exception):
        def __init__(self, error_message: str, error_code: int = 8010, *args, **kwargs):
            super(RedisQueue.Disconnected, self).__init__(error_message, *args)
            self.error_code = error_code

            for kwarg in kwargs:
                setattr(self, kwarg, kwargs[kwarg])

    def __init__(self, connection: 'RedisQueue.redis.Redis', key: str = None, max_size: Optional[int] = None, refresh_interval: float = 1.0):
        self.redis_conn = connection
        self.redis_args = connection.connection_pool.connection_kwargs
        if key is None:
            self.key = f'Queue:{uuid.uuid4()}'
        else:
            self.key = key

        if max_size is not None:
            assert max_size > 0, 'Invalid max size'

        self.max_size = max_size
        self.refresh_interval = refresh_interval

    def validate_key(self):
        keys = self.redis_conn.keys(pattern=self.key)
        if keys:
            LOGGER.warning(f'Key {self.key} already exist in redis')

    def put(self, message, block: bool = True, timeout: Optional[int] = None):
        while True:
            try:
                start_time = time.time()

                if self.max_size is not None:
                    # check block
                    if block:
                        while True:
                            size = self.redis_conn.llen(self.key)
                            if size >= self.max_size:
                                # block indefinitely
                                if timeout is None:
                                    pass
                                # check timeout
                                else:
                                    if time.time() - start_time > timeout:
                                        raise self.Full(f'<redis://{self.redis_args.get("host")}:{self.redis_args.get("port")}/> Redis Queue {self.key} Full!')
                                time.sleep(self.refresh_interval)
                            else:
                                break
                    # check non block
                    else:
                        size = self.redis_conn.llen(self.key)
                        if size >= self.max_size:
                            raise self.Full(f'<redis://{self.redis_args.get("host")}:{self.redis_args.get("port")}/> Redis Queue {self.key} Full!')
                        else:
                            pass
                else:
                    pass

                self.redis_conn.rpush(self.key, pickle.dumps(message))
                break
            except self.redis.exceptions.ConnectionError:
                self.reconnect()

            time.sleep(self.refresh_interval)

    def get(self, block: bool = False, timeout: Optional[int] = None):
        while True:
            try:
                # check block
                if block:
                    # block indefinitely
                    if timeout is None:
                        message = self.redis_conn.blpop(keys=self.key, timeout=0)
                    else:
                        if timeout == 0:
                            raise self.Invalid('Invalid timeout!')

                        message = self.redis_conn.blpop(keys=self.key, timeout=int(np.ceil(timeout)))
                # check non block
                else:
                    message = self.redis_conn.lpop(self.key)

                if message is None:
                    raise self.Empty(f'<redis://{self.redis_args.get("host")}:{self.redis_args.get("port")}/> Redis Queue {self.key} Empty!')

                break
            except self.redis.exceptions.ConnectionError:
                self.reconnect()

            time.sleep(self.refresh_interval)

        return pickle.loads(message[1])

    def force_put(self, message):
        while True:
            try:
                if self.max_size is not None:
                    self.redis_conn.rpush(self.key, pickle.dumps(message))
                    self.redis_conn.ltrim(self.key, -self.max_size, -1)
                break
            except self.redis.exceptions.ConnectionError:
                self.reconnect()

    def close(self, delete_queue=False):
        try:
            if delete_queue:
                self.redis_conn.delete(self.key)
            self.redis_conn.close()
        except self.redis.exceptions.ConnectionError:
            pass

    def reconnect(self):
        try:
            self.redis_conn.close()
        finally:
            self.redis_conn = self.redis.Redis(**self.redis_args)
            try:
                self.redis_conn.ping()
            except self.redis.exceptions.ConnectionError:
                raise self.Disconnected(f'<redis://{self.redis_args.get("host")}:{self.redis_args.get("port")}/> Redis disconnected!')

    @property
    def queue(self):
        result = deque()

        while True:
            try:
                message_list = self.redis_conn.lrange('test', 0, -1)
                break
            except self.redis.exceptions.ConnectionError:
                self.reconnect()

        for message in message_list:
            result.append(pickle.loads(message))

        return result
