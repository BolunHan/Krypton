import functools
import inspect
import os
from collections import defaultdict
from typing import Optional

import pandas as pd

from . import GLOBAL_LOGGER

LOGGER = GLOBAL_LOGGER.getChild(os.path.basename(os.path.splitext(__file__)[0]))
__all__ = ['Decorator', 'NestedClass']


class Decorator(object):

    @classmethod
    def get_real_args(cls, function, *args, **kwargs):
        signature = inspect.signature(function)
        arg_names = list(signature.parameters)
        arg_dict = {}

        for k in range(len(args)):
            arg_dict[arg_names[k]] = args[k]
        arg_dict.update(kwargs)

        for arg_name in arg_names:
            if arg_name not in arg_dict:
                default_value = signature.parameters[arg_name].default
                if default_value is not inspect.Parameter.empty:
                    arg_dict[arg_name] = default_value

        return arg_dict

    @classmethod
    def empty(cls, function):
        """
        An empty decorator for testing purpose
        :param function:
        :return:
        """

        @functools.wraps(function)
        def wrapper(*args, **kwargs):
            result = function(*args, **kwargs)
            return result

        return wrapper

    @classmethod
    def dynamic(cls, wrapper_name: str, with_self: bool = False, with_cls: bool = False):
        """
        A dynamic decorator to extract the actual wrapper from self class and execute
        :param wrapper_name: the name string of the actual wrapper attribute
        :param with_self: pass self to the decorated function as first of args if True
        :param with_cls: pass self.__class__ to the decorated function as first of args if True
        :return: a function decorator
        :return:
        """

        if with_self and with_cls:
            raise ValueError('with_self and with_cls MUST NOT be both True')

        def decorator(function: callable):
            @functools.wraps(function)
            def wrapper(self, *args, **kwargs):
                dynamic_wrapper = getattr(self, wrapper_name, None)
                arg_dict = cls.get_real_args(function, *args, **kwargs)

                if dynamic_wrapper is None:
                    if with_self:
                        result = function(self, **arg_dict)
                    elif with_cls:
                        result = function(self.__class__, **arg_dict)
                    else:
                        result = function(**arg_dict)
                else:
                    if with_self:
                        result = dynamic_wrapper(function)(self, **arg_dict)
                    elif with_cls:
                        result = dynamic_wrapper(function)(self.__class__, **arg_dict)
                    else:
                        result = dynamic_wrapper(function)(**arg_dict)
                return result

            return wrapper

        return decorator

    @classmethod
    def with_interval(cls, interval: int, reset_index: Optional[pd.Index] = None):
        """
        calculate factor with interval parameter
        only detect args with name 'open_price', 'close_price', 'high_price', 'low_price', 'volume', 'notional' or '*_df',
        skip args with name '*_static'
        :param interval:
        :param reset_index:
        :return:
        """
        if reset_index is None:
            need_update = True
        else:
            need_update = False

        def decorator(function: callable):
            def wrapper(*args, **kwargs):
                nonlocal reset_index

                arg_dict = cls.get_real_args(function, *args, **kwargs)

                target_value_dict = defaultdict(list)
                result_dict = {}
                for i in range(interval):
                    interval_kwargs = {}

                    for attribute_name, attribute_value in arg_dict.items():
                        if '_static' in attribute_name:
                            interval_kwargs[attribute_name] = attribute_value
                        elif attribute_name in ['open_price', 'close_price', 'high_price', 'low_price', 'volume', 'notional'] \
                                or '_df' in attribute_name:

                            if need_update:
                                reset_index = attribute_value.index

                            if 'high' in attribute_name:
                                # noinspection SpellCheckingInspection
                                target_attribute_value = attribute_value.reset_index(drop=True).fillna(method='ffill').rolling(window=interval).max().iloc[i::interval]
                            elif 'low' in attribute_name:
                                # noinspection SpellCheckingInspection
                                target_attribute_value = attribute_value.reset_index(drop=True).fillna(method='ffill').rolling(window=interval).min().iloc[i::interval]
                            elif 'notional' in attribute_name or 'volume' in attribute_name:
                                # noinspection SpellCheckingInspection
                                target_attribute_value = attribute_value.reset_index(drop=True).fillna(method='ffill').rolling(window=interval).sum().iloc[i::interval]
                            else:
                                # noinspection SpellCheckingInspection
                                target_attribute_value = attribute_value.reset_index(drop=True).fillna(method='ffill').iloc[i::interval]

                            interval_kwargs[attribute_name] = target_attribute_value
                        else:
                            interval_kwargs[attribute_name] = attribute_value

                    result = function(**interval_kwargs)

                    if isinstance(result, dict):
                        for key, value in result.items():
                            target_value_dict[key].append(value)
                    else:
                        key = '@LoneEntry'
                        target_value_dict[key].append(result)

                for key, target_value_list in target_value_dict.items():
                    target_value = pd.concat(target_value_list, axis=0, sort=False).sort_index()
                    target_value.index = reset_index
                    result_dict[key] = target_value

                if '@LoneEntry' in result_dict:
                    return result_dict['@LoneEntry']
                else:
                    return dict(result_dict)

            return wrapper

        return decorator

    @classmethod
    def deprecated(cls, message: str, logger=None):
        """
        Raise a deprecated warning
        :param message: the deprecation warning message
        :param logger: a logger to log this warning, if exist
        :return:
        """

        def decorator(function: callable):
            @functools.wraps(function)
            def wrapper(*args, **kwargs):
                if logger:
                    logger.warning(DeprecationWarning(message), stacklevel=2)

                result = function(*args, **kwargs)
                return result

            return wrapper

        return decorator


class NestedClass(object):
    SubClass = Decorator.dynamic(wrapper_name='_subclassing', with_self=False, with_cls=False)

    def _subclassing(self, cls):
        parent = self

        class SubClass(cls):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)

                if hasattr(self, 'parent'):
                    if self.parent == parent:
                        LOGGER.debug(f'{cls.__name__} already hasattr [parent] with default value!')
                    else:
                        LOGGER.warning(f'{cls.__name__} already hasattr [parent] with different value!')
                else:
                    self.parent = parent

        SubClass.__name__ = cls.__name__

        return SubClass

    @staticmethod
    def subclass(with_self=False, with_cls=False):
        return Decorator.dynamic(wrapper_name='_subclassing', with_self=False, with_cls=False)
