import datetime
import threading
import time
import traceback
from typing import Optional, Dict, Union

from ._Telemetric import LOGGER
from ..Res.ToolKit import pretty_timedelta, EventHook as EventHookBase, EventEngine as EventEngineBase, Topic

__all__ = ['EVENT_ENGINE']
LOGGER = LOGGER.getChild('Base.EventEngine')


class EventHook(EventHookBase):
    def __init__(self, topic, handler):
        self.logger = LOGGER.getChild(f'EventHook.{topic}')
        super().__init__(topic=topic, handler=handler)

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
                self.logger.error(traceback.format_exc())


class EventEngine(EventEngineBase):
    def __init__(self, max_size=0):
        super().__init__(max_size=max_size)
        self.timer: Dict[Union[float, str], threading.Thread] = {}

    def register_handler(self, topic, handler):
        topic = Topic.cast(topic)
        super().register_handler(topic=topic, handler=handler)

    def publish(self, topic, block: bool = True, timeout: float = None, args=None, kwargs=None):
        topic = Topic.cast(topic)
        super().publish(topic=topic, block=block, timeout=timeout, args=args, kwargs=kwargs)

    def unregister_hook(self, topic) -> None:
        topic = Topic.cast(topic)
        super().unregister_hook(topic=topic)

    def unregister_handler(self, topic, handler) -> None:
        topic = Topic.cast(topic)

        try:
            super().unregister_handler(topic=topic, handler=handler)
        except ValueError as _:
            raise ValueError(f'unregister topic {topic} failed! handler {handler} not found!')

    def get_timer(self, interval: Union[datetime.timedelta, float, int], activate_time: Optional[datetime.datetime] = None) -> Topic:
        """
        Start a timer, if not exist, and get topic of the timer event
        :param interval: timer event interval in seconds
        :param activate_time: UTC, timer event only start after active_time. This arg has no effect if timer already started.
        :return: the topic of timer event hook
        """
        if isinstance(interval, datetime.timedelta):
            interval = interval.total_seconds()

        if interval == 1:
            topic = Topic('EventEngine.Internal.Timer.Second')
            timer = threading.Thread(target=self._second_timer, kwargs={'topic': topic})
        elif interval == 60:
            topic = Topic('EventEngine.Internal.Timer.Minute')
            timer = threading.Thread(target=self._minute_timer, kwargs={'topic': topic})
        else:
            topic = Topic(f'EventEngine.Internal.Timer.{interval}')
            timer = threading.Thread(target=self._run_timer, kwargs={'interval': interval, 'topic': topic, 'activate_time': activate_time})

        if interval not in self.timer:
            self.timer[interval] = timer
            timer.start()
        else:
            if activate_time is not None:
                LOGGER.debug(f'Timer thread with interval [{pretty_timedelta(datetime.timedelta(seconds=interval))}] already initialized! Argument [activate_time] takes no effect!')

        return topic

    def _run_timer(self, interval: Union[datetime.timedelta, float, int], topic: Topic, activate_time: Optional[datetime.datetime] = None) -> None:
        if isinstance(interval, datetime.timedelta):
            interval = interval.total_seconds()

        if activate_time is None:
            scheduled_time = datetime.datetime.utcnow()
        else:
            scheduled_time = activate_time

        while self._active:
            sleep_time = (scheduled_time - datetime.datetime.utcnow()).total_seconds()

            if sleep_time > 0:
                time.sleep(sleep_time)
            self.put(topic=topic)

            while scheduled_time < datetime.datetime.utcnow():
                scheduled_time += datetime.timedelta(seconds=interval)

    def _minute_timer(self, topic: Topic):
        while self._active:
            t = datetime.datetime.utcnow()
            sleep_time = 60 - (t.second + t.microsecond / 1000000.0)
            time.sleep(sleep_time)
            self.put(topic=topic)

    def _second_timer(self, topic: Topic):
        while self._active:
            t = datetime.datetime.utcnow()
            sleep_time = 1 - t.microsecond / 1000000.0
            time.sleep(sleep_time)
            self.put(topic=topic)

    def stop(self) -> None:
        super().stop()

        for timer in self.timer.values():
            timer.join()


EVENT_ENGINE = EventEngine()
EVENT_ENGINE.start()
