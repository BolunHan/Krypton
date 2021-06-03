import datetime
import gzip
import json
import traceback
import uuid
from typing import Optional, Dict

import aiohttp

from . import LOGGER
from ...Res.ToolKit import WebsocketsClient

LOGGER = LOGGER.getChild('WebsocketsClient')


class _WebsocketsClient(object):
    def __init__(
            self,
            topic: str,
            url: str,
            logger,
            proxy: Optional[str] = None,
            **kwargs
    ):
        self.topic = topic
        self.url = url
        self.logger = logger
        self.proxy = f'http://{proxy}' if proxy else None
        self.topic_id = kwargs.pop('websockets_id', uuid.uuid4().int)
        self.on_open_jobs = kwargs.pop('on_open_jobs', [])
        self.on_reconnect_jobs = kwargs.pop('on_reconnect_jobs', [])

        self.websockets = WebsocketsClient(
            url=self.url,
            proxy=self.proxy,
            on_open=self._on_open,
            on_close=self._on_close,
            on_message=self._on_message,
            on_reconnect=self._on_reconnect,
            logger=self.logger.getChild('aiohttp'),
            auto_reconnect=True
        )

        self.sub_callbacks = []

        self.status = 'idle'
        self.running = False
        self.last_ping = None
        self.retries = 0

    def _sub_message(self, opt='SUBSCRIBE'):
        message = {
            "method": opt,
            "params": [self.topic],
            "id": self.topic_id
        }
        return message

    def _on_open(self, websockets: WebsocketsClient):
        self.logger.info(f'{websockets} connected!')
        self.status = {}
        self.retries = 0

        if self.on_open_jobs:
            for job in self.on_open_jobs:
                job(self)

    def _on_reconnect(self, websockets: WebsocketsClient):
        self.logger.info(f'{websockets} reconnected!')

        self.status = {}
        self.retries = 0

        if self.on_reconnect_jobs:
            for job in self.on_reconnect_jobs:
                job(self)

        self.subscribe()

    def _on_message(self, websockets: WebsocketsClient, msg: WebsocketsClient.MESSAGE):
        if msg.type == aiohttp.WSMsgType.BINARY:
            message = gzip.decompress(msg.data).decode('utf-8')
        elif msg.type == aiohttp.WSMsgType.TEXT:
            message = msg.data
        else:
            self.logger.warning(f'Invalid message {msg}')
            return

        if 'ping' in message:
            tick = json.loads(message)
            if 'action' in tick and tick.get('action') == 'ping':
                self.last_ping = datetime.datetime.utcfromtimestamp(float(tick['data']['ts']) / 1000)
                respond = {'action': 'pong', 'data': {'ts': tick['data']['ts']}}
                # respond = "{\"action\": \"pong\",\"data\": {\"ts\": " + str(tick['data']['ts']) + "}}"
            elif 'op' in tick and tick.get('op') == 'ping':
                self.last_ping = datetime.datetime.utcfromtimestamp(float(tick['ts']) / 1000)
                respond = {'op': 'pong', 'ts': tick.get('ts')}
            elif 'ping' in tick:
                self.last_ping = datetime.datetime.utcfromtimestamp(float(tick['ping']) / 1000)
                respond = {'pong': tick.get('ping')}
            else:
                self.last_ping = datetime.datetime.utcnow()
                respond = message.replace('ping', 'pong')

            self.logger.debug(f'last ping at {self.last_ping}')
            websockets.send(respond)
        else:
            try:
                result = json.loads(message)
                self._handle_message(message=result)
            except json.JSONDecodeError as _:
                self.logger.error(f'Invalid message received! Message: {msg}')

    def _on_error(self, websockets: WebsocketsClient, error):
        self.logger.error(f'Websockets {websockets} Error, {error}')

    def _on_close(self, websockets: WebsocketsClient):
        self.logger.info(f'Websockets {websockets} closed!')

    def _handle_message(self, message: dict):
        # handle operation feedback
        if 'error' in message:
            self.logger.error(f'<{self.url}> error! {message["error"]}')
        elif 'result' in message:
            if message['result'] is None:
                topic_id = message['id']

                if topic_id == self.topic_id:
                    if self.status == 'subscribing':
                        self.status = 'subscribed'
                        self.logger.info(f'<{self.url}> subscribe {self.topic} successful!')
                    elif self.status == 'unsubscribing':
                        self.status = 'idle'
                        self.logger.info(f'<{self.url}> unsubscribe {self.topic} successful!')
                else:
                    LOGGER.error(f'topic_id not match! origin {self.topic}, got {topic_id}')
        else:
            callbacks = self.sub_callbacks
            for callback in callbacks:
                # noinspection PyBroadException
                try:
                    callback(message, topic=self.topic)
                except Exception as _:
                    self.logger.error(f'<{self.url}> push {self.topic} processing {message} failed! traceback: {traceback.format_exc()}')

    def add_callback(self, callback):
        self.sub_callbacks.append(callback)

    def pop_callback(self, callback):
        self.sub_callbacks.remove(callback)

    def subscribe(self):
        if self.connected:
            if self.status not in ['subscribed', 'subscribing']:
                self.websockets.send(self._sub_message(opt='SUBSCRIBE'))
                self.status = 'subscribing'
            else:
                self.logger.warning(f'{self.topic} already {self.status}!')
        else:
            self.logger.warning('Socket not connected, abort!')

    def unsubscribe(self):
        if self.connected:
            if self.status not in ['idle', 'unsubscribing']:
                self.websockets.send(self._sub_message(opt='UNSUBSCRIBE'))
                self.status = 'unsubscribing'
            else:
                self.logger.warning(f'{self.topic} already {self.status}!')
        else:
            self.logger.warning('Socket not connected, abort!')

    def connect(self, **kwargs):
        if not self.running:
            self.running = True
            self.websockets.connect(**kwargs)
        else:
            self.logger.warning('Client already running.')

    def disconnect(self):
        self.websockets.disconnect()
        self.running = False

    @property
    def connected(self):
        return self.websockets.connected


class BinanceWebsocketsClient(object):
    def __init__(self, base_url, proxy=None):
        self.base_url = base_url
        self.websockets: Dict[str, _WebsocketsClient] = {}
        self.proxy = proxy
        self.logger = LOGGER.getChild(f'Websockets')

    def subscribe(self, topic: str, callback, **kwargs):
        if topic in self.websockets:
            self.logger.info(f'{topic} already subscribed')
            self.websockets[topic].add_callback(callback)
            return

        client = _WebsocketsClient(
            topic=topic,
            url=f'{self.base_url}/ws/{topic}',
            logger=self.logger.getChild(topic),
            proxy=self.proxy,
            websockets_id=len(self.websockets),
            **kwargs
        )

        self.websockets[topic] = client
        client.add_callback(callback=callback)
        client.connect()
        client.subscribe()

    def unsubscribe(self, topic: str):
        if topic not in self.websockets:
            self.logger.warning(f'{topic} not subscribed')
            return

        _ = self.websockets.get(topic)
        _.unsubscribe()
        _.disconnect()
