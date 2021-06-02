import base64
import datetime
import gzip
import hashlib
import hmac
import json
import traceback
import urllib.parse
import uuid
from collections import defaultdict
from typing import Optional, Dict, List, Union, Callable, Iterable

import aiohttp

from . import LOGGER
from ...Res.ToolKit import WebsocketsClient

LOGGER = LOGGER.getChild('WebsocketsClient')


class HuobiWebsocketsClient(object):
    def __init__(
            self,
            name: str,
            url: str,
            proxy: Optional[str] = None,
            **kwargs
    ):
        self.name = name
        self.url = url
        self.proxy = f'http://{proxy}' if proxy else None
        self.access_key = kwargs.pop('access_key', None)
        self.secret_key = kwargs.pop('secret_key', None)
        self.on_open_jobs = kwargs.pop('on_open_jobs', [])
        self.on_reconnect_jobs = kwargs.pop('on_reconnect_jobs', [])

        self.logger = LOGGER.getChild('Websockets')

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

        self.subscription = {}

        self.req_callbacks = defaultdict(list)
        self.sub_callbacks = defaultdict(list)

        self.status = {}
        self.running = False
        self.last_ping = None
        self.retries = 0

    # noinspection DuplicatedCode
    @staticmethod
    def _create_sign(secret_key: str, parameters: dict, method: str, request_url: str):
        host_name = urllib.parse.urlparse(request_url).hostname
        host_url = host_name.lower()
        request_path = urllib.parse.urlparse(request_url).path
        sorted_params = sorted(parameters.items(), key=lambda d: d[0], reverse=False)
        encode_params = urllib.parse.urlencode(sorted_params)
        payload = [method, host_url, request_path, encode_params]
        payload = '\n'.join(payload)
        payload = payload.encode(encoding='UTF8')
        secret_key = secret_key.encode(encoding='UTF8')

        digest = hmac.new(secret_key, payload, digestmod=hashlib.sha256).digest()
        signature = base64.b64encode(digest)
        signature = signature.decode()
        return signature

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

        for topic, message in self.subscription.items():
            self.subscribe(topic=topic, message=message)

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

    # noinspection SpellCheckingInspection
    def _handle_message(self, message):
        # handle operation feedback
        if 'op' in message:
            # authentication
            if message['op'] == 'auth':
                operation = 'authentication'
                topic = None
                if message['err-code'] == 0:
                    error_msg = None
                else:
                    error_msg = {'error-code': message['err-code'], 'error-msg': message['err-msg']}
            elif message['op'] == 'sub':
                operation = 'subscription'
                topic = message['topic']
                if message['err-code'] == 0:
                    error_msg = None
                else:
                    error_msg = {'error-code': message['err-code'], 'error-msg': message['err-msg']}
            elif message['op'] == 'notify':
                operation = 'push'
                topic = message['topic']
                error_msg = None
            else:
                operation = 'unknown'
                topic = None
                error_msg = {'error-msg': message}
        # handle subscribe response
        elif 'subbed' in message:
            operation = 'subscription'
            topic = message['subbed']
            if message['status'] == 'ok':
                error_msg = None
            else:
                error_msg = {'error-code': message['err-code'], 'error-msg': message['err-msg']}
        # handle unsubscribe response
        elif 'unsubbed' in message:
            operation = 'unsubscription'
            topic = message['unsubbed']
            if message['status'] == 'ok':
                error_msg = None
            else:
                error_msg = {'error-code': message['err-code'], 'error-msg': message['err-msg']}
        # handle v2 response
        elif 'action' in message:
            topic = message['ch']
            event = message['action']
            if event == 'push':
                operation = 'push'
                error_msg = None
            elif event == 'sub':
                operation = 'subscription'
                error_msg = None
            elif event == 'req' and topic == 'auth':
                operation = 'authentication'
                topic = None
                if message['code'] == 200:
                    error_msg = None
                else:
                    error_msg = {'error-code': message['err-code'], 'error-msg': message['err-msg']}
            else:
                operation = 'unknown'
                error_msg = {'error-msg': message}
        # handle subscription push
        elif 'ch' in message:
            operation = 'push'
            topic = message['ch']
            error_msg = None
        # handle request message
        elif 'req' in message:
            operation = 'request'
            topic = message['req']
            if message['status'] == 'ok':
                error_msg = None
            else:
                error_msg = {'error-code': message['err-code'], 'error-msg': message['err-msg']}
        # handle unrecognized message
        else:
            operation = 'unknown'
            topic = None
            error_msg = {'error-msg': message}

        if operation == 'authentication':
            if error_msg:
                self.logger.error(f'<{self.url}> Authentication failed! CODE {error_msg["error-code"]}: {error_msg["error-msg"]}')
            else:
                self.logger.info(f'<{self.url}> authentication successful!')
        elif operation == 'subscription':
            if error_msg:
                self.logger.error(f'<{self.url}> subscribe {topic} failed! CODE {error_msg["error-code"]}: {error_msg["error-msg"]}')
                self.status[topic] = 'sub_error'
            else:
                self.logger.info(f'<{self.url}> subscribe {topic} successful!')
                self.status[topic] = 'subscribed'
        elif operation == 'unsubscription':
            if error_msg:
                self.logger.error(f'<{self.url}> subscribe {topic} failed! CODE {error_msg["error-code"]}: {error_msg["error-msg"]}')
                self.status[topic] = 'unsub_error'
            else:
                self.logger.info(f'<{self.url}> unsubscription {topic} successful!')
                self.status.pop(topic)
        elif operation == 'push':
            callbacks = self.sub_callbacks.get(topic, [])
            for callback in callbacks:
                # noinspection PyBroadException
                try:
                    callback(message)
                except Exception as _:
                    self.logger.error(f'<{self.url}> push {topic} processing failed! traceback: {traceback.format_exc()}')
        elif operation == 'request':
            callbacks = self.req_callbacks.pop(topic, [])
            for callback in callbacks:
                # noinspection PyBroadException
                try:
                    callback(message)
                except Exception as _:
                    self.logger.error(f'<{self.url}> request {topic} processing failed! traceback: {traceback.format_exc()}')
        elif operation == 'unknown':
            self.logger.error(f'<{self.url}> unknown message {error_msg["error-msg"]}')
        else:
            self.logger.error(f'<{self.url}> unregistered operation {operation} with topic {topic}, err {error_msg}')

    # noinspection DuplicatedCode
    def auth(self, version='2.1'):
        self.logger.info(f'<{self.url}> acquiring authentication...')
        if self.access_key and self.secret_key:
            params = {
                'action': 'req',
                'ch': 'auth'
            }
            if version == '2.1':
                auth_params = {
                    "accessKey": self.access_key,
                    "signatureMethod": "HmacSHA256",
                    "signatureVersion": "2.1",
                    "timestamp": datetime.datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S')
                }
                sign = self._create_sign(self.secret_key, auth_params, 'GET', self.url)
                auth_params['signature'] = sign
                auth_params['authType'] = "api"
                params["params"] = auth_params
            elif version == '2':
                params = {
                    "AccessKeyId": self.access_key,
                    "SignatureMethod": "HmacSHA256",
                    "SignatureVersion": "2",
                    "Timestamp": datetime.datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S')
                }
                sign = self._create_sign(self.secret_key, params, 'GET', self.url)
                params["Signature"] = sign
                params["op"] = "auth"
                params["type"] = "api"
                params["cid"] = str(uuid.uuid1())

            self.websockets.send(params)

    # noinspection DuplicatedCode
    def subscribe(self, topic: Optional[str] = None, message: Optional[dict] = None, callback: Optional[Union[List[Callable], Callable]] = None):
        if topic:
            if message is None:
                message = {
                    'sub': topic,
                    'id': str(uuid.uuid4())
                }
        elif message:
            if 'ch' in message:
                topic = message['ch']
            elif 'topic' in message:
                topic = message['topic']
            else:
                raise ValueError('No topic for subscription')
        else:
            raise ValueError('Subscribe method requires value for topic or message')

        self.subscription[topic] = message

        if self.connected:
            if callback:
                self.sub_callbacks[topic].append(callback)

            if self.status.get(topic) not in ['subscribed', 'subscribing']:
                self.websockets.send(message)
                self.status[topic] = 'subscribing'
        else:
            self.logger.warning('Socket not connected, abort!')

    def unsubscribe(self, topic: str, message: Optional[dict] = None):
        if topic in self.subscription:
            self.subscription.pop(topic)

            if message is None:
                message = {
                    'unsub': topic,
                    'id': str(uuid.uuid4())
                }

        if self.connected:
            self.websockets.send(message)
            self.status[topic] = 'unsubscribing'
            self.sub_callbacks.pop(topic, None)
        else:
            self.logger.warning('Socket not connected, abort!')

    def request(self, topic: Dict[str, str], start_time: datetime.datetime, end_time: datetime.datetime, callback: Optional[Callable] = None):
        message = {
            'unsub': topic,
            'id': str(uuid.uuid4()),
            'from': int(start_time.timestamp()),
            'to': int(end_time.timestamp()),
        }

        if self.connected:
            self.websockets.send(message)
            if callback:
                if isinstance(callback, Iterable):
                    self.req_callbacks[topic].extend(callback)
                else:
                    self.req_callbacks[topic].append(callback)
        else:
            self.logger.warning('Client not connected, abort!')

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
