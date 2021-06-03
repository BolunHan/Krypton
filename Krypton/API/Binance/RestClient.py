import base64
import datetime
import hashlib
import hmac
import json
import time
import traceback
import urllib.parse
from typing import Optional

import aiohttp
import requests

from . import LOGGER

LOGGER = LOGGER.getChild('RestClient')


class BinanceRestClient(object):
    def __init__(
            self,
            name: str,
            url: str,
            proxy: Optional[str] = None,
            **kwargs
    ):
        self.name = name
        self.url = url

        if proxy is None:
            self.proxy = None
        else:
            self.proxy = {
                "http": f"http://{proxy}",
                "https": f"https://{proxy}",
                "ftp": f"ftp://{proxy}"
            }

        self.access_key = kwargs.pop('access_key', None)
        self.secret_key = kwargs.pop('secret_key', None)

        self.logger = LOGGER.getChild('Websockets')
        self.session = None

    async def async_request(self, request: str, url: str, params=None, callback=None, **kwargs):
        self.start_session()
        async with self.session.request(method=request, url=url, params=params, proxy=self.proxy['http'], **kwargs) as response:
            if response.content_type == 'application/json':
                msg = await response.json()
            else:
                msg = await response.text()

            if response.status != 200:
                self.logger.error(f'HTTP {request} {url} {params} failed!')

            if callback:
                # noinspection PyBroadException
                try:
                    callback(msg)
                except Exception as _:
                    self.logger.error(traceback.format_exc())

            return msg

    # noinspection DuplicatedCode
    async def async_request_signed(self, request: str, url: str, params=None, callback=None, **kwargs):
        timestamp = int(time.time() * 1000)

        if not params:
            params = {}

        params.update(
            {
                'timestamp': timestamp
            }
        )

        params['Signature'] = self._create_sign(params)
        return await self.async_request(request=request, url=url, params=params, callback=callback, **kwargs)

    def request(self, request: str, url: str, params=None, callback=None, **kwargs):
        response = requests.request(method=request, url=url, params=params, proxies=self.proxy, **kwargs)
        msg = response.json()

        if response.status_code != 200:
            self.logger.error(f'HTTP {request} {url} {json} failed!')

        if callback:
            # noinspection PyBroadException
            try:
                callback(msg)
            except Exception as _:
                self.logger.error(traceback.format_exc())

        return msg

    def request_signed(self, request: str, url: str, params=None, callback=None, **kwargs):
        timestamp = int(time.time() * 1000)

        if not params:
            params = {}

        params.update(
            {
                'timestamp': timestamp
            }
        )

        params['Signature'] = self._create_sign(params)
        return self.request(request=request, url=url, params=params, callback=callback, **kwargs)

    def _create_sign(self, parameters):
        encode_params = urllib.parse.urlencode(parameters)
        payload = encode_params.encode(encoding='utf-8')
        secret_key = self.secret_key.encode(encoding='utf-8')
        signature = hmac.new(secret_key, payload, digestmod=hashlib.sha256).hexdigest()
        return signature

    def start_session(self):
        if self.session is None:
            self.session = aiohttp.ClientSession()
            # if os.name == 'nt' and self.proxy:
            #     policy = asyncio.WindowsSelectorEventLoopPolicy()
            #     asyncio.set_event_loop_policy(policy)

    async def close_session(self):
        if self.session:
            await self.session.close()
