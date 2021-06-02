import base64
import datetime
import hashlib
import hmac
import json
import traceback
import urllib.parse
from typing import Optional

import aiohttp
import requests

from . import LOGGER

LOGGER = LOGGER.getChild('RestClient')


class HuobiRestClient(object):
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
        method = request.upper()
        timestamp = datetime.datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S')
        params.update(
            {
                'AccessKeyId': self.access_key,
                'SignatureMethod': 'HmacSHA256',
                'SignatureVersion': '2',
                'Timestamp': timestamp
            }
        )

        host_name = urllib.parse.urlparse(url).hostname
        host_name = host_name.lower()
        path = urllib.parse.urlparse(url).path
        params['Signature'] = self._create_sign(params, method, host_name, path)
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
        method = request.upper()
        timestamp = datetime.datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S')

        if not params:
            params = {}

        if request == 'GET':
            sign_dict = params
        else:
            # according to the API doc https://huobiapi.github.io/docs/spot/v1/cn/#c64cd15fdc
            # post params should not added to the signature dict,
            # but real-world test shows otherwise. Probably a bug
            sign_dict = params

        sign_dict.update(
            {
                'AccessKeyId': self.access_key,
                'SignatureMethod': 'HmacSHA256',
                'SignatureVersion': '2',
                'Timestamp': timestamp
            }
        )

        host_name = urllib.parse.urlparse(url).hostname
        host_name = host_name.lower()
        path = urllib.parse.urlparse(url).path
        signature = self._create_sign(sign_dict, method, host_name, path)
        sign_dict['Signature'] = signature
        # params.update(sign_dict)
        # return self.request(request=request, url=url, params=params, callback=callback, headers={"Accept": "application/json", 'Content-Type': 'application/json'}, **kwargs)

        if method == 'GET':
            params.update(sign_dict)
            return self.request(request=request, url=url, params=params, callback=callback, **kwargs)
        else:
            new_url = fr'{url}?{urllib.parse.urlencode(sign_dict)}'
            return self.request(request=request, url=new_url, json=params, callback=callback, **kwargs)

    def _create_sign(self, parameters, method, host_url, request_path):
        sorted_params = sorted(parameters.items(), key=lambda d: d[0], reverse=False)
        encode_params = urllib.parse.urlencode(sorted_params)
        payload = [method, host_url, request_path, encode_params]
        payload = '\n'.join(payload)
        payload = payload.encode(encoding='utf-8')
        secret_key = self.secret_key.encode(encoding='utf-8')

        digest = hmac.new(secret_key, payload, digestmod=hashlib.sha256).digest()
        signature = base64.b64encode(digest).decode()
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
