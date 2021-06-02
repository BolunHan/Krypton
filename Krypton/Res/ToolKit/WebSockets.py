import asyncio
import inspect
import logging
import os
import threading
import time
import traceback
from typing import Optional

import aiohttp

from . import GLOBAL_LOGGER
from .Collections import AttrDict
from .ConsoleUtils import count_ordinal

LOGGER = GLOBAL_LOGGER.getChild(os.path.basename(os.path.splitext(__file__)[0]))
__all__ = ['WebsocketsClient']


class WebsocketsClient(object):
    MESSAGE = aiohttp.WSMessage

    def __init__(
            self,
            url,
            proxy=None,
            auto_reconnect: bool = False,
            logger: Optional[logging.Logger] = None,
            **kwargs
    ):
        self.url = url
        self.proxy = proxy
        self._callback = AttrDict(
            on_open=kwargs.get('on_open', None),
            on_reconnect=kwargs.get('on_reconnect', None),
            on_message=kwargs.get('on_message', None),
            on_error=kwargs.get('on_error', None),
            on_close=kwargs.get('on_close', None),
        )

        self.auto_reconnect = auto_reconnect
        self.logger = logger if logger is not None else LOGGER.getChild('Websockets.Client')

        self.websockets_thread: Optional[threading.Thread] = None
        self.loop = None
        self.session = None
        self.websockets = None
        self.running = False
        self.retries = 0
        self.max_retry = kwargs.pop('max_retry', 10)

    def __repr__(self):
        return f'<AsyncWebsocketsClient>{{url: {self.url}, proxy: {self.proxy}, auto_reconnect: {self.auto_reconnect}, status: {"running" if self.running else "pending"}}}'

    def __str__(self):
        return self.__repr__()

    async def _on_callback(self, flag: str, *args):
        self.logger.debug(f'Websockets {self.url} {flag} successfully')

        if self._callback.get(flag) is not None:
            # noinspection PyBroadException
            try:
                if inspect.isawaitable(self._callback.get(flag)):
                    await self._callback.get(flag)(*args)
                else:
                    self._callback.get(flag)(*args)
            except Exception as _:
                self.logger.warning(traceback.format_exc())

    async def _session_thread(self, **kwargs):
        self.running = True
        self.loop = asyncio.get_event_loop()
        self.retries = 0

        async with aiohttp.ClientSession() as self.session:
            while not self.connected:
                # noinspection PyBroadException
                try:
                    async with self.session.ws_connect(url=self.url, proxy=self.proxy, **kwargs) as self.websockets:
                        # on open
                        await self._on_callback('on_open', self)

                        # on reconnect
                        if self.retries:
                            await self._on_callback('on_reconnect', self)

                        # reset retry counter
                        self.retries = 0

                        # listen
                        while self.running:
                            message = await self.websockets.receive()
                            self.logger.debug(f'Websockets {self.url} received {message}')

                            if message.type == aiohttp.WSMsgType.CLOSED:
                                await self._on_callback('on_close', self)
                                break
                            elif message.type == aiohttp.WSMsgType.CLOSE:
                                break
                            elif message.type == aiohttp.WSMsgType.ERROR:
                                await self._on_callback('on_error', self, ConnectionError(message.data))
                                break
                            else:
                                await self._on_callback('on_message', self, message)
                except Exception as _:
                    self.logger.error(traceback.format_exc())

                # break loop if error or disconnect
                if not self.auto_reconnect:
                    break

                # graceful shutdown
                await asyncio.sleep(0.250)

                # check max retry limit
                if self.retries > self.max_retry:
                    self.logger.error(f'Websockets {self.url} max retry exceeded, {count_ordinal(self.retries)} connection abort!')
                    break

                self.retries += 1
                self.logger.debug(f'Websockets {self.url} disconnected, {count_ordinal(self.retries)} reconnect retry!')

    def send(self, message, **kwargs):
        if isinstance(message, str):
            self.loop.create_task(self.websockets.send_str(message, **kwargs))
        elif isinstance(message, bytes):
            self.loop.create_task(self.websockets.send_bytes(message, **kwargs))
        elif isinstance(message, dict):
            self.loop.create_task(self.websockets.send_json(message, **kwargs))
        else:
            self.logger.error(TypeError(f'Websockets {self.url} send message failed! Message must be str, bytes or json(dict)'))

    def connect(self, **kwargs):

        if os.name == 'nt' and self.proxy:
            policy = asyncio.WindowsSelectorEventLoopPolicy()
            asyncio.set_event_loop_policy(policy)

        if self.connected:
            self.logger.debug(f'Websockets {self.url} already connected')
            return

        self.websockets_thread = threading.Thread(target=asyncio.run, args=[self._session_thread(**kwargs)])
        self.websockets_thread.start()

        while not self.connected:
            time.sleep(0.01)

        self.logger.debug(f'Websockets {self.url} connected!')

    def disconnect(self):
        if not self.connected:
            self.logger.warning(f'Websockets {self.url} already closed')
            return

        oar = self.auto_reconnect
        self.auto_reconnect = False
        self.running = False
        self.loop.create_task(self.websockets.close())

        while self.connected:
            time.sleep(0.01)

        self.auto_reconnect = oar
        self.websockets_thread.join()

    @property
    def connected(self):
        if self.websockets is None:
            # self.logger.info('websockets not initialized!')
            return False
        else:
            return not self.websockets.closed
