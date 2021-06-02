import io
import json
import logging
import logging.handlers
import os
import queue
import socket
import sys
import threading
import time
import traceback
from multiprocessing.dummy import Pool
from typing import Optional, Union

import slack

from . import GLOBAL_LOGGER
from .Collections import AttrDict

LOGGER = GLOBAL_LOGGER.getChild(os.path.basename(os.path.splitext(__file__)[0]))
__all__ = ['init_logger', 'temp_log', 'SlackMessage', 'SlackChannelHandler', 'ColoredFormatter']


class ColoredFormatter(logging.Formatter):
    """Logging Formatter to add colors and count warning / errors"""

    def __init__(self, fmt=None, datefmt=None, style='{', validate=True):
        self.format_str = '[{asctime} {name} - {threadName} - {levelname}] {message}' if fmt is None else fmt
        self.date_fmt = '%Y-%m-%d %H:%M:%S' if datefmt is None else datefmt
        self.style = style

        super().__init__(fmt=fmt, datefmt=datefmt, style=style, validate=validate)

    def _get_format(self, level: int, select=False):
        bold_red = f"\33[31;1;3;4{';7' if select else ''}m"
        red = f"\33[31;1{';7' if select else ''}m"
        green = f"\33[32;1{';7' if select else ''}m"
        yellow = f"\33[33;1{';7' if select else ''}m"
        blue = f"\33[34;1{';7' if select else ''}m"
        reset = "\33[0m"

        if level <= logging.NOTSET:
            fmt = self.format_str
        elif level <= logging.DEBUG:
            fmt = blue + self.format_str + reset
        elif level <= logging.INFO:
            fmt = green + self.format_str + reset
        elif level <= logging.WARNING:
            fmt = yellow + self.format_str + reset
        elif level <= logging.ERROR:
            fmt = red + self.format_str + reset
        else:
            fmt = bold_red + self.format_str + reset

        return fmt

    def format(self, record):
        log_fmt = self._get_format(level=record.levelno)
        formatter = logging.Formatter(log_fmt, datefmt=self.date_fmt, style=self.style)
        return formatter.format(record)


class SlackMessage(object):
    def __init__(self, token: str, pool: Optional[Pool] = None):
        self.token = token
        self.client = slack.WebClient(token=self.token)
        self.pool = pool
        self.conversations = self.list_conversations()

        # disable slack deprecation warning
        # noinspection SpellCheckingInspection
        os.environ["SLACKCLIENT_SKIP_DEPRECATION"] = '1'

    def send_message_async(self, message: Union[str, dict], channel: str):
        return self.pool.apply_async(self.send_message, (message, channel))

    def send_file_async(self, file_name: str, file_io: io.BytesIO, channel: str):
        return self.pool.apply_async(self.send_file, (file_name, file_io, channel))

    def send_message(self, message: Union[str, dict], channel: str):
        channel = str(channel).strip().lstrip('#')
        _ = self.client.conversations_join(channel=self.conversations[channel].id)
        if isinstance(message, str):
            response = self.client.chat_postMessage(channel=self.conversations[channel].id, text=message)
        else:
            response = self.client.chat_postMessage(channel=self.conversations[channel].id, **message)
        if not response["ok"]:
            LOGGER.warning(f'Sending slack message failed!, {message}')

    def send_file(self, file_name: str, file_io: io.IOBase, channel: str):
        channel = str(channel).strip().lstrip('#')
        _ = self.client.conversations_join(channel=self.conversations[channel].id)
        response = self.client.api_call("files.upload", files={"file": file_io}, data={'title': file_name, 'channels': self.conversations[channel].id, 'as_user': True})
        if not response["ok"]:
            LOGGER.warning(f'Sending slack file failed!, {file_name}')

    def list_conversations(self) -> AttrDict:
        response = self.client.conversations_list()
        conversation_dict = AttrDict(alphanumeric=False)
        if response.status_code == 200:
            data = response.data
            if data['ok']:
                channel_list = data['channels']

                for channel in channel_list:
                    conversation_dict[channel['name']] = AttrDict(channel, alphanumeric=False)
            else:
                raise ValueError('Invalid slack channels data')
        else:
            raise ValueError('Invalid slack response')

        return conversation_dict

    def check_channel(self, channel: str):
        channel = str(channel).strip().lstrip('#')
        channels = self.list_conversations()

        if channel not in channels:
            raise ValueError(f'Channel {channel} does not exist')


class SlackChannelHandler(logging.Handler):
    COLORS = {
        'CRITICAL': '#DE5B49',
        'ERROR': '#E37B40',
        'WARN': '#F0CA4D',
        'WARNING': '#F0CA4D',
        'INFO': '#4180A8',
        'DEBUG': '#46B29D',
        'NOTSET': '#B2B2B2',
    }

    def __init__(self, channel: str, slack: Optional[SlackMessage] = None, token: Optional[str] = None, pool: Optional[Pool] = None, schedule: float = .0, _schedule_batch: int = 10, **kwargs):
        super(SlackChannelHandler, self).__init__()

        if slack is None and token is None:
            raise ValueError('Must assign a slack client or a token')
        elif slack is None:
            self.token = token
            self.slack = SlackMessage(token)
        elif token is None:
            self.token = slack.token
            self.slack = slack
        else:
            if slack.token == token:
                self.token = token
                self.slack = slack
            else:
                raise ValueError('Token not match!')

        self.channel = str(channel).strip().lstrip('#')
        self.pool = Pool(processes=5) if pool is None else pool
        self.schedule = schedule
        self._message_queue = queue.Queue()
        self._schedule_batch = _schedule_batch
        self._message_extras = kwargs

        self.slack.check_channel(self.channel)
        self.slack.pool = self.pool
        self.host = socket.gethostname()

        if schedule > 0:
            threading.Thread(target=self.scheduled_emit, daemon=True).start()

    def emit(self, record):

        if self.schedule:
            try:
                self._message_queue.put(record, timeout=3)
            except queue.Full:
                LOGGER.warning('Slack logging handler message queue full!')
            return

        message = {}

        attachments = {
            'fallback': self.format(record),
            'color': self.COLORS.get(record.levelname, self.COLORS['NOTSET']),
            'text': self.format(record),
            'footer': f'{record.levelname} from {self.host}, pid-{record.process}'
        }

        message.update(self._message_extras)
        message['attachments'] = json.dumps([attachments])

        if self.slack.pool:
            self.slack.send_message_async(channel=self.channel, message=message)
        else:
            self.slack.send_message(channel=self.channel, message=message)

    def scheduled_emit(self):
        while True:
            text_messages = []
            level = None
            pid = None

            # noinspection PyBroadException
            try:
                while not self._message_queue.empty():
                    record = self._message_queue.get(block=False)

                    if (level is None or level == record.levelname) and (pid is None or pid == record.process) and len(text_messages) <= self._schedule_batch:
                        text_messages.append(self.format(record))
                        level = record.levelname
                        pid = record.process
                    else:
                        break

                if text_messages:
                    message = {}

                    attachments = {
                        'color': self.COLORS.get(level, self.COLORS['NOTSET']),
                        'text': '\n'.join(text_messages),
                        'footer': f'{level} from {self.host}, pid-{pid}'
                    }

                    message.update(self._message_extras)
                    message['attachments'] = json.dumps([attachments])

                    if self.slack.pool:
                        self.slack.send_message_async(channel=self.channel, message=message)
                    else:
                        self.slack.send_message(channel=self.channel, message=message)
            except [SystemExit, KeyboardInterrupt] as _:
                break
            except Exception as _:
                LOGGER.warning(traceback.format_exc())
            time.sleep(self.schedule)


def temp_log(logger: logging.Logger, level, msg, *args, **kwargs):
    def amend_handler(lg: logging.Logger):
        handler_dict = {
            'parent': None,
            'original_handler': [],
            'disabled_handler': [],
        }

        for handler in lg.handlers:
            if isinstance(handler, logging.FileHandler):
                lg.removeHandler(handler)
                handler_dict['disabled_handler'].append(handler)
            elif isinstance(handler, logging.StreamHandler):
                handler_dict['original_handler'].append((handler, handler.terminator))
                handler.terminator = '\r'
            else:
                lg.removeHandler(handler)
                handler_dict['disabled_handler'].append(handler)

        if lg.parent:
            handler_dict['parent'] = amend_handler(lg.parent)

        return handler_dict

    def restore_handler(lg: logging.Logger, handler_dict: dict):
        for handler, terminator in handler_dict['original_handler']:
            handler.terminator = terminator

        for handler in handler_dict['disabled_handler']:
            logger.addHandler(handler)

        if handler_dict['parent']:
            restore_handler(lg.parent, handler_dict['parent'])

    _ = amend_handler(logger)
    logger.log(level, msg, *args, **kwargs)
    restore_handler(logger, _)


def init_logger(logger_name: str, level: int = 20, **kwargs) -> logging.Logger:
    """
    init a logger with stream handler, rotating file handler and slack message handler
    :param logger_name: logger name
    :param level: logging level, default = 20, which is INFO
    :keyword formatter (logging.Formatter): specify logging format
    :keyword stream_io (io): default stream handler io, default is sys.stdout
    :keyword file_name (str): rotation file handler name
    :keyword file_size (int): rotation size, default is 102400 bytes
    :keyword file_backup (int): rotation backups, default is 10
    :keyword slack_token (str): slack token
    :keyword slack_channel (str): slack channel, default is 'general'
    :keyword slack_schedule (float): slack update schedule, in seconds, default is 1.0
    :keyword pool (Pool): threading / processing pool
    :return: a logger object, with specific handlers
    """
    formatter = kwargs.get('formatter', logging.Formatter('[{asctime} {name} - {threadName} - {levelname}] {message}', datefmt='%Y-%m-%d %H:%M:%S', style='{'))
    str_formatter = kwargs.get('str_formatter', ColoredFormatter())
    stream_io = kwargs.get('stream_io', sys.stdout)
    file_name = kwargs.get('file_name')
    file_size = kwargs.get('file_size', 1024 * 1024 * 16)
    file_backup = kwargs.get('file_backup', 10)
    slack_token_and_channel = kwargs.get('slack_token_and_channel')
    slack_token = kwargs.get('slack_token')
    slack_channel = kwargs.get('slack_channel', 'general')
    slack_schedule = kwargs.get('slack_schedule', 1.)
    pool = kwargs.get('pool')

    if slack_token_and_channel is not None:
        LOGGER.warning(DeprecationWarning('[slack_token_and_channel] depreciated! Use [slack_token] and [slack_channel] instead!'), stacklevel=2)
        if slack_token is None or slack_token == slack_token_and_channel[0]:
            slack_token = slack_token_and_channel[0]
        else:
            raise ValueError('slack token not match!')

        slack_channel = slack_token_and_channel[1]

    names = logger_name.split('.')

    root_name = names[0]

    logger = logging.getLogger(name=root_name)
    logger.setLevel(level=level)

    # noinspection SpellCheckingInspection
    logging.Formatter.converter = time.gmtime
    if formatter is None:
        formatter = logging.Formatter('[{asctime} {name} - {threadName} - {levelname}] {message}', datefmt='%Y-%m-%d %H:%M:%S', style='{')

    # add StreamHandler
    if stream_io:
        have_handler = False
        for handler in logger.handlers:
            # noinspection PyUnresolvedReferences
            if type(handler) == logging.StreamHandler and handler.stream == stream_io:
                have_handler = True
                break

        if not have_handler:
            logger_ch = logging.StreamHandler(stream=stream_io)
            logger_ch.setLevel(level=level)
            logger_ch.setFormatter(fmt=str_formatter)
            logger.addHandler(logger_ch)

    # add RotatingFileHandler
    if file_name:
        real_path = os.path.realpath(file_name)
        have_handler = False
        for handler in logger.handlers:
            # noinspection PyUnresolvedReferences
            if type(handler) == logging.handlers.RotatingFileHandler and handler.baseFilename == real_path:
                have_handler = True
                break

        if not have_handler:
            logger_fh = logging.handlers.RotatingFileHandler(filename=real_path, maxBytes=file_size, backupCount=file_backup)
            logger_fh.setLevel(level=level)
            logger_fh.setFormatter(fmt=formatter)
            logger.addHandler(logger_fh)

    # add SlackChannelHandler
    if slack_token:
        have_handler = False
        for handler in logger.handlers:
            # noinspection PyUnresolvedReferences
            if type(handler) == SlackChannelHandler and handler.token == slack_token and handler.channel == slack_channel:
                have_handler = True
                break

        if not have_handler:
            logger_sh = SlackChannelHandler(token=slack_token, channel=slack_channel, schedule=slack_schedule, pool=pool)
            logger_sh.setLevel(level=level)
            logger_sh.setFormatter(fmt=formatter)
            logger.addHandler(logger_sh)

    logger = logging.getLogger(name=logger_name)

    return logger
