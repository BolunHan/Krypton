import inspect
import os
import pathlib
import platform
import sys

from . import GLOBAL_LOGGER
from .ConsoleUtils import count_ordinal

LOGGER = GLOBAL_LOGGER.getChild(os.path.basename(os.path.splitext(__file__)[0]))
__all__ = ['get_platform', 'get_current_path', 'get_debug_flag']


def get_platform():
    uname = platform.uname()

    if uname.system == 'Linux':
        if 'microsoft' in uname.release:
            return 'wsl'
        elif 'generic' in uname.release:
            return 'Linux'
        # manjaro release
        elif 'manjaro' in uname.release.lower():
            return 'Linux'
        else:
            LOGGER.warning(f'Unknown Linux release {uname.release}')
            return 'Linux'
    elif uname.system == 'Windows':
        return 'Windows'
    else:
        raise ValueError(f'Unknown system {uname.system}')


def get_current_path(idx: int = 1) -> pathlib.Path:
    stacks = inspect.stack()
    if len(stacks) < 1:
        raise ValueError(f'Can not go back to {count_ordinal(idx)} stack')
    else:
        return pathlib.Path(stacks[idx].filename)


def get_debug_flag():
    get_trace = getattr(sys, 'gettrace', None)
    if get_trace is not None and get_trace():
        return True
    else:
        return False
