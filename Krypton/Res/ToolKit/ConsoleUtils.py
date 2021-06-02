import io
import logging
import shutil
import sys
import threading
import time
from enum import Enum
from typing import Callable, Iterable, Union, Sized, Optional

from .LoggerUtils import temp_log

__all__ = ['Progress', 'GetInput', 'count_ordinal', 'TerminalStyle']


# noinspection SpellCheckingInspection
class TerminalStyle(Enum):
    CEND = '\33[0m'
    CBOLD = '\33[1m'
    CITALIC = '\33[3m'
    CURL = '\33[4m'
    CBLINK = '\33[5m'
    CBLINK2 = '\33[6m'
    CSELECTED = '\33[7m'

    CBLACK = '\33[30m'
    CRED = '\33[31m'
    CGREEN = '\33[32m'
    CYELLOW = '\33[33m'
    CBLUE = '\33[34m'
    CVIOLET = '\33[35m'
    CBEIGE = '\33[36m'
    CWHITE = '\33[37m'

    CBLACKBG = '\33[40m'
    CREDBG = '\33[41m'
    CGREENBG = '\33[42m'
    CYELLOWBG = '\33[43m'
    CBLUEBG = '\33[44m'
    CVIOLETBG = '\33[45m'
    CBEIGEBG = '\33[46m'
    CWHITEBG = '\33[47m'

    CGREY = '\33[90m'
    CRED2 = '\33[91m'
    CGREEN2 = '\33[92m'
    CYELLOW2 = '\33[93m'
    CBLUE2 = '\33[94m'
    CVIOLET2 = '\33[95m'
    CBEIGE2 = '\33[96m'
    CWHITE2 = '\33[97m'

    CGREYBG = '\33[100m'
    CREDBG2 = '\33[101m'
    CGREENBG2 = '\33[102m'
    CYELLOWBG2 = '\33[103m'
    CBLUEBG2 = '\33[104m'
    CVIOLETBG2 = '\33[105m'
    CBEIGEBG2 = '\33[106m'
    CWHITEBG2 = '\33[107m'

    @staticmethod
    def color_table():
        """
        prints table of formatted text format options
        """
        for style in range(8):
            for fg in range(30, 38):
                s1 = ''
                for bg in range(40, 48):
                    _format = ';'.join([str(style), str(fg), str(bg)])
                    s1 += '\x1b[%sm %s \x1b[0m' % (_format, _format)
                print(s1)
            print('\n')


class Progress(object):
    DEFAULT = '{prompt} [{bar}] {progress:>7.2%} {eta}{done}'
    MINI = '{prompt} {progress:.2%}'
    FULL = '{prompt} [{bar}] {done_tasks}/{total_tasks} {progress:>7.2%}, {remaining} to go {eta}{done}'

    def __init__(self, tasks: Union[int, Iterable], prompt: str = 'Progress:', format_spec: str = DEFAULT, **kwargs):
        self.prompt = prompt
        self.format_spec = format_spec
        self._width = kwargs.pop('width', None)
        self.tick_size = kwargs.pop('tick_size', 0.0001)
        self.progress_symbol = kwargs.pop('progress_symbol', '=')
        self.blank_symbol = kwargs.pop('blank_symbol', ' ')

        if isinstance(tasks, int):
            self.total_tasks = tasks
            self.tasks = range(self.total_tasks)
        elif isinstance(tasks, (Sized, Iterable)):
            self.total_tasks = len(tasks)
            self.tasks = tasks

        if 'outputs' not in kwargs:
            self.outputs = [sys.stdout]
        else:
            outputs = kwargs.pop('outputs')
            if outputs is None:
                self.outputs = []
            elif isinstance(outputs, Iterable):
                self.outputs = outputs
            else:
                self.outputs = [outputs]

        self.start_time = time.time()
        self.done_tasks = 0
        self.done_time = None
        self.iter_task = None
        self.last_output = -1

    @property
    def eta(self):
        remaining = self.total_tasks - self.done_tasks
        time_cost = time.time() - self.start_time

        if self.done_tasks == 0:
            eta = float('inf')
        else:
            eta = time_cost / self.done_tasks * remaining

        return eta

    @property
    def work_time(self):
        if self.done_time:
            work_time = self.done_time - self.start_time
        else:
            work_time = time.time() - self.start_time

        return work_time

    @property
    def is_done(self):
        return self.done_tasks == self.total_tasks

    @property
    def progress(self):
        return self.done_tasks / self.total_tasks

    @property
    def remaining(self):
        return self.total_tasks - self.done_tasks

    @property
    def width(self):
        if self._width:
            width = self._width
        else:
            width = shutil.get_terminal_size().columns

        return width

    def format_progress(self):

        if self.is_done:
            eta = ''
            done = f'All done in {self.work_time:,.2f} seconds'
        else:
            eta = f'ETA: {self.eta:,.2f} seconds'
            done = ''

        args = {
            'total_tasks': self.total_tasks,
            'done_tasks': self.done_tasks,
            'progress': self.progress,
            'remaining': self.remaining,
            'work_time': self.work_time,
            'eta': eta,
            'done': done,
            'prompt': self.prompt,
            'bar': '',
        }

        bar_size = max(10, self.width - len(self.format_spec.format_map(args)))
        progress_size = round(bar_size * self.progress)
        args['bar'] = self.progress_symbol * progress_size + self.blank_symbol * (bar_size - progress_size)
        progress_str = self.format_spec.format_map(args)
        return progress_str

    def reset(self):
        self.done_tasks = 0
        self.done_time = None
        self.last_output = -1

    def output(self):
        progress_str = self.format_progress()
        self.last_output = self.progress

        for output in self.outputs:
            if isinstance(output, Callable):
                output(progress_str)
            elif isinstance(output, logging.Logger):
                temp_log(logger=output, level=logging.INFO, msg=progress_str)
            elif isinstance(output, (io.TextIOBase, logging.Logger)):
                print('\r' + progress_str, file=output, end='')
            else:
                pass

    def __call__(self, *args, **kwargs):
        return self.format_progress()

    def __next__(self):
        try:
            if (not self.tick_size) or self.progress >= self.tick_size + self.last_output:
                self.output()
            self.done_tasks += 1
            return self.iter_task.__next__()
        except StopIteration:
            self.done_tasks = self.total_tasks
            self.output()
            raise StopIteration()

    def __iter__(self):
        self.reset()
        self.start_time = time.time()
        self.iter_task = self.tasks.__iter__()
        return self


class GetInput(object):
    def __init__(self, timeout=5, prompt_message: Optional[str] = None, default_value: Optional[str] = None):

        if prompt_message is None:
            prompt_message = f'Please respond in {timeout} seconds: '

        self.timeout = timeout
        self.default_value = default_value
        self.prompt_message = prompt_message
        self._input = None
        self.input_thread: Optional[threading.Thread] = None
        self.show()

    def show(self):
        self.input_thread = threading.Thread(target=self.get_input)
        self.input_thread.daemon = True
        self.input_thread.start()
        self.input_thread.join(timeout=self.timeout)
        # input_thread.terminate()

        if self._input is None:
            print(f"No input was given within {self.timeout} seconds. Use {self.default_value} as default value.")
            self._input = self.default_value

    def get_input(self):
        self._input = None
        self._input = input(self.prompt_message)
        return

    @property
    def input(self):
        return self._input


def count_ordinal(n: int) -> str:
    """
    Convert an integer into its ordinal representation::
    make_ordinal(0)   => '0th'
    make_ordinal(3)   => '3rd'
    make_ordinal(122) => '122nd'
    make_ordinal(213) => '213th'
    """
    n = int(n)
    suffix = ['th', 'st', 'nd', 'rd', 'th'][min(n % 10, 4)]
    if 11 <= (n % 100) <= 13:
        suffix = 'th'
    return str(n) + suffix
