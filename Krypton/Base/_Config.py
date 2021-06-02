import configparser
import json
import pathlib

import dateutil
import pandas as pd

from ._Statics import GlobalStatics
from ..Res.ToolKit import AttrDict

CONFIG = AttrDict()
CWD = pathlib.Path(GlobalStatics.WORKING_DIRECTORY.value)


def from_ini(file_path):
    parser = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation(), allow_no_value=True)
    parser.optionxform = str
    parser.read(file_path)

    for section_name in parser.sections():
        section = parser[section_name]
        for key in section:
            value = section[key]
            address = section_name.split('.') + key.split('.')
            key = address[-1]

            sub_dict = CONFIG
            for prefix in address[:-1]:
                sub_dict = sub_dict[prefix]

            if isinstance(value, str):

                # noinspection PyBroadException
                try:
                    value = json.loads(value)
                except Exception as _:
                    pass

                if value is not None:
                    try:
                        # noinspection PyUnresolvedReferences
                        value = dateutil.parser(value)
                    except Exception as _:
                        pass

                    try:
                        value = pd.to_numeric(value).item()
                    except Exception as _:
                        value = value

            sub_dict[key] = value


from_ini(CWD.joinpath('config.ini'))
