from flask import Flask
from werkzeug.middleware.dispatcher import DispatcherMiddleware
from werkzeug.serving import run_simple

from Base import Telemetric, CONFIG

__all__ = ['start_app']
__version__ = "0.1.0"

LOGGER = Telemetric.LOGGER.getChild('WebApp')
APP = Flask(__name__)
HOSTNAME = CONFIG.get('WebApp', 'HOST', fallback='0.0.0.0')
PORT = CONFIG.getint('WebApp', 'PORT', fallback=80)

import WebApp.Monitor
import WebApp.FileServer

mounts = {
    '/Monitor': WebApp.Monitor.FLASK_APP,
    '/FileServer': WebApp.FileServer.FLASK_APP,
}


def start_app():
    application = DispatcherMiddleware(APP, mounts)

    if __name__ == '__main__':
        for mount_path in mounts:
            LOGGER.info(f'WebApp running on http://{HOSTNAME}:{PORT}/{mount_path}')

        run_simple(
            hostname=HOSTNAME,
            port=PORT,
            application=application
        )


if __name__ == '__main__':
    start_app()
