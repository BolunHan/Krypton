import datetime

from ..Base import LOGGER

LOCAL_TIMEZONE = datetime.timezone.utc
LOGGER = LOGGER.getChild('API')
__all__ = ['LOCAL_TIMEZONE', 'LOGGER', 'Template']
