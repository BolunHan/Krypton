import dash

from ...Base import LOGGER

LOGGER = LOGGER.getChild('WebApp.Monitor.Trade')


def wind(clicked):
    if clicked:
        LOGGER.info('Wind button clicked!')
    return dash.no_update


def unwind(clicked):
    if clicked:
        LOGGER.info('Unwind button clicked!')
    return dash.no_update


def cancel(clicked):
    if clicked:
        LOGGER.info('Cancel button clicked!')
    return dash.no_update


def refresh(clicked):
    if clicked:
        LOGGER.info('Refresh button clicked!')
    return dash.no_update

