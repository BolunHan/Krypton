import datetime

import pandas as pd

__all__ = ['to_timestamp', 'pretty_timedelta', 'add_months']


def to_timestamp(t):
    if isinstance(t, datetime.timedelta):
        ts = t.total_seconds()
    elif isinstance(t, datetime.datetime):
        ts = t.timestamp()
    elif isinstance(t, datetime.date):
        ts = datetime.datetime.combine(t, datetime.datetime.min.time()).timestamp()
    elif isinstance(t, (int, float)):
        ts = t
    elif isinstance(t, pd.Timestamp):
        ts = t.timestamp()
    else:
        raise TypeError(f'Invalid market time {t}')

    return ts


def pretty_timedelta(td_object):
    seconds = int(td_object.total_seconds())
    periods = [
        ('year', 60 * 60 * 24 * 365),
        ('month', 60 * 60 * 24 * 30),
        ('day', 60 * 60 * 24),
        ('hour', 60 * 60),
        ('minute', 60),
        ('second', 1)
    ]

    strings = []
    for period_name, period_seconds in periods:
        if seconds >= period_seconds:
            period_value, seconds = divmod(seconds, period_seconds)
            has_s = 's' if period_value > 1 else ''
            strings.append(f"{period_value:,} {period_name}{has_s}")

    return ", ".join(strings)


def add_months(source_date: datetime.date, months: int) -> datetime.date:
    """
    Add months to specific date
    :param source_date: the given date
    :param months: the month to add
    :return: the calculated date
    """
    import calendar
    month = source_date.month - 1 + months
    year = source_date.year + month // 12
    month = month % 12 + 1
    day = min(source_date.day, calendar.monthrange(year, month)[1])
    return datetime.date(year, month, day)
