import datetime as dt

def interval_str_to_time(interval_str):
    if interval_str == '1m':
        return 1
    if interval_str == '3m':
        return 3
    if interval_str == '5m':
        return 5
    if interval_str == '15m':
        return 15
    if interval_str == '1d':
        return 60 * 24

def min_timestamp(interval_str):
    if interval_str == '1m':
        return dt.datetime.now() - dt.timedelta(hours=3)
    if interval_str == '3m':
        return dt.datetime.now() - dt.timedelta(hours=12)
    if interval_str == '5m':
        return dt.datetime.now() - dt.timedelta(hours=18)
    if interval_str == '15m':
        return dt.datetime.now() - dt.timedelta(hours=36)
    if interval_str == '1d':
        return dt.datetime.now() - dt.timedelta(days=60)