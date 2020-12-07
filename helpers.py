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
    if interval_str == '30m':
        return 30

def max_candles(interval_str):
    if interval_str == '1m':
        return 100
    if interval_str == '3m':
        return 80
    if interval_str == '5m':
        return 60
    if interval_str == '15m':
        return 60
    if interval_str == '30m':
        return 60