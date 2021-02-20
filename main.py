from dotenv import load_dotenv
load_dotenv(verbose=True)

from bnc import client
from trader import Trader
import datetime as dt
import requests
import urllib3
import socket
import binance

def safe_loop():
    try:
        t = Trader()
        print('$', t.portfolio['balance'])
        t.scan_market_loop()
    except Exception as e:
        print(e)
        if type(e).__name__ != 'KeyboardInterrupt':
            safe_loop()

if __name__ == '__main__':
    safe_loop()