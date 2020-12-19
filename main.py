from dotenv import load_dotenv
load_dotenv(verbose=True)

from bnc import client
from trader import Trader
from analyzer import Analyzer
import datetime as dt

if __name__ == '__main__':
    # a = Analyzer()
    # a.scan_markets()
    t = Trader()
    print('$', t.portfolio['balance'])
    t.scan_market_loop()
