from dotenv import load_dotenv
load_dotenv(verbose=True)

from bnc import client
from robot import Robot
import datetime as dt

if __name__ == '__main__':
    r = Robot()
    r.scan_market_loop()
