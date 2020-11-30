"""
The robot will run for ??? hours, playing the markets while managing risk.
Once finished, it will sell everything and store it in USDT.

Init:
    1. Get portfolio (USDT)
    2. Set goal at 2% of portfolio
    3. Identify markets with +-2% price change in last 24 hours
    4. Start scan loop

Scan:
    1. Check if bull/bear VWAP
    2. Calculate ABCDs
    3. If currently hitting a C in bull market, place bet

"""
import time
import datetime as dt
import numpy as np
import math
import matplotlib.pyplot as plt
import helpers
from itertools import chain
from scipy.signal import argrelextrema
from bnc import client
from simplification.cutil import (
    simplify_coords,
    simplify_coords_vw
)

plt.style.use('seaborn-whitegrid')

interval_str = '1m'
min_goal = 5
min_bet = 10

min_change_percent = 2
min_volume = 500000
min_trades = 50000
min_atr = 0.5


class Robot():
    def __init__(self, *args, **kwargs):
        super(Robot, self).__init__(*args, **kwargs)
        self.portfolio = self.get_portfolio()
        self.goal = max(self.portfolio['balance'] * 0.0002, min_goal)
        self.min_timestamp = helpers.min_timestamp(interval_str)

    def get_portfolio(self):
        return {
            'trade_fee': 0.01, # assume fees are 1% of trade
            'balance': float(client.get_asset_balance(asset='USDT')['free'])
        }

    def get_active_markets(self):
        tickers = client.get_ticker()
        print(len(tickers))
        # Tether markets only
        tickers = [x for x in tickers if x['symbol'].endswith('USDT')]
        print(len(tickers))
        # Fluctuating price ranges
        tickers = [x for x in tickers if abs(float(x['priceChangePercent'])) >= min_change_percent]
        print(len(tickers))
        # High volume
        tickers = [x for x in tickers if float(x['volume']) >= min_volume]
        print(len(tickers))
        # High amount of trades
        tickers = [x for x in tickers if float(x['count']) >= min_trades]
        print(len(tickers))
        # Good ATR
        tickers = [x for x in tickers if self.calc_atr(x) > min_atr]
        # FIXME: this grabs ETHUSDT for testing purposes
        # tickers = [x for x in tickers if x['symbol'] == 'ETHUSDT'] 
        # if len(tickers) == 0:
        #     tickers.append({ 'symbol': 'ETHUSDT' })
        return [{'symbol': x['symbol'], 'price_change': abs(float(x['priceChange'])), 'weighted_avg_price': float(x['weightedAvgPrice']) } for x in tickers]

    def scan_market_loop(self):
        while (True):
            print('Scanning markets...')
            self.markets = self.get_active_markets()
            self.scan_markets()
            # FIXME: Don't sleep for 15 minutes.
            # Calculate time until next quarter of hour.
            time.sleep(helpers.interval_str_to_time(interval_str) * 60)

    def scan_markets(self):
        for market in self.markets:
            try:
                print('Searching', market['symbol'])
                candles = client.get_klines(symbol=market['symbol'], interval=interval_str)
                full_vwaps = self.get_vwaps(candles)
                short_vwaps = self.get_vwaps([x for x in candles if dt.datetime.fromtimestamp(x[0] / 1000) >= self.min_timestamp])
                candles_sanitized = [{'timestamp': dt.datetime.fromtimestamp(x[0] / 1000), 'open': float(x[1]), 'close': float(x[4]) } for x in candles]

                min_price = min(np.array([x['close'] for x in candles_sanitized]))
                max_price = max(np.array([x['close'] for x in candles_sanitized]))

                vw_checks = [4000, 2000, 1000]
                dp_checks = [10, 5, 2]

                abc = None
                abcd = None

                for vw_check in vw_checks:
                    abc, abcd = self.identify_abcds(market, candles_sanitized, full_vwaps, short_vwaps, simplify_coords_vw, (max_price - min_price) / 100, vw_check)
                    if abcd:
                        break

                if not abcd:
                    for dp_check in dp_checks:
                        abc, abcd = self.identify_abcds(market, candles_sanitized, full_vwaps, short_vwaps, simplify_coords, (max_price - min_price) / 100, dp_check)
                        if abcd:
                            break
                
                if abc:
                    # Track it
                    print('GOT A LIVE ONE')
                    plt.clf()
                    plt.plot(np.array([x['timestamp'] for x in candles_sanitized]), np.array([x['close'] for x in candles_sanitized]))
                    plt.plot(np.array([dt.datetime.fromtimestamp(x[0]) for x in abc]), np.array([x[1] for x in abc]), linewidth=3)
                    plt.savefig('./tracking/{}.jpg'.format(market['symbol']))

                if abcd:
                    # Make bet
                    print('MAKING A BET')
                    plt.clf()
                    plt.plot(np.array([x['timestamp'] for x in candles_sanitized]), np.array([x['close'] for x in candles_sanitized]))
                    plt.plot(np.array([dt.datetime.fromtimestamp(x[0]) for x in abcd]), np.array([x[1] for x in abcd]), linewidth=3)
                    plt.savefig('./bets/{}.jpg'.format(market['symbol']))
            except Exception as e:
                print(e)
        print('Done')

    def get_vwaps(self, candles):
        # https://academy.binance.com/en/articles/volume-weighted-average-price-vwap-explained
        results = []
        volume = 0

        for index, candle in enumerate(candles):
            typical_price = (float(candle[2]) + float(candle[3]) + float(candle[4])) / 3
            n = typical_price * float(candle[5])
            if index > 0:
                n += results[index - 1]['n']
            volume += float(candle[5])
            vwap = n / max(volume, 1)
            results.append({ 'timestamp': dt.datetime.fromtimestamp(candle[0] / 1000), 'n': n, 'vwap': vwap})

        return results

    def calc_atr(self, ticker):
        # FIXME: This is TR calculation, not ATR
        # current high - previous close
        a = abs(float(ticker['highPrice']) - float(ticker['prevClosePrice']))
        # current low - previous close
        b = abs(float(ticker['lowPrice']) - float(ticker['prevClosePrice']))
        # current high - current low
        c = abs(float(ticker['highPrice']) - float(ticker['lowPrice']))

        return max(a, b, c)

    def identify_abcds(self, market, candles, full_vwaps, short_vwaps, simplification_algo, ratio, factor):
        plt.clf()

        # Cut timeframes for plotting and pattern calculation
        candles = [x for x in candles if x['timestamp'] >= self.min_timestamp]
        full_vwaps = [x for x in full_vwaps if x['timestamp'] >= self.min_timestamp]

        timestamps = [x['timestamp'] for x in candles]
        prices = [x['close'] for x in candles]
        full_vwaps = [x for x in full_vwaps]

        coords = [list(pair) for pair in zip([x.timestamp() for x in timestamps], prices)]
        simplified = simplification_algo(coords, ratio * factor)
        s_timestamps = [dt.datetime.fromtimestamp(x[0]) for x in simplified]
        s_prices = [x[1] for x in simplified]

        local_maximas = argrelextrema(np.array(s_prices), np.greater)[0].tolist()
        local_minimas = argrelextrema(np.array(s_prices), np.less)[0].tolist()

        # Append start and end prices
        if timestamps[local_maximas[0]] < timestamps[local_minimas[0]]:
            local_minimas.insert(0, 0)
        else:
            local_maximas.insert(0, 0)

        if timestamps[local_maximas[len(local_maximas) - 1]] > timestamps[local_minimas[len(local_minimas) - 1]]:
            local_minimas.append(len(s_prices) - 1)
        else:
            local_maximas.append(len(s_prices) - 1)

        # extremas = list(chain(*zip(local_maximas, local_minimas)))
        plt.plot(np.array(timestamps), np.array([x['vwap'] for x in full_vwaps]), linewidth=0.5)
        plt.plot(np.array(timestamps), np.array([x['vwap'] for x in short_vwaps]), linewidth=0.5)
        plt.plot(np.array(timestamps), np.array(prices), linewidth=1)
        plt.plot(np.array(s_timestamps), np.array(s_prices), linewidth=3)
        # plt.plot(np.array([x for i, x in enumerate(timestamps) if i in extremas]), np.array([x for i, x in enumerate(prices) if i in extremas]))
        # plt.plot(np.array([x for i, x in enumerate(s_timestamps) if i in local_maximas]), np.array([x for i, x in enumerate(s_prices) if i in local_maximas]))
        # plt.plot(np.array([x for i, x in enumerate(s_timestamps) if i in local_minimas]), np.array([x for i, x in enumerate(s_prices) if i in local_minimas]))

        # Get trend for past 6 hours
        # https://stackoverflow.com/questions/10048571/python-finding-a-trend-in-a-set-of-numbers

        # Maybe calculate ABCDs using trend lines?
        # https://towardsdatascience.com/programmatic-identification-of-support-resistance-trend-lines-with-python-d797a4a90530

        abcs = []

        # BUYING
        for i in local_minimas:
            bx = s_timestamps[i].timestamp()
            by = s_prices[i]

            # Split surrounding maximas that are greater than current extrema
            prev_maximas = [x for x in local_maximas if s_timestamps[x].timestamp() < bx and s_prices[x] > by][-1:]
            next_maximas = [x for x in local_maximas if s_timestamps[x].timestamp() > bx and s_prices[x] > by][:1]

            for j in prev_maximas:
                ax = s_timestamps[j].timestamp()
                ay = s_prices[j]

                # Calculate length
                len_ab = math.sqrt(pow(by - ay, 2) + pow(bx - ax, 2))

                # Calculate price diff
                diff_price_ab = ay - by

                for k in next_maximas:
                    cx = s_timestamps[k].timestamp()
                    cy = s_prices[k]

                    len_bc = math.sqrt(pow(cy - by, 2) + pow(cx - bx, 2))
                    diff_price_cb = cy - by

                    # In goldilocks zone?
                    if len_ab > len_bc:
                        if 0.5 <= diff_price_cb / diff_price_ab <= 0.8:
                            abcs.append([[ax, ay], [bx, by], [cx, cy]])

        abcds = []

        for abc in abcs:
            pt_a = abc[0]
            ax = pt_a[0]
            ay = pt_a[1]

            pt_b = abc[1]
            bx = pt_b[0]
            by = pt_b[1]

            pt_c = abc[2]
            cx = pt_c[0]
            cy = pt_c[1]

            diff_price_ab = ay - by
            slope_ab = (by - ay) / (bx - ax)

            next_minimas = [x for x in local_minimas if s_timestamps[x].timestamp() > cx and s_prices[x] < cy][:1]

            for i in next_minimas:
                dx = s_timestamps[i].timestamp()
                dy = s_prices[i]

                diff_price_cd = cy - dy
                slope_cd = (dy - cy) / (dx - cx)

                # Calc VWAP directions

                # In goldilocks zone?
                if 0.5 <= slope_cd / slope_ab <= 1.5:
                    if 0.8 <= diff_price_cd / diff_price_ab <= 1.618:
                        abcds.append([[ax, bx, cx, dx], [ay, by, cy, dy]])

        for abcd in abcds:
            plt.plot(np.array([dt.datetime.fromtimestamp(x) for x in abcd[0]]), np.array(abcd[1]), linewidth=5)

        plt.savefig("./snapshots/{}-{}-{}.jpg".format(market['symbol'], simplification_algo.__name__, factor))

        abc = None
        abcd = None

        # Return latest ABC & ABCD
        if len(abcs) > 0:
            abc = [x for x in abcs if x[2][0] == s_timestamps[-1].timestamp()]
            if len(abc) > 0:
                abc = abc[-1]
        if len(abcds) > 0:
            abcd = [x for x in abcds if x[1][3] == s_timestamps[-1].timestamp()]
            if len(abcd) > 0:
                abcd = abcd[-1]

        return abc, abcd
