"""
The robot will run for ??? hours, playing the markets while managing risk.
Once finished, it will sell everything and store it in USDT.

A: Identify active markets
    - already handled with get_active_markets()
B: Identify trend in market
    - long MA (500 candles) and short MA (20 points? or 100 candles?)
    - HH's and HL's
C: Identify patterns in market
    - ABC fibonacci pattern (0.3 or 0.5 or 0.618)
    - 1st or 2nd retracement only
    - Avg. or below avg. volume at C (buffer above avg line at 5-10%)
D: Manage trade
    - simple sell at profit or stop

"""
import os
import time
import datetime as dt
import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd
import json
import itertools
from findiff import FinDiff
from itertools import chain
from scipy.signal import argrelextrema
from bnc import client

plt.style.use('seaborn-whitegrid')

min_balance = 90

ignore_list = ['XRPUPUSDT']

min_change_percent = 2
min_volume = 500000 / 2
min_trades = 50000 / 2
min_atr = 0.5

class Analyzer():
    def __init__(self, *args, **kwargs):
        super(Analyzer, self).__init__(*args, **kwargs)
        os.system('rm -r ./graphs/ > /dev/null 2>&1')
        os.system('mkdir ./graphs')
        self.portfolio = self.get_portfolio()

    def get_portfolio(self):
        balance = float(client.get_asset_balance(asset='USDT')['free'])

        return {
            'trade_fee': 0.0015, # assume fees are 0.15% of trade
            'balance': balance
        }

    def get_active_markets(self):
        markets = client.get_ticker()
        # Tether markets only
        tickers = [x for x in markets if x['symbol'].endswith('USDT')]
        # Not in ignore list
        tickers = [x for x in tickers if x['symbol'] not in ignore_list]
        # Fluctuating price ranges
        tickers = [x for x in tickers if abs(float(x['priceChangePercent'])) >= min_change_percent]
        # High volume
        tickers = [x for x in tickers if float(x['volume']) >= min_volume]
        # High amount of trades
        tickers = [x for x in tickers if float(x['count']) >= min_trades]
        # Good ATR
        tickers = [x for x in tickers if self.calc_atr(x) > min_atr]
        # FIXME: this grabs ETHUSDT for testing purposes
        # tickers = [x for x in tickers if x == 'ETHUSDT'] 
        # if len(tickers) == 0:
        #     tickers.append({ 'symbol': 'ETHUSDT' })
        return [x['symbol'] for x in tickers]

    def scan_markets(self):
        print('=> Analyzing markets')
        self.markets = self.get_active_markets()
        intervals = ['15m', '5m', '3m', '1m']

        for market_symbol in self.markets:
            print('=> Searching', market_symbol)

            # Get overall market trend
            candles = client.get_klines(symbol=market_symbol, interval='30m', limit=100)
            candles = [{ 'dt': dt.datetime.fromtimestamp(x[0] / 1000), 'close': float(x[4]), 'volume': float(x[5]) } for x in candles]
            price_range = max([x['close'] for x in candles]) - min([x['close'] for x in candles])
            moving_avgs = self.get_moving_averages(candles, 10)

            for interval in intervals:
                # Download candles
                interval_candles = client.get_klines(symbol=market_symbol, interval=interval, limit=101)
                interval_candles = [{ 'dt': dt.datetime.fromtimestamp(x[0] / 1000), 'open': float(x[1]), 'close': float(x[4]), 'volume': float(x[5]) } for x in interval_candles]
                interval_candles = interval_candles[:-1]
                price_range = max([x['close'] for x in interval_candles]) - min([x['close'] for x in interval_candles])
                interval_moving_avgs = self.get_moving_averages(interval_candles, 10)

                # Find extremas
                minima_ids, maxima_ids = self.get_extremas([x['close'] for x in interval_candles])
                extremas = [{ **x, 'category': 'minima' if i in minima_ids else 'maxima' } for i, x in enumerate(interval_candles) if i in minima_ids or i in maxima_ids]
                extremas = sorted(extremas, key=lambda x: x['dt']) 

                # Look for HH's and HL's
                hh_hl_count = self.get_hhs_and_hls(extremas)

                # Calculate average volume
                avg_volume = sum([x['volume'] for x in interval_candles]) / len(interval_candles)

                # Look for successful pullbacks
                pullbacks = self.find_successful_pullbacks(market_symbol, interval, interval_candles, interval_moving_avgs, extremas, avg_volume, price_range, hh_hl_count)
                pullbacks = [x for x in pullbacks if self.calc_pullback_trade(market_symbol, interval, x)]

                if len(pullbacks) > 0:
                    print('Got some!')
                    os.system('mkdir -p ./graphs/{}'.format(market_symbol))
                    os.system('mkdir ./graphs/{}/{}'.format(market_symbol, interval))

                    # Market graph
                    plt.figure(figsize=(20, 10))
                    plt.plot(np.array([x['dt'] for x in candles]), np.array([x['close'] for x in candles]))
                    plt.plot(np.array([x['dt'] for x in moving_avgs]), np.array([x['close'] for x in moving_avgs]))
                    plt.savefig('./graphs/{}/market.jpg'.format(market_symbol))
                    plt.clf()

                    # Interval graph
                    plt.figure(figsize=(20, 10))
                    plt.plot(np.array([x['dt'] for x in interval_candles]), np.array([x['close'] for x in interval_candles]))
                    plt.plot(np.array([x['dt'] for x in interval_moving_avgs]), np.array([x['close'] for x in interval_moving_avgs]))
                    plt.savefig('./graphs/{}/{}/candles.jpg'.format(market_symbol, interval))
                    plt.clf()

                    # Volume graph
                    plt.figure(figsize=(20, 10))
                    plt.plot(np.array([x['dt'] for x in interval_candles]), np.array([x['volume'] for x in interval_candles]))
                    plt.axhline(y=avg_volume, linestyle='--')
                    plt.savefig('./graphs/{}/{}/volume.jpg'.format(market_symbol, interval))
                    plt.clf()

                    # Pullback graph
                    plt.figure(figsize=(20, 10))
                    plt.plot(np.array([x['dt'] for x in extremas]), np.array([x['close'] for x in extremas]))
                    plt.plot(np.array([x['dt'] for x in interval_moving_avgs]), np.array([x['close'] for x in interval_moving_avgs]))
                    for pullback in pullbacks:
                        plt.plot(np.array([x['dt'] for x in pullback]), np.array([x['close'] for x in pullback]), linewidth=3)
                    plt.savefig('./graphs/{}/{}/pullbacks.jpg'.format(market_symbol, interval))
                    plt.clf()
        print('Done')

    def get_moving_averages(self, candles, periods):
        results = []
        divisions = math.ceil(len(candles) / periods)

        for i, candle in enumerate(candles):
            price = candle['close']
            if i % divisions == 0:
                results.append({ 'dt': candle['dt'], 'close': price, 'count': 1})
            else:
                results[len(results) - 1]['close'] += price
                results[len(results) - 1]['count'] += 1

            if i % divisions == (len(candles) / periods) - 1 or i == len(candles) - 1:
                results[len(results) - 1]['dt'] = candle['dt']
                results[len(results) - 1]['close'] /= max(results[len(results) - 1]['count'], 1)

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

    def get_time_to_next_time_period(self, period=1):
        delta = dt.timedelta(minutes=period)
        now = dt.datetime.now()
        next_minute = (now + delta).replace(microsecond=0, second=1)
        return next_minute - now

    def get_extremas(self, prices):
        h = prices
        d_dx = FinDiff(0, 1, 1)
        d2_dx2 = FinDiff(0, 1, 2)
        clarr = np.asarray(prices)
        mom = d_dx(clarr)
        momacc = d2_dx2(clarr)
        def get_extrema(isMin):
            return [x for x in range(len(mom))
                if (momacc[x] > 0 if isMin else momacc[x] < 0) and
                (mom[x] == 0 or
                    (x != len(mom) - 1 and
                    (mom[x] > 0 and mom[x+1] < 0 and
                    h[x] >= h[x+1] or
                    mom[x] < 0 and mom[x+1] > 0 and
                    h[x] <= h[x+1]) or
                    x != 0 and
                    (mom[x-1] > 0 and mom[x] < 0 and
                    h[x-1] < h[x] or
                    mom[x-1] < 0 and mom[x] > 0 and
                    h[x-1] > h[x])))]
        return get_extrema(True), get_extrema(False)

    def get_hhs_and_hls(self, extremas):
        e_max = 0
        e_min = 0
        chain = 0
        # Start from latest extrema
        for x in reversed(extremas):
            if x['category'] == 'maxima':
                if e_max == 0:
                    e_max = x['close']
                else:
                    if x['close'] < e_max:
                        e_max = x['close']
                        chain += 1
                    else:
                        break
            else:
                if e_min == 0:
                    e_min = x['close']
                else:
                    if x['close'] < e_min:
                        e_min = x['close']
                        chain += 1
                    else:
                        break
        # Should be part of jagged uptrend
        # Require minimum 2: while there COULD be a reversal, we want to wait until trend is established
        # Require maximum 5: we want to jump in on 1st or 2nd wave (max 4), but there might be a rogue extrema, so this is loosened a bit
        return chain

    def is_acceptable_risk_reward(self, market_symbol, buy_price, price_range, stop_price, profit_price):
        fee = self.portfolio['trade_fee']
        balance = self.portfolio['balance']

        # How many coins would you have if you bought now?
        coins = (balance / buy_price) * (1 - fee)

        # How much money would be gained if sold at profit price?
        reward = ((profit_price - buy_price) * coins) * (1 - fee)

        # How much money lost if sold at stop price?
        risk = ((buy_price - stop_price) * coins) * (1 + fee)

        return reward > (2 * (balance * fee)) and reward / risk >= 1.8

    def calc_pullback_trade(self, market_symbol, interval, pullback):
        price_range = pullback[1]['close'] - pullback[0]['close']
        stop_price = pullback[2]['close'] - (price_range * 0.1)
        profit_price = pullback[2]['close'] + (price_range * 0.9)
        buy_price = pullback[2]['close'] + (price_range * 0.1)

        if self.is_acceptable_risk_reward(market_symbol, buy_price, price_range, stop_price, profit_price):
            return True
        return False

    def find_successful_pullbacks(self, market_symbol, interval, candles, moving_avgs, extremas, avg_volume, price_range, hh_hl_count):
        # Normalize
        max_dt = max([x['dt'].timestamp() for x in extremas])
        min_dt = min([x['dt'].timestamp() for x in extremas])
        max_close = max([x['close'] for x in extremas])
        min_close = min([x['close'] for x in extremas])
        norm_extremas = [{ 'category': x['category'], 'dt': (x['dt'].timestamp() - min_dt) / (max_dt - min_dt), 'close': (x['close'] - min_close) / (max_close - min_close), 'index': i } for i, x in enumerate(extremas)]

        # Find clean pullbacks
        pullbacks = []
        first_minima_idx = next(i for i, x in enumerate(norm_extremas) if x['category'] == 'minima')
        try:
            for x in norm_extremas[first_minima_idx::2]:
                n_a = x
                n_b = next(y for y in norm_extremas if y['index'] == x['index'] + 1)
                n_c = next(y for y in norm_extremas if y['index'] == x['index'] + 2)
                n_d = next(y for y in norm_extremas if y['index'] == x['index'] + 3)

                len_ab = abs(math.sqrt(((n_b['close'] - n_a['close']) ** 2) + ((n_b['dt'] - n_a['dt']) ** 2)))
                len_bc = abs(math.sqrt(((n_c['close'] - n_b['close']) ** 2) + ((n_c['dt'] - n_b['dt']) ** 2)))
                len_cd = abs(math.sqrt(((n_d['close'] - n_c['close']) ** 2) + ((n_d['dt'] - n_c['dt']) ** 2)))

                if 0.3 <= len_bc / len_ab <= 0.7 and len_cd / len_ab >= 0.9:
                    pullbacks.append([extremas[n_a['index']], extremas[n_b['index']], extremas[n_c['index']], extremas[n_d['index']]])
        except StopIteration:
            pass

        return pullbacks
