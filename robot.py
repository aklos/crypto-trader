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
min_volume = 500000
min_trades = 50000
min_atr = 0.5

class Robot():
    def __init__(self, *args, **kwargs):
        super(Robot, self).__init__(*args, **kwargs)
        self.portfolio = self.get_portfolio()
        self.trading = False
        self.trade = {}
        try:
            self.tradebook = pd.read_csv('./tradebook.csv', index=False)
        except:
            self.tradebook = None
        os.system('rm ./graphs/*.jpg')

    def get_portfolio(self):
        return {
            'trade_fee': 0.0015, # assume fees are 0.15% of trade
            'balance': float(client.get_asset_balance(asset='USDT')['free'])
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

    def scan_market_loop(self):
        sleep_time = self.get_time_to_next_time_period()
        print('=> Starting in {} seconds'.format(sleep_time.seconds))
        time.sleep(sleep_time.seconds)

        while (True):
            self.markets = self.get_active_markets()
            print('=> Scanning {} active markets'.format(len(self.markets)))
            self.scan_markets()
            if self.trading:
                break
            sleep_time = self.get_time_to_next_time_period()
            print('=> Sleeping for {} seconds'.format(sleep_time.seconds))
            time.sleep(sleep_time.seconds)
        
        if self.trade is not None:
            print('=> Making a trade in {}'.format(self.trade['market_symbol']))
            self.buy_coins()
            while (True):
                self.manage_trade()
                if not self.trading:
                    break
                time.sleep(1)

        if self.portfolio['balance'] <= min_balance:
            quit()

        self.scan_market_loop()

    def manage_trade(self):
        price = float(client.get_klines(symbol=self.trade['market_symbol'], interval='1m', limit=1)[0][4])

        print('curr', price, 'stop', self.trade['stop_price'], 'profit', self.trade['profit_price'])

        # TODO: Catch reversals here with volume comparison
        # TODO: Use average market price here for support check

        if price <= self.trade['stop_price']:
            self.sell_coins(price)
            return

        if price >= self.trade['profit_price']:
            self.sell_coins(price)
            return

    def update_tradebook(self, data):
        if self.tradebook is not None:
            self.tradebook = self.tradebook.append(data, ignore_index=True)
        else:
            self.tradebook = pd.DataFrame([data])
        self.tradebook.to_csv('./tradebook.csv', index=False)

    def buy_coins(self):
        # TODO: Setup marketable limit order
        balance = self.portfolio['balance']
        fee = self.portfolio['trade_fee']
        coins = (balance / self.trade['buy_price']) * (1 - fee)
        self.update_tradebook({ 
            'timestamp': dt.datetime.now(), 
            'action': 'BUY', 
            'market': self.trade['market_symbol'], 
            'price': self.trade['buy_price'], 
            'amount': coins, 
            'total_balance': balance 
            })

    def sell_coins(self, price):
        fee = self.portfolio['trade_fee']
        coins = self.tradebook.tail(1).iloc[0]['amount']
        balance = (coins * price) * (1 - fee)
        self.portfolio['balance'] = balance
        self.update_tradebook({ 
            'timestamp': dt.datetime.now(), 
            'action': 'SELL', 
            'market': self.trade['market_symbol'], 
            'price': price, 
            'amount': coins, 
            'total_balance': balance
            })
        self.trade = {}
        self.trading = False

    def scan_markets(self):
        # TODO: Try with 30m snapshot
        intervals = ['5m', '3m', '1m']

        for market_symbol in self.markets:
            print('=> Searching', market_symbol)

            # Get overall market trend
            # TODO: Include latest unfinished candle or not?
            candles = client.get_klines(symbol=market_symbol, interval='15m', limit=100)
            candles = [{ 'dt': dt.datetime.fromtimestamp(x[0] / 1000), 'close': float(x[4]), 'volume': float(x[5]) } for x in candles]
            price_range = max([x['close'] for x in candles]) - min([x['close'] for x in candles])
            moving_avgs = self.get_moving_averages(candles, 20)

            plt.plot(np.array([x['dt'] for x in candles]), np.array([x['close'] for x in candles]))
            plt.plot(np.array([x['dt'] for x in moving_avgs]), np.array([x['close'] for x in moving_avgs]))
            plt.savefig('./graphs/{}-15m-snapshot.jpg'.format(market_symbol))
            plt.clf()

            # Skip if trending less than 5% upwards
            if (moving_avgs[-1]['close'] - moving_avgs[-2]['close']) / price_range < 0.05:
                continue 

            for interval in intervals:
                # Download candles
                candles = client.get_klines(symbol=market_symbol, interval=interval, limit=101)
                candles = [{ 'dt': dt.datetime.fromtimestamp(x[0] / 1000), 'close': float(x[4]), 'volume': float(x[5]) } for x in candles]
                candles = candles[:-1]
                price_range = max([x['close'] for x in candles]) - min([x['close'] for x in candles])
                moving_avgs = self.get_moving_averages(candles, 10)

                # Skip if trending less than 10% upwards for last 20% of graph
                if (moving_avgs[-1]['close'] - moving_avgs[-3]['close']) / price_range < 0.1:
                    continue

                # Find extremas
                minima_ids, maxima_ids = self.get_extremas([x['close'] for x in candles])
                extremas = [{ **x, 'category': 'minima' if i in minima_ids else 'maxima' } for i, x in enumerate(candles) if i in minima_ids or i in maxima_ids]
                extremas = sorted(extremas, key=lambda x: x['dt']) 

                # Look for HH's and HL's
                if not self.has_hhs_and_hls(extremas):
                    continue

                plt.plot(np.array([x['dt'] for x in extremas]), np.array([x['close'] for x in extremas]))
                plt.plot(np.array([x['dt'] for x in moving_avgs]), np.array([x['close'] for x in moving_avgs]))
                plt.savefig('./graphs/{}-{}-uptrend.jpg'.format(market_symbol, interval))
                plt.clf()

                # Calculate average volume
                avg_volume = sum([x['volume'] for x in candles]) / len(candles)

                plt.plot(np.array([x['dt'] for x in candles]), np.array([x['volume'] for x in candles]))
                plt.axhline(y=avg_volume, linestyle='--')
                plt.savefig('./graphs/{}-{}-volumes.jpg'.format(market_symbol, interval))
                plt.clf()

                # Look for trade entry
                try:
                    pullback = self.find_pullback(market_symbol, interval, candles, extremas, avg_volume)

                    plt.plot(np.array([x['dt'] for x in extremas]), np.array([x['close'] for x in extremas]))
                    plt.plot(np.array([x['dt'] for x in pullback]), np.array([x['close'] for x in pullback]), linewidth=3)
                    plt.savefig('./graphs/{}-{}-pullback.jpg'.format(market_symbol, interval))
                    plt.clf()

                    self.calc_pullback_trade(market_symbol, interval, pullback)

                    if self.trading:
                        df_candles = pd.DataFrame(candles)
                        df_candles.to_csv('./test_data/{}_{}_{}_candles.csv'.format(dt.datetime.now().strftime('%Y%m%d_%H%M'), market_symbol, interval))
                        return
                except AssertionError as e:
                    print('{} -'.format(interval), e)
                except Exception as e:
                    print('ERROR(pullback)', e)
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

    def has_hhs_and_hls(self, extremas):
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
        # Should be start of jagged uptrend
        return 1 <= chain <= 4

    def is_acceptable_risk_reward(self, market_symbol, buy_price, price_range, stop_price, profit_price):
        fee = self.portfolio['trade_fee']

        # How many coins would you have if you bought now?
        coins = (self.portfolio['balance'] / buy_price) * (1 - fee)

        # How much money would be gained if sold at profit price?
        reward = ((profit_price - buy_price) * coins) * (1 - fee)

        # How much money lost if sold at stop price?
        risk = ((buy_price - stop_price) * coins) * (1 + fee)

        return reward >= 0.1 and reward / risk >= 1.8

    def calc_pullback_trade(self, market_symbol, interval, pullback):
        price_range = pullback[1]['close'] - pullback[0]['close']
        stop_price = pullback[2]['close'] - (price_range * 0.1)
        profit_price = pullback[2]['close'] + (price_range * 0.9)
        buy_price = float(client.get_klines(symbol=market_symbol, interval='1m', limit=1)[0][4])

        if self.is_acceptable_risk_reward(market_symbol, buy_price, price_range, stop_price, profit_price):
            self.trade = { 'market_symbol': market_symbol, 'interval': interval, 'buy_price': buy_price, 'stop_price': stop_price, 'profit_price': profit_price }
            self.trading = True

    def find_pullback(self, market_symbol, interval, candles, extremas, avg_volume):
        # End on minima
        assert extremas[-1]['category'] == 'minima', 'does not end on minima'

        # Last candle should be bullish
        assert candles[-1]['close'] > extremas[-1]['close'], 'last candle not bullish'

        # C volume should be less than or equal to avg (+2%)
        assert extremas[-1]['volume'] / avg_volume <= 1.02, 'above avg volume'

        # Normalize
        max_dt = max([x['dt'].timestamp() for x in extremas])
        min_dt = min([x['dt'].timestamp() for x in extremas])
        max_close = max([x['close'] for x in extremas])
        min_close = min([x['close'] for x in extremas])
        norm_extremas = [{ 'dt': (x['dt'].timestamp() - min_dt) / (max_dt - min_dt), 'close': (x['close'] - min_close) / (max_close - min_close) } for x in extremas]

        n_a = norm_extremas[-3]
        n_b = norm_extremas[-2]
        n_c = norm_extremas[-1]

        # Calculate lengths of segments 
        # math.sqrt(((y2 - y1) ** 2)) + ((x2 - x1) ** 2))
        len_ab = abs(math.sqrt(((n_b['close'] - n_a['close']) ** 2) + ((n_b['dt'] - n_a['dt']) ** 2)))
        len_bc = abs(math.sqrt(((n_c['close'] - n_b['close']) ** 2) + ((n_c['dt'] - n_b['dt']) ** 2)))

        print(len_ab, len_bc, len_bc / len_ab)
        print('a', n_a)
        print('b', n_b)
        print('c', n_c)

        # Should match fibonacci rule
        assert 0.38 <= len_bc / len_ab <= 0.618, 'failed fibonacci rule'

        a = extremas[-3]
        b = extremas[-2]
        c = extremas[-1]

        return [a, b, c]
