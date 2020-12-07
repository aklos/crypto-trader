"""
The robot will run for ??? hours, playing the markets while managing risk.
Once finished, it will sell everything and store it in USDT.

"""
import os
import time
import datetime as dt
import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd
import helpers
import json
import itertools
from findiff import FinDiff
from itertools import chain
from scipy.signal import argrelextrema
from bnc import client

plt.style.use('seaborn-whitegrid')

min_goal = 5
min_bet = 10

ignore_list = ['XRPUPUSDT']

min_change_percent = 2
min_volume = 500000
min_trades = 50000
min_atr = 0.5

# TODO: Implement these options
max_change_percent = 3
max_wobble_percent = 20
min_spikes = 3

class Robot():
    def __init__(self, *args, **kwargs):
        super(Robot, self).__init__(*args, **kwargs)
        self.portfolio = self.get_portfolio()
        self.goal = max(self.portfolio['balance'] * 0.0002, min_goal)
        self.trackbook = []
        self.trade = None
        try:
            self.trade_history = pd.read_csv('./trade_history.csv')
        except:
            self.trade_history = None
        os.system('rm ./market_graphs/*.jpg')
        os.system('rm ./possible_consolidations/*.jpg')
        os.system('rm ./possible_pullbacks/*.jpg')

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
        # Add tracked markets, if any
        ticker_market_symbols = [x['symbol'] for x in tickers]
        for tracked in self.trackbook:
            if tracked['market'] not in ticker_market_symbols:
                ticker_data = [x for x in markets if x['symbol'] == tracked['market']][0]
                tickers.append(ticker_data)
                ticker_market_symbols.append(tracked['market'])
        # FIXME: this grabs ETHUSDT for testing purposes
        # tickers = [x for x in tickers if x['symbol'] == 'ETHUSDT'] 
        # if len(tickers) == 0:
        #     tickers.append({ 'symbol': 'ETHUSDT', 'priceChange': '2', 'weightedAvgPrice': '500' })
        return [{'symbol': x['symbol'], 'price_change': abs(float(x['priceChange'])), 'weighted_avg_price': float(x['weightedAvgPrice']) } for x in tickers]

    def scan_market_loop(self):
        sleep_time = self.get_time_to_next_time_period()
        print('=> Starting in {} seconds'.format(sleep_time.seconds))
        time.sleep(sleep_time.seconds)

        while (True):
            self.markets = self.get_active_markets()
            print('=> Scanning {} active markets'.format(len(self.markets)))
            self.scan_markets()
            if self.trade is not None:
                break
            sleep_time = self.get_time_to_next_time_period()
            print('=> Sleeping for {} seconds'.format(sleep_time.seconds))
            time.sleep(sleep_time.seconds)
        
        if self.trade is not None:
            print('=> Making a trade in {}'.format(self.trade['market']))
            self.buy_coins()
            while (True):
                self.manage_trade()
                if self.trade is None:
                    break
                time.sleep(1)

        # TODO: Stop if lost 3 times in a row?

        self.scan_market_loop()

    def manage_trade(self):
        price = float(client.get_klines(symbol=self.trade['market'], interval='1m', limit=1)[0][4])

        print('curr', price, 'stop', self.trade['stop_price'], 'profit', self.trade['profit_price'])

        if price <= self.trade['stop_price']:
            # TODO: Check if price is A LOT lower than stop price, if so, hold
            self.sell_coins(price)
            return

        if price >= self.trade['profit_price']:
            self.sell_coins(price)
            return

    def buy_coins(self):
        # TODO: Setup marketable limit order
        balance = self.portfolio['balance']
        fee = self.portfolio['trade_fee']
        coins = (balance / self.trade['buy_price']) * (1 - fee)
        data = { 
            'timestamp': dt.datetime.now(), 
            'action': 'BUY', 
            'market': self.trade['market'], 
            'price': self.trade['buy_price'], 
            'amount': coins, 
            'total_balance': balance 
            }

        if self.trade_history is not None:
            self.trade_history = self.trade_history.append(data, ignore_index=True)
        else:
            self.trade_history = pd.DataFrame([data])
        self.trade_history.to_csv('./trade_history.csv')

    def sell_coins(self, price):
        fee = self.portfolio['trade_fee']
        coins = self.trade_history.tail(1).iloc[0]['amount']
        # Convert coins to balance at current price
        balance = (coins * price) * (1 - fee)
        self.portfolio['balance'] = balance
        
        data = { 
            'timestamp': dt.datetime.now(), 
            'action': 'SELL', 
            'market': self.trade['market'], 
            'price': price, 
            'amount': coins, 
            'total_balance': balance
            }

        if self.trade_history is not None:
            self.trade_history = self.trade_history.append(data, ignore_index=True)
        else:
            self.trade_history = pd.DataFrame([data])
        self.trade_history.to_csv('./trade_history.csv')
        self.trade = None

    def scan_markets(self):
        time_ranges = ['30m', '15m', '5m', '3m', '1m']

        for market in self.markets:
            print('=> Searching', market['symbol'])
            for time_range in time_ranges:
                # Download candles
                candles = client.get_klines(symbol=market['symbol'], interval=time_range, limit=101)
                # Remove current live candle
                candles = candles[:-1]
                # Calculate moving average prices
                moving_averages = self.get_moving_averages(candles, 20)
                # Calculate moving average volumes
                # moving_volume_averages = self.get_moving_volume_averages(candles, 20)
                # Serialize candles to dicts
                candles_sanitized = [{'timestamp': dt.datetime.fromtimestamp(x[0] / 1000), 'open': float(x[1]), 'close': float(x[4]), 'volume': float(x[5]) } for x in candles]

                trackbook_entries = [x for x in self.trackbook if x['market'] == market['symbol'] and x['time_range'] == time_range]

                if len(trackbook_entries) == 0:
                    # Pullbacks
                    try:
                        a_price, b_price, c_price = self.find_pullback(market, time_range, candles_sanitized, moving_averages)
                        print(a_price, b_price, c_price)
                        self.track_pullback(a_price, b_price, c_price, candles_sanitized, market['symbol'], time_range)
                        if self.trade is not None:
                            return
                    except AssertionError:
                        pass
                    except Exception as e:
                        print('({})'.format(time_range), e)

                    # Consolidations
                    try:
                        res, sup, start = self.find_possible_consolidation(market, time_range, candles_sanitized, moving_averages)
                        print(res, sup, start)
                        # Add to trackbook
                        self.trackbook.append({ 'market': market['symbol'], 'time_range': time_range, 'resistance': res, 'support': sup, 'start': start })
                    except AssertionError:
                        pass
                    except Exception as e:
                        print('({})'.format(time_range), e)
                else:
                    self.track_consolidation(trackbook_entries[0], candles_sanitized)
                    if self.trade is not None:
                        return
        print('Done')

    # def get_moving_volume_averages(self, candles, chunks):
    #     results = []
    #     divisions = math.ceil(len(candles) / chunks)

    #     for i, candle in enumerate(candles):
    #         volume = float(candle[5])
    #         if i % divisions == 0:
    #             results.append({ 'timestamp': dt.datetime.fromtimestamp(candle[0] / 1000), 'volume': volume, 'count': 1})
    #         else:
    #             results[len(results) - 1]['volume'] += volume
    #             results[len(results) - 1]['count'] += 1

    #         if i % divisions == (len(candles) / chunks) - 1 or i == len(candles) - 1:
    #             results[len(results) - 1]['timestamp'] = dt.datetime.fromtimestamp(candle[0] / 1000)
    #             results[len(results) - 1]['volume'] /= max(results[len(results) - 1]['count'], 1)

    #     return results

    def get_moving_averages(self, candles, chunks):
        results = []
        divisions = math.ceil(len(candles) / chunks)

        for i, candle in enumerate(candles):
            price = float(candle[4])
            if i % divisions == 0:
                results.append({ 'timestamp': dt.datetime.fromtimestamp(candle[0] / 1000), 'price': price, 'count': 1})
            else:
                results[len(results) - 1]['price'] += price
                results[len(results) - 1]['count'] += 1

            if i % divisions == (len(candles) / chunks) - 1 or i == len(candles) - 1:
                results[len(results) - 1]['timestamp'] = dt.datetime.fromtimestamp(candle[0] / 1000)
                results[len(results) - 1]['price'] /= max(results[len(results) - 1]['count'], 1)

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

    def calc_avg_slope_to_pt(self, df, end_dt=None):
        slopes = []
        if end_dt:
            df = df[df['timestamp'].lt(end_dt)]
        for (indx1, row1), (indx2, row2) in zip(df[:-1].iterrows(), df[1:].iterrows()):
            slope = (row2['price'] - row1['price']) / (row2['timestamp'].timestamp() - row1['timestamp'].timestamp())
            slopes.append(slope)

        return sum(slopes) / max(len(slopes), 1)

    def get_time_to_next_time_period(self, period=1):
        delta = dt.timedelta(minutes=period)
        now = dt.datetime.now()
        next_minute = (now + delta).replace(microsecond=0, second=1)
        return next_minute - now

    def get_extremas(self, df):
        h = df.close.tolist()
        d_dx = FinDiff(0, 1, 1)
        d2_dx2 = FinDiff(0, 1, 2)
        clarr = np.asarray(df.close)
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

    def find_pullback(self, market, time_frame, candles, moving_averages):
        df_candles = pd.DataFrame(candles)
        df_moving_averages = pd.DataFrame(moving_averages)

        # 1. Determine trend
        start = moving_averages[0]
        end = moving_averages[-1]
        slope = (end['price'] - start['price']) / (end['timestamp'].timestamp() - start['timestamp'].timestamp())

        # Global trend must be positive
        assert(slope > 0)

        # FIXME: Is this useful?
        lead_up_averages = moving_averages[-3:]
        start = lead_up_averages[0]
        end = lead_up_averages[-1]
        lead_up_slope = (end['price'] - start['price']) / (end['timestamp'].timestamp() - start['timestamp'].timestamp())

        # Lead up trend must be higher than global trend
        assert(lead_up_slope > slope)

        # 2. Get extrema
        minima_idxs, maxima_idxs = self.get_extremas(df_candles)
        df_maximas = df_candles.loc[df_candles.index.isin(maxima_idxs),:]
        df_maximas['type'] = 'maxima'
        df_minimas = df_candles.loc[df_candles.index.isin(minima_idxs),:]
        df_minimas['type'] = 'minima'
        df_extremas = pd.concat([df_maximas, df_minimas]).sort_values('timestamp')

        # plt.plot(np.array(df_candles.timestamp), np.array(df_candles.close))
        # plt.plot(np.array(df_extremas.timestamp), np.array(df_extremas.close), linewidth=2)
        # plt.savefig('./market_graphs/{}-{}.jpg'.format(market['symbol'], time_frame))
        # plt.clf()

        # 3. Find all pullbacks
        started = False
        tuples = [[]]
        for i, extrema in df_extremas.iterrows():
            if extrema['type'] == 'minima':
                started = True

            if started:
                tuples[-1].append(extrema)
                if len(tuples[-1]) == 3:
                    tuples.append([])

        tuples = [x for x in tuples if len(x) == 3]
        
        pullbacks = []
        for x in tuples:
            a = x[0]
            b = x[1]
            c = x[2]
            diff_ab = b['close'] - a['close']
            if diff_ab > 0:
                diff_bc = b['close'] - c['close']
                if 0.618 >= diff_bc / diff_ab >= 0.4:
                    pullbacks.append(x)
        
        assert(len(pullbacks) > 0)
        latest_pullback = pullbacks[-1]

        plt.plot(np.array(df_extremas.timestamp), np.array(df_extremas.close))
        plt.plot(np.array(df_moving_averages.timestamp), np.array(df_moving_averages.price))
        plt.plot(np.array([x['timestamp'] for x in latest_pullback]), np.array([x['close'] for x in latest_pullback]), linewidth=2)
        plt.savefig('./possible_pullbacks/{}-{}.jpg'.format(market['symbol'], time_frame))
        plt.clf()

        # Latest pullback should end at current time
        assert(latest_pullback[2]['timestamp'] == df_extremas.tail(1).iloc[0]['timestamp'])

        a_price = latest_pullback[0]['close']
        b_price = latest_pullback[1]['close']
        c_price = latest_pullback[2]['close']

        return a_price, b_price, c_price

    def find_possible_consolidation(self, market, time_frame, candles, moving_averages):
        prices = [x['close'] for x in candles]

        min_price = min(prices)
        max_price = max(prices)
        price_range = max_price - min_price

        df_candles = pd.DataFrame(candles)
        df_moving_averages = pd.DataFrame(moving_averages)

        # 1. Get extrema
        minima_idxs, maxima_idxs = self.get_extremas(df_candles)
        df_maximas = df_candles.loc[df_candles.index.isin(maxima_idxs),:]
        df_maximas['type'] = 'maxima'
        df_minimas = df_candles.loc[df_candles.index.isin(minima_idxs),:]
        df_minimas['type'] = 'minima'
        df_extremas = pd.concat([df_maximas, df_minimas]).sort_values('timestamp')

        # plt.plot(np.array(df_moving_averages.timestamp), np.array(df_moving_averages.price))
        # plt.plot(np.array(df_extremas.timestamp), np.array(df_extremas.close), linewidth=2)
        # plt.savefig('./market_graphs/{}-{}.jpg'.format(market['symbol'], time_frame))
        # plt.clf()

        # 2. Find lulls in the moving average
        chains = []
        for (idx_a, a), (idx_b, b) in zip(df_moving_averages[:-1].iterrows(),df_moving_averages[1:].iterrows()):
            if len(chains) == 0:
                chains.append({ 'idxs': [], 'rows': []})

            percent_change = 100 * ((b['price'] - a['price']) / price_range)

            if abs(percent_change) <= 3:
                if idx_a not in chains[-1]['idxs']:
                    chains[-1]['idxs'].append(idx_a)
                    chains[-1]['rows'].append(a)
                if idx_b not in chains[-1]['idxs']:
                    chains[-1]['idxs'].append(idx_b)
                    chains[-1]['rows'].append(b)
            else:
                chains.append({ 'idxs': [], 'rows': []})

        assert(len([x for x in chains if len(x['idxs']) >= 3]) > 0)

        last_chain = [x for x in chains if len(x['idxs']) >= 3][-1]

        a = last_chain['rows'][0]
        b = last_chain['rows'][-1]

        # 3. Check that average volume in range is below global average
        global_avg_volume = sum([x['volume'] for x in candles]) / len(candles)
        candles_in_range = [x for x in candles if x['timestamp'] >= a['timestamp'] and x['timestamp'] <= b['timestamp']]
        avg_volume = sum([x['volume'] for x in candles_in_range]) / len(candles_in_range)

        assert(avg_volume / global_avg_volume < 1)
        
        # 4. Get ratio of extremas above and below trend line
        slope = (b['price'] - a['price']) / (b['timestamp'].timestamp() - a['timestamp'].timestamp())

        df_extremas_in_range = df_extremas[df_extremas['timestamp'].ge(a['timestamp']) & df_extremas['timestamp'].le(b['timestamp'])]

        # FIXME: MA points might cut off first or last extrema
        # Need at least 3 spikes, so require 5 extremas
        assert(len(df_extremas_in_range) >= 5)

        above_count = 0
        below_count = 0

        for i, extrema in df_extremas_in_range.iterrows():
            # Point-slope formula
            y = (slope * (extrema['timestamp'].timestamp() - a['timestamp'].timestamp())) + a['price']

            if extrema['close'] > y:
                above_count += 1
            if extrema['close'] < y:
                below_count += 1

        ratio = above_count / below_count

        # Allow 25% lean in either direction
        assert(ratio <= 1.25 and ratio >= 0.75)

        # Only allow through if ends at current time
        end_dt = df_extremas_in_range.tail(1).iloc[0]['timestamp']
        last_extrema = df_extremas.tail(1).iloc[0]

        assert(end_dt >= last_extrema['timestamp'])
        
        # PASSED ALL CHECKS
        resistance = float(df_extremas_in_range.iloc[df_extremas_in_range['close'].argmax()]['close'])
        support = float(df_extremas_in_range.iloc[df_extremas_in_range['close'].argmin()]['close'])
        start_dt = df_extremas_in_range.head(1).iloc[0]['timestamp']

        plt.plot(np.array(df_extremas.timestamp), np.array(df_extremas.close), linewidth=1)
        plt.plot(np.array(df_moving_averages.timestamp), np.array(df_moving_averages.price))
        plt.plot(np.array(df_extremas_in_range.timestamp), np.array(df_extremas_in_range.close), linewidth=3)
        plt.axhline(y=resistance, linestyle='--')
        plt.axhline(y=support, linestyle='--')
        plt.savefig('./possible_consolidations/{}-{}.jpg'.format(market['symbol'], time_frame))
        plt.clf()

        return resistance, support, start_dt

    def drop_from_trackbook(self, entry):
        self.trackbook = [x for x in self.trackbook if x['market'] != entry['market'] or x['time_range'] != entry['time_range']]

    def track_pullback(self, a_price, b_price, c_price, candles, market, time_range):
        latest_closed_candle = candles[-1]

        if latest_closed_candle['close'] > c_price:
            # Calculate bet
            price_range = b_price - a_price
            stop_price = c_price - (price_range * 0.1)
            profit_price = c_price + price_range

            buy_price = float(client.get_klines(symbol=market, interval='1m', limit=1)[0][4])
            fee = self.portfolio['trade_fee']

            # How many coins would you have if you bought now?
            coins = (self.portfolio['balance'] / buy_price) * (1 - fee)

            # How much money would be gained if sold at profit price?
            reward = ((profit_price - buy_price) * coins) * (1 - fee)

            # How much money lost if sold at stop price?
            risk = ((buy_price - stop_price) * coins) * (1 + fee)

            if reward >= 0.1 and reward / risk >= 1.8:
                # Make the bet!
                self.trade = {'market': market, 'time_range': time_range, 'buy_price': buy_price, 'stop_price': stop_price, 'profit_price': profit_price}
                self.trackbook = []
                return

    def track_consolidation(self, entry, candles):
        trade = False
        last_closed_candle = candles[-1]
        candles_in_range = [x for x in candles if x['timestamp'] >= entry['start']]

        print('({})'.format(entry['time_range']), 'tracking', entry['support'], entry['resistance'], last_closed_candle['close'], last_closed_candle['timestamp'])

        # TODO: Add VWAP and MA support
        price_range = entry['resistance'] - entry['support']

        # Candle closed below support
        if last_closed_candle['close'] < entry['support']:
            percent_drop = (entry['support'] - last_closed_candle['close']) / price_range
            if percent_drop > 0.04:
                print('dropping: closed below support')
                self.drop_from_trackbook(entry)
                return
        
        # Check for volume increase
        avg_volume = sum([x['volume'] for x in candles]) / len(candles)
        lead_up_candles = candles_in_range[-3:]
        avg_lead_up_volume = sum([x['volume'] for x in lead_up_candles]) / len(lead_up_candles)
        
        if last_closed_candle['close'] > entry['resistance']:
            percent_increase = (last_closed_candle['close'] - entry['resistance']) / price_range
            if percent_increase >= 0.05:
                if last_closed_candle['open'] > entry['resistance']:
                    prev_candle = candles_in_range[-2]
                    if last_closed_candle['close'] > prev_candle['close']:
                        trade = True
                    else:
                        print('dropping: no momentum')
                        self.drop_from_trackbook(entry)
                        return
                else:
                    # Get percentage of candle above resistance
                    candle_range = last_closed_candle['close'] - last_closed_candle['open']
                    above_res = last_closed_candle['close'] - entry['resistance']

                    a = avg_lead_up_volume > avg_volume
                    b = (above_res / candle_range) >= 0.2 or (candle_range / price_range) >= 0.6

                    if a or b:
                        trade = True

        if trade:
            # Calculate bet
            price_range = entry['resistance'] - entry['support']
            stop_price = entry['resistance'] - (price_range * 0.1)
            profit_price = entry['resistance'] + price_range

            # buy_price = float(client.get_avg_price(symbol=entry['market'])['price'])
            buy_price = float(client.get_klines(symbol=entry['market'], interval='1m', limit=1)[0][4])
            fee = self.portfolio['trade_fee']

            # How many coins would you have if you bought now?
            coins = (self.portfolio['balance'] / buy_price) * (1 - fee)

            # How much money would be gained if sold at profit price?
            reward = ((profit_price - buy_price) * coins) * (1 - fee)

            # How much money lost if sold at stop price?
            risk = ((buy_price - stop_price) * coins) * (1 + fee)

            if reward >= 0.1 and reward / risk >= 1.8:
                # Make the bet!
                self.trade = {'market': entry['market'], 'time_range': entry['time_range'], 'buy_price': buy_price, 'stop_price': stop_price, 'profit_price': profit_price}
                self.trackbook = []
                return
            else:
                print('dropping: reward too low')
                self.drop_from_trackbook(entry)
                return
