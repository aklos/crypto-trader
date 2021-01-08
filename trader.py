import os
import time
import datetime as dt
import numpy as np
import math
import pandas as pd
import json
import itertools
import finplot as fplt
from findiff import FinDiff
from itertools import chain
from scipy.signal import argrelextrema
from bnc import client
from playsound import playsound

min_balance = 90

ignore_list = ['XRPUPUSDT']

min_change_percent = 2
min_volume = 500000
min_trades = 50000
min_atr = 0.5

class Trader():
    def __init__(self, *args, **kwargs):
        super(Trader, self).__init__(*args, **kwargs)
        self.trading = False
        self.trade = {}
        try:
            self.tradebook = pd.read_csv('./tradebook.csv', index_col=False)
        except:
            self.tradebook = None
        self.portfolio = self.get_portfolio()

    def get_portfolio(self):
        balance = None
        if self.tradebook is not None:
            balance = self.tradebook.tail(1).iloc[0]['total_balance']
        else:
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
        tickers = [x for x in tickers if x == 'BTCUSDT'] 
        # if not next((x for x in tickers if x['symbol'] == 'BTCUSDT'), None):
        #     tickers.append({ 'symbol': 'BTCUSDT'})
        if len(tickers) == 0:
            tickers.append({ 'symbol': 'BTCUSDT' })
        return [x['symbol'] for x in tickers]

    def scan_market_loop(self):
        # sleep_time = self.get_time_to_next_time_period()
        # print('=> Starting in {} seconds'.format(sleep_time.seconds))
        # time.sleep(sleep_time.seconds)

        while (True):
            self.markets = self.get_active_markets()
            print('=> Scanning {} active markets'.format(len(self.markets)))
            self.scan_markets()
            if self.trading:
                break
            if True:
                return
            sleep_time = self.get_time_to_next_time_period()
            print('=> Sleeping for {} seconds'.format(sleep_time.seconds))
            time.sleep(sleep_time.seconds)
        
        if self.trade is not None:
            print('=> Making a trade in {} at {}'.format(self.trade['market_symbol'], self.trade['interval']))
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
        pass

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
            'reversal': None,
            'total_balance': balance 
            })

    def sell_coins(self, price, reversal=False):
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
            'reversal': reversal,
            'total_balance': balance
            })
        self.trade = {}
        self.trading = False

    def get_candles(self, symbol, interval, limit):
        candles = client.get_klines(symbol=symbol, interval=interval, limit=limit + 1)
        candles = [{ 
            'open_dt': dt.datetime.fromtimestamp(x[0] / 1000), 
            'close_dt': dt.datetime.fromtimestamp(x[6] / 1000), 
            'time': dt.datetime.fromtimestamp(x[6] / 1000), 
            'open': float(x[1]),
            'high': float(x[2]),
            'low': float(x[3]),
            'close': float(x[4]), 
            'volume': float(x[5]) 
            } for x in candles]
        candles = candles[:-1]
        return candles

    def plot_accumulation_distribution(self, df, ax):
        ad = (2*df.close-df.high-df.low) * df.volume / (df.high - df.low)
        ad.cumsum().ffill().plot(ax=ax, legend='Accum/Dist', color='#f00000')

    def calculate_accumulation_distribution(self, df):
        ad = (2*df.close-df.high-df.low) * df.volume / (df.high - df.low)
        ad = ad.cumsum().ffill()
        return ad

    def calculate_rsi(self, df):
        diff = df.close.diff().values
        gains = diff
        losses = -diff
        with np.errstate(invalid='ignore'):
            gains[(gains<0)|np.isnan(gains)] = 0.0
            losses[(losses<=0)|np.isnan(losses)] = 1e-10 # we don't want divide by zero/NaN
        n = 14
        m = (n-1) / n
        ni = 1 / n
        g = gains[n] = np.nanmean(gains[:n])
        l = losses[n] = np.nanmean(losses[:n])
        gains[:n] = losses[:n] = np.nan
        for i,v in enumerate(gains[n:],n):
            g = gains[i] = ni*v + m*g
        for i,v in enumerate(losses[n:],n):
            l = losses[i] = ni*v + m*l
        rs = gains / losses
        df['rsi'] = 100 - (100/(1+rs))
        return df

    def draw_candlestick_chart(self, symbol, candles, levels):
        os.system('mkdir -p ./graphs')
        ax, ax2, ax3 = fplt.create_plot(symbol, rows=3)
        df = pd.DataFrame(candles)
        df = df.astype({'time':'datetime64[ns]'})
        fplt.candlestick_ochl(df[['time', 'open', 'close', 'high', 'low']])

        # Plot MA
        fplt.plot(df['time'], df['close'].rolling(25).mean(), ax=ax, legend='ma-25')

        # Plot RSI
        df = self.calculate_rsi(df)
        df.rsi.plot(ax=ax, legend='RSI')
        fplt.set_y_range(0, 100, ax=ax2)
        fplt.add_band(30, 70, ax=ax2)

        # Plot accu/dist
        self.plot_accumulation_distribution(df, ax3)

        min_date = min([x['time'] for x in candles])
        max_date = max([x['time'] for x in candles])
        for level in levels:
            fplt.add_line([min_date, level['value']], [max_date, level['value']], ax=ax)
        def save():
            fplt.screenshot(open('./graphs/{}.png'.format(symbol), 'wb'))
            fplt.app.exit()
        fplt.timer_callback(save, 1, single_shot=True) # wait some until we're rendered
        fplt.show()

    def scan_markets(self):
        # Find levels in 1 day, 4 hour, and 1 hour intervals
        # Find levels in 1m interval
        # Calculate RSI line
        # Calculate A/D line
        for market_symbol in self.markets:
            print('=> Searching', market_symbol)

            levels = []
            
            for interval in ['1d', '4h']:
                candles = self.get_candles(market_symbol, interval, 300)
                price_range = max([x['high'] for x in candles]) - min([x['low'] for x in candles])
                extremas = self.get_extremas([x['close'] for x in candles], candles)

                # Take latest levels, because newer swings will be more accurate?

                highs = [x['high'] for x in extremas]
                highs.reverse()
                lows = [x['low'] for x in extremas]
                lows.reverse()

                for high in highs:
                    if next((x for x in levels if abs(high - x['value']) / price_range <= 0.03), None) is None:
                        levels.append({ 'swing': 'high', 'value': high })
                
                for low in lows:
                    if next((x for x in levels if abs(low - x['value']) / price_range <= 0.03), None) is None:
                        levels.append({ 'swing': 'low', 'value': low })

            candles = self.get_candles(market_symbol, '1m', 300)

            # Find major levels in current graph (> 2 touches)
            price_range = max([x['high'] for x in candles]) - min([x['low'] for x in candles])
            extremas = self.get_extremas([x['close'] for x in candles], candles)
            highs = [x['high'] for x in extremas]
            lows = [x['low'] for x in extremas]

            def compare_prices(x, y, range):
                return abs(x - y) / range <= 0.01

            for high in highs:
                if sum([1 for x in highs if compare_prices(x, high, price_range)]) > 2:
                    if next((x for x in levels if abs(high - x['value']) / price_range <= 0.01), None) is None:
                        levels.append({ 'swing': 'high', 'value': high })

            for low in lows:
                if sum([1 for x in lows if compare_prices(x, low, price_range)]) > 2:
                    if next((x for x in levels if abs(low - x['value']) / price_range <= 0.01), None) is None:
                        levels.append({ 'swing': 'low', 'value': low })

            # Calculate RSI and AD
            df = pd.DataFrame(candles)
            df = self.calculate_rsi(df)
            rsi = df.rsi.to_list()
            ad = self.calculate_accumulation_distribution(df).to_list()

            print(rsi[-1], ad[-1])
            # print(abspath('./alert.wav'))
            playsound('./alert.wav')

            if rsi[-2] < 30 and rsi[-1] > 30 and ad[-1] > 0:
                self.draw_candlestick_chart(market_symbol, candles, levels)

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

    def get_extremas(self, prices, candles):
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
        minima_ids = get_extrema(True)
        maxima_ids = get_extrema(False)
        extremas = [{ **x, 'category': 'minima' if i in minima_ids else 'maxima' } for i, x in enumerate(candles) if i in minima_ids or i in maxima_ids]
        extremas = sorted(extremas, key=lambda x: x['open_dt'])
        return extremas

    def get_hh_and_hl_length(self, extremas):
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

        print('risk/reward', risk, reward, 'fee', 2 * (balance * fee), 'ratio', reward / risk)

        return reward > (2 * (balance * fee)) and reward / risk >= 1.8
