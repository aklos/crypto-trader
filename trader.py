import os
import time
import datetime as dt
import numpy as np
import math
import pandas as pd
# import finplot as fplt
from findiff import FinDiff
from bnc import client
# from playsound import playsound

min_balance = 10

ignore_list = ['XRPUPUSDT', 'BTCUSDT']

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
        tickers = [x for x in tickers if float(x['priceChangePercent']) >= min_change_percent]
        # High volume
        tickers = [x for x in tickers if float(x['volume']) >= min_volume]
        # High amount of trades
        tickers = [x for x in tickers if float(x['count']) >= min_trades]
        # Good ATR
        tickers = [x for x in tickers if self.calc_atr(x) > min_atr]
        return [x['symbol'] for x in tickers]

    def scan_market_loop(self):
        last_trade = None
        try:
            last_trade = self.tradebook.tail(1).iloc[0]
        except:
            pass

        if last_trade is None or len(client.get_open_orders(symbol=last_trade['market'])) == 0:
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
        else:
            self.trading = True
            self.trade = {
                'market_symbol': last_trade['market'],
                'profit_price': last_trade['profit'],
                'stop_price': last_trade['stop'],
                'interval': '5m',
                'crossed_scale': False
            }
        
        if self.trade is not None:
            print('=> Making a trade in {} at {}'.format(self.trade['market_symbol'], self.trade['interval']))

            if last_trade is None or len(client.get_open_orders(symbol=last_trade['market'])) == 0:
                self.buy_coins()

            while (True):
                sleep_time = self.get_time_to_next_time_period()
                print('=> Sleeping for {} seconds'.format(sleep_time.seconds))
                time.sleep(sleep_time.seconds)
                self.manage_trade()
                if not self.trading:
                    break

        if self.portfolio['balance'] <= min_balance:
            quit()

        self.scan_market_loop()

    def manage_trade(self):
        [current_candle, candles] = self.get_candles(self.trade['market_symbol'], '5m', 300)

        # Calculate RSI
        df = pd.DataFrame(candles)
        df = self.calculate_rsi(df)
        rsi = df.rsi.to_list()

        # Calculate scale exit (75%)
        price_range = self.trade['profit_price'] - self.trade['stop_price']
        percentage = (candles[-1]['close'] - self.trade['stop_price']) / price_range
        
        if percentage >= 0.75 and self.trade['crossed_scale'] is False:
            self.trade['crossed_scale'] = True
        elif percentage <= 0.75 and self.trade['crossed_scale'] is True:
            self.sell_coins()
            return

        if rsi[-1] > 69:
            self.sell_coins()
            return

        if len(client.get_open_orders(symbol=self.trade['market_symbol'])) == 0:
            new_portfolio = self.get_portfolio()

            self.portfolio = new_portfolio
            self.trade = {}
            self.trading = False

    def update_tradebook(self, data):
        if self.tradebook is not None:
            self.tradebook = self.tradebook.append(data, ignore_index=True)
        else:
            self.tradebook = pd.DataFrame([data])
        self.tradebook.to_csv('./tradebook.csv', index=False)

    def create_oco_order(self, price, stop, stop_limit, precision, step, sub):
        try:
            quantity = float(client.get_asset_balance(asset=self.trade['market_symbol'].replace('USDT', ''))['free'])
            quantity = quantity - sub
            quantity = round(quantity, precision)
            client.order_oco_sell(symbol=self.trade['market_symbol'], quantity=quantity, price=price, stopPrice=stop, stopLimitPrice=stop_limit, stopLimitTimeInForce=client.TIME_IN_FORCE_GTC)
        except:
            self.create_oco_order(price, stop, stop_limit, precision, step, sub + step)

    def buy_coins(self):
        balance = float(client.get_asset_balance(asset='USDT')['free'])
        info = client.get_exchange_info()
        filters = [x['filters'] for x in info['symbols'] if x['symbol'] == self.trade['market_symbol']][0]
        step_size = next((x['stepSize'] for x in filters if x['filterType'] == 'LOT_SIZE'))
        tick_size = next((x['tickSize'] for x in filters if x['filterType'] == 'PRICE_FILTER'))
        precision_limit = int(round(-math.log(float(tick_size), 10), 0))
        precision = int(round(-math.log(float(step_size), 10), 0))
        trades = client.get_recent_trades(symbol=self.trade['market_symbol'])
        coins = (balance / float(trades[0]['price'])) * 0.98
        coins = round(coins, precision)
        client.order_market_buy(symbol=self.trade['market_symbol'], quantity=coins)

        price = round(self.trade['profit_price'], precision_limit)
        stop = round(self.trade['stop_price'], precision_limit)
        stop_limit = round(self.trade['stop_price_limit'], precision_limit)
        self.create_oco_order(price, stop, stop_limit, precision, float(step_size), 0)

        self.update_tradebook({ 
            'timestamp': dt.datetime.now(), 
            'action': 'BUY', 
            'market': self.trade['market_symbol'], 
            'price': self.trade['buy_price'], 
            'quantity': coins, 
            'profit': self.trade['profit_price'],
            'stop': self.trade['stop_price'],
            'total_balance': balance,
            'precision': precision,
            'crossed_scale': False
            })

    def sell_coins(self):
        orders = client.get_open_orders(symbol=self.trade['market_symbol'])

        for order in orders:
            client.cancel_order(symbol=self.trade['market_symbol'], orderId=order['orderId'])

        coins = float(client.get_asset_balance(asset=self.trade['market_symbol'].replace('USDT', ''))['free'])
        coins = coins * 0.995
        coins = round(coins, self.trade['precision'])
        client.order_market_sell(symbol=self.trade['market_symbol'], quantity=coins)

        self.portfolio = self.get_portfolio()
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
        current_candle = candles[-1]
        candles = candles[:-1]
        return [current_candle, candles]

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

    # def draw_candlestick_chart(self, symbol, candles, levels):
    #     os.system('mkdir -p ./graphs')
    #     ax, ax2 = fplt.create_plot(symbol, rows=2)
    #     df = pd.DataFrame(candles)
    #     df = df.astype({'time':'datetime64[ns]'})
    #     fplt.candlestick_ochl(df[['time', 'open', 'close', 'high', 'low']])

    #     # Plot MA
    #     # fplt.plot(df['time'], df['close'].rolling(25).mean(), ax=ax, legend='ma-25')

    #     # Plot RSI
    #     df = self.calculate_rsi(df)
    #     df.rsi.plot(ax=ax2, legend='RSI')
    #     fplt.set_y_range(0, 100, ax=ax2)
    #     fplt.add_band(30, 70, ax=ax2)

    #     # Plot accu/dist
    #     # self.plot_accumulation_distribution(df, ax3)

    #     min_date = min([x['time'] for x in candles])
    #     max_date = max([x['time'] for x in candles])
    #     fplt.add_line([min_date, 50], [max_date, 50], ax=ax2, color='#0000bb')
    #     for level in levels:
    #         fplt.add_line([min_date, level['value']], [max_date, level['value']], ax=ax)
    #     fplt.show()

    def scan_markets(self):
        for market_symbol in self.markets:
            print('=> Searching', market_symbol)

            levels = []

            [current_candle, candles] = self.get_candles(market_symbol, '5m', 300)

            # Find levels in current graph
            price_range = max([x['high'] for x in candles]) - min([x['low'] for x in candles])
            extremas = self.get_extremas([x['close'] for x in candles], candles)
            highs = [x['high'] for x in extremas]
            lows = [x['low'] for x in extremas]

            def compare_prices(x, y, range):
                return abs(x - y) / range <= 0.01

            for high in highs:
                if sum([1 for x in highs if compare_prices(x, high, price_range)]) > 1:
                    if next((x for x in levels if abs(high - x['value']) / price_range <= 0.1), None) is None:
                        levels.append({ 'swing': 'high', 'value': high })

            for low in lows:
                if sum([1 for x in lows if compare_prices(x, low, price_range)]) > 1:
                    if next((x for x in levels if abs(low - x['value']) / price_range <= 0.1), None) is None:
                        levels.append({ 'swing': 'low', 'value': low })

            min_price = min([x for x in lows])
            max_price = max([x for x in highs])

            if next((x for x in levels if abs(min_price - x['value']) / price_range <= 0.1), None) is None:
                levels.append({ 'swing': 'low', 'value': min_price })

            if next((x for x in levels if abs(max_price - x['value']) / price_range <= 0.1), None) is None:
                levels.append({ 'swing': 'high', 'value': max_price })

            # Calculate RSI & AD
            df = pd.DataFrame(candles)
            df = self.calculate_rsi(df)
            rsi = df.rsi.to_list()
            ad = self.calculate_accumulation_distribution(df)
            ad = ad.to_list()

            if rsi[-2] <= 29 and rsi[-1] > 30 and ad[-1] > 0:
                # Calculate win at next level above 
                levels.sort(key=lambda x: x['value'])
                profit_price = next((x for x in levels if x['value'] > current_candle['close'] and abs(current_candle['close'] - x['value']) / price_range > 0.05), None)
                if profit_price is None:
                    continue
                profit_price = profit_price['value'] - (price_range * 0.01)

                # Calculate loss at 1% of range below prev minimum
                stop_price = lows[-1] - (price_range * 0.01)

                if self.is_acceptable_risk_reward(market_symbol, current_candle['close'], price_range, stop_price, profit_price):
                    self.trading = True
                    self.trade = {
                        'market_symbol': market_symbol,
                        'interval': '5m',
                        'buy_price': current_candle['close'],
                        'stop_price': stop_price,
                        'stop_price_limit': stop_price - (price_range * 0.01),
                        'profit_price': profit_price,
                        'crossed_scale': False
                    }
                    return

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

        return reward > (2 * (balance * fee)) and reward / risk >= 1.8
