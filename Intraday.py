# Intraday.py
# Pull intraday data along with buy/sell signals from any ticker available in yfinance

import yfinance as yf
import pandas as pd
import numpy as np
import ta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
from datetime import datetime, timedelta

# Today's date + tomorrow's date
today = datetime.today().strftime('%Y-%m-%d')
tomorrow = (datetime.today() + timedelta(days=1)).strftime('%Y-%m-%d')

ticker = input("Enter stock ticker to view intraday data: \n")

# Download intraday data from yfinance
df = yf.download(ticker, start=today, end=tomorrow, interval='1m')

# Force UTC to EST conversion
if df.index.tz is None:
    df.index = df.index.tz_localize('UTC')
df.index = df.index.tz_convert('America/New_York')

# Intraday trading hours only
df = df.between_time('09:30', '16:00')
# Bollinger bands calculation
df['ma_20'] = df['Close'].rolling(window=20).mean()
df['vol'] = df['Close'].rolling(window=20).std()
df['upper_bb'] = df['ma_20'] + (2 * df['vol'])
df['lower_bb'] = df['ma_20'] - (2 * df['vol'])

# RSI calculation with ta
df['rsi'] = ta.momentum.rsi(df['Close'], window=6)

# Buy/Sell Signals based on both RSI And Bollinger Bands
conditions = [(df['rsi'] < 30) & (df['Close'] < df['lower_bb']),
              (df['rsi'] > 70) & (df['Close'] > df['upper_bb'])]
choices = ['Buy', 'Sell']

df['signal'] = np.select(conditions, choices)
df['signal'] = df['signal'].shift()
df.dropna(inplace=True)

# Track buy/sell points
buydates, selldates = [], []
buyprices, sellprices = [], []
position = False

for index, row in df.iterrows():
    if not position and row['signal'] == 'Buy':
        buydates.append(index)
        buyprices.append(row['Open'])
        position = True
    
    if position:
        if row['signal'] == 'Sell' or row['Close'] < 0.6 * buyprices[-1]:
            selldates.append(index)
            sellprices.append(row['Open'])
            position = False

# Check buy/sell prices, output sample return percentage
if buyprices and sellprices:
    s = (pd.Series([(sell - buy) / buy for sell, buy in zip(sellprices, buyprices)]) + 1).prod() - 1
    print(f"Total Return: {'%.2f' % (s * 100)}%")
else:
    print("No buy or sell signals generated, or missing data.")

# Plot options
plt.figure(figsize=(12, 6))
plt.plot(df['Close'], label= ticker+' Close Price')

if buydates:
    plt.scatter(df.loc[buydates].index, df.loc[buydates]['Close'], marker='^', color='green', label='Buy Signal', alpha=1)
if selldates:
    plt.scatter(df.loc[selldates].index, df.loc[selldates]['Close'], marker='v', color='red', label='Sell Signal', alpha=1)


plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M', tz=df.index.tz))  
plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=1))  # Set major ticks every hour
plt.gca().xaxis.set_minor_locator(mdates.MinuteLocator(interval=15))  # Set minor ticks every 15 minutes

plt.title(ticker+"  Price with Buy/Sell Signals")
plt.xlabel("Time in EST, 9:30-16:00")
plt.legend()
plt.tight_layout()  # Prevent label clipping
plt.show()
