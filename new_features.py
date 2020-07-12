# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

print("====> Using VADER Data")
# df = pd.read_pickle('data/vader_out.pickle')
df = pd.read_pickle('data/vader_out_clean.pickle')
df['datetime'] = df.datetime.astype('datetime64')

def plot(name: str = 'fig.png'):
    plt.savefig(name)
    plt.close()


print(pd.date_range(start='2019-01-01', end='2019-12-31').difference(df.datetime.dt.date))

import yfinance as yf

raw_yahoo = yf.Ticker('TSLA').history(start='2019-01-01', end='2019-12-31', interval='1D')
raw_prices = raw_yahoo['Close']