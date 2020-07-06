# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import multiprocessing as mp

df = pd.read_pickle('data/vader_out.pickle')
# df = pd.read_pickle('data/afinn_alldata_2019')

df['datetime'] = df.datetime.astype('datetime64')
# df['sentiment'] = df.scores
# df = df.drop(['date', 'scores'], axis=1)


def plot(name: str = 'fig.png'):
    plt.savefig(name)
    plt.close()


# %%
pd.date_range(start='2019-01-01', end='2019-12-31').difference(df.datetime.dt.date)

# %%
import yfinance as yf

raw_prices = yf.Ticker('TSLA').history(start='2019-01-01', end='2019-12-31', interval='1D')['Close']
daily_return = raw_prices.pct_change()

# %%
# Backfill
daily_return = daily_return.asfreq('D', method='bfill').fillna(0)

# %%
daily_df = df.groupby(df.datetime.dt.date).agg(['mean', 'std']).fillna(0)
daily_df = daily_df.reset_index()

#%%
sns.scatterplot(daily_df.loc[1:363, 'sentiment']['mean'], daily_return.reset_index(drop=True))
plt.xlabel('Vader sentiment per day')
plt.ylabel('Same-day stock return')
plot()

#%%
daily_df['datetime'] = daily_df['datetime'].astype('datetime64')

#%%
sns.lineplot(x=daily_return.iloc[0:30].index, y=daily_df.loc[1:30, 'sentiment']['mean'], label='daily Vader sentiment')
sns.lineplot(x=daily_return.iloc[0:30].index, y=daily_return.iloc[0:30], label='Daily Stock return')
plt.legend()
plot('fig2.png')