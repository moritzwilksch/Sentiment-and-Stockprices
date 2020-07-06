#%%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option('display.max_colwidth', -1)
#%%
def plot():
    plt.savefig('fig.png')
    plt.close()

#%%
df = pd.read_pickle('data/preprocessed_all2019.pickle').reset_index(drop=True)


#%%
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import multiprocessing as mp
analyzer = SentimentIntensityAnalyzer()


#%%
def get_vader_compound(tweet: str):
    return analyzer.polarity_scores(tweet).get('compound', pd.NA)

with mp.Pool(mp.cpu_count()) as p:
    result = p.map(get_vader_compound, df.tweet)

df['sentiment'] = pd.Series(result)

#%%
out = pd.DataFrame({'datetime': df.date.astype('datetime64'), 'tweet': df.tweet, 'sentiment': df.sentiment})
out.to_pickle('data/vader_out.pickle')

#%%
daily = df.groupby(df.datetime.dt.date)['sentiment'].agg(['mean', 'std'])

#%%
plt.figure(figsize=(20, 5))
sns.lineplot(data=df, x=df.datetime.dt.week, y='sentiment')
plot()
