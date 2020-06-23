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
df = pd.read_pickle('data/preprocessed_all2019.pickle')


#%%
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()

sentis = df.tweet.apply(lambda tw: analyzer.polarity_scores(tw).get('compound', pd.NA))
df['sentiment'] = sentis

#%%
daily = df.groupby(df.datetime.dt.date)['sentiment'].agg(['mean', 'std'])

#%%
plt.figure(figsize=(20, 5))
sns.lineplot(data=df, x=df.datetime.dt.week, y='sentiment')
plot()