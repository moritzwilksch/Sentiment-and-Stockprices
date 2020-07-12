# %%
import multiprocessing as mp
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option('display.max_colwidth', -1)
# %%


def plot():
    plt.savefig('fig.png')
    plt.close()


# %%
df = pd.read_pickle('data/preprocessed_all2019.pickle').reset_index(drop=True)

clean = True
if clean:
    # remove links
    df['tweet'] = df.tweet.str.replace('http\S+|www.\S+', '')
    df['tweet'] = df.tweet.str.replace('@\w+', '')

# %%
analyzer = SentimentIntensityAnalyzer()


# %%
def get_vader_compound(tweet: str):
    return analyzer.polarity_scores(tweet).get('compound', pd.NA)


with mp.Pool(mp.cpu_count()) as p:
    result = p.map(get_vader_compound, df.tweet)

df['sentiment'] = pd.Series(result)

# %%
social_interaction_cols = ['replies_count', 'nreplies',
                           'likes_count', 'nlikes', 'retweets_count', 'nretweets']
df[social_interaction_cols] = df[social_interaction_cols].fillna(0)
df['likes'] = df['likes_count'] + df['nlikes']
df['replies'] = df['replies_count'] + df['nreplies']
df['retweets'] = df['retweets_count'] + df['nretweets']
df = df.drop(social_interaction_cols, axis=1)

# %%
out = pd.DataFrame({
    'datetime': df.date.astype('datetime64'),
    'tweet': df.tweet,
    'sentiment': df.sentiment,
    'likes': df.likes,
    'retweets': df.retweets,
    'replies': df.replies
})


if not clean:
    out.to_pickle('data/vader_out.pickle')
elif clean:
    out.to_pickle('data/vader_out_clean.pickle')
