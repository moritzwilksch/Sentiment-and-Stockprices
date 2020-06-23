# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# %%
df: pd.DataFrame = pd.read_pickle('data/jan_til_jun2019.pickle')
df.head()


# %%
def plot():
    plt.savefig('fig.png')
    plt.close()


cols_to_drop = ['place', 'near', 'geo', 'source', 'user_rt_id', 'user_rt', 'retweet_id', 'retweet_date', 'translate',
                'trans_src', 'trans_dest']

df = df.drop(cols_to_drop, axis=1)

# %%
# df['tweet'].str.match('(\$TSLA | \$tsla)')

df['num_of_cashtags'] = df.cashtags.apply(lambda s: len(s[1:-1].split(',')))

# %%
df.info()

# %%
sns.countplot(df.num_of_cashtags)
plot()

#%%
df.groupby()