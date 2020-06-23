#%%
import twint
import pandas as pd
#import nest_asyncio

#nest_asyncio.apply()


def get_data(since, until):
    c = twint.Config
    c.Search = '$TSLA'
    c.Since = since
    c.Until = until
    c.Pandas = True
    c.Count = True
    c.Hide_output = True
    twint.run.Search(c)
    df = twint.storage.panda.Tweets_df
    return df

#%%
df = get_data('2019-01-01 0:0:0', '2019-02-01 0:0:0')
df.to_csv("data/jan2019.csv")
#%%