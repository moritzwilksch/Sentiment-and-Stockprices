#%%
import pandas as pd
pd.set_option('display.max_colwidth', None)
import numpy as np

#%%
sw = pd.read_pickle('data/sentiwords_out.pickle')
sw['tweet'] = sw['tweet'].astype('string')
sw = sw.rename({'sentiment': 'swn_sentiment'}, axis=1)

vader = pd.read_pickle('data/vader_out.pickle')
vader['tweet'] = vader['tweet'].astype('string')
vader = vader.rename({'sentiment': 'vader_sentiment'}, axis=1)

afinn = pd.read_pickle('data/afinn_alldata2019')
afinn['tweet'] = afinn['tweet'].astype('string')
afinn = afinn.rename({'scores': 'afinn_sentiment'}, axis=1)


#%%
df = pd.DataFrame({
    'datetime': sw.datetime.reset_index(drop=True),
    'tweet': sw.tweet.reset_index(drop=True),
    'swn_sentiment': sw.swn_sentiment.fillna(0).astype('float64').reset_index(drop=True),
    'vader_sentiment': vader.vader_sentiment.reset_index(drop=True),
    'afinn_sentiment': afinn.afinn_sentiment.reset_index(drop=True)
})

#%%
df.corr()

#%%
# df.to_pickle('data/sentis_merged.pickle')