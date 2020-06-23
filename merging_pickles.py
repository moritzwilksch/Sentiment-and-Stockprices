import pandas as pd

#%%
df1 = pd.read_pickle('data/jan_til_jun2019.pickle')
df1['datetime'] = df1.date.astype('datetime64')
df2 = pd.read_pickle('data/jul_to_dec2019.pickle')
df2['datetime'] = df2.date.astype('datetime64')
df = pd.concat([df1,df2])
df = df.sort_values('datetime').reset_index(drop=True)
df.drop('date', axis=1)

#%%
df.to_pickle("data/all2019.pickle")