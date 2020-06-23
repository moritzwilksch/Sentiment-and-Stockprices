#%%
import pandas as pd

#%%
df = pd.read_csv("~/Desktop/SMBAdata/jan2019.csv")
other_files = ['feb2019.csv', 'mar2019.csv', 'apr2019.csv', 'may2019.csv', 'jun2019.csv']
for file in other_files:
    next_df = pd.read_csv(f"~/Desktop/SMBAdata/{file}")
    df = pd.concat([df, next_df])
df = df.sort_values('date').reset_index(drop=True)

#%%
df.to_pickle("data/jan_til_jun2019.pickle")