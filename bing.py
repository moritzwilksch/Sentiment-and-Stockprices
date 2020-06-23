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
df = pd.read_pickle('data/EDA_all2019.pickle')

#%%
# convert to lower case
df['tweet'] = df.tweet.str.lower()

#%%
# remove links
df['tweet'] = df.tweet.str.replace('http\S+|www.\S+', '')

#%%
