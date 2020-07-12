# %%
import matplotlib.pyplot as plt
import pandas as pd
import multiprocessing as mp
import numpy as np
pd.set_option('display.max_colwidth', -1)


# %%
def plot():
    plt.savefig('fig.png')
    plt.close()


# %%
df: pd.DataFrame = pd.read_pickle('data/preprocessed_all2019.pickle')
df = df.reset_index(drop=True)
# convert to lower case
df['tweet'] = df.tweet.str.lower()
# remove links
df['tweet'] = df.tweet.str.replace('http\S+|www.\S+', '')


# %%
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn
from nltk.corpus import sentiwordnet as swn
from nltk import sent_tokenize, word_tokenize, pos_tag
from nltk.stem import PorterStemmer
import string
import time

# init stemmer and stopwords
ps = PorterStemmer()
en_sw = {'a',
         'about',
         # 'above',
         'after',
         'again',
         'against',
         'ain',
         'all',
         'am',
         'an',
         'and',
         'any',
         'are',
         'aren',
         # "aren't",
         'as',
         'at',
         'be',
         'because',
         'been',
         'before',
         'being',
         # 'below',
         'between',
         'both',
         'but',
         'by',
         'can',
         'couldn',
         # "couldn't",
         'd',
         'did',
         'didn',
         # "didn't",
         'do',
         'does',
         'doesn',
         # "doesn't",
         'doing',
         'don',
         # "don't",
         # 'down',
         'during',
         'each',
         'few',
         'for',
         'from',
         'further',
         'had',
         'hadn',
         # "hadn't",
         'has',
         'hasn',
         # "hasn't",
         'have',
         'haven',
         # "haven't",
         'having',
         'he',
         'her',
         'here',
         'hers',
         'herself',
         'him',
         'himself',
         'his',
         'how',
         'i',
         'if',
         'in',
         'into',
         'is',
         'isn',
         # "isn't",
         'it',
         "it's",
         'its',
         'itself',
         'just',
         'll',
         'm',
         'ma',
         'me',
         'mightn',
         # "mightn't",
         'more',
         'most',
         'mustn',
         # "mustn't",
         'my',
         'myself',
         'needn',
         # "needn't",
         # 'no',
         # 'nor',
         # 'not',
         'now',
         'o',
         'of',
         'off',
         'on',
         'once',
         'only',
         'or',
         'other',
         'our',
         'ours',
         'ourselves',
         'out',
         # 'over',
         'own',
         're',
         's',
         'same',
         'shan',
         # "shan't",
         'she',
         "she's",
         'should',
         "should've",
         'shouldn',
         # "shouldn't",
         'so',
         'some',
         'such',
         't',
         'than',
         'that',
         "that'll",
         'the',
         'their',
         'theirs',
         'them',
         'themselves',
         'then',
         'there',
         'these',
         'they',
         'this',
         'those',
         'through',
         'to',
         'too',
         # 'under',
         'until',
         # 'up',
         've',
         'very',
         'was',
         'wasn',
         # "wasn't",
         'we',
         'were',
         'weren',
         # "weren't",
         'what',
         'when',
         'where',
         'which',
         'while',
         'who',
         'whom',
         'why',
         'will',
         'with',
         'won',
         # "won't",
         'wouldn',
         # "wouldn't",
         'y',
         'you',
         "you'd",
         "you'll",
         "you're",
         "you've",
         'your',
         'yours',
         'yourself',
         'yourselves'}
en_sw = en_sw.union(set(string.punctuation))

# %%
def stem_and_drop_sw(tweet):
    return " ".join(ps.stem(word) for word in word_tokenize(tweet) if word not in en_sw)

#%%
starttime = time.time()
tmp = df.head(50_000)['tweet'].apply(stem_and_drop_sw)
endtime = time.time()
print(endtime - starttime)

#%%
starttime = time.time()
with mp.Pool(mp.cpu_count()) as p:
    out = p.map(stem_and_drop_sw, df.head(50_000)['tweet'])
endtime = time.time()
print(endtime - starttime)
