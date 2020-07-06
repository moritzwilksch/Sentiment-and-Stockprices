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

with mp.Pool(mp.cpu_count()) as p:
    result = pd.Series(p.map(stem_and_drop_sw, df.tweet))
df['tweet'] = pd.Series(result)

# %%


lemmatizer = WordNetLemmatizer()


def penn_to_wn(tag):
    """
    Convert between the PennTreebank tags to simple Wordnet tags
    """
    if tag.startswith('J'):
        return wn.ADJ
    elif tag.startswith('N'):
        return wn.NOUN
    elif tag.startswith('R'):
        return wn.ADV
    elif tag.startswith('V'):
        return wn.VERB
    return None


def clean_text(text):
    text = text.replace("<br />", " ")
    # text = text.decode("utf-8")

    return text


def swn_polarity(text):
    """
    Return a sentiment polarity: 0 = negative, 1 = positive
    """
    try:
        sentiment = 0.0
        tokens_count = 0

        text = clean_text(text)

        raw_sentences = sent_tokenize(text)
        for raw_sentence in raw_sentences:
            tagged_sentence = pos_tag(word_tokenize(raw_sentence))

            for word, tag in tagged_sentence:
                wn_tag = penn_to_wn(tag)
                if wn_tag not in (wn.NOUN, wn.ADJ, wn.ADV):
                    continue

                lemma = lemmatizer.lemmatize(word, pos=wn_tag)
                if not lemma:
                    continue

                synsets = wn.synsets(lemma, pos=wn_tag)
                if not synsets:
                    continue

                # Take the first sense, the most common
                synset = synsets[0]
                swn_synset = swn.senti_synset(synset.name())

                sentiment += swn_synset.pos_score() - swn_synset.neg_score()
                tokens_count += 1

        # judgment call ? Default to positive or negative
        if not tokens_count:
            return 0

        """# sum greater than 0 => positive sentiment
        if sentiment >= 0:
            return 1"""
        return sentiment
        # negative sentiment
        # return 0
    except:
        return pd.NA


# %%
print(df.tweet.head(10))
df.tweet.head(10).apply(swn_polarity)

#%%
df.head(1000)['tweet'].apply(swn_polarity)

# %%
import multiprocessing as mp

with mp.Pool(mp.cpu_count()) as p:
    result = p.map(swn_polarity, df['tweet'])

#%%
sentis = pd.Series(result)

# %%
out = pd.DataFrame({'datetime': df.datetime, 'tweet': df.tweet, 'sentiment': sentis})
out.to_pickle('data/sentiwords_out.pickle')