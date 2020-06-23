# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

pd.set_option('display.max_colwidth', -1)


# %%
def plot():
    plt.savefig('fig.png')
    plt.close()


# %%
df = pd.read_pickle('data/preprocessed_all2019.pickle')

# %%
# convert to lower case
df['tweet'] = df.tweet.str.lower()

# %%
# remove links
df['tweet'] = df.tweet.str.replace('http\S+|www.\S+', '')

#%%
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
ps = PorterStemmer()

def stem(tweet):
    return " ".join(ps.stem(word) for word in word_tokenize(tweet))

df['tweet'] = df.tweet.apply(lambda tw: " ".join(ps.stem(word) for word in word_tokenize(tw)))

# %%
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn
from nltk.corpus import sentiwordnet as swn
from nltk import sent_tokenize, word_tokenize, pos_tag

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
    #text = text.decode("utf-8")

    return text


def swn_polarity(text):
    """
    Return a sentiment polarity: 0 = negative, 1 = positive
    """

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

#%%
print(df.tweet.head(10))
df.tweet.head(10).apply(swn_polarity)

#%%
print(df.tweet.tail(10))
df.tweet.tail(10).apply(swn_polarity)

#%%
df['sentiment'] = df.tweet.apply(swn_polarity)