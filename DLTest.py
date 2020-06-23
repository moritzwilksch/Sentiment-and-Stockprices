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
# Convert to lower
df['tweet'] = df.tweet.str.lower()
# Remove Hyperlinks
df['tweet'] = df.tweet.str.replace('http\S+|www.\S+', '')

#%%
import yfinance as yf
close = yf.Ticker('TSLA').history(start="2019-01-01", end="2019-12-31")['Close']
daily_return = close.pct_change().shift(-1)
daily_return = daily_return.asfreq('D', method='bfill').fillna(0)



#%%
df['date'] = df['datetime'].dt.date
df['date'] = df['date'].astype('datetime64')
df = pd.merge(df, daily_return, left_on='date', right_index=True)
df = df.rename({'Close': 'next_return'}, axis=1)


#%%
import keras
from sklearn.model_selection import train_test_split

tok = keras.preprocessing.text.Tokenizer()
tok.fit_on_texts(df.tweet)
sequences = tok.texts_to_sequences(df.tweet)
X = keras.preprocessing.sequence.pad_sequences(sequences, 280, padding='post')
y = df['next_return'] > 0

X_train, X_test, y_train, y_test = train_test_split(X, y)

#%%
vocab_size = len(tok.word_index)

model = keras.Sequential([
    keras.layers.Embedding(input_dim=vocab_size+1, output_dim=20, input_length=280),
    # keras.layers.Flatten(),
    keras.layers.AveragePooling1D(280),
    keras.layers.Flatten(),
    keras.layers.Dense(units=100, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(units=20, activation='relu'),
    keras.layers.Dense(units=1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()
#%%
model.fit(X_train, y_train, epochs=10, batch_size=4096, validation_data=(X_test, y_test))
pd.DataFrame({'test': model.history.history['val_accuracy'], 'train': model.history.history['accuracy']}).plot()
plot()

#%%
from xgboost import XGBClassifier

type(y_test)

clf = XGBClassifier()
clf.fit(X_train, y_train)
