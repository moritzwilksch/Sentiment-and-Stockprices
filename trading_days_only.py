# %%
from itertools import product
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from rich.console import Console

c = Console()

print("====> Using VADER Data")
vader_df = pd.read_pickle('data/vader_out_clean.pickle')
vader_df['datetime'] = vader_df.datetime.astype('datetime64')


print(pd.date_range(start='2019-01-01',
                    end='2019-12-31').difference(vader_df.datetime.dt.date))


raw_yahoo = yf.Ticker('TSLA').history(
    start='2019-01-01', end='2019-12-31', interval='1D')
raw_prices = raw_yahoo['Close']

# %%
# Backfill
daily_return = raw_prices.pct_change().shift(-1)
# daily_return = daily_return.asfreq('D', method='ffill').fillna(0)

# %%
daily_sentiment: pd.Series = vader_df.groupby(vader_df.datetime.dt.date).agg(
    lambda x: np.average(x, weights=vader_df.loc[x.index, "likes"]))['sentiment']
pct_pos = vader_df.groupby(vader_df.datetime.dt.date)[
    'sentiment'].agg(lambda s: sum(s > 0.05)/len(s))
pct_neg = vader_df.groupby(vader_df.datetime.dt.date)[
    'sentiment'].agg(lambda s: sum(s < -0.05)/len(s))
num_of_tweets = vader_df.groupby(vader_df.datetime.dt.date)['sentiment'].size()

#%%
df = pd.merge(daily_return, daily_sentiment, left_index=True, right_index=True)
df = (df
    .assign(pct_pos=pct_pos[df.index])
    .assign(pct_neg=pct_neg[df.index])
    .assign(num_tweets=num_of_tweets[df.index])
    .assign(volume=raw_yahoo['Volume'])
    .assign(prev0=raw_prices.pct_change().shift(0).fillna(0))
    .assign(prev1=raw_prices.pct_change().shift(1).fillna(0))
    .assign(prev2=raw_prices.pct_change().shift(2).fillna(0))
    .dropna()
)

c.print("[bold underline]Data Frame")
df.info()

# Sanity check
jan3rd = df.iloc[1, :]
assert jan3rd['sentiment'] - 1.11126 < 0.001
assert jan3rd['num_tweets'] - 1402 < 0.001
assert jan3rd['pct_pos'] - 0.4265 < 0.001
# assert df.iloc[12, :]

y_binary = (df.Close > 0).astype('int8')
y = df.Close.copy()


# %%
ss = StandardScaler()
pf = PolynomialFeatures()


def prep(dataf, scale, poly):
    dataf = dataf.copy()

    if poly:
        dataf = pf.fit_transform(dataf)

    if scale:
        dataf = ss.fit_transform(dataf)

    return dataf


# %%
SCORING = 'roc_auc'
cv = KFold(10)

svc2 = GridSearchCV(SVC(probability=True,),
                    param_grid={'kernel': ['rbf', 'poly' 'sigmoid'],
                                'C': [0.01, 0.1, 0.5, 1, 5, 10]
                                },
                    n_jobs=-1,
                    cv=cv,
                    scoring=SCORING
                    )


combos = [c for c in product((True, False), (True, False))]
results = {
    'combos': combos,
    'svc_score': [],
    'lr_score': [],
    'nb_score': [],
    'rf_score': [],
}


for combo in combos:
    X = prep(df.drop('Close', axis=1), *combo)
    c.print(f"[bold underline]Fitting combo {combo}")

    # LR
    lr = LinearRegression()
    results['lr_score'].append(-np.mean(cross_val_score(lr, X,
                                                        y, n_jobs=-1, cv=cv, scoring='neg_mean_absolute_error')))

    # NB
    nb2 = GaussianNB()
    results['nb_score'].append(
        np.mean(cross_val_score(nb2, X, y_binary,
                                cv=cv, n_jobs=-1, scoring=SCORING))
    )
    print(" -> fitted Naive Bayes!")

    # SVC
    svc2.fit(X, y_binary)
    print(" -> fitted SVC!")
    results['svc_score'].append(svc2.best_score_)


# %%
c.print(
    f"Best [bold underline]LR[/] combo = {results['combos'][np.argmin(results['lr_score'])]} yielding [blue on white]MAE = {np.min(results['lr_score'])}")

c.print(
    f"Best [bold underline]NB[/] combo = {results['combos'][np.argmax(results['nb_score'])]} yielding [blue on white]{SCORING} = {np.max(results['nb_score'])}")

c.print(
    f"Best [bold underline]SVC[/] combo = {results['combos'][np.argmax(results['svc_score'])]} yielding [blue on white]{SCORING} = {np.max(results['svc_score'])}")


# %%


