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


use_vader = True

vader = pd.read_pickle('data/vader_out_clean.pickle')
vader['datetime'] = vader.datetime.astype('datetime64')

if use_vader:
    print("====> Using VADER Data")
    senti_df = vader.copy()
else:
    print("====> Using AFINN Data")
    afinn = pd.read_pickle('data/afinn_alldata_2019')
    afinn['sentiment'] = afinn.scores.copy()
    afinn = afinn.drop(['scores'], axis=1)
    afinn['datetime'] = afinn.date.astype('datetime64')
    afinn = afinn.reset_index(drop=True)
    afinn = pd.concat([afinn, vader[['likes', 'retweets', 'replies']]], axis=1)
    senti_df = afinn.copy()





print(pd.date_range(start='2019-01-01',
                    end='2019-12-31').difference(senti_df.datetime.dt.date))


raw_yahoo = yf.Ticker('TSLA').history(
    start='2019-01-01', end='2019-12-31', interval='1D')
raw_prices = raw_yahoo['Close']

# %%
# Backfill
daily_return = raw_prices.pct_change().shift(-1)
daily_return = daily_return.asfreq('D', method='ffill').fillna(0)

# %%
daily_sentiment: pd.Series = senti_df.groupby(senti_df.datetime.dt.date).agg(
    lambda x: np.average(x, weights=senti_df.loc[x.index, "likes"]))['sentiment']
pct_pos = senti_df.groupby(senti_df.datetime.dt.date)[
    'sentiment'].agg(lambda s: sum(s > 0.05)/len(s))
pct_neg = senti_df.groupby(senti_df.datetime.dt.date)[
    'sentiment'].agg(lambda s: sum(s < -0.05)/len(s))
num_of_tweets = senti_df.groupby(senti_df.datetime.dt.date)['sentiment'].size()


# %%
df = pd.DataFrame({
    'sentiment': daily_sentiment[1:364].reset_index(drop=True),
    'pct_pos': pct_pos[1:364].reset_index(drop=True),
    'pct_neg': pct_neg[1:364].reset_index(drop=True),
    'num_tweets': num_of_tweets[1:364].reset_index(drop=True),
    'volume': raw_yahoo['Volume'].asfreq('D', method='ffill').fillna(0).values,
    'prev0': raw_prices.pct_change().shift(0).asfreq('D', method='ffill').reset_index(drop=True).fillna(0),
    'prev1': raw_prices.pct_change().shift(1).asfreq('D', method='ffill').reset_index(drop=True).fillna(0),
    'prev2': raw_prices.pct_change().shift(2).asfreq('D', method='ffill').reset_index(drop=True).fillna(0),
    # 'prev4': raw_prices.pct_change().shift(3).asfreq('D', method='ffill').reset_index(drop=True).fillna(0),
}
)
c.print("[bold underline]Data Frame")
df.info()

# Sanity check
jan3rd = df.iloc[1, :]
assert jan3rd['sentiment'] - 1.11126 < 0.001
assert jan3rd['num_tweets'] - 1402 < 0.001
assert jan3rd['pct_pos'] - 0.4265 < 0.001
# assert df.iloc[12, :]

y_binary = (daily_return > 0).astype('int8')
y = daily_return


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
# ATTENTION! For experimenting!

financial_only = False

if financial_only:
    c.print("[red] CAUTION: dropping all sentiment features, using financial only!")
    df = df.drop(['sentiment', 'num_tweets', 'pct_neg', 'pct_pos', 'prev0', 'prev1', 'prev2'], axis=1)

# df = df.drop(['prev0', 'prev1', 'prev2'], axis=1)



SCORING = 'accuracy'
cv = KFold(10)

svc2 = GridSearchCV(SVC(probability=True,),
                    param_grid={'kernel': ['rbf', 'sigmoid'],
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
    X = prep(df, *combo)
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


