# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import numpy as np

use_vader = False

if use_vader:
    print("====> Using VADER Data")
    # df = pd.read_pickle('data/vader_out.pickle')
    df = pd.read_pickle('data/vader_out_clean.pickle')
    df['datetime'] = df.datetime.astype('datetime64')

else:
    print("====> Using AFINN Data")
    df = pd.read_pickle('data/afinn_alldata_2019')
    df['sentiment'] = df.scores
    df = df.drop(['scores'], axis=1)
    df['datetime'] = df.date.astype('datetime64')


def plot(name: str = 'fig.png'):
    plt.savefig(name)
    plt.close()


print(pd.date_range(start='2019-01-01', end='2019-12-31').difference(df.datetime.dt.date))

import yfinance as yf

raw_yahoo = yf.Ticker('TSLA').history(start='2019-01-01', end='2019-12-31', interval='1D')
raw_prices = raw_yahoo['Close']

# Backfill
daily_return = raw_prices.pct_change().shift(-1)
daily_return = daily_return.asfreq('D', method='ffill').fillna(0)

daily_df = df.groupby(df.datetime.dt.date).agg(['mean', 'std']).fillna(0)
daily_df = daily_df.reset_index()
daily_df.columns = ['date', 'mean', 'std']

# %%
from sklearn.preprocessing import StandardScaler, PolynomialFeatures


def prep(polynomial_features, standard_scale, add_volume, add_previous_days):
    X = daily_df.loc[1:363, ['mean', 'std']]

    if add_volume:
        X['volume'] = raw_yahoo['Volume'].asfreq('D', method='ffill').fillna(0).values

    if add_previous_days:
        X['prev_1'] = raw_prices.pct_change().shift(1).asfreq('D', method='ffill').reset_index(drop=True).fillna(0)
        X['prev_1'] = X['prev_1'].fillna(0)

        X['prev_2'] = raw_prices.pct_change().shift(2).asfreq('D', method='ffill').reset_index(drop=True).fillna(0)
        X['prev_2'] = X['prev_2'].fillna(0)

        X['prev_3'] = raw_prices.pct_change().shift(3).asfreq('D', method='ffill').reset_index(drop=True).fillna(0)
        X['prev_3'] = X['prev_3'].fillna(0)


    # ATTENTION!!! ONLY FOR ONE OF THE RUNS
    # X = X.drop(['mean', 'std'], axis=1)
    print(X.columns)

    if polynomial_features:
        # Polynomial Features
        poly = PolynomialFeatures(2)
        X = poly.fit_transform(X)

    if standard_scale:
        ss = StandardScaler()
        X = ss.fit_transform(X)

    y = daily_return
    y_binary = (daily_return > 0).astype('int8')

    return X, y, y_binary


# %%
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from itertools import product
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

results = {
    'combos': [x for x in product([True, False], repeat=4)],
    'lr_mae': [],
    'nb_roc': [],
    'svc_roc': []
}

# Never add financial indicators - sentiment only
"""results['combos'] = [
    (False, False, False, False),
    (False, True, False, False),
    (True, False, False, False),
    (True, True, False, False)
]"""

# Financial indicators only - dont forget do add a .drop into prep
"""results['combos'] = [(True, True, True, True),
                     (True, True, True, False),
                     (True, True, False, True),
                     # (True, True, False, False), must not occur i.O. to drop sentiment
                     (True, False, True, True),
                     (True, False, True, False),
                     (True, False, False, True),
                     # (True, False, False, False), must not occur i.O. to drop sentiment
                     (False, True, True, True),
                     (False, True, True, False),
                     (False, True, False, True),
                     # (False, True, False, False), must not occur i.O. to drop sentiment
                     (False, False, True, True),
                     (False, False, True, False),
                     (False, False, False, True),
                     # (False, False, False, False) must not occur i.O. to drop sentiment
                     ]"""
from sklearn.model_selection import TimeSeriesSplit, KFold

for combo in results['combos']:
    X, y, y_binary = prep(*combo)

    tss = TimeSeriesSplit(n_splits=10)
    # cv = [(train_idx, test_idx) for train_idx, test_idx in tss.split(X)]
    cv = KFold(10)

    lr2 = LinearRegression()
    results['lr_mae'].append(-np.mean(cross_val_score(lr2, X, y, n_jobs=-1, cv=10, scoring='neg_mean_absolute_error')))

    nb2 = GaussianNB()
    results['nb_roc'].append(np.mean(cross_val_score(nb2, X, y_binary, cv=cv, n_jobs=-1, scoring='roc_auc')))

    svc2 = GridSearchCV(SVC(probability=True),
                        param_grid={'kernel': ['rbf', 'poly', 'sigmoid'],
                                    'C': [0.01, 0.1, 0.5, 1, 5, 10]
                                    },
                        n_jobs=-1,
                        cv=cv,
                        scoring='roc_auc'
                        )
    svc2.fit(X, y_binary)
    results['svc_roc'].append(svc2.best_score_)

print(f"Best Linear Regression combo = {results['combos'][np.argmin(results['lr_mae'])]} yielding MAE = {np.min(results['lr_mae'])}")

print(
    f"Best Naive Bayes combo = {results['combos'][np.argmax(results['nb_roc'])]} yielding Accuracy = {np.max(results['nb_roc'])}")

print(
    f"Best SVC combo = {results['combos'][np.argmax(results['svc_roc'])]} yielding Accuracy = {np.max(results['svc_roc'])}")
