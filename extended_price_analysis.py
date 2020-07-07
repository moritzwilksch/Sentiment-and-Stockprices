# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import multiprocessing as mp

use_vader = False

if use_vader:
    print("====> Using VADER Data")
    df = pd.read_pickle('data/vader_out.pickle')
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

# %%
# Backfill
daily_return = raw_prices.pct_change().shift(-1)
daily_return = daily_return.asfreq('D', method='ffill').fillna(0)

# %%
daily_df = df.groupby(df.datetime.dt.date).agg(['mean', 'std']).fillna(0)
daily_df = daily_df.reset_index()
daily_df.columns = ['date', 'mean', 'std']

# %%
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

# Splitting the data
X = daily_df.loc[1:363, ['mean', 'std']]

polynomial_features = False  # Dont use poly features
standard_scale = True
add_volume = True
add_previous_days = False  # False delivers GREAT SVC for AFINN

# Best until now:
# AFINN + F/T/T/F => SVC 0.583
# Vader + T/T/T/F => SVC 0.578

if add_volume:
    X['volume'] = raw_yahoo['Volume'].asfreq('D', method='ffill').fillna(0).values


if add_previous_days:
    X['prev_1'] = raw_prices.pct_change().shift(1).asfreq('D', method='ffill').reset_index(drop=True).fillna(0)
    X['prev_1'] = X['prev_1'].fillna(0)

    X['prev_2'] = raw_prices.pct_change().shift(2).asfreq('D', method='ffill').reset_index(drop=True).fillna(0)
    X['prev_2'] = X['prev_2'].fillna(0)

if polynomial_features:
    # Polynomial Features
    poly = PolynomialFeatures(2)
    X = poly.fit_transform(X)


if standard_scale:
    ss = StandardScaler()
    X = ss.fit_transform(X)

y = daily_return
y_binary = (daily_return > 0).astype('int8')

# %%
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

# SENTIMENT & POLARITY
print("=== LINEAR REGRESSION ===")
print("--- ALL ---")
lr2 = LinearRegression()

print(">> CV RESULTS <<")
print(f" R^2 = {np.mean(cross_val_score(lr2, X, y, n_jobs=-1, cv=10, scoring='r2'))}")
print(f" RMSE = {-np.mean(cross_val_score(lr2, X, y, n_jobs=-1, cv=10, scoring='neg_root_mean_squared_error'))}")

# %%
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, roc_auc_score

print("=== GAUSSIAN NAIVE BAYES ===")

# SENTIMENT & POLARITY
print("--- ALL ---")
nb2 = GaussianNB()

print(f"AUC = {np.mean(cross_val_score(nb2, X, y_binary, cv=10, n_jobs=-1, scoring='roc_auc'))}")
nb2.fit(X, y_binary)
print(f"Confusion Matrix \n {confusion_matrix(y_binary, nb2.predict(X))}")

# %%
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

# SENTIMENT & POLARITY

print("=== Support Vector Classifier ===")
print("--- ALL ---")

svc2 = GridSearchCV(SVC(probability=True),
                    param_grid={'kernel': ['rbf', 'poly', 'sigmoid'],
                                'C': [0.01, 0.1, 0.5, 1, 5, 10]
                                },
                    n_jobs=-1,
                    cv=10,
                    scoring='roc_auc'
                    )
"""from sklearn.preprocessing import MinMaxScaler
mm = MinMaxScaler()
sample_weight=mm.fit_transform(np.abs(y).values.reshape(-1,1)).flatten()
"""
svc2.fit(X, y_binary)
print(f"Best params: {svc2.best_params_} yielding {svc2.scoring} = {svc2.best_score_}")
print(f"Confusion Matrix \n {confusion_matrix(y_binary, svc2.predict(X))}")

# %%
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier

print("=== RANDOM FOREST Classifier ===")
print("--- ALL ---")

rf = GridSearchCV(XGBClassifier(),
                  param_grid={
                      'max_depth': [2, 3, 4],
                      'gamma': [0, 0.01, 0.1, 1, 10],
                      'lambda': [1, 5, 10],
                      'alpha': [1, 5, 10]
                  },
                  n_jobs=-1,
                  cv=5,
                  scoring='roc_auc'
                  )

rf.fit(X, y_binary)
print(f"Best params: {rf.best_params_} yielding {rf.scoring} = {rf.best_score_}")
print(f"Confusion Matrix \n {confusion_matrix(y_binary, rf.predict(X))}")


#%%

# Testsing Significance of ROC AUC of SVC
n_runs = 1000
roc_aucs = []

for _ in range(n_runs):
    np.random.shuffle(y_binary)
    model = SVC(**svc2.best_params_)
    roc_aucs.append(
        np.mean(cross_val_score(model, X, y_binary, cv=10, n_jobs=-1, scoring='roc_auc'))
    )

print(f"p-value = {sum(np.array(roc_aucs) >= 0.58336) / 1000}")

# reset y_binary
y_binary = (daily_return > 0).astype('int8')
#%%
