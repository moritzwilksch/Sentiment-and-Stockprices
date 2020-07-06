# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import multiprocessing as mp

df = pd.read_pickle('data/vader_out.pickle')
# df = pd.read_pickle('data/afinn_alldata_2019')

df['datetime'] = df.datetime.astype('datetime64')
# df['sentiment'] = df.scores
# df = df.drop(['date', 'scores'], axis=1)


def plot(name: str = 'fig.png'):
    plt.savefig(name)
    plt.close()


# %%
pd.date_range(start='2019-01-01', end='2019-12-31').difference(df.datetime.dt.date)

# %%
import yfinance as yf

raw_prices = yf.Ticker('TSLA').history(start='2019-01-01', end='2019-12-31', interval='1D')['Close']
daily_return = raw_prices.pct_change().shift(-1)

# %%
# Backfill
daily_return = daily_return.asfreq('D', method='bfill').fillna(0)

# %%
daily_df = df.groupby(df.datetime.dt.date).agg(['mean', 'std']).fillna(0)
daily_df = daily_df.reset_index()
daily_df.columns = ['date', 'mean', 'std']

# %%
import scipy.stats as stats
from sklearn.model_selection import GridSearchCV

print("=== CORRELATION - SENTIMENT ONLY ===")
for lag in range(6):
    daily_return = raw_prices.pct_change().shift(-lag)
    daily_return = daily_return.asfreq('D', method='bfill').fillna(0)
    print(f"==== LAG = {lag} ====")
    print(stats.pearsonr(daily_df.loc[1:363, 'mean'], daily_return))
    # print(stats.spearmanr(daily_df.loc[1:363, 'sentiment']['mean'], daily_return))

print("=== CORRELATION - POLARITY ONLY ===")
for lag in range(6):
    daily_return = raw_prices.pct_change().shift(-lag)
    daily_return = daily_return.asfreq('D', method='bfill').fillna(0)
    print(f"==== LAG = {lag} ====")
    print(stats.pearsonr(daily_df.loc[1:363, 'std'], daily_return))
    # print(stats.spearmanr(daily_df.loc[1:363, 'sentiment']['std'], daily_return))

#%%
daily_df.head()



# %%
# Splitting the data
# Use Jan - Nov as train, December as test
X = daily_df.loc[1:363, ['mean', 'std']]
x_train = daily_df.loc[1:333, ['mean', 'std']]
x_test = daily_df.loc[334:363, ['mean', 'std']]

y = daily_return
y_binary = (daily_return > 0).astype('int8')
y_train = daily_return[:pd.to_datetime('2019-11-30')]
y_train_binary = (y_train > 0).astype('int8')
y_test = daily_return[pd.to_datetime('2019-12-01'):]
y_test_binary = (y_test > 0).astype('int8')

# %%
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score


print("=== LINEAR REGRESSION ===")

# SENTIMENT ONLY
print("--- SENTIMENT ONLY ---")
lr = LinearRegression()
print(">> CV RESULTS <<")
print(f" R^2 = {np.mean(cross_val_score(lr, X['mean'].values.reshape(-1, 1), y, n_jobs=-1, cv=10, scoring='r2'))}")
print(f" RMSE = {-np.mean(cross_val_score(lr, X['mean'].values.reshape(-1, 1), y, n_jobs=-1, cv=10, scoring='neg_root_mean_squared_error'))}")


# SENTIMENT & POLARITY
print("--- SENTIMENT & POLARITY ---")
lr2 = LinearRegression()

print(">> CV RESULTS <<")
print(f" R^2 = {np.mean(cross_val_score(lr2, X, y, n_jobs=-1, cv=10, scoring='r2'))}")
print(f" RMSE = {-np.mean(cross_val_score(lr2, X, y, n_jobs=-1, cv=10, scoring='neg_root_mean_squared_error'))}")

# %%
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, plot_confusion_matrix, roc_auc_score

print("=== GAUSSIAN NAIVE BAYES ===")

# SENTIMENT ONLY
print("--- SENTIMENT ONLY ---")
nb = GaussianNB()
# nb.fit(x_train['mean'].values.reshape(-1, 1), (y_train > 0).astype('int8'))
print(">> CV RESULTS <<")
print(f"AUC = {np.mean(cross_val_score(nb, X, y_binary, cv=10, n_jobs=-1, scoring='roc_auc'))}")
nb.fit(X, y_binary)
print(f"Confusion Matrix \n {confusion_matrix(y_binary, nb.predict(X))}")


# SENTIMENT & POLARITY
print("--- SENTIMENT & POLARITY ---")
nb2 = GaussianNB()

print(f"AUC = {np.mean(cross_val_score(nb2, X, y_binary, cv=10, n_jobs=-1, scoring='roc_auc'))}")
nb2.fit(X, y_binary)
print(f"Confusion Matrix \n {confusion_matrix(y_binary, nb2.predict(X))}")


# %%
from sklearn.svm import SVC

print("=== Support Vector Classifier ===")

scoring = 'roc_auc'

# SENTIMENT ONLY
print("--- SENTIMENT ONLY ---")
# svc = SVC(C=10) # Nice, GridSearch!!!!!!

svc = GridSearchCV(SVC(probability=True),
                   param_grid={'kernel': ['rbf', 'poly', 'sigmoid'],
                               'C': [0.01, 0.1, 0.5, 1, 5, 10]
                               },
                   n_jobs=-1,
                   cv=3,
                   scoring=scoring
                   )
# svc.fit(x_train['mean'].values.reshape(-1, 1), (y_train > 0).astype('int8'))
svc.fit(X['mean'].values.reshape(-1, 1), y_binary)
print(f"Best params: {svc.best_params_} yielding {svc.scoring} = {svc.best_score_}")
print(f"Confusion Matrix \n {confusion_matrix(y_binary, svc.predict(X['mean'].values.reshape(-1, 1)))}")
print(f"ROC AUC = {roc_auc_score(y_binary, svc.predict(X['mean'].values.reshape(-1, 1)))}")


# SENTIMENT & POLARITY

print("--- SENTIMENT & POLARITY ---")

svc2 = GridSearchCV(SVC(probability=True),
                    param_grid={'kernel': ['rbf', 'poly', 'sigmoid'],
                                'C': [0.01, 0.1, 0.5, 1, 5, 10]
                                },
                    n_jobs=-1,
                    cv=5,
                    scoring=scoring
                    )

svc2.fit(X, y_binary)
print(f"Best params: {svc2.best_params_} yielding {svc2.scoring} = {svc2.best_score_}")
print(f"Confusion Matrix \n {confusion_matrix(y_binary, svc2.predict(X))}")
print(f"ROC AUC = {roc_auc_score(y_binary, svc2.predict(X))}")


# %%
from sklearn.metrics import classification_report
print(classification_report(y_binary, svc.predict(X['mean'].values.reshape(-1, 1))))
