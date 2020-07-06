# %%
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from typing import List
import pandas as pd


# %%


def init_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """Just returns a copy of the df"""
    return df.copy()


def drop_cols(df: pd.DataFrame, col_list: List[str]) -> pd.DataFrame:
    """Drops all columns thet are in the list"""
    try:
        return df.drop(col_list, axis=1)
    except:
        print("Could not drop column. Returning initial df")
        return df


def fix_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Fixes datatypes"""
    try:
        string_cols = ['username', 'name', 'tweet', 'link']
        df[string_cols] = df[string_cols].astype('string')
    except:
        print("Could not convert to string.")
        return df

    return df

def drop_many_cashtags(df: pd.DataFrame) -> pd.DataFrame:
    df['num_cashtags'] = df.cashtags.apply(lambda s: len(s[1:-1].split(',')))
    # Drop tweets with more than 5 CTs
    mask = df['num_cashtags'] < 5
    return df[mask]

# %%
cols_to_drop = ['place', 'near', 'geo', 'source', 'user_rt_id', 'user_rt', 'retweet_id', 'retweet_date', 'translate',
                'trans_src', 'trans_dest', 'retweet']

pipe = Pipeline([
    ('Init Pipeline', FunctionTransformer(init_pipeline)),
    ('drop cols', FunctionTransformer(drop_cols, kw_args={'col_list': cols_to_drop})),
    ('fix dtypes', FunctionTransformer(fix_dtypes)),
    ('drop many CTs', FunctionTransformer(drop_many_cashtags))

])

# %%
raw = pd.read_pickle('data/all2019.pickle')
df: pd.DataFrame = pipe.fit_transform(raw)

#%%
df.to_pickle('data/preprocessed_all2019.pickle')
