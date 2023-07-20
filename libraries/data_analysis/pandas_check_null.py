import pandas as pd

def enhance_describe(df:pd.DataFrame, columns:str=None):
    if columns is None:
        columns = df.columns
    df = df[columns]
    df1 = df.describe(include = 'all')
    df1.loc['dtype'] = df.dtypes
    df1.loc['size'] = len(df)
    df1.loc['null% count'] = df.isnull().mean()
    df1.loc['null count'] = X.isnull().sum(axis=0)
    return df1