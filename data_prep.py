import pandas as pd


def load_data():
    return pd.read_csv('data/train_users.csv', parse_dates=[1, 2])


def transform_data(df):
    df.age.fillna(-1, inplace=True)
    df = dummy_gender(df)
    return df


def dummy_gender(df):
    df['gender'] = df.gender.str.lower()
    df.gender.replace({'-unknown-': 'unk', 'other': 'unk'}, inplace=True)
    return pd.get_dummies(data=df, columns=['gender'], prefix='gender')
