import pandas as pd


def load_data(train=True):
    origin = 'train' if train else 'test'
    return pd.read_csv('data/{}_users.csv'.format(origin), parse_dates=[1, 2])


def transform_data(df):
    deal_with_nans(df)
    df = dummy_gender(df)
    return df


def deal_with_nans(df):
    df.age.fillna(-1, inplace=True)
    df['already_booked'] = df.date_first_booking.isnull()


def dummy_gender(df):
    df['gender'] = df.gender.str.lower()
    df.gender.replace({'-unknown-': 'unk', 'other': 'unk'}, inplace=True)
    return pd.get_dummies(data=df, columns=['gender'], prefix='gender')
