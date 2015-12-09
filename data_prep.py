import pandas as pd


def load_data(train=True):
    '''
    Input:  Bool - Whether or not to load train data, loads test otherwise
    Output: DataFrame
    '''
    origin = 'train' if train else 'test'
    return pd.read_csv('data/{}_users.csv'.format(origin), parse_dates=[1, 2])


def transform_data(df):
    '''
    Input:  DataFrame - likely loaded directly from csv
    Output: DataFrame - with newly engineered features
    '''
    deal_with_nans(df)
    df = dummy_gender(df)
    return df


def deal_with_nans(df):
    '''
    Input:  DataFrame
    Output: DataFrame - with some column's nans filled
                        and some column's nans turned into new column of bools
    '''
    df.age.fillna(-1, inplace=True)
    df['already_booked'] = df.date_first_booking.isnull()


def dummy_gender(df):
    '''
    Input:  DataFrame - with gender column
    Output: DataFrame - with information from gender column turned into dummied columns
    '''
    df['gender'] = df.gender.str.lower()
    df.gender.replace({'-unknown-': 'unk'}, inplace=True)
    return pd.get_dummies(data=df, columns=['gender'], prefix='gender')
