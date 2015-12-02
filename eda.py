import pandas as pd
from data_prep import transform


def load_data():
    return pd.read_csv('data/train_users.csv', parse_dates=[1, 2])


if __name__ == '__main__':
    df = load_data()
    df = transform(df)
