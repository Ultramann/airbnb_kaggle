import pandas as pd
from data_prep import load_data


def category_column_correlation(column, target):
    '''
    Input:  Series - Column to be dummied, Series - Target, undummified
    Output: DataFrame - Correlation coefficient matrix dummies of column and target
                        Non-redundant
    '''
    column_dummies = pd.get_dummies(column)
    target_dummies = pd.get_dummies(target)
    correlation_df = pd.concat([column_dummies, target_dummies], axis=1).corr()
    return correlation_df.ix[column_dummies.columns, target_dummies.columns]


if __name__ == '__main__':
    df = load_data()
