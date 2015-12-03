import numpy as np
from data_prep import load_data, transform_data
from sklearn.ensemble import RandomForestClassifier


def prep_for_modeling(df, columns_to_drop=['id', 'date_account_created', 'timestamp_first_active', 'date_first_booking', 'signup_method', 'signup_flow', 'language', 'affiliate_channel', 'affiliate_provider', 'first_affiliate_tracked', 'signup_app', 'first_device_type', 'first_browser', 'country_destination']):
    X = df.drop(labels=columns_to_drop, axis=1).values
    y = df.country_destination.values
    return X, y


def ndcg_score(y_pred, y_true, k=5):
    relevance = (y_pred[:k] == y_true).astype(int)
    dcg_numerator = 2 ** relevance - 1
    dcg_denominator = np.log2(np.arange(relevance.shape[0]) + 2)
    return np.sum(dcg_numerator / dcg_denominator)


def modeling_exclaimation_point(df):
    X, y = prep_for_modeling(df)
    rfc = RandomForestClassifier(n_estimators=20)
    return score


if __name__ == '__main__':
    df = load_data()
    df = transform_data(df)
    score = modeling_exclaimation_point(df)
