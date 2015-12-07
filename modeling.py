from data_prep import load_data, transform_data
from cross_validation import ndcg_cross_val_score
from sklearn.ensemble import RandomForestClassifier


def prep_for_modeling(df, columns_to_drop):
    X = df.drop(labels=columns_to_drop, axis=1).values
    y = df.country_destination.values
    return X, y


def modeling_exclaimation_point(df):
    df = transform_data(df)
    columns_to_drop =['id', 'date_account_created', 'timestamp_first_active', 'date_first_booking', 
                      'signup_method', 'signup_flow', 'language', 'affiliate_channel', 
                      'affiliate_provider', 'first_affiliate_tracked', 'signup_app', 
                      'first_device_type', 'first_browser', 'country_destination']
    X, y = prep_for_modeling(df, columns_to_drop=columns_to_drop)
    rfc = RandomForestClassifier(n_estimators=20)
    scores = ndcg_cross_val_score(rfc, X, y)
    return scores


if __name__ == '__main__':
    df = load_data()
    scores = modeling_exclaimation_point(df)
