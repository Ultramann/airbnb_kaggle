import pickle
from data_prep import load_data, transform_data
from cross_validation import ndcg_cross_val_score
from sklearn.ensemble import RandomForestClassifier


columns_to_drop = ['id', 'date_account_created', 'timestamp_first_active', 'date_first_booking', 
                   'signup_method', 'signup_flow', 'language', 'affiliate_channel', 
                   'affiliate_provider', 'first_affiliate_tracked', 'signup_app', 
                   'first_device_type', 'first_browser', 'country_destination']
model_path = 'models/{}.pkl'


def prep_for_modeling(df, column, columns_to_drop):
    '''
    Input:  DataFrame, Str - name of column to return separately, List
    Output: NpArray - data from df without columns_to_drop, NpArray - single specified column
    '''
    X = df.drop(labels=columns_to_drop, axis=1).values
    single_column = df[column].values
    return X, single_column


def modeling_exclaimation_point(df):
    '''
    Input:  DataFrame
    Output: NpArray - scores from k-fold tests on current data transformation

    Function to quickly test engineered features.
    '''
    df = transform_data(df)
    X, y = prep_for_modeling(df, columns_to_drop=columns_to_drop)
    rfc = RandomForestClassifier(n_estimators=100, n_jobs=-1)
    scores = ndcg_cross_val_score(rfc, X, y)
    return scores


def make_model(df, model, model_name):
    '''
    Input:  DataFrame, Model Instance - implementing fit method, Str - of name for pickled model file
    Output: None
    '''
    df = transform_data(df)
    X, y = prep_for_modeling(df, column='country_destination', columns_to_drop=columns_to_drop)
    model.fit(X, y)
    with open(model_path.format(model_name), 'w+') as model_file:
        pickle.dump(model, model_file)


if __name__ == '__main__':
    df = load_data()
    scores = modeling_exclaimation_point(df)
