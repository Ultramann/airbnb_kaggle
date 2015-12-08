import pickle
import numpy as np
import pandas as pd
from data_prep import load_data, transform_data
from modeling import columns_to_drop, model_path, prep_for_modeling


def get_test_data():
    df = load_data(train=False)
    df = transform_data(df)
    X, ids = prep_for_modeling(df, column='id', columns_to_drop=columns_to_drop[:-1])
    return X, ids


def get_model(model_name):
    with open(model_path.format(model_name), 'r') as model_file:
        model = pickle.load(model_file)
    return model


def get_predictions(model, X_test):
    target_classes = model.classes_
    test_probs = model.predict_proba(X_test)
    test_predictions = target_classes[np.argsort(test_probs)[:, :-6:-1]]
    return test_predictions


def prep_submission(predictions_df):
    submissions_df = predictions_df.stack().reset_index(level=0)
    submissions_df.columns = ['id', 'country']
    return submissions_df


def make_submission(model_name):
    X_test, ids = get_test_data()
    model = get_model(model_name)
    test_predictions = get_predictions(model, X_test)
    predictions_df = pd.DataFrame(test_predictions, index=ids)
    submissions_df = prep_submission(predictions_df)
    submissions_df.to_csv('csv_submissions/{}.csv'.format(model_name), index=False)


if __name__ == '__main__':
    make_submission('random_forest_1000')
    
