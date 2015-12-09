import numpy as np
from sklearn.cross_validation import KFold


def ndcg_score(y_preds, y_true, k=5):
    '''
    Input:  NpArray - predicted categories, 
            NpArray - actual category, 
            Int - number of preditions to consider
    Output: Int - normalized discounted cumulative gain for categorical predictions
    '''
    relevance = (y_preds[:, :k] == y_true[:, None]).astype(int)
    dcg_denominator = np.log2(np.arange(relevance.shape[1]) + 2)
    ndcg = relevance / dcg_denominator[None, :]
    return np.sum(ndcg, axis=1)


def ndcg_cross_val_score(model, X, y, n_folds=5):
    '''
    Input:  Model Instance - implementing fit method, NpArray, NpArray, Int
    Output: NpArray - ndcg score for each fold
    '''
    kf = KFold(y.shape[0], n_folds=n_folds)
    target_classes = np.sort(np.unique(y))

    def make_score_fold(train_idx, test_idx):
        '''
        Helper method to make list of test predictions to pass to ndcg_score for each fold.
        '''
        model.fit(X[train_idx], y[train_idx])
        test_probs = model.predict_proba(X[test_idx])
        test_predictions = target_classes[np.argsort(test_probs)[:, ::-1]]
        return ndcg_score(test_predictions, y[test_idx]).mean()

    return np.array([make_score_fold(train_idx, test_idx) for train_idx, test_idx in kf])
