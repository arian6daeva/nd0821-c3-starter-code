from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from starter.ml.data import process_data


# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """
    rf_model = RandomForestClassifier()
    rf_model.fit(X_train, y_train)
    return rf_model


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : ???
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    preds = model.predict(X)
    return preds


def compute_metrics_on_slices(
        model,
        data,
        feature,
        encoder,
        lb,
        categorical_features):
    """
    Computes the performance metrics for each unique value of a given categorical feature.

    Inputs
    ------
    model : Trained machine learning model.
    data : pd.DataFrame
        Original DataFrame containing test data.
    feature : str
        Categorical feature to hold fixed.
    encoder : OneHotEncoder
        Fitted encoder to transform the categorical features.
    lb : LabelBinarizer
        Fitted label binarizer to transform the target variable.
    Returns
    -------
    slice_metrics : dict
        Dictionary containing the performance metrics for each unique value of the categorical feature.
    """
    slice_metrics = {}
    feature_values = data[feature].unique()

    for value in feature_values:
        data_slice = data[data[feature] == value]
        X_slice, y_slice, _, _ = process_data(
            data_slice, categorical_features=categorical_features, label="salary", training=False, encoder=encoder, lb=lb)

        if len(X_slice) == 0:
            continue

        preds = model.predict(X_slice)
        precision, recall, fbeta = compute_model_metrics(y_slice, preds)
        slice_metrics[value] = {
            'precision': precision,
            'recall': recall,
            'fbeta': fbeta}

    return slice_metrics
