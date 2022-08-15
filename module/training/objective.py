import lightgbm as lgb
import pandas as pd
import numpy as np
import sklearn


def optuna_objective(
    trial,
    train_x: pd.DataFrame,
    train_y: pd.Series,
    valid_x: pd.DataFrame,
    valid_y: pd.Series,
) -> float:
    """
    optuna objective function
    Args:
        trial
        train_x (pd.DataFrame):
        train_y (pd.Series):
        valid_x (pd.DataFrame):
        valid_y (pd.Series):
        weight_train (pd.Series):

    Returns:
        float: rmse score
    """
    dtrain = lgb.Dataset(data=train_x, label=train_y)
    param = {
        "objective": "binary",  # e.g. "rmse" for regression
        "metric": "binary_logloss",  # e.g. "rmse" for regression
        "verbosity": -1,
        "boosting_type": "gbdt",
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 2, 256),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
    }

    gbm = lgb.train(param, dtrain)
    preds = gbm.predict(valid_x)

    ## binary classification
    pred_labels = np.rint(preds)
    accuracy = sklearn.metrics.accuracy_score(valid_y, pred_labels)

    ## regression
    # rmse = np.sqrt(np.mean((valid_y - preds) ** 2))

    return accuracy
