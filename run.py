import numpy as np
import pandas as pd
import sklearn
import optuna
import lightgbm as lgb

from module.utils.read_data import read_data
from module.utils.common import normalize_list
from module.feature_engineering.groupby import groupby_feature
from module.feature_engineering.string_encoder import (
    create_label_map,
    transform_categorical_cols,
)
from module.training.objective import optuna_objective


read_inputs = {
    "train_df": [f"data/train.csv"],
    "test_df": [f"data/test.csv"],
}
df_dict = read_data(read_inputs)
train_df = df_dict["train_df"]
test_df = df_dict["test_df"]
sample_submission_df = df_dict["sample_submission_df"]

train_df.columns = normalize_list(train_df.columns)
test_df.columns = normalize_list(test_df.columns)

category_cols = []
target_cols = ["y"]

label_map = create_label_map(train_df)
train_df = transform_categorical_cols(train_df, label_map)
test_df = transform_categorical_cols(test_df, label_map)

groupby_dict, cols = groupby_feature(
    df=train_df,
    category_cols=category_cols,
    target_cols=target_cols,
    func_dict={
        "mean": np.mean,
        "median": np.median,
    },
)

for key, item in groupby_dict.items():
    item_name = item.name
    index_name = item.index.name
    _groupby_df = (
        pd.DataFrame(item).reset_index(drop=False).rename({item_name: key}, axis=1)
    )
    train_df = train_df.merge(_groupby_df, on=index_name, how="left")
    test_df = test_df.merge(_groupby_df, on=index_name, how="left")

_train_df = train_df.sample(frac=0.7)
_valid_df = train_df.loc[~train_df["id"].isin(_train_df["id"])]

train_x_df = train_df.drop(columns=target_cols)
_train_x_df = _train_df.drop(columns=target_cols)
_valid_x_df = _valid_df.drop(columns=target_cols)

result_dict = {}
for target_col in target_cols:
    train_y_df = train_df[target_col]
    _train_y_df = _train_df[target_col]
    _valid_y_df = _valid_df[target_col]

    def hp_tune(trial):
        return optuna_objective(
            trial,
            train_x=_train_x_df,
            train_y=_train_y_df,
            valid_x=_valid_x_df,
            valid_y=_valid_y_df,
        )

    study = optuna.create_study(direction="minimize")
    study.optimize(hp_tune, n_trials=5)
    print("Best trial:")
    trial = study.best_trial
    print("  Value: {}".format(trial.value))
    print("================ OPT PARAMS ================")
    opt_params = {
        "objective": "binary",
        "metric": "binary",
        "verbosity": -1,
        "boosting_type": "gbdt",
    }
    print(study.best_value)
    opt_params.update(study.best_params)
    print(opt_params)
    print("=============================================")
    model_params = dict(
        params=opt_params,
        train_set=lgb.Dataset(train_x_df, label=train_y_df),
        verbose_eval=1,
    )
    final_mdl = lgb.train(**model_params)
    test_y_hat = pd.Series(final_mdl.predict(test_df))
    result_dict.update({target_col: test_y_hat})

train_y_hat = pd.Series(final_mdl.predict(train_x_df))
train_accuracy_score = sklearn.metrics.accuracy_score(train_y_df, np.rint(train_y_hat))
train_log_loss = sklearn.metrics.log_loss(train_y_df, np.rint(train_y_hat))
train_confuxion_matrix = sklearn.metrics.confusion_matrix(
    train_y_df, np.rint(train_y_hat)
)
print(
    f"""
    accuracy score : {train_accuracy_score}
    log loss : {train_log_loss}
    confucion matirx : \n {train_confuxion_matrix}
    """
)
