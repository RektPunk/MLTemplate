from typing import List, Dict, Callable
import pandas as pd
import optuna
import lightgbm as lgb
from module.training.objective import optuna_regression_objective


def training(
    train_df: pd.DataFrame,
    target_cols: List[str],
    frac_ratio: float,
    id_col: str,
    hp_tune_trials: int,
) -> Dict[str, Callable]:
    """
    Return trained models
    Args:
        train_df (pd.DataFrame)
        target_cols (List[str])
        frac_ratio (float) : [0, 1] float for train valid random split
        id_col (str): id colum name
        hp_tune_trials (int): the number of hyperparameter trials
    Returns:
        Dict[str, Callable]: {f"{target_col}": model, ...}
    """
    _train_df = train_df.sample(frac=frac_ratio)
    _valid_df = train_df.loc[~train_df[id_col].isin(_train_df[id_col])]

    train_x_df = train_df.drop(columns=target_cols)
    _train_x_df = _train_df.drop(columns=target_cols)
    _valid_x_df = _valid_df.drop(columns=target_cols)

    mdl_dict = {}
    for target_col in target_cols:
        train_y_df = train_df[target_col]
        _train_y_df = _train_df[target_col]
        _valid_y_df = _valid_df[target_col]

        def hp_tune(trial):
            return optuna_regression_objective(
                trial,
                train_x=_train_x_df,
                train_y=_train_y_df,
                valid_x=_valid_x_df,
                valid_y=_valid_y_df,
            )

        study = optuna.create_study(direction="minimize")
        study.optimize(hp_tune, n_trials=hp_tune_trials)
        opt_params = {
            "objective": "rmse",
            "metric": "rmse",
            "verbosity": -1,
            "boosting_type": "gbdt",
        }
        opt_params.update(study.best_params)
        model_params = dict(
            params=opt_params,
            train_set=lgb.Dataset(train_x_df, label=train_y_df),
            verbose_eval=1,
        )
        final_mdl = lgb.train(**model_params)
        mdl_dict.update({f"{target_col}": final_mdl})

    return mdl_dict
