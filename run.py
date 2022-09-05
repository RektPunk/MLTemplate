from typing import Any, Dict
import logging
import numpy as np
from models import BaselineClassificationModel
from models import BaselineRegressionModel


def run_model(
    ld_input: Dict[str, Any],
    fe_input: Dict[str, Any],
    tr_input: Dict[str, Any],
    task: str,
):
    """
    Run model with task
    Args:
        ld_input (Dict[str, Any)]):
        fe_input (Dict[str, Any)]):
        tr_input (Dict[str, Any)]):
        task (str):
    """
    try:
        if task == "classification":
            ClassificationModel = BaselineClassificationModel(
                ld_input=ld_input,
                fe_input=fe_input,
                tr_input=tr_input,
            )
            output = ClassificationModel.fit()
        elif task == "regression":
            RegressionModel = BaselineRegressionModel(
                ld_input=ld_input,
                fe_input=fe_input,
                tr_input=tr_input,
            )
            output = RegressionModel.fit()
        else:
            return False
    except Exception as e:
        logging.error(e)
        return False
    return True


if __name__ == "__main__":
    task = "classification"
    ld_input = {
        "train_df": [f"data/train.csv"],
        "test_df": [f"data/test.csv"],
        "sample_submission_df": [f"data/sample_submission.csv"],
    }
    fe_input = {
        "category_cols": ["country", "store", "product"],
        "target_cols": ["num_sold"],
        "func_dict": {
            "mean": np.mean,
            "median": np.median,
        },
    }
    tr_input = {
        "frac_ratio": 0.7,
        "id_col": "id",
        "hp_tune_trials": 3,
    }

    run_model(ld_input=ld_input, fe_input=fe_input, tr_input=tr_input, task=task)
