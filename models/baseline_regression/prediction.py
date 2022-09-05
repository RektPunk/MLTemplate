from typing import Dict, Callable
import pandas as pd


def prediction(
    test_df: pd.DataFrame,
    mdls: Dict[str, Callable],
) -> Dict[str, pd.Series]:
    """
    Prediction with mdls
    Args:
        test_df (pd.DataFrame)
        mdls (Dict[str, Callable)])
    Returns:
        Dict[str, pd.Series]
    """
    predict_dict = {}
    for target_col, mdl in mdls.items():
        test_y_hat = pd.Series(mdl.predict(test_df))
        predict_dict.update({target_col: test_y_hat})

    return predict_dict
