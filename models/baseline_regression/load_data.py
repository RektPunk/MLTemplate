from typing import Dict, List, Tuple
import pandas as pd
from module.utils.read_data import read_data


def load_data(
    ld_inputs: Dict[str, List[str]],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load data from ld inputs
    ld inputs including dataframe name and paths
    >>> ld_inputs
    {
        "train_df": [f"data/train.csv"],
        "test_df" : [f"data/test.csv"],
    }
    Args:
        ld_inputs (Dict[str, List[str])]): load data name and path
    """
    df_dict = read_data(ld_inputs)
    train_df = df_dict["train_df"]
    test_df = df_dict["test_df"]
    return train_df, test_df
