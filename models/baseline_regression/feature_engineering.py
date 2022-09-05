from typing import List, Dict, Any, Callable
import pandas as pd

from module.utils.common import normalize_list
from module.feature_engineering.groupby import groupby_feature
from module.feature_engineering.string_encoder import (
    create_label_map,
    transform_categorical_cols,
)


def feature_engineering(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    category_cols: List[str],
    target_cols: List[str],
    func_dict: Dict[str, Callable],
) -> Dict[str, Any]:
    """
    Feature engineering for train and test df
    Args:
        train_df (pd.DataFrame)
        test_df (pd.DataFrame)
        category_cols (List[str])
        target_cols (List[str])
        func_dict (Dict[str, Callable])
    Returns
        Dict[str, Any]: {"train_df": ..., "test_df": ..., "label_map": ...}
    """
    train_df.columns = normalize_list(train_df.columns)
    test_df.columns = normalize_list(test_df.columns)

    label_map = create_label_map(train_df)
    train_df = transform_categorical_cols(train_df, label_map)
    test_df = transform_categorical_cols(test_df, label_map)

    groupby_dict, _ = groupby_feature(
        df=train_df,
        category_cols=category_cols,
        target_cols=target_cols,
        func_dict=func_dict,
    )

    for key, item in groupby_dict.items():
        item_name = item.name
        index_name = item.index.name
        _groupby_df = (
            pd.DataFrame(item).reset_index(drop=False).rename({item_name: key}, axis=1)
        )
        train_df = train_df.merge(_groupby_df, on=index_name, how="left")
        test_df = test_df.merge(_groupby_df, on=index_name, how="left")

    output_dict = {
        "train_df": train_df,
        "test_df": test_df,
        "label_map": label_map,
    }
    return output_dict
