from typing import Tuple, Dict
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


class LGBMLabelEncoder(object):
    """
    Label Encoder for LGBM
    """

    def __init__(self):
        self.label_encoder = LabelEncoder()

    def fit(self, series):
        """
        Fit series with "NaN" and "Unseen" values
        """
        self.label_encoder.fit(list(series[~series.isna()]) + ["NaN", "Unseen"])

    def transform(self, series):
        """
        Transform series after changing na >> "NaN", not seen variable to "Unseen"
        """
        _pre_transformed_series = np.select(
            [series.isna(), ~series.isin(self.label_encoder.classes_)],
            [
                "NaN",
                "Unseen",
            ],
            series,
        )
        return self.label_encoder.transform(_pre_transformed_series)

    def inverse_transform(self, series):
        """
        Inverse transoform series with label encoder
        """
        return self.label_encoder.inverse_transform(series)


def create_label_map(
    df: pd.DataFrame,
) -> Dict[str, LGBMLabelEncoder]:
    """
    Create label map
    Args:
        df (pd.DataFrame): train dataframe
    Returns
        Dict[str, LGBMLabelEncoder]: {f"{col_name}": LGBMLabelEncoder}
    """
    label_map = dict()
    for col in df.dtypes[(df.dtypes == "object") & (df.dtypes != "bool")].keys():
        _label_map = LGBMLabelEncoder()
        _label_map.fit(df[col].astype(str))
        label_map.update({col: _label_map})

    return label_map


def transform_categorical_cols(
    df: pd.DataFrame,
    label_map: Dict[str, LGBMLabelEncoder],
) -> pd.DataFrame:
    """
    Convert categorical variable using label map
    Args:
        df (pd.DataFrame): input dataframe
        label_map: label map from create label map
    Returns:
        pd.DataFrame: converted dataFrame
    """
    for key, item in label_map.items():
        df = df.assign(**{key: item.transform(df[key].astype(str))})
    return df


def inverse_transform_categorical_cols(
    df: pd.DataFrame,
    label_map=Dict[str, LGBMLabelEncoder],
) -> Tuple[pd.DataFrame, Dict[str, LGBMLabelEncoder]]:
    """
    Convert categorical variable using label map
    Args:
        df (pd.DataFrame): input dataframe
        label_map: label map from create label map
    Returns:
        pd.DataFrame: converted dataFrame
    """
    for key, item in label_map.items():
        df = df.assign(**{key: item.inverse_transform(df[key])})
    return df
