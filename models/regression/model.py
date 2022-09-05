from typing import Any, Dict, Callable
import pandas as pd
from models.regression.load_data import load_data
from models.regression.feature_engineering import feature_engineering
from models.regression.training import training
from models.regression.prediction import prediction


class BaselineRegressionModel:
    """
    Define MLprocess in general
    Attributes:
        ld_input (Dict[str, Any]): load input
        fe_input (Dict[str, Any]): feature engineering inputs
        tr_input (Dict[str, Any]): training inputs
    Methods:
        fit(inputs, output): run entire process and return transformed (data, outputs, model)
        predict(inputs): return predicted value with inputs
    """

    def __init__(
        self,
        ld_input: Dict[str, Any],
        fe_input: Dict[str, Any],
        tr_input: Dict[str, Any],
    ):
        self.ld_input = ld_input
        self.fe_input = fe_input
        self.tr_input = tr_input

        self.target_cols = fe_input["target_cols"]
        self.train_df, self.test_df = load_data(ld_input)

    def _feature_engineering(self) -> Dict[str, Any]:
        """
        Feature engineering
        Returns:
            Dict[str, Any]
        """
        _fe_output_dict = feature_engineering(
            train_df=self.train_df,
            test_df=self.test_df,
            category_cols=self.fe_input["category_cols"],
            target_cols=self.fe_input["target_cols"],
        )
        self.train_refined_df = _fe_output_dict["train_df"]
        self.test_refined_df = _fe_output_dict["test_df"]
        self.label_map = _fe_output_dict["label_map"]
        return _fe_output_dict

    def _training(self) -> Dict[str, Callable]:
        """
        Trainining model
        Returns:
            Dict[str, Callable]
        """
        self.mdls = training(
            train_df=self.train_refined_df,
            target_cols=self.target_cols,
            frac_ratio=self.tr_input["frac_ratio"],
            id_col=self.tr_input["id_col"],
            hp_tune_trials=self.tr_input["hp_tune_trials"],
        )
        return self.mdls

    def _prediction(self) -> Dict[str, pd.Series]:
        """
        Predict for given inputs including new data
        Returns:
            Dict[str, pd.Series]: predicted value
        """
        self.preds = prediction(test_df=self.test_refined_df, mdls=self.mdls)
        return self.preds

    def fit(
        self,
    ) -> Dict[str, Any]:
        """
        Model fit
        Returns:
            Dict[str, Any]
        """
        fe_output = self._feature_engineering(self)
        tr_output = self._training(self)
        pr_output = self._prediction(self)
        return {
            "fe_output": fe_output,
            "tr_output": tr_output,
            "pr_output": pr_output,
        }
