"""
This is a boilerplate pipeline 'data_science'
generated using Kedro 0.18.3
"""

import logging

from typing import Dict, Tuple, Union
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split


def split_data(data: pd.DataFrame, parameters: Dict) -> Tuple:

    """
    Splits data into features and targets training and test sets

    Args:
        data: Data containing features and target.
        parameters: Parameters defined in parameters/data_science.yml.

    Returns:
        Split data.
    """

    X = data[parameters["features"]]
    y = data["price"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=parameters["test_size"], random_state=parameters["random_state"]
    )

    return X_train, X_test, y_train, y_test


MachineModel = Union[LinearRegression, RandomForestRegressor]

def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> MachineModel:

    """
    Trains the linear regression model.

    Args:
        X_train: Training data of independent features.
        y_train: Training data for price.
    """

    regressor = RandomForestRegressor()
    regressor.fit(X_train, y_train)

    return regressor


def evaluate_model(regressor: MachineModel, X_test: pd.DataFrame, y_test: pd.Series):

    """
    Calculates and logs the coefficient of determination

    Args:
        regressor: Trained model.
        X_test: Testing data of independent features.
        y_test: Testing dta for price.
    """

    y_pred = regressor.predict(X_test)
    score = r2_score(y_test, y_pred)

    logger = logging.getLogger(__name__)
    logger.info("Model has an R^2 coefficient of %.3f on test data", score)
