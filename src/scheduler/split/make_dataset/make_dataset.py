# -*- coding: utf-8 -*-
from typing import Tuple
import pandas as pd
from sklearn.model_selection import train_test_split


def read_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def split_train_val_data(
    data: pd.DataFrame, test_size: float
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_data, val_data = train_test_split(data, test_size=test_size,)

    return train_data, val_data
