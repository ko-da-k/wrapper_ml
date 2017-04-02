#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
回帰分析を行う
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns
from .mldata import MlData


class Regression(MlData):
    """線形重回帰分析"""

    def __init__(self, dataFrame: pd.DataFrame, y_column: str, x_columns: list):
        super().__init__(dataFrame, y_column, x_columns)

    def cross_validation(self, cv: int = 5, scoring: str ="r2", n_jobs=-1):
        cross_val_score(self._return_base_model(), self.x_values.as_matrix(), self.y_values.as_matrix(),
                        cv=cv, scoring=scoring, n_jobs=n_jobs)
