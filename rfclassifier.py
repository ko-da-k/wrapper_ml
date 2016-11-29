#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
from .classification import Classification
from .balanced_random_forest import BalancedRandomForestClassifier


class RFClassifier(Classification):
    """RandomForestによる分類"""

    def __init__(self, dataFrame: pd.DataFrame, y_column: str, x_columns: list):
        super().__init__(dataFrame, y_column, x_columns)

    def _return_base_model(self):
        return RandomForestClassifier(n_estimators=1000, n_jobs=-1, oob_score=True,
                                      class_weight="balanced")

    def learn(self, isplot=True):
        self._clf = self._return_base_model()
        self._clf.fit(self.x_values().as_matrix(), self.y_values().as_matrix())
        self.predict = lambda x: self._clf.predict(x)

        self.importance = pd.DataFrame({"key": self.x_columns,
                                        "value": self._clf.feature_importances_})
        if isplot:
            sns.barplot(x='value', y='key', data=self.importance)
            plt.show()


class BRFClassifier(RFClassifier):
    """RandomForestによる分類"""

    def __init__(self, dataFrame: pd.DataFrame, y_column: str, x_columns: list,
                 n_estimator: int, n_samples: int, bootstrap: bool = True):
        """
        :param n_estimator: number of tree
        :param n_samples: number of sampling data
        :param bootstrap: how to resample
        """
        super().__init__(dataFrame, y_column, x_columns)
        self.n_estimator = n_estimator
        self.n_samples = n_samples
        self.bootstrap = bootstrap

    def _return_base_model(self):
        """
        :param n_estimator: number of tree
        :param n_sample: number of sampling data
        :param bootstrap: how to resample
        :return: BRF model
        """
        return BalancedRandomForestClassifier(self.n_estimator, self.n_samples, self.bootstrap)
