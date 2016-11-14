#!/usr/bin/env python
# -*- coding: utf-8 -*-

from collections import Counter
from random import sample
from math import sqrt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.externals.joblib import Parallel, delayed
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.utils import resample
from tools import cr_to_df
from analysis.classification import Classification


def _build_tree(train: np.ndarray, label: np.ndarray):
    tree = DecisionTreeClassifier()
    tree.fit(train, label)
    return tree


def _sampling_equal(y_values: np.ndarray, n_samples: int, bootstrap: bool = True):
    """
    :param y_values: label data
    :param n_samples: number of samples
    :param bootstrap: whether bootstrap or not
    :return: sampling index
    """
    if bootstrap:
        return [i for v in [resample(pd.Series(y_values)[pd.Series(y_values) == uq].index,
                                     n_samples=n_samples) for uq in np.unique(y_values)] for i in v]
    else:
        return [i for v in [sample(list(pd.Series(y_values)[pd.Series(y_values) == uq].index),
                                   n_samples) for uq in np.unique(y_values)] for i in v]


class BalancedRandomForestClassifier():
    """クラスごとのサンプル数を一定にして木を作ることで不均衡データに対応"""

    def __init__(self, n_estimator: int, n_samples: int, bootstrap: bool = True, max_features: int = 0):
        """
        :param n_estimator: number of tree
        :param n_samples: number of sampling data
        :param bootstrap: how to resample
        :param max_features: number of feature
        """
        self.n_estimator = n_estimator
        self.n_samples = n_samples
        self.bootstrap = bootstrap
        self.max_features = max_features

    def fit(self, x_values: np.ndarray, y_values: np.ndarray):
        """
        :param x_values: train data
        :param y_values: label data
        """
        if self.max_features == 0:
            self.max_features = round(sqrt(x_values.shape[1]))
        # bootstrap等で木に学習させるデータを選択
        index_list = [_sampling_equal(y_values, self.n_samples, self.bootstrap) for i in range(0, self.n_estimator)]
        # 木ごとの特徴量を選択
        self.feature_list = [sample(range(0, x_values.shape[1]), self.max_features) for i in range(0, self.n_estimator)]
        # 上記に基づいて木を構築
        self.forest = Parallel(n_jobs=-1, backend="threading")(
            delayed(_build_tree)(x_values[np.ix_(index, feature)], y_values[index])
            for index, feature in zip(index_list, self.feature_list))
        # 各木の予測の多数決で決定
        self.predict = lambda x: [Counter(item).most_common(1)[0][0]
                                  for item in np.array([tree.predict(x[:,feature])
                                                        for tree,feature in zip(self.forest,self.feature_list)]).T]
        # ここからはimportanceの計算
        count = np.zeros(x_values.shape[1])
        feature_importances = np.zeros(x_values.shape[1])
        for tree, feature in zip(self.forest, self.feature_list):
            count[feature] += 1
            feature_importances[feature] += tree.feature_importances_
        self.feature_importances_ = feature_importances / count



