#!/usr/bin/env python
# -*- coding: utf-8 -*-

from collections import Counter
from random import sample
from math import sqrt
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.externals.joblib import Parallel, delayed
from sklearn.utils import resample


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
    unique_list = np.unique(y_values)
    y_series = pd.Series(y_values)  # indexを取得するのにpandasの方が扱いやすいため
    # ラベルごとにindexをサンプリング
    if bootstrap:
        each_label = [resample(y_series[y_series == uq].index.tolist(), n_samples=n_samples) for uq in unique_list]
    else:
        each_label = [sample(y_series[y_series == uq].index.tolist(), n_samples) for uq in unique_list]

    return [inner for outer in each_label for inner in outer]  # 2重リストを1重に


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

        # ここからはimportanceの計算
        count = np.zeros(x_values.shape[1])  # 特徴量の出現回数集計用の初期定義
        feature_importances = np.zeros(x_values.shape[1])
        for tree, feature in zip(self.forest, self.feature_list):
            count[feature] += 1
            feature_importances[feature] += tree.feature_importances_
        self.feature_importances_ = feature_importances / count

    def predict(self, x_values: np.ndarray) -> list:
        """
        入力データのラベルを予測 1つであっても2次元arrayで
        :param x_values: 予測したいデータ
        :return: predict label
        """
        each_tree_predict = np.array([tree.predict(x_values[:, feature]) for tree, feature
                                      in zip(self.forest, self.feature_list)])

        # 転地させないとラベルごとの予測結果にならないことに注意
        return [Counter(item).most_common(1)[0][0] for item in each_tree_predict.T]
