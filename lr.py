#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
scikit-learnの機械学習でよく使われるものを関数化
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cross_validation import StratifiedKFold
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import LogisticRegression
import seaborn as sns
from classification import Classification


class LRClassifier(Classification):
    """使用関数のクラス化"""

    def __init__(self, dataFrame: pd.DataFrame, y_column: str, x_columns: list):
        super().__init__(dataFrame, y_column, x_columns)

    def lr_gridsearch(self, n: int = 5,
                      C_range: np.ndarray = np.r_[np.logspace(0, 2, 10), np.logspace(2, 3, 10)],
                      plot: bool = True):
        """
        ロジスティック回帰分類のgridsearch
        パラメータは、lpfgs法の場合(多クラス分類に使用)はl2ノルムしか取れない
        :param n: 交差数
        :param C_range: Cパラメータの範囲
        :param plot: whether plot or not
        :return:
        """
        parameters = {'penalty': ["l2"],
                      'C': C_range,
                      'class_weight': [None, "balanced"]}
        """
        parameters = {'penalty': ["l1", "l2"],
                      'C': C_range,
                      'class_weight': [None, "balanced"]}
        """
        cv = StratifiedKFold(self.y_values(), n_folds=n, shuffle=True)
        clf = GridSearchCV(LogisticRegression(multi_class='multinomial', solver='lbfgs'), parameters, cv=cv, n_jobs=-1)

        print("grid search...")
        clf.fit(self.x_values().as_matrix(), self.y_values().as_matrix())
        self._return_base_model = lambda: clf.best_estimator_
        scores = [x[1] for x in clf.grid_scores_]
        scores = np.array(scores).reshape(len(C_range), 2)
        self.scores = pd.DataFrame(scores, index=C_range, columns=parameters["class_weight"])
        self.scores.index.name = "C"
        self.scores.columns.name = "class_weight"
        if plot:
            sns.heatmap(self.scores, annot=True, cmap="Blues")
            sns.plt.show()
