#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.cross_validation import StratifiedKFold
from sklearn.grid_search import GridSearchCV
import seaborn as sns
from classification import Classification


class SVM(Classification):
    """使用関数のクラス化"""

    def __init__(self, dataFrame: pd.DataFrame, y_column: str, x_columns: list):
        super().__init__(dataFrame, y_column, x_columns)

    def svm_gridsearch(self, n: int = 5,
                       C_range: np.ndarray = np.r_[np.logspace(0, 2, 10), np.logspace(2, 3, 10)],
                       gamma_range: np.ndarray = np.logspace(-4, -2, 10),
                       plot: bool = False):
        """
        gridsearchを行う関数
        :param n: 交差数
        :param C_range: Cパラメータの範囲
        :param gamma_range: gammaパラメータの範囲
        :param plot: 可視化するかしないか
        :return:
        """
        tuned_parameters = [{'kernel': ['rbf'], 'gamma': gamma_range,
                             'C': C_range},
                            {'kernel': ['linear'], 'C': C_range}]

        cv = StratifiedKFold(self.y_values(), n_folds=n, shuffle=True)
        clf = GridSearchCV(SVC(probability=True, class_weight='balanced', decision_function_shape='ovr'),
                           tuned_parameters, cv=cv, n_jobs=-1)

        print("grid search...")
        clf.fit(self.x_values().as_matrix(), self.y_values().as_matrix())

        self._return_base_model = lambda: clf.best_estimator_

        scores = [x[1] for x in clf.grid_scores_[:len(C_range) * len(gamma_range)]]
        scores = np.array(scores).reshape(len(C_range), len(gamma_range))
        self.rbf_scores = pd.DataFrame(scores, index=C_range, columns=gamma_range)
        self.rbf_scores.index.name = "C"
        self.rbf_scores.columns.name = "gamma"
        self.linear_scores = [x[1] for x in clf.grid_scores_[len(C_range) * len(gamma_range):]]

        if plot:
            sns.heatmap(self.rbf_scores, annot=True, cmap="Blues")
            sns.plt.show()
            plt.plot(C_range, self.linear_scores)
            plt.xlabel("C")
            plt.ylabel("accuracy")
            plt.show()

        print(clf.best_estimator_)
        print("set classifier")
