#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
回帰分析を行う
"""

import numpy as np
import pandas as pd
from pandas.tools.plotting import scatter_matrix
from pandas.tools.plotting import lag_plot
from pandas.tools.plotting import autocorrelation_plot
from sklearn import linear_model, svm
from sklearn.ensemble import RandomForestRegressor
import statsmodels.api as sm
import statsmodels.formula.api as smf
import statsmodels.stats.outliers_influence as oti
import statsmodels.tsa.api as tsa
import matplotlib.pyplot as plt
import seaborn as sns
from mldata import MlData

class Regression(MlData):
    """線形重回帰分析"""

    def __init__(self, dataFrame: pd.DataFrame, y_column: str, x_columns: list):
        super().__init__(dataFrame, y_column, x_columns)

    def linear_regression(self):
        self.rm = linear_model.LinearRegression(fit_intercept=True, normalize=False,
                                                copy_X=True, n_jobs=-1)
        """
        :param fit_intercept: Falseにすると切片を求めない(原点を通る場合に有効)
        :param normalize: Trueにすると説明変数を正規化する
        :param copy_X: メモリないでデータを複製するかどうか
        """
        self.rm.fit(self.x_values(), self.y_values())
        self.coef = pd.DataFrame({"Name": self.x_columns,
                                  "Coefficients": self.rm.coef_})
        self.intercept = self.rm.intercept_
        self.predict = lambda x: self.rm.predict(x)

    def svr(self):
        self.rm = svm.SVR(kernel='rbf', C=1, gamma=0.1)
        self.rm.fit(self.x_values(), self.y_values())
        self.predict = lambda x: self.rm.predict(x)

    def random_forest_regression(self):
        self.rm = RandomForestRegressor(n_estimators=100, criterion="mse",
                                        max_features="auto", n_jobs=-1)
        self.rm.fit(self.x_values(), self.y_values())

        features_importance_rank = np.argsort(self.rm.feature_importances_)[::-1]
        features_importance_value = self.rm.feature_importances_[features_importance_rank]
        features_importance_key = self.data[features_importance_rank].keys()
        importance = pd.DataFrame(
            {
                "key": features_importance_key,
                "value": features_importance_value
            }
        )
        sns.barplot(x='value', y='key', data=importance)
        plt.show()

    def analysis(self):
        """http://necochan.com/2014/06/07/python-for-economist-6/"""
        eq = self.y_column + "~" + "+".join(self.x_columns)
        self.rm = smf.ols(formula=eq, data=self.data).fit()
        print(self.rm.summary())
        # ADF test, H0: Non-stationary
        tsa.adfuller(self.rm.resid, regression='nc')

        # Autocorrel plot of resid
        autocorrelation_plot(self.rm.resid)  # Show ACF of residuals
        ACF_resid = tsa.acf(self.rm.resid)  # Keep ACF of residuals
        # Checking Multicolinearity by VIF
        VIF = pd.DataFrame(
            [oti.variance_inflation_factor(self.rm.model.exog, i) for i in range(1, self.rm.model.exog.shape[1])],
            index=self.rm.model.exog_names[1:], columns=['VIF'])  # VIF&gt;10 should be cared
