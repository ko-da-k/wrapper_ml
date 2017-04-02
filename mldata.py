#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd


class MlData:
    """データセット"""

    def __init__(self, dataFrame: pd.DataFrame, y_column: str, x_columns: list):
        """
        :param dataFrame: データセット
        :param y_column:  目的変数
        :param x_columns: 説明変数,単回帰でもlist
        """
        self.data = dataFrame
        self.x_columns = x_columns
        self.y_column = y_column
        self.x_values = self.data[self.x_columns]
        self.y_values = self.data[self.y_column]

    def set_x_columns(self, x_columns: list):
        """
        カラム名を更新したときに参照する値も更新する
        :param x_columns: 新しい説明変数,単回帰でもlist
        """
        self.x_columns = x_columns
        self.x_values = self.data[self.x_columns]

    def set_y_column(self, y_column: str):
        """
        カラム名を更新した時に参照する値も更新する
        :param y_column: 目的変数
        """
        self.y_column = y_column
        self.y_values = self.data[self.y_column]

    def _return_base_model(self):
        """CVなどに用いるためのbase model(学習させていないパラメータのみのモデル)を返す"""
        raise NotImplementedError()

    def learn_all_data(self):
        """DataFrameのすべてのデータを学習させる"""
        self._clf = self._return_base_model()
        self._clf.fit(self.x_values.as_matrix(), self.y_values.as_matrix())
        self.predict = lambda x: self._clf.predict(x)
