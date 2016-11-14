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
        self.x_values = lambda: self.data[self.x_columns]
        self.y_values = lambda: self.data[self.y_column]
