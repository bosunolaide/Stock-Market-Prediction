from __future__ import annotations

import numpy as np
from sklearn.preprocessing import StandardScaler


class MultiDimensionScaler:
    
    def __init__(self):
        self.scalers = []

    def fit(self, X):
        self.scalers = []
        for i in range(X.shape[2]):
            s = MinMaxScaler()
            s.fit(X[:, :, i])
            self.scalers.append(s)
        return self

    def transform(self, X):
        for i in range(X.shape[2]):
            X[:, :, i] = self.scalers[i].transform(X[:, :, i])
        return X
