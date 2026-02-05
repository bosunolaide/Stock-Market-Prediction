from __future__ import annotations

import numpy as np
from sklearn.preprocessing import StandardScaler


class MultiDimensionScaler:
    """A simple scaler for 3D tensors shaped (batch, timesteps, features).

    It works by flattening the first two dims into one, fitting a StandardScaler
    over the feature dimension, then reshaping back on transform/inverse_transform.

    This matches a very common pattern used in sequence models and also allows us to
    load legacy pickles that referenced MultiDimensionScaler.
    """

    def __init__(self):
        self.scaler = StandardScaler()
        self.n_features_: int | None = None

    def fit(self, X: np.ndarray):
        X = np.asarray(X)
        if X.ndim != 3:
            raise ValueError(f"Expected 3D array (batch,timesteps,features), got shape={X.shape}")
        b, t, f = X.shape
        self.n_features_ = f
        X2 = X.reshape(b * t, f)
        self.scaler.fit(X2)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X)
        if X.ndim != 3:
            raise ValueError(f"Expected 3D array, got shape={X.shape}")
        b, t, f = X.shape
        if self.n_features_ is None:
            # allow transform for legacy pickles if set on scaler
            self.n_features_ = f
        X2 = X.reshape(b * t, f)
        X2t = self.scaler.transform(X2)
        return X2t.reshape(b, t, f)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X)
        if X.ndim != 3:
            raise ValueError(f"Expected 3D array, got shape={X.shape}")
        b, t, f = X.shape
        X2 = X.reshape(b * t, f)
        X2t = self.scaler.inverse_transform(X2)
        return X2t.reshape(b, t, f)

    # pickle compatibility: expose get_params/set_params-like shape
    def __getstate__(self):
        return {"scaler": self.scaler, "n_features_": self.n_features_}

    def __setstate__(self, state):
        self.scaler = state.get("scaler", StandardScaler())
        self.n_features_ = state.get("n_features_", None)
