from __future__ import annotations

import numpy as np


class MultiDimensionScaler:
    """Scale 3D time-series tensors feature-wise.

    Input shape: (batch, timesteps, n_features)

    This project has *legacy* scaler artifacts produced with very old scikit-learn.
    Some of those scalers were (incorrectly) fitted with `n_features_in_ == timesteps`
    (i.e., treating each timestep as a feature). Newer, correct behavior is fitting
    each feature on a single column vector (batch*timesteps, 1).

    This class supports both at inference time:
      - If a per-feature scaler expects 1 feature -> we reshape to (-1, 1)
      - If it expects `timesteps` features -> we pass (batch, timesteps) directly
    """

    def __init__(self):
        self.scalers: list[object] = []

    def fit(self, X: np.ndarray):
        """Fit one scaler per feature, correctly as a single column."""
        from sklearn.preprocessing import MinMaxScaler  # local import (lighter + avoids pickle issues)

        X = np.asarray(X)
        if X.ndim != 3:
            raise ValueError(
                f"MultiDimensionScaler.fit expects 3D array (batch,timesteps,features), got shape {X.shape}"
            )

        self.scalers = []
        n_features = X.shape[2]
        for i in range(n_features):
            s = MinMaxScaler()
            col = X[:, :, i].reshape(-1, 1)
            s.fit(col)
            self.scalers.append(s)
        return self

    def _expected_n_features(self, scaler: object) -> int | None:
        # sklearn >=0.24 provides n_features_in_
        nfi = getattr(scaler, "n_features_in_", None)
        if isinstance(nfi, (int, np.integer)):
            return int(nfi)

        # fallback: infer from learned params if present
        scale_ = getattr(scaler, "scale_", None)
        if isinstance(scale_, np.ndarray) and scale_.ndim == 1 and scale_.size > 0:
            return int(scale_.size)

        min_ = getattr(scaler, "min_", None)
        if isinstance(min_, np.ndarray) and min_.ndim == 1 and min_.size > 0:
            return int(min_.size)

        return None

    def transform(self, X: np.ndarray):
        X = np.asarray(X)
        if X.ndim != 3:
            raise ValueError(
                f"MultiDimensionScaler.transform expects 3D array (batch,timesteps,features), got shape {X.shape}"
            )
        n_features = X.shape[2]
        if not self.scalers:
            raise ValueError("MultiDimensionScaler has no fitted scalers. Call .fit() first.")
        if len(self.scalers) != n_features:
            raise ValueError(
                f"Feature mismatch: got {n_features} features but feature_scaler has {len(self.scalers)} scalers."
            )

        X_out = X.astype(np.float32, copy=True)
        b, t, _ = X_out.shape

        for i, scaler in enumerate(self.scalers):
            # Compatibility for ancient MinMaxScaler pickles (<0.24) where 'clip' may be missing
            if scaler.__class__.__name__ == "MinMaxScaler" and not hasattr(scaler, "clip"):
                try:
                    setattr(scaler, "clip", False)
                except Exception:
                    pass

            expected = self._expected_n_features(scaler)
            col2d = X_out[:, :, i]  # shape (batch, timesteps)

            if expected is None or expected == 1:
                # correct, per-feature scaling
                col = col2d.reshape(-1, 1)  # (batch*timesteps, 1)
                col_t = scaler.transform(col)
                X_out[:, :, i] = col_t.reshape(b, t)
            elif expected == t:
                # legacy: scaler was fitted treating each timestep as a feature
                col_t = scaler.transform(col2d)
                X_out[:, :, i] = col_t
            else:
                raise ValueError(
                    f"Scaler for feature index {i} expects {expected} features, but got timesteps={t}. "
                    "This indicates incompatible scaler artifacts. Rebuild scalers using the training pipeline."
                )

        return X_out
