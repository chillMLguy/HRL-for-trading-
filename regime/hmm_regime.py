"""
HMM-based market regime detector.

Fits a Gaussian HMM on 2D observations (log returns, rolling 20-day vol).
States are sorted by increasing emission volatility after fitting, so:
  state 0 = lowest-vol regime  →  calm / trending
  state 1 = mid-vol regime     →  normal
  state 2 = highest-vol regime →  crisis / turbulent
"""

import numpy as np
import joblib
from hmmlearn.hmm import GaussianHMM


class HMMRegimeDetector:

    def __init__(self, n_states=3, n_iter=200, covariance_type="full",
                 random_state=42):
        self.n_states = n_states
        self.n_iter = n_iter
        self.covariance_type = covariance_type
        self.random_state = random_state
        self.model = None

    # ── Feature construction ───────────────────────────────────────

    @staticmethod
    def build_observations(prices):
        """
        Build the HMM input matrix from a price series.

        Parameters
        ----------
        prices : pd.Series or np.ndarray of close prices

        Returns
        -------
        obs : np.ndarray of shape (T, 2)
            Column 0 = daily log returns, column 1 = 20-day rolling vol
            (annualised with sqrt(252)).
        valid_start : int
            Index into the *original* price array where valid obs begins.
            (Accounts for the 1-bar return lag + 19-bar rolling window.)
        """
        import pandas as pd
        p = pd.Series(prices.values if hasattr(prices, "values")
                       else np.asarray(prices, dtype=np.float64))
        log_ret = np.log(p / p.shift(1))
        roll_vol = log_ret.rolling(20).std() * np.sqrt(252)

        # First valid index = 20 (1 for return + 19 more for rolling window)
        valid_mask = ~(log_ret.isna() | roll_vol.isna())
        valid_start = int(valid_mask.values.argmax())  # first True

        obs = np.column_stack([
            log_ret.values[valid_start:],
            roll_vol.values[valid_start:],
        ]).astype(np.float64)

        return obs, valid_start

    # ── Fitting ────────────────────────────────────────────────────

    def fit(self, observations):
        """
        Fit HMM and sort states by increasing emission volatility.
        """
        model = GaussianHMM(
            n_components=self.n_states,
            covariance_type=self.covariance_type,
            n_iter=self.n_iter,
            random_state=self.random_state,
            verbose=False,
        )
        model.fit(observations)

        # Sort states by the mean rolling-vol dimension (column 1)
        order = np.argsort(model.means_[:, 1])
        model.means_ = model.means_[order]
        model.startprob_ = model.startprob_[order]
        model.transmat_ = model.transmat_[order][:, order]

        if self.covariance_type == "full":
            model.covars_ = model.covars_[order]
        elif self.covariance_type == "diag":
            model.covars_ = model.covars_[order]
        elif self.covariance_type == "spherical":
            model.covars_ = model.covars_[order]
        # tied: single matrix, no reorder needed

        self.model = model
        return self

    # ── Inference ──────────────────────────────────────────────────

    def predict_proba(self, observations):
        """
        Forward-algorithm posterior probabilities (no future look-ahead).

        Returns array of shape (T, n_states).
        """
        if self.model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        return self.model.predict_proba(observations)

    def decode(self, observations):
        """
        Viterbi decoding — for visualisation / analysis only.
        Uses the full sequence (looks ahead), so NOT suitable for
        generating real-time trading weights.

        Returns array of shape (T,) with integer state labels.
        """
        if self.model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        _, states = self.model.decode(observations)
        return states

    # ── Persistence ────────────────────────────────────────────────

    def save(self, path):
        joblib.dump({
            "model": self.model,
            "n_states": self.n_states,
            "covariance_type": self.covariance_type,
        }, path)

    @classmethod
    def load(cls, path):
        data = joblib.load(path)
        det = cls(n_states=data["n_states"],
                  covariance_type=data["covariance_type"])
        det.model = data["model"]
        return det

    # ── Diagnostics ────────────────────────────────────────────────

    def summary(self):
        """Print interpretable summary of the fitted model."""
        if self.model is None:
            print("Model not fitted yet.")
            return
        m = self.model
        print(f"\n{'═'*60}")
        print(f"  HMM Summary — {self.n_states} states, "
              f"cov_type={self.covariance_type}")
        print(f"{'═'*60}")
        for s in range(self.n_states):
            label = ["low-vol", "mid-vol", "high-vol"][s] \
                if self.n_states == 3 else f"state-{s}"
            mu_ret, mu_vol = m.means_[s]
            print(f"  State {s} ({label}):  "
                  f"mean_ret={mu_ret:+.5f}  mean_vol={mu_vol:.4f}")
        print(f"\n  Transition matrix:")
        for row in m.transmat_:
            print(f"    [{', '.join(f'{v:.3f}' for v in row)}]")
        diag = np.diag(m.transmat_)
        print(f"  Diagonal (persistence): "
              f"[{', '.join(f'{v:.3f}' for v in diag)}]")
        if np.any(diag < 0.7):
            print("  ⚠  WARNING: Some states are not persistent "
                  "(diagonal < 0.7). Consider fewer states.")
        print(f"{'═'*60}\n")
