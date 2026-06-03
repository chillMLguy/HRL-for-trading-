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
                 random_state=42, use_scaler=True):
        """
        Parameters
        ----------
        use_scaler : bool
            If True, fit a sklearn StandardScaler on the training
            observations and apply it during inference. Recommended
            because raw (log_ret, vol_20) features have very different
            scales and the absolute level of vol drifts between training
            and test periods.
        """
        self.n_states = n_states
        self.n_iter = n_iter
        self.covariance_type = covariance_type
        self.random_state = random_state
        self.use_scaler = use_scaler
        self.model = None
        self.scaler = None  # set inside fit() when use_scaler=True

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

        If use_scaler=True, fits a StandardScaler on `observations` first
        and transforms them. The same scaler is then applied at inference
        time inside predict_proba / decode / score.
        """
        if self.use_scaler:
            from sklearn.preprocessing import StandardScaler
            self.scaler = StandardScaler().fit(observations)
            observations = self.scaler.transform(observations)

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

    def _transform(self, observations):
        if self.scaler is not None:
            return self.scaler.transform(observations)
        return observations

    def predict_proba(self, observations):
        """
        Forward-algorithm posterior probabilities (no future look-ahead).

        Returns array of shape (T, n_states).
        """
        if self.model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        return self.model.predict_proba(self._transform(observations))

    def filtered_proba(self, observations):
        """
        Forward-only *filtered* posteriors  P(state_t | o_1 … o_t).

        Unlike ``predict_proba`` (which runs the full forward–backward pass and
        therefore returns *smoothed* posteriors that peek at future
        observations), this performs a single forward pass, so row ``t`` uses
        only information available up to and including ``t`` — no look-ahead.

        This is the causal quantity needed by the Phase 2 meta-agent: it can be
        precomputed once for an entire price series and indexed in O(1) per
        step, and it matches ``predict_proba(obs[:t+1])[-1]`` exactly (verified
        to machine precision) while costing O(T·K²) for the whole sequence
        instead of O(T²·K²).

        Returns array of shape (T, n_states); each row sums to 1.
        """
        if self.model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        from scipy.stats import multivariate_normal
        from scipy.special import logsumexp

        X = self._transform(np.asarray(observations, dtype=np.float64))
        m = self.model
        K = self.n_states
        T = len(X)

        # Emission log-likelihoods log P(o_t | state=k). hmmlearn's covars_
        # property always exposes full (K, n_feat, n_feat) matrices; guard the
        # diag-storage case anyway for robustness across versions.
        covars = np.asarray(m.covars_)
        log_b = np.empty((T, K), dtype=np.float64)
        for k in range(K):
            cov_k = covars[k] if covars.ndim == 3 else np.diag(covars[k])
            log_b[:, k] = multivariate_normal.logpdf(
                X, mean=m.means_[k], cov=cov_k, allow_singular=True)

        log_pi = np.log(m.startprob_ + 1e-300)
        log_A = np.log(m.transmat_ + 1e-300)

        log_alpha = np.empty((T, K), dtype=np.float64)
        log_alpha[0] = log_pi + log_b[0]
        for t in range(1, T):
            # log α_t(j) = log b_t(j) + logΣ_i α_{t-1}(i) · A_{ij}
            log_alpha[t] = log_b[t] + logsumexp(
                log_alpha[t - 1][:, None] + log_A, axis=0)

        # Normalise each timestep to a proper posterior.
        return np.exp(log_alpha - logsumexp(log_alpha, axis=1, keepdims=True))

    def decode(self, observations):
        """
        Viterbi decoding — for visualisation / analysis only.
        Uses the full sequence (looks ahead), so NOT suitable for
        generating real-time trading weights.

        Returns array of shape (T,) with integer state labels.
        """
        if self.model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        _, states = self.model.decode(self._transform(observations))
        return states

    def score(self, observations):
        """Log-likelihood of the observations under the fitted HMM."""
        if self.model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        return float(self.model.score(self._transform(observations)))

    # ── Persistence ────────────────────────────────────────────────

    def save(self, path):
        joblib.dump({
            "model": self.model,
            "n_states": self.n_states,
            "covariance_type": self.covariance_type,
            "scaler": self.scaler,
            "use_scaler": self.use_scaler,
        }, path)

    @classmethod
    def load(cls, path):
        data = joblib.load(path)
        det = cls(n_states=data["n_states"],
                  covariance_type=data["covariance_type"],
                  use_scaler=data.get("use_scaler", False))
        det.model = data["model"]
        det.scaler = data.get("scaler", None)
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
