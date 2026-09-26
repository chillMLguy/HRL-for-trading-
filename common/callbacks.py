"""
F10 — model selection on validation SHARPE, not shaped reward.

The stock EvalCallback picks best_model by mean shaped reward over a single
deterministic episode, which (a) is not the metric the thesis reports and
(b) is very noisy.  SharpeEvalCallback rolls the current policy over the
validation window deterministically, computes the annualized Sharpe of the
realized net returns, and keeps the best checkpoint by that number.
"""
import os

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


def rollout_net_returns(model, env, deterministic=True):
    """Roll `model` over one full episode of `env`; return realized net rets."""
    obs, _ = env.reset()
    rets = []
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, _, done, truncated, info = env.step(action)
        rets.append(info["net_ret"])
        done = done or truncated
    return np.asarray(rets, dtype=np.float64)


class SharpeEvalCallback(BaseCallback):
    def __init__(self, eval_env, ann=252, eval_freq=10_000,
                 best_model_save_path=None, verbose=0):
        super().__init__(verbose)
        self.eval_env = eval_env            # plain gym env in eval_mode
        self.ann = ann
        self.eval_freq = max(1, int(eval_freq))
        self.save_path = best_model_save_path
        self.best_sharpe = -np.inf
        self.history = []

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq != 0:
            return True
        rets = rollout_net_returns(self.model, self.eval_env)
        sd = rets.std()
        sharpe = float(np.sqrt(self.ann) * rets.mean() / sd) if sd > 1e-12 \
            else 0.0
        total = float(np.prod(1.0 + rets) - 1.0)
        self.history.append((self.num_timesteps, sharpe, total))
        self.logger.record("eval/val_sharpe", sharpe)
        self.logger.record("eval/val_total_ret", total)
        if self.verbose:
            print(f"    [eval @ {self.num_timesteps:>8,}] "
                  f"val Sharpe {sharpe:+.3f}  total {total:+.2%}")
        if sharpe > self.best_sharpe:
            self.best_sharpe = sharpe
            if self.save_path is not None:
                os.makedirs(self.save_path, exist_ok=True)
                self.model.save(os.path.join(self.save_path, "best_model"))
        return True
