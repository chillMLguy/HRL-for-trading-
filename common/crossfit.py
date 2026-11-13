"""
Cross-fitted meta training: the shared bookkeeping.

The meta-agent has to learn from what the agents do on data they have NOT seen,
because that is all it will ever get in the test year.  So the last part of the
training window is cut into consecutive blocks.  Each block's proposals come
from agents (and an HMM) fitted only on the data before that block:

    bars:  [ history .................. | block 1 | block 2 | block 3 | block 4 ]
    fold 1 fits on  [ history ]           proposes on block 1
    fold 2 fits on  [ history + block 1 ]           proposes on block 2
    ...

The blocks joined together are the out-of-sample proposal stream.  train_crossfit.py
writes it to <outdir>/crossfit/proposals.csv and train_phase2.py --crossfit trains
on it: every block but the last for training, the last one for validation.
"""
import os

import numpy as np
import pandas as pd


def fold_edges(n_bars, folds, min_train_frac):
    """
    [(start, end), ...] bar indices of each proposal block.  Blocks are
    consecutive, cover the part of the window after the first min_train_frac,
    and a fold fits only on bars before its own start.
    """
    if folds < 2:
        raise ValueError("Cross-fitting needs at least 2 folds: the last block "
                         "is validation, the others are training.")
    if not 0.0 < min_train_frac < 1.0:
        raise ValueError("min_train_frac must be between 0 and 1")
    first = int(n_bars * min_train_frac)
    edges = np.linspace(first, n_bars, folds + 1).astype(int)
    return [(int(edges[i]), int(edges[i + 1])) for i in range(folds)]


def stream_path(rundir):
    return os.path.join(rundir, "crossfit", "proposals.csv")


def load_stream(rundir):
    """The proposal stream, indexed by the date each decision was made."""
    path = stream_path(rundir)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{path} not found. Run train_crossfit.py first.")
    return pd.read_csv(path, parse_dates=["date"]).set_index("date").sort_index()


def _align(stream, prices, columns):
    """Stream columns aligned to the bars of `prices`; NaN where it has no row."""
    if not columns:
        return None
    return stream[columns].reindex(prices.index).values.astype(np.float64)


def stream_arrays(stream, prices, agent_order):
    """(proposals, hmm_posteriors) as arrays with one row per bar of `prices`."""
    proposals = _align(stream, prices, [f"a_{a}" for a in agent_order])
    hmm = _align(stream, prices, [c for c in stream.columns if c.startswith("hmm_")])
    return proposals, hmm


def stream_windows(stream, prices, warm_bars):
    """
    The training and validation slices of the stream.  Each slice of `prices`
    starts warm_bars early, so features are warm when trading starts, and runs
    one bar past its last decision, so that decision has a return to earn.

    Returns {"train": (prices_slice, stream_slice, start_index), "val": ...}.
    """
    last_fold = stream["fold"].max()
    parts = {"train": stream[stream["fold"] < last_fold],
             "val": stream[stream["fold"] == last_fold]}
    out = {}
    for name, part in parts.items():
        first = prices.index.searchsorted(part.index[0])
        last = prices.index.searchsorted(part.index[-1])
        lo = max(0, first - warm_bars)
        out[name] = (prices.iloc[lo: min(last + 2, len(prices))], part, first - lo)
    return out
