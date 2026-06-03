#!/usr/bin/env bash
# Phase 2 walk-forward: train + evaluate + plot the meta-RL allocator on both
# splits. Phase 0 sub-agents and the Phase 1 HMM must already exist under
# split1/models and split2/models (they do). CNN auto-disables per split
# (no split*/models/cnn_features), matching how those sub-agents were trained.
set -e
source venv/bin/activate

COST=0.0005   # matches split*/models/phase1/config.json

echo "######## SPLIT 1 — train 2011-2021 -> test 2022 (bear) ########"
python -u train_phase2.py    --train_start 2011-01-01 --train_end 2021-12-31 \
    --modeldir split1 --outdir split1 --cost_pct $COST --full
python -u evaluate_phase2.py --test_start 2022-01-01 --test_end 2022-12-31 \
    --modeldir split1 --outdir split1 --cost_pct $COST
python -u plot_phase2.py --datadir split1

echo "######## SPLIT 2 — train 2013-2023 -> test 2024 (bull) ########"
python -u train_phase2.py    --train_start 2013-01-01 --train_end 2023-12-31 \
    --modeldir split2 --outdir split2 --cost_pct $COST --full
python -u evaluate_phase2.py --test_start 2024-01-01 --test_end 2024-12-31 \
    --modeldir split2 --outdir split2 --cost_pct $COST
python -u plot_phase2.py --datadir split2

echo "######## WALK-FORWARD COMPLETE ########"
