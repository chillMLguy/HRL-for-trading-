"""
Network size, weight decay and training length, shared by every training script.

    net_arch      hidden layers of the actor and of both critics, e.g. [32, 32];
                  [] makes both linear in the inputs
    weight_decay  decoupled weight decay (AdamW) on the actor and the critics;
                  0 is plain Adam, which is what runs before R19 used
    passes        training steps = passes × training bars, so every training day
                  is seen about `passes` times whatever the window length

A run records all three, and the resulting steps, in each model's config.json.
"""
import argparse

import torch

import config


def parse_net_arch(text):
    """'32,32' → [32, 32];  'linear' or '' → []."""
    if isinstance(text, (list, tuple)):
        return [int(h) for h in text]
    text = str(text).strip().lower()
    if text in ("", "linear", "none", "[]"):
        return []
    try:
        arch = [int(h) for h in text.replace("[", "").replace("]", "").split(",")
                if h.strip()]
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"--net_arch takes sizes separated by commas, e.g. 32,32, or 'linear'; "
            f"got {text!r}")
    if any(h <= 0 for h in arch):
        raise argparse.ArgumentTypeError("--net_arch sizes must be positive")
    return arch


def arch_label(arch):
    return "linear" if not arch else "[" + ",".join(str(h) for h in arch) + "]"


def add_training_args(parser, default_arch, default_passes):
    """--net_arch, --weight_decay, --passes, --steps, --quick on one script."""
    g = parser.add_argument_group("network and training length")
    g.add_argument("--net_arch", type=parse_net_arch, default=list(default_arch),
                   help=f"Hidden layers, e.g. 32,32 or 64,64, or 'linear' "
                        f"(default {arch_label(default_arch)})")
    g.add_argument("--weight_decay", type=float, default=config.WEIGHT_DECAY,
                   help=f"AdamW weight decay, 0 = plain Adam "
                        f"(default {config.WEIGHT_DECAY})")
    g.add_argument("--passes", type=float, default=default_passes,
                   help=f"Training steps = passes × training bars "
                        f"(default {default_passes})")
    g.add_argument("--steps", type=int, default=0,
                   help="Fixed number of training steps instead of --passes")
    g.add_argument("--quick", action="store_true",
                   help=f"{config.QUICK_PASSES} passes: checks that the pipeline "
                        f"runs, not a result")
    return g


def training_steps(n_bars, passes, steps=0, quick=False):
    """Steps for `n_bars` training bars: --steps wins, then --quick, then --passes."""
    if steps and steps > 0:
        return int(steps), "fixed steps"
    p = config.QUICK_PASSES if quick else passes
    if p <= 0:
        raise SystemExit("--passes must be positive")
    return max(config.MIN_STEPS, int(round(p * n_bars))), \
        f"{p:g} passes{' (quick)' if quick else ''}"


def sac_policy_kwargs(net_arch, weight_decay):
    """SB3 policy_kwargs: the network size and AdamW for the actor and critics."""
    return dict(net_arch=list(net_arch), optimizer_class=torch.optim.AdamW,
                optimizer_kwargs=dict(weight_decay=float(weight_decay)))


def training_settings(net_arch, weight_decay, timesteps, n_bars, how):
    """What config.json records about the network and the training length."""
    return {"net_arch": list(net_arch), "weight_decay": float(weight_decay),
            "optimizer": "AdamW", "timesteps": int(timesteps),
            "train_bars": int(n_bars), "passes": round(timesteps / max(n_bars, 1), 2),
            "budget": how}


def forward_args(args):
    """The same settings as command-line arguments for a child script."""
    out = ["--net_arch", ",".join(str(h) for h in args.net_arch) or "linear",
           "--weight_decay", str(args.weight_decay), "--passes", str(args.passes)]
    if args.steps:
        out += ["--steps", str(args.steps)]
    if args.quick:
        out.append("--quick")
    return out
