"""
sweep_configs.py — Automated training sweep over compression configs.

Usage:
    # Print Pareto table (estimated params only, no training)
    python sweep_configs.py --dry_run

    # Train the primary 6-level sweep (CL0 through CL5)
    python sweep_configs.py --sweep primary

    # Train a single config
    python sweep_configs.py --config CL1

    # Train full sweep including depthwise variants
    python sweep_configs.py --sweep full

After training, collect results:
    python sweep_configs.py --collect_results
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch

from compression_configs import CONFIGS, PRIMARY_SWEEP, FULL_SWEEP
from model_compressed import CompressedModelConfig, WorldModelCompressed


# ============================================================
# Pareto table (no training required)
# ============================================================

def print_pareto_table():
    """Print estimated param counts and sizes for all configs."""
    print(f"\n{'Config':<10} | {'gru':>4} | {'feat':>4} | {'ctx':>3} | {'DW':>2} | "
          f"{'TotalP':>9} | {'InfP':>9} | {'InfMB':>6} | {'Compression':>11}")
    print("-" * 85)

    baseline_inf = None
    for name in FULL_SWEEP:
        kwargs = dict(CONFIGS[name])
        cfg    = CompressedModelConfig(**kwargs)
        m      = WorldModelCompressed(cfg)
        if baseline_inf is None:
            baseline_inf = m.inference_param_count
        ratio = baseline_inf / m.inference_param_count
        print(
            f"{name:<10} | {cfg.gru_hidden_dim:>4} | {cfg.encoder_feat_dim:>4} | "
            f"{cfg.context_dim:>3} | {'Y' if cfg.use_depthwise else 'N':>2} | "
            f"{m.total_param_count:>9,} | {m.inference_param_count:>9,} | "
            f"{m.model_size_mb:>6.3f} | {ratio:>8.1f}×"
        )
    print()


# ============================================================
# Single training run
# ============================================================

def train_config(config_name: str, epochs: int = 40, seed: int = 42) -> bool:
    """Launch train_compressed.py as a subprocess. Returns True on success."""
    cmd = [
        sys.executable, "train_compressed.py",
        "--config", config_name,
        "--epochs", str(epochs),
        "--seed", str(seed),
    ]
    print(f"\n{'='*60}")
    print(f"Training: {config_name}  (epochs={epochs}, seed={seed})")
    print(f"Command:  {' '.join(cmd)}")
    print(f"{'='*60}\n")

    result = subprocess.run(cmd, cwd=Path(__file__).parent)
    return result.returncode == 0


# ============================================================
# Run eval on all trained checkpoints
# ============================================================

def collect_results(sweep: list[str] = PRIMARY_SWEEP) -> None:
    """Print a summary table from all training_summary.json files."""
    print(f"\n{'Config':<10} | {'InfP':>9} | {'InfMB':>6} | {'BestLoss':>9} | {'BestJoint':>9} | {'TrainMin':>8}")
    print("-" * 70)
    for name in sweep:
        p = Path("checkpoints") / name / "training_summary.json"
        if not p.exists():
            print(f"{name:<10} | NOT TRAINED")
            continue
        d = json.loads(p.read_text())
        print(
            f"{name:<10} | {d.get('inference_params', 0):>9,} | "
            f"{d.get('inference_size_mb', 0):>6.3f} | "
            f"{d.get('best_val_loss', 0):>9.5f} | "
            f"{d.get('best_val_joint_acc', 0):>9.5f} | "
            f"{d.get('total_training_min', 0):>8.1f}"
        )
    print()


def collect_eval_results(results_dir: str = "./results") -> None:
    """Print evaluation summary from all *_summary.json in results/."""
    results_path = Path(results_dir)
    jsons = sorted(results_path.glob("*_summary.json"))
    if not jsons:
        print(f"No eval summaries found in {results_dir}")
        return

    print(f"\n{'Config':<20} | {'InfP':>9} | {'MB':>5} | {'Succ%':>6} | "
          f"{'Coll%':>5} | {'ms/step':>7} | {'BelErr%':>7}")
    print("-" * 80)
    for j in jsons:
        d  = json.loads(j.read_text())
        c  = d.get("config", {})
        sr = d.get("success_rate", 0)
        cr = d.get("collision_rate", 0)
        ms = d.get("mean_ms_per_step", 0)
        be = d.get("mean_belief_error_rate", 0)
        print(
            f"{j.stem:<20} | {c.get('inference_params',0):>9,} | "
            f"{c.get('inference_size_mb',0):>5.2f} | "
            f"{sr*100:>5.1f}% | {cr*100:>4.1f}% | "
            f"{ms:>7.2f} | {be*100:>6.1f}%"
        )
    print()


# ============================================================
# Main
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(description="Compression sweep manager")
    p.add_argument("--dry_run",         action="store_true", help="Print Pareto table only")
    p.add_argument("--sweep",           type=str,  default=None,
                   choices=["primary", "full"],    help="Run a predefined sweep")
    p.add_argument("--config",          type=str,  default=None, help="Train a single config")
    p.add_argument("--epochs",          type=int,  default=40)
    p.add_argument("--seed",            type=int,  default=42)
    p.add_argument("--collect_results", action="store_true", help="Print training result table")
    p.add_argument("--collect_eval",    action="store_true", help="Print eval result table")
    p.add_argument("--results_dir",     type=str,  default="./results")
    return p.parse_args()


def main():
    args = parse_args()

    if args.dry_run:
        print_pareto_table()
        return

    if args.collect_results:
        sweep = PRIMARY_SWEEP if not args.sweep else (PRIMARY_SWEEP if args.sweep == "primary" else FULL_SWEEP)
        collect_results(sweep)
        return

    if args.collect_eval:
        collect_eval_results(args.results_dir)
        return

    if args.config:
        success = train_config(args.config, epochs=args.epochs, seed=args.seed)
        sys.exit(0 if success else 1)

    if args.sweep:
        sweep = PRIMARY_SWEEP if args.sweep == "primary" else FULL_SWEEP
        print(f"\nStarting {'primary' if args.sweep == 'primary' else 'full'} sweep: {sweep}")
        print_pareto_table()

        failed = []
        for name in sweep:
            ok = train_config(name, epochs=args.epochs, seed=args.seed)
            if not ok:
                failed.append(name)
                print(f"[FAILED] {name}")

        print(f"\nSweep complete. Failed: {failed if failed else 'none'}")
        collect_results(sweep)
        sys.exit(0 if not failed else 1)

    # Default: just print the table
    print_pareto_table()
    print("Use --sweep primary|full to start training, or --config CL1 for a single run.")


if __name__ == "__main__":
    main()
