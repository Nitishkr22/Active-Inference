"""
compression_configs.py — Canonical ablation configs for the compression study.

Each entry maps a config name to kwargs for CompressedModelConfig.
Import CONFIGS here rather than hardcoding dims in multiple scripts.

Naming scheme:
  CL<N>       — standard CNN encoder, compression level N
  CL<N>-DW    — depthwise-separable encoder at same dim as CL<N>

Research questions each group answers:
  CL0 vs CL1-CL5  →  Pareto curve: how much can we compress before accuracy drops?
  CL1 vs CL1-DW   →  Does depthwise encoder help at same dim budget?
  CL2 vs CL3      →  Does context_dim (32→8) matter at small gru/feat?
"""
from __future__ import annotations

CONFIGS: dict[str, dict] = {
    # ---- Primary compression axis ----
    "CL0": dict(gru_hidden_dim=256, encoder_feat_dim=128, context_dim=32, use_depthwise=False),
    "CL1": dict(gru_hidden_dim=128, encoder_feat_dim=64,  context_dim=16, use_depthwise=False),
    "CL2": dict(gru_hidden_dim=64,  encoder_feat_dim=32,  context_dim=16, use_depthwise=False),
    "CL3": dict(gru_hidden_dim=64,  encoder_feat_dim=32,  context_dim=8,  use_depthwise=False),
    "CL4": dict(gru_hidden_dim=32,  encoder_feat_dim=16,  context_dim=8,  use_depthwise=False),
    "CL5": dict(gru_hidden_dim=16,  encoder_feat_dim=8,   context_dim=4,  use_depthwise=False),

    # ---- Depthwise encoder variants ----
    "CL1-DW": dict(gru_hidden_dim=128, encoder_feat_dim=64, context_dim=16, use_depthwise=True),
    "CL2-DW": dict(gru_hidden_dim=64,  encoder_feat_dim=32, context_dim=16, use_depthwise=True),

    # ---- Context-only ablations (GRU/feat fixed at CL2, context varies) ----
    "CL2-C32": dict(gru_hidden_dim=64, encoder_feat_dim=32, context_dim=32, use_depthwise=False),
    "CL2-C8":  dict(gru_hidden_dim=64, encoder_feat_dim=32, context_dim=8,  use_depthwise=False),
    "CL2-C4":  dict(gru_hidden_dim=64, encoder_feat_dim=32, context_dim=4,  use_depthwise=False),
}

# Shorthand for the primary sweep (train these first)
PRIMARY_SWEEP = ["CL0", "CL1", "CL2", "CL3", "CL4", "CL5"]

# Full sweep including architecture variants
FULL_SWEEP = list(CONFIGS.keys())
