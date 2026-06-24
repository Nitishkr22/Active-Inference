"""
world_model_training/config.py — Hyperparameters for RSSM world model training.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class EncoderConfig:
    rgb_in_channels: int = 3
    depth_in_channels: int = 1
    img_size: int = 64           # resize input to this before CNN
    conv_channels: Tuple[int, ...] = (32, 64, 128, 256)
    z_dim: int = 256             # stochastic latent dimension (μ, logvar each z_dim)


@dataclass
class RSSMConfig:
    h_dim: int = 256             # GRU hidden state dimension
    z_dim: int = 256             # must match EncoderConfig.z_dim
    action_dim: int = 4          # FORWARD / TURN_L / TURN_R / STOP
    prior_hidden: int = 256      # MLP hidden for prior p(z_t | h_t)


@dataclass
class DecoderConfig:
    h_dim: int = 256
    z_dim: int = 256
    img_size: int = 64


@dataclass
class OccupancyHeadConfig:
    h_dim: int = 256
    z_dim: int = 256
    grid_size: int = 64          # local occupancy grid (grid_size × grid_size)


@dataclass
class TrainConfig:
    # Data
    data_dir: str = "Symbolic_AIF_v6/world_model_training/data"
    batch_size: int = 16
    seq_len: int = 20            # BPTT sequence length
    num_workers: int = 4
    val_split: float = 0.05

    # Optimiser
    lr: float = 1e-4
    lr_warmup_steps: int = 500
    weight_decay: float = 1e-5
    grad_clip: float = 100.0

    # Loss weights
    w_rgb: float = 1.0
    w_depth: float = 1.0
    w_occ: float = 2.0           # emphasise occupancy — most relevant for nav
    w_kl: float = 1.0
    kl_free_nats: float = 3.0   # free nats to prevent KL from collapsing early

    # Training schedule
    num_epochs: int = 50
    ckpt_every_epochs: int = 5
    log_every_steps: int = 100

    # Checkpoint
    ckpt_dir: str = "Symbolic_AIF_v6/world_model_training/checkpoints"
    resume_ckpt: str = ""        # path to checkpoint to resume from (empty = train from scratch)

    # Architecture (references)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    rssm: RSSMConfig = field(default_factory=RSSMConfig)
    decoder: DecoderConfig = field(default_factory=DecoderConfig)
    occupancy: OccupancyHeadConfig = field(default_factory=OccupancyHeadConfig)


@dataclass
class CollectConfig:
    # Habitat scene
    scene_path: str = ""         # leave empty → HabitatEnv default
    total_steps: int = 50_000
    episode_steps: int = 200     # steps per episode before resetting
    save_dir: str = "Symbolic_AIF_v6/world_model_training/data"

    # Random-walk policy
    p_forward: float = 0.70
    p_left: float = 0.15         # p_right = 1 - p_forward - p_left
    seed: int = 42

    # Occupancy GT
    occ_resolution: float = 0.1  # metres per cell
    occ_grid_size: int = 64      # cells per side of local grid
