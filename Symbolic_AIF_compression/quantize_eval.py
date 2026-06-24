"""
quantize_eval.py — Post-training quantization study for compressed AIF models.

Tests FP32 → FP16 → INT8 precision on the inference modules (encoder, GRU,
factor_heads, transition_model). The decoder is excluded since it is never
deployed.

Usage:
    python quantize_eval.py --checkpoint ./checkpoints/CL1/best_model.pt
    python quantize_eval.py --checkpoint ./checkpoints/CL2/best_model.pt --trials 50
    python quantize_eval.py --checkpoint ./checkpoints/CL1/best_model.pt --noise_sigma 0.10

Output:
    results/<config>_quant_summary.json — comparison table: FP32/FP16/INT8

INT8 note:
    We use PyTorch dynamic quantization (linear + GRU layers quantized at runtime).
    This is the pragmatic approach for CPU/ARM deployment without needing calibration
    data. For GPU INT8 (e.g. TensorRT), a separate calibration step would be needed.
    The ICRA paper can position dynamic INT8 as the edge-deployment baseline and
    TensorRT-style static INT8 as a future work direction.
"""
from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.quantization as tq

from model_compressed import CompressedModelConfig, WorldModelCompressed
from simulator import TinyIndoorEnv, Pose
from aif_efe_planner_v3 import EFEPlannerV3
from aif_efe_planner_v2 import EFEPlannerConfig
from aif_planner_v3 import IDX_TO_HEADING, most_likely_state_from_belief, masked_joint_from_belief
from evaluate_compressed import (
    load_model, sample_start_goal_pairs, build_reachable_mask,
    shortest_path_distances, get_free_cells, belief_entropy_masked,
    apply_corruption, _make_result, EvalConfig,
)


# ============================================================
# Quantized inference wrapper
# ============================================================

class InferenceOnlyModel:
    """
    Wraps WorldModelCompressed for inference-only deployment:
    - Decoder is removed from the computation graph
    - Optional FP16 or INT8 quantization applied to linear/GRU layers

    Exposes only forward_step_online and rollout_from_filtered_state,
    which are the only methods needed at deployment.
    """

    def __init__(
        self,
        base_model: WorldModelCompressed,
        precision:  str = "fp32",    # "fp32", "fp16", "int8"
    ):
        self.cfg       = base_model.cfg
        self.precision = precision

        # Deep-copy everything except the decoder
        self.action_embedding  = copy.deepcopy(base_model.action_embedding)
        self.encoder           = copy.deepcopy(base_model.encoder)
        self.filter_input_proj = copy.deepcopy(base_model.filter_input_proj)
        self.filter_gru        = copy.deepcopy(base_model.filter_gru)
        self.factor_heads      = copy.deepcopy(base_model.factor_heads)
        self.transition_model  = copy.deepcopy(base_model.transition_model)

        # Move to eval mode
        for m in self._modules():
            m.eval()

        self._apply_precision(precision)

    def _modules(self):
        return [
            self.action_embedding, self.encoder, self.filter_input_proj,
            self.filter_gru, self.factor_heads, self.transition_model,
        ]

    def _apply_precision(self, precision: str) -> None:
        if precision == "fp32":
            pass  # default

        elif precision == "fp16":
            for m in self._modules():
                m.half()

        elif precision == "int8":
            # Dynamic quantization: Linear and GRU layers quantized to INT8 at call time.
            # encoder conv layers remain FP32 (static quant needs calibration data).
            for attr in ("filter_input_proj", "factor_heads", "transition_model"):
                mod = getattr(self, attr)
                q   = tq.quantize_dynamic(mod, {torch.nn.Linear}, dtype=torch.qint8)
                setattr(self, attr, q)
            # GRU quantization
            self.filter_gru = tq.quantize_dynamic(
                self.filter_gru, {torch.nn.GRU}, dtype=torch.qint8,
            )
        else:
            raise ValueError(f"Unknown precision: {precision!r}. Use 'fp32', 'fp16', or 'int8'.")

    @property
    def inference_param_count(self) -> int:
        total = 0
        for m in self._modules():
            for p in m.parameters():
                total += p.numel()
        return total

    def model_size_bytes(self) -> int:
        """Approximate model size in bytes at the current precision."""
        bits_per_param = {"fp32": 32, "fp16": 16, "int8": 8}.get(self.precision, 32)
        return self.inference_param_count * bits_per_param // 8

    @torch.no_grad()
    def forward_step_online(
        self,
        obs_t:       torch.Tensor,
        action_prev: Optional[torch.Tensor],
        h_prev:      Optional[torch.Tensor]            = None,
        prev_belief: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        import torch.nn.functional as F

        device = obs_t.device

        if self.precision == "fp16":
            obs_t = obs_t.half()

        feat = self.encoder(obs_t)

        if action_prev is None:
            action_prev = torch.zeros(1, dtype=torch.long, device=device)
        a_emb = self.action_embedding(action_prev)

        if self.precision == "fp16":
            a_emb = a_emb.half()
            feat  = feat.half()

        x = self.filter_input_proj(torch.cat([feat, a_emb], dim=-1))

        if self.precision == "fp16":
            x = x.half()
            h_prev_in = h_prev.half() if h_prev is not None else None
        else:
            h_prev_in = h_prev

        h_seq, h_new = self.filter_gru(x.float().unsqueeze(1) if self.precision == "int8" else x.unsqueeze(1), h_prev_in)
        h_t = h_seq[:, 0, :]

        factors = self.factor_heads(h_t.float() if self.precision == "int8" else h_t)

        if prev_belief is not None and self.cfg.use_predictive_prior:
            eps = 1e-8
            # FP16: pass half tensors to half-weight transition model, cast outputs to float after.
            # INT8: transition model is dynamically quantized on Linear layers (FP32 input/output).
            # FP32: no cast needed.
            if self.precision == "fp16":
                cast_in  = lambda t: t.half()
                cast_out = lambda t: t.float()
            elif self.precision == "int8":
                cast_in  = lambda t: t.float()
                cast_out = lambda t: t
            else:
                cast_in  = lambda t: t
                cast_out = lambda t: t

            b_row = cast_in(prev_belief["row_probs"])
            b_col = cast_in(prev_belief["col_probs"])
            b_hdg = cast_in(prev_belief["heading_probs"])
            b_ctx = cast_in(prev_belief["context"])

            tr = self.transition_model.forward_step(
                b_row, b_col, b_hdg, b_ctx, cast_in(a_emb),
            )
            tr = {k: cast_out(v) for k, v in tr.items()}
            row_probs     = F.softmax(torch.log(tr["next_row_probs"].clamp_min(eps))     + F.log_softmax(factors["row_logits"].float(),     dim=-1), dim=-1)
            col_probs     = F.softmax(torch.log(tr["next_col_probs"].clamp_min(eps))     + F.log_softmax(factors["col_logits"].float(),     dim=-1), dim=-1)
            heading_probs = F.softmax(torch.log(tr["next_heading_probs"].clamp_min(eps)) + F.log_softmax(factors["heading_logits"].float(), dim=-1), dim=-1)
        else:
            row_probs     = torch.softmax(factors["row_logits"].float(),     dim=-1)
            col_probs     = torch.softmax(factors["col_logits"].float(),     dim=-1)
            heading_probs = torch.softmax(factors["heading_logits"].float(), dim=-1)

        belief = {
            "row_probs":      row_probs,
            "col_probs":      col_probs,
            "heading_probs":  heading_probs,
            "context":        factors["context"].float(),
            "context_mu":     factors["context_mu"].float(),
            "context_logvar": factors["context_logvar"].float(),
        }
        return belief, h_new.float()

    @torch.no_grad()
    def rollout_from_filtered_state(
        self,
        row_probs_start, col_probs_start, heading_probs_start,
        context_start, action_seq, skip_recon=True,
    ) -> Dict[str, torch.Tensor]:
        import torch.nn.functional as F
        B, K = action_seq.shape

        # Ensure inputs match transition_model weight dtype
        if self.precision == "fp16":
            cast_in  = lambda t: t.half()
            cast_out = lambda t: t.float()
        elif self.precision == "int8":
            cast_in  = lambda t: t.float()
            cast_out = lambda t: t
        else:
            cast_in  = lambda t: t
            cast_out = lambda t: t

        q_row = cast_in(row_probs_start)
        q_col = cast_in(col_probs_start)
        q_hdg = cast_in(heading_probs_start)
        ctx   = cast_in(context_start)

        row_logits_roll, col_logits_roll, hdg_logits_roll = [], [], []
        row_probs_roll, col_probs_roll, hdg_probs_roll    = [], [], []
        ctx_roll, coll_roll = [], []

        for k in range(K):
            a_emb = cast_in(self.action_embedding(action_seq[:, k]))
            step  = self.transition_model.forward_step(q_row, q_col, q_hdg, ctx, a_emb)
            q_row = step["next_row_probs"]
            q_col = step["next_col_probs"]
            q_hdg = step["next_heading_probs"]
            ctx   = step["next_context"]
            row_logits_roll.append(cast_out(step["next_row_logits"]))
            col_logits_roll.append(cast_out(step["next_col_logits"]))
            hdg_logits_roll.append(cast_out(step["next_heading_logits"]))
            row_probs_roll.append(cast_out(q_row))
            col_probs_roll.append(cast_out(q_col))
            hdg_probs_roll.append(cast_out(q_hdg))
            ctx_roll.append(cast_out(ctx))
            coll_roll.append(cast_out(step["collision_logits"]))

        return {
            "row_logits_roll":       torch.stack(row_logits_roll, dim=1),
            "col_logits_roll":       torch.stack(col_logits_roll, dim=1),
            "heading_logits_roll":   torch.stack(hdg_logits_roll, dim=1),
            "row_probs_roll":        torch.stack(row_probs_roll,  dim=1),
            "col_probs_roll":        torch.stack(col_probs_roll,  dim=1),
            "heading_probs_roll":    torch.stack(hdg_probs_roll,  dim=1),
            "context_roll":          torch.stack(ctx_roll,        dim=1),
            "collision_logits_roll": torch.stack(coll_roll,       dim=1),
            "recon_roll":            None,
        }


# ============================================================
# Single-trial eval for quantized model
# ============================================================

def run_quant_trial(
    trial_id:    int,
    qmodel:      InferenceOnlyModel,
    device:      torch.device,
    start_pose:  Tuple[int, int, str],
    goal_pos:    Tuple[int, int],
    cfg:         EvalConfig,
) -> Dict[str, Any]:
    env            = TinyIndoorEnv(seed=cfg.seed + trial_id)
    num_rows       = qmodel.cfg.num_row_classes
    num_cols       = qmodel.cfg.num_col_classes
    reachable_mask = build_reachable_mask(env, num_rows, num_cols).to(device)
    dist_t         = shortest_path_distances(env, goal_pos, num_rows, num_cols).to(device)

    efe_planner = EFEPlannerV3(
        model=qmodel, dist_t=dist_t, reachable_mask=reachable_mask, cfg=cfg.efe_cfg,
    )

    pose_in = Pose(row=start_pose[0], col=start_pose[1], heading=start_pose[2])
    obs, info = env.reset(start_pose=pose_in, goal_pos=goal_pos, use_goal=True)
    shortest_dist = env.shortest_path_length((start_pose[0], start_pose[1]), goal_pos)

    obs_history    = [np.asarray(obs, dtype=np.float32)]
    recent_states  = [(int(info["row"]), int(info["col"]), str(info["heading"]))]
    h_state, prev_belief, last_action = None, None, None
    reached_goal   = False
    total_coll     = 0
    belief_errors  = 0
    inf_times_ms:  List[float] = []
    entropies:     List[float] = []

    for a in (2, 3, 3, 2):
        step_result = env.step(a)
        obs_raw, done, info = step_result.obs, step_result.done, step_result.info
        true_pose = (int(info["row"]), int(info["col"]), str(info["heading"]))
        obs_np    = np.asarray(obs_raw, dtype=np.float32)
        obs_history.append(obs_np)
        recent_states.append(true_pose)
        if info["collision"]: total_coll += 1
        if done or info["reached_goal"]: reached_goal = True; break
        obs_t = torch.from_numpy(obs_np).unsqueeze(0).unsqueeze(0).to(device)
        act_t = torch.tensor([a], dtype=torch.long, device=device)
        prev_belief, h_state = qmodel.forward_step_online(obs_t, last_action, h_state, prev_belief)
        last_action = act_t

    if reached_goal:
        return _make_result(trial_id, start_pose, goal_pos, True, 0, total_coll,
                            shortest_dist, 0, [], [], [], cfg)

    for _ in range(cfg.max_steps):
        t0     = time.perf_counter()
        obs_np = obs_history[-1]
        obs_c  = apply_corruption(obs_np, cfg.noise_sigma, cfg.mask_fraction)
        obs_t  = torch.from_numpy(obs_c).unsqueeze(0).unsqueeze(0).to(device)
        prev_belief, h_state = qmodel.forward_step_online(obs_t, last_action, h_state, prev_belief)
        inf_times_ms.append((time.perf_counter() - t0) * 1000.0)

        entropies.append(belief_entropy_masked(prev_belief["row_probs"][0], prev_belief["col_probs"][0], reachable_mask))
        pred_r, pred_c, pred_h = most_likely_state_from_belief(
            prev_belief["row_probs"][0], prev_belief["col_probs"][0],
            prev_belief["heading_probs"][0], reachable_mask,
        )
        if (pred_r, pred_c, IDX_TO_HEADING[pred_h]) != recent_states[-1]:
            belief_errors += 1

        scored   = efe_planner.score_action_sequences_pure(belief=prev_belief, recent_true_states=recent_states)
        best_seq = scored.get("best_sequence") or [0]
        action   = int(best_seq[0])

        step_result = env.step(action)
        obs_raw, done, info = step_result.obs, step_result.done, step_result.info
        true_pose = (int(info["row"]), int(info["col"]), str(info["heading"]))
        obs_history.append(np.asarray(obs_raw, dtype=np.float32))
        recent_states.append(true_pose)
        last_action = torch.tensor([action], dtype=torch.long, device=device)
        if len(recent_states) > 20: recent_states = recent_states[-20:]
        if info["collision"]: total_coll += 1
        if done or info["reached_goal"]: reached_goal = True; break

    return _make_result(trial_id, start_pose, goal_pos, reached_goal,
                        len(inf_times_ms), total_coll, shortest_dist, belief_errors,
                        entropies, [], inf_times_ms, cfg)


# ============================================================
# Run one precision level
# ============================================================

def eval_precision(
    base_model: WorldModelCompressed,
    precision:  str,
    pairs:      list,
    device:     torch.device,
    cfg:        EvalConfig,
) -> Dict[str, Any]:
    print(f"\n--- Evaluating {precision.upper()} ---")
    qmodel = InferenceOnlyModel(base_model, precision=precision)
    size_b = qmodel.model_size_bytes()
    print(f"  Params: {qmodel.inference_param_count:,}  |  Size: {size_b/1024:.1f} KB")

    results = []
    for i, (start_pose, goal_pos) in enumerate(pairs):
        r = run_quant_trial(i, qmodel, device, start_pose, goal_pos, cfg)
        results.append(r)
        print(f"  [{i+1:03d}/{len(pairs)}] {'OK' if r['success'] else 'FAIL'} | steps={r['steps']:02d} | ms={r['mean_ms_per_step']:.2f}")

    n         = len(results)
    successes = [r for r in results if r["success"]]
    mean      = lambda vs: float(np.mean(vs)) if vs else float("nan")
    return {
        "precision":          precision,
        "inference_params":   qmodel.inference_param_count,
        "model_size_kb":      round(size_b / 1024, 2),
        "success_rate":       len(successes) / max(n, 1),
        "collision_rate":     mean([float(r["collisions"] > 0) for r in results]),
        "avg_steps_success":  mean([r["steps"] for r in successes]),
        "mean_path_efficiency": mean([r["path_efficiency"] for r in successes]),
        "mean_belief_error_rate": mean([r["belief_error_rate"] for r in results]),
        "mean_entropy":       mean([r["mean_entropy"] for r in results]),
        "mean_ms_per_step":   mean([r["mean_ms_per_step"] for r in results]),
    }


# ============================================================
# Main
# ============================================================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",    type=str, required=True)
    p.add_argument("--trials",        type=int, default=50)
    p.add_argument("--max_steps",     type=int, default=80)
    p.add_argument("--seed",          type=int, default=42)
    p.add_argument("--noise_sigma",   type=float, default=0.0)
    p.add_argument("--mask_fraction", type=float, default=0.0)
    p.add_argument("--output_dir",    type=str, default="./results")
    p.add_argument("--precisions",    type=str, default="fp32,fp16,int8",
                   help="Comma-separated precision levels to test")
    return p.parse_args()


def main():
    args      = parse_args()
    device    = torch.device("cpu")   # quantization most relevant on CPU/embedded
    print(f"Device: {device}  (INT8 dynamic quant is CPU-only in PyTorch)")

    base_model = load_model(args.checkpoint, device)

    cfg = EvalConfig(
        checkpoint_path=args.checkpoint,
        num_trials=args.trials,
        max_steps=args.max_steps,
        seed=args.seed,
        noise_sigma=args.noise_sigma,
        mask_fraction=args.mask_fraction,
    )

    template_env = TinyIndoorEnv(seed=cfg.seed)
    pairs        = sample_start_goal_pairs(template_env, cfg.num_trials, cfg.seed)
    precisions   = [p.strip() for p in args.precisions.split(",")]

    all_results = []
    for prec in precisions:
        res = eval_precision(base_model, prec, pairs, device, cfg)
        all_results.append(res)

    # Print comparison table
    print("\n" + "=" * 80)
    print("QUANTIZATION COMPARISON")
    print("=" * 80)
    print(f"  Config: gru={base_model.cfg.gru_hidden_dim}, feat={base_model.cfg.encoder_feat_dim}, "
          f"ctx={base_model.cfg.context_dim}")
    print(f"  Trials: {args.trials}")
    print()
    print(f"{'Prec':<6} | {'Params':>9} | {'KB':>8} | {'Succ%':>6} | {'Coll%':>5} | {'ms/step':>7} | {'BelErr%':>7}")
    print("-" * 65)
    for r in all_results:
        print(
            f"{r['precision']:<6} | {r['inference_params']:>9,} | "
            f"{r['model_size_kb']:>7.1f} | {r['success_rate']*100:>5.1f}% | "
            f"{r['collision_rate']*100:>4.1f}% | {r['mean_ms_per_step']:>7.2f} | "
            f"{r['mean_belief_error_rate']*100:>6.1f}%"
        )

    # Save
    ckpt_stem  = Path(args.checkpoint).parent.name
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path   = output_dir / f"{ckpt_stem}_quant_summary.json"
    payload    = {
        "checkpoint":      args.checkpoint,
        "model_config": {
            "gru_hidden_dim":  base_model.cfg.gru_hidden_dim,
            "encoder_feat_dim": base_model.cfg.encoder_feat_dim,
            "context_dim":     base_model.cfg.context_dim,
            "use_depthwise":   base_model.cfg.use_depthwise,
        },
        "trials":         args.trials,
        "noise_sigma":    args.noise_sigma,
        "mask_fraction":  args.mask_fraction,
        "results":        all_results,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
