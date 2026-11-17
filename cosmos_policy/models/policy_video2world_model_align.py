"""Vid2Robot auxiliary alignment losses (arXiv 2403.12943, Sec. II-E) on the WATCH retrieval policy.

Used with the Vid2Robot-style layout (EgoVerseRetV3StateDataset no_ret_action + no_future): the demo (prompt)
video and the current image condition the policy, and the action chunk is the only generated slot. On top of the
rectified-flow loss:

  TCC  (Sec. II-E (2), Eq. 1-2)  temporal cycle consistency between the per-frame embeddings of the demo video and
                                  of the query video, MSE on the cycled-back frame index (normalized to [0, 1]),
                                  both directions averaged.
  VVCL (Sec. II-E (3), Eq. 3)    SigLIP between one embedding per demo video and per query video, batch gathered
                                  over all data-parallel ranks. Pairs with the same fine task class are not used as
                                  negatives (the batch holds many same-task windows).

Features: the output of DiT block `align_block`, averaged over the spatial tokens of each latent frame.
  demo video  = its latent slots in the training forward (clean conditioning, timestep 0)
  query video = a second forward of the same net on [state slot | query video], all slots clean conditioning,
                the query video in the demo-video role (the shared "prompt encoder" of Vid2Robot)
The second forward runs on every rank at every step (FSDP needs the same collectives on all ranks); invalid
samples (video condition dropped by the conditioner, demo blanked) only get weight 0.

The query video comes from the dataset (align_query_video=True: align_video / align_t_positions /
align_slot_roles / align_cls / align_valid). Heads and weights: networks/align_dit.py (net.align_cfg).
"""
from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn.functional as F

from cosmos_policy._src.imaginaire.utils import log
from cosmos_policy._src.predict2.conditioner import DataType
from cosmos_policy.models.policy_text2world_model import replace_latent_with_proprio
from cosmos_policy.models.policy_video2world_model_realrobot_ret import (
    CosmosPolicyRealRobotRetConfigRectifiedFlow,
    CosmosPolicyRealRobotRetModelRectifiedFlow,
)

_PASS_KEYS = ("t5_text_embeddings", "t5_text_mask", "fps", "padding_mask", "image_size")


def tcc_loss(u: torch.Tensor, v: torch.Tensor, temp: float) -> torch.Tensor:
    """Eq. 1-2 for one pair: u (L1, d) cycles through v (L2, d) and back; MSE of the normalized index."""
    a = torch.softmax(-torch.cdist(u, v).pow(2) / temp, dim=1)          # (L1, L2) soft neighbour in v
    v_t = a @ v                                                          # (L1, d)
    b = torch.softmax(-torch.cdist(v_t, u).pow(2) / temp, dim=1)        # (L1, L1) cycle back to u
    idx = torch.arange(u.shape[0], device=u.device, dtype=u.dtype) / max(u.shape[0] - 1, 1)
    return ((b @ idx - idx) ** 2).mean()


def _gather(x: torch.Tensor) -> list[torch.Tensor]:
    if not (dist.is_available() and dist.is_initialized()) or dist.get_world_size() == 1:
        return [x]
    out = [torch.empty_like(x) for _ in range(dist.get_world_size())]
    dist.all_gather(out, x.detach().contiguous())
    out[dist.get_rank()] = x                                             # local slice keeps its gradient
    return out


def siglip_local(z_a: torch.Tensor, z_b_all: torch.Tensor, pos_col: torch.Tensor, neg_ok: torch.Tensor,
                 scale: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """SigLIP rows of the local samples: z_a (B, d) vs z_b_all (N, d); pos_col (B,) column of each positive;
    neg_ok (B, N) which off-diagonal pairs count as negatives. Sum over columns, mean over rows."""
    logits = z_a @ z_b_all.t() * scale.exp() + bias
    lab = -torch.ones_like(logits)
    rows = torch.arange(z_a.shape[0], device=z_a.device)
    lab[rows, pos_col] = 1.0
    w = neg_ok.float(); w[rows, pos_col] = 1.0
    return -(F.logsigmoid(lab * logits) * w).sum(1)


class CosmosPolicyRealRobotRetConfigRectifiedFlowAlign(CosmosPolicyRealRobotRetConfigRectifiedFlow):
    pass


class CosmosPolicyRealRobotRetModelRectifiedFlowAlign(CosmosPolicyRealRobotRetModelRectifiedFlow):

    def get_data_and_condition(self, data_batch):
        raw_state, latent_state, condition = super().get_data_and_condition(data_batch)
        self._align_main_condition = condition       # to see whether the video condition was dropped
        return raw_state, latent_state, condition

    # ---- the second forward: query video in the demo-video role, everything clean conditioning ----
    def _query_pass(self, data_batch):
        db = {k: data_batch[k] for k in _PASS_KEYS if k in data_batch}
        db["video"] = data_batch["align_video"]
        self._normalize_video_databatch_inplace(db, input_key="video")
        latent = self.encode(db["video"]).contiguous().float()
        B, T = latent.shape[0], latent.shape[2]
        no_drop = {k: 0.0 for k in ("use_video_condition", "text") if k in self.conditioner.embedders}
        cond = self.conditioner(db, override_dropout_rate=no_drop)
        cond = cond.edit_data_type(DataType.VIDEO)
        cond = cond.set_video_condition(
            gt_frames=latent.to(**self.tensor_kwargs),
            random_min_num_conditional_frames=T, random_max_num_conditional_frames=T,
            num_conditional_frames=torch.full((B,), T, device=latent.device, dtype=torch.long),
            conditional_frames_probs=None,
            t_positions=data_batch["align_t_positions"], slot_roles=data_batch["align_slot_roles"],
        )
        if "proprio" in data_batch and torch.all(data_batch["current_proprio_latent_idx"] == 0):
            cond.gt_frames = replace_latent_with_proprio(
                cond.gt_frames, data_batch["proprio"], proprio_indices=torch.zeros(B, dtype=torch.long, device=latent.device))
        xt = cond.gt_frames.to(**self.tensor_kwargs)
        self.denoise(noise=torch.zeros_like(xt), xt_B_C_T_H_W=xt,
                     timesteps_B_T=torch.zeros(B, 1, **self.tensor_kwargs_fp32), condition=cond)

    def training_step(self, data_batch, iteration):
        cfg = getattr(self.net, "align_cfg", None)
        if cfg is None or "align_video" not in data_batch:
            return super().training_step(data_batch, iteration)
        cap = []
        hook = self.net.blocks[cfg["block"]].register_forward_hook(lambda m, i, o: cap.append(o))
        try:
            output_batch, flow_loss = super().training_step(data_batch, iteration)
            n_main = len(cap)
            self._query_pass(data_batch)
        finally:
            hook.remove()
        assert n_main >= 1 and len(cap) > n_main, f"align hook captured {n_main} / {len(cap)} block outputs"
        x_main, x_q = cap[n_main - 1], cap[-1]                     # (B, T, H, W, D)
        heads = self.net.align_heads

        rs, re = data_batch["retrieved_video_start_latent_idx"], data_batch["retrieved_video_end_latent_idx"]
        assert bool((rs == rs[0]).all() and (re == re[0]).all()), "align: mixed demo lengths in one batch"
        f_ref = x_main[:, int(rs[0]):int(re[0])].mean(dim=(2, 3))  # (B, k, D)
        f_q = x_q[:, 1:].mean(dim=(2, 3))                          # (B, g, D)

        valid = data_batch["align_valid"].to(f_ref.device).float()
        ucond = getattr(self._align_main_condition, "use_video_condition", True)
        if isinstance(ucond, torch.Tensor):
            ucond = bool(ucond.all())
        valid = valid * float(bool(ucond))

        # TCC (both directions)
        e_ref, e_q = heads.frame_embed(f_ref), heads.frame_embed(f_q)
        t = cfg["tcc_temp"]
        tcc = torch.stack([0.5 * (tcc_loss(e_ref[b], e_q[b], t) + tcc_loss(e_q[b], e_ref[b], t))
                           for b in range(e_ref.shape[0])])
        tcc = (tcc * valid).sum() / valid.sum().clamp(min=1.0)

        # VVCL: SigLIP over the gathered batch, same-class pairs are not negatives
        z_ref, z_q = heads.video_embed(f_ref), heads.video_embed(f_q)
        cls = data_batch["align_cls"].to(f_ref.device).long()
        Z_ref, Z_q = torch.cat(_gather(z_ref)), torch.cat(_gather(z_q))
        C, V = torch.cat(_gather(cls)), torch.cat(_gather(valid))
        r = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        pos = torch.arange(z_ref.shape[0], device=f_ref.device) + r * z_ref.shape[0]
        neg_ok = (cls[:, None] != C[None, :]) & (V[None, :] > 0)
        s, bb = heads.logit_scale, heads.logit_bias
        vv = 0.5 * (siglip_local(z_ref, Z_q, pos, neg_ok, s, bb) + siglip_local(z_q, Z_ref, pos, neg_ok, s, bb))
        vv = (vv * valid).sum() / valid.sum().clamp(min=1.0)

        loss = flow_loss + cfg["tcc_weight"] * tcc + cfg["vv_weight"] * vv
        output_batch["align_tcc_loss"] = tcc.detach()
        output_batch["align_vv_loss"] = vv.detach()
        output_batch["flow_loss"] = flow_loss.detach()
        if iteration % 50 == 0 and r == 0:
            log.info(f"[align] iter {iteration}: flow {flow_loss.item():.4f}  tcc {tcc.item():.4f}  vv {vv.item():.4f}  "
                     f"valid {int(valid.sum().item())}/{valid.numel()}  scale {s.exp().item():.2f} bias {bb.item():.2f}")
        return output_batch, loss
