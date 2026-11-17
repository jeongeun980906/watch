"""DiT with the projection heads of the Vid2Robot auxiliary alignment losses (arXiv 2403.12943, Sec. II-E).

The heads live inside the net so that the optimizer (built on model.net), FSDP and the checkpoint all cover them
without special cases. A post-train with the plain MinimalV1LVGDiT loads such a checkpoint unchanged: the .pt
warm start is strict=False, so the `align_heads.*` keys are ignored.

  align_heads.tcc      Phi of Sec. II-E (2): 2-layer MLP on the per-frame (spatially pooled) block features
  align_heads.vv_*     attention pooling (one learned query) over the per-frame features -> one embedding per
                       video, for the SigLIP prompt-video / robot-video contrastive loss (Sec. II-E (3))
  align_heads.logit_*  SigLIP's learnable temperature (log scale) and bias, init log(10) and -10 (SigLIP paper)

The losses themselves are computed in models/policy_video2world_model_align.py.
"""
import math

import torch
from torch import nn
from torch.distributed._composable.fsdp import fully_shard

from cosmos_policy._src.predict2.networks.minimal_v1_lvg_dit import MinimalV1LVGDiT


class AlignHeads(nn.Module):
    def __init__(self, d_model: int, dim: int = 128, hidden: int = 1024, num_heads: int = 16):
        super().__init__()
        self.tcc = nn.Sequential(nn.Linear(d_model, hidden), nn.GELU(), nn.Linear(hidden, dim))
        self.vv_query = nn.Parameter(torch.zeros(1, 1, d_model))
        self.vv_attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.vv_proj = nn.Linear(d_model, dim)
        self.logit_scale = nn.Parameter(torch.zeros(()))
        self.logit_bias = nn.Parameter(torch.zeros(()))

    def init_weights(self) -> None:
        for m in (self.tcc[0], self.tcc[2], self.vv_proj):
            m.reset_parameters()
        nn.init.normal_(self.vv_query, std=0.02)
        self.vv_attn._reset_parameters()
        with torch.no_grad():
            self.logit_scale.fill_(math.log(10.0))
            self.logit_bias.fill_(-10.0)

    def frame_embed(self, f_B_L_D: torch.Tensor) -> torch.Tensor:
        """Per-frame embedding for TCC, L2-normalized: (B, L, D) -> (B, L, dim)."""
        return nn.functional.normalize(self.tcc(f_B_L_D).float(), dim=-1)

    def video_embed(self, f_B_L_D: torch.Tensor) -> torch.Tensor:
        """One embedding per video for SigLIP, L2-normalized: (B, L, D) -> (B, dim)."""
        q = self.vv_query.expand(f_B_L_D.shape[0], -1, -1).to(f_B_L_D.dtype)
        z, _ = self.vv_attn(q, f_B_L_D, f_B_L_D, need_weights=False)
        return nn.functional.normalize(self.vv_proj(z[:, 0]).float(), dim=-1)


class MinimalV1LVGDiTAlign(MinimalV1LVGDiT):
    """MinimalV1LVGDiT + AlignHeads. `align_*` arguments are read by the align model (net.align_cfg)."""

    def __init__(self, *args, align_block: int = 14, align_tcc_weight: float = 1.0, align_vv_weight: float = 1.0,
                 align_tcc_temp: float = 0.1, align_dim: int = 128, **kwargs):
        super().__init__(*args, **kwargs)
        assert 0 <= align_block < len(self.blocks), f"align_block {align_block} not in [0, {len(self.blocks)})"
        self.align_cfg = dict(block=int(align_block), tcc_weight=float(align_tcc_weight),
                              vv_weight=float(align_vv_weight), tcc_temp=float(align_tcc_temp))
        self.align_heads = AlignHeads(self.model_channels, dim=align_dim)
        self.align_heads.init_weights()          # MiniTrainDIT.__init__ already ran init_weights before the heads existed

    def init_weights(self):
        super().init_weights()
        if hasattr(self, "align_heads"):
            self.align_heads.init_weights()

    def fully_shard(self, mesh):
        super().fully_shard(mesh)
        fully_shard(self.align_heads, mesh=mesh, reshard_after_forward=True)
