"""Targeted self-attention capture for the GR1 v6 policy (ICL and plain/BC arms).

Why not ``analysis_tools/attn_capture.py``: that one materializes the full
(B, H, S, S) weight matrix, which is what forced its callers down to a handful of
layers and 4 heads. We only ever look at ONE query block — the action latent — so
we compute just those rows:

    logits = q[action_slot] @ k_all^T / sqrt(d)      (B, H, HW, S)
    weights = softmax(logits, dim=-1)                 normalized over the FULL sequence
    map = weights.mean(query rows).view(H, T, Hs, Ws)

At 448 px the GR1 layouts are tiny in token terms (patch_spatial=2 over a 28x28
VAE latent => 14x14 = 196 tokens per slot; ICL T=20 => S=3920, BC T=11 => S=2156),
so the targeted form costs ~50 MB of transient activation and lets us keep ALL 28
blocks and ALL 16 heads instead of a subset.

Token order is ``b (t h w) d`` (minimal_v4_dit.py: block forward rearranges
``b t h w d -> b (t h w) d`` before self-attn), so slot ``t`` owns the contiguous
row range ``[t*H*W, (t+1)*H*W)``. ``video_size`` is handed to ``compute_attention``
by the block itself, so T/H/W are read, never guessed.

The rectified-flow sampler calls the net once per denoising step. We accumulate a
running mean over steps and also keep the final step, since attention at high noise
is not the same object as attention at the end of the trajectory.

Usage::

    with Gr1AttnCapture(model.net, query_slot=11) as cap:
        latents = model.generate_samples_from_batch(...)
    out = cap.result()      # {"mean": (L,H,T,Hs,Ws), "last": ..., "layers": [...]}
"""

from __future__ import annotations

import math
import re
from typing import Optional

import torch
from einops import rearrange


class Gr1AttnCapture:
    """Capture action-query self-attention maps from every DiT block.

    Args:
        net: the DiT module (``model.net``). Searched recursively for ``self_attn``
            ``Attention`` modules living under ``blocks.<i>``.
        query_slot: latent index of the ACTION slot (ICL v6: 11, plain/BC v6: 2).
            Pass the value the sample itself carries (``action_latent_idx``) rather
            than a literal, so a layout change cannot silently point this at the
            wrong tokens.
        layers: block indices to capture. None = all.
        head_reduce: "mean" stores the head-averaged map (small); "none" keeps every
            head. Both keep all layers.
    """

    def __init__(
        self,
        net: torch.nn.Module,
        query_slot: int,
        layers: Optional[list[int]] = None,
        head_reduce: str = "mean",
        step_stride: int = 1,
        head_chunk: int = 4,
    ) -> None:
        if head_reduce not in ("mean", "none"):
            raise ValueError(f"head_reduce must be 'mean' or 'none', got {head_reduce!r}")
        self.net = net
        self.query_slot = int(query_slot)
        self.layers = set(layers) if layers is not None else None
        self.head_reduce = head_reduce
        # Capture every Nth denoising step. The sampler calls the net once per step
        # (~36 for the eval default) and the attention structure moves slowly across
        # them, so striding is a cheap way to keep the rollout near its normal speed.
        self.step_stride = max(1, int(step_stride))
        # Heads are processed in groups so the transient logits tensor stays small.
        self.head_chunk = max(1, int(head_chunk))

        self._patched: list[tuple[torch.nn.Module, object]] = []
        self._sum: dict[int, torch.Tensor] = {}
        self._last: dict[int, torch.Tensor] = {}
        self._n: dict[int, int] = {}
        self._calls: dict[int, int] = {}
        self._shape: Optional[tuple[int, int, int]] = None   # (T, H, W)

    # ---- module discovery -------------------------------------------------
    @staticmethod
    def _block_index(name: str) -> Optional[int]:
        m = re.search(r"blocks\.(\d+)\.", name)
        return int(m.group(1)) if m else None

    # ---- the patched op ---------------------------------------------------
    def _make_wrapper(self, layer_idx: int, original):
        def wrapped(_self, q, k, v, video_size=None, kv_cache_cfg=None):
            c = self._calls.get(layer_idx, 0)
            self._calls[layer_idx] = c + 1
            if c % self.step_stride:
                return original(q, k, v, video_size=video_size, kv_cache_cfg=kv_cache_cfg)
            try:
                self._record(layer_idx, q, k, video_size)
            except Exception as exc:  # never let instrumentation break the rollout
                # Surface once per layer rather than silently producing empty maps.
                if layer_idx not in self._n:
                    print(f"[gr1_attn_capture] layer {layer_idx} capture failed: {exc!r}")
                    self._n[layer_idx] = 0
            return original(q, k, v, video_size=video_size, kv_cache_cfg=kv_cache_cfg)

        return wrapped

    @torch.no_grad()
    def _record(self, layer_idx: int, q, k, video_size) -> None:
        if video_size is None:
            raise RuntimeError("video_size is None — cannot locate the action slot")
        T, H, W = int(video_size.T), int(video_size.H), int(video_size.W)
        HW = H * W
        S = q.shape[1]
        if S != T * HW:
            raise RuntimeError(f"seq len {S} != T*H*W ({T}*{H}*{W}) — layout assumption broken")
        if not (0 <= self.query_slot < T):
            raise RuntimeError(f"query_slot {self.query_slot} out of range for T={T}")
        self._shape = (T, H, W)

        lo, hi = self.query_slot * HW, (self.query_slot + 1) * HW
        # (B,S,Hh,D) -> (B,Hh,S,D); q restricted to the action rows. The matmul runs in
        # the tensors' own dtype — the same precision the real attention op uses, so this
        # is closer to what the model computes than an fp32 recompute would be, and it is
        # several times faster. Only the softmax is promoted to fp32.
        qa = rearrange(q[:, lo:hi], "b s h d -> b h s d")
        kkT = rearrange(k, "b s h d -> b h d s")
        scale = 1.0 / math.sqrt(qa.shape[-1])
        n_head = qa.shape[1]
        parts = []
        for h0 in range(0, n_head, self.head_chunk):
            h1 = min(h0 + self.head_chunk, n_head)
            logits = torch.matmul(qa[:, h0:h1], kkT[:, h0:h1]) * scale   # (B,hc,HWq,S)
            w = logits.float().softmax(dim=-1)
            # average over the action-slot query tokens -> per-key attention mass
            parts.append(w.mean(dim=2)[0])                               # (hc,S)
            del logits, w
        m = torch.cat(parts, dim=0).view(-1, T, H, W)                    # (Hh,T,H,W)
        if self.head_reduce == "mean":
            m = m.mean(dim=0, keepdim=True)                              # (1,T,H,W)
        m = m.to("cpu", dtype=torch.float32)

        if layer_idx in self._sum:
            self._sum[layer_idx] += m
            self._n[layer_idx] += 1
        else:
            self._sum[layer_idx] = m
            self._n[layer_idx] = 1
        self._last[layer_idx] = m

    # ---- context manager --------------------------------------------------
    def __enter__(self) -> "Gr1AttnCapture":
        for name, module in self.net.named_modules():
            if module.__class__.__name__ != "Attention":
                continue
            if "self_attn" not in name:
                continue
            idx = self._block_index(name)
            if idx is None:
                continue
            if self.layers is not None and idx not in self.layers:
                continue
            original = module.compute_attention
            module.compute_attention = self._make_wrapper(idx, original).__get__(
                module, module.__class__
            )
            self._patched.append((module, original))
        if not self._patched:
            raise RuntimeError("no self-attention modules matched — capture would be empty")
        return self

    def __exit__(self, *_exc) -> None:
        for module, original in self._patched:
            module.compute_attention = original
        self._patched.clear()

    # ---- readout ----------------------------------------------------------
    def result(self) -> dict:
        """Return stacked maps. ``mean``/``last`` are (L, Hh, T, Hs, Ws) float32,
        with Hh == 1 when ``head_reduce='mean'``. Rows are normalized over the FULL
        key sequence, so summing a slot's map gives that slot's share of the action
        token's attention."""
        layers = sorted(k for k in self._sum if self._n.get(k, 0) > 0)
        if not layers:
            raise RuntimeError("capture produced nothing — was the model actually run?")
        mean = torch.stack([self._sum[i] / self._n[i] for i in layers])
        last = torch.stack([self._last[i] for i in layers])
        T, H, W = self._shape
        return {
            "mean": mean,
            "last": last,
            "layers": layers,
            "steps": self._n[layers[0]],
            "query_slot": self.query_slot,
            "T": T, "H": H, "W": W,
        }

    def reset(self) -> None:
        self._sum.clear()
        self._last.clear()
        self._n.clear()
        self._calls.clear()
