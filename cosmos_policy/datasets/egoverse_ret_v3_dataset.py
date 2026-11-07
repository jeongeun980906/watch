"""EgoVerse mid-training V3 — v2 compact layout WITHOUT grid padding.

Ablation vs v2 (docs/egoverse_mid_v2_design.md): every video token is the PRIMARY
camera decoded FULL-FRAME at final_image_size (224), for ALL mid embodiments
(human AND MimicDroid) — no 2x2 grid, no side cams, no black quadrants.
Everything else (compact layout, slot_roles, t_positions, native lengths,
actions/state, buckets) is inherited from EgoVerseRetV2Dataset unchanged.

Purpose: measure the image-format transfer gap of full-frame mid-training into
2x2-grid GR1 post-training (probe: /mnt/ddn/tmp/v3_gate_probe.py sections C/C2).
"""
from __future__ import annotations

import numpy as np

from cosmos_policy.datasets.egoverse_ret_dataset import _decode
from cosmos_policy.datasets.egoverse_ret_v2_dataset import EgoVerseRetV2Dataset


class EgoVerseRetV3Dataset(EgoVerseRetV2Dataset):
    """V2 with full-frame single-video packing (no grid) — see module doc."""

    def _grid_frame(self, g, i: int) -> np.ndarray:
        """Primary cam decoded full-frame at final_image_size; no grid, no side cams."""
        S = self.final_image_size
        b = g.frame(int(i))
        return _decode(b, S) if b else np.zeros((S, S, 3), np.uint8)
