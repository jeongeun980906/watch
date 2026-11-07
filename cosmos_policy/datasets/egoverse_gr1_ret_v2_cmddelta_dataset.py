# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""COMMAND-space delta GR1 post-train dataset ("cd", 2026-07-18).

Delta target defined on the recorded COMMAND sequence (abs_action57) so that
differentiate -> integrate -> execute all live in one (command) space:

    dpos_t = R^cmd_t^T (p^cmd_{t+1} - p^cmd_t)      (eef blocks, local cmd frame)
    dR_t   = R^cmd_t^T R^cmd_{t+1}                  (6D)
    head9  = identity delta; hand15 = ABS command hand pose at t+1

Round-trip exact; |dpos| p50 0.55cm / p99.5 1.7cm. Server: --pred-target
cmd_delta57 (command-anchor chained integration; t0 anchor = ctrl-offset-mapped
measured pose — exact at rest). Manifest WIN=36 guarantees w0+33 rows.
"""
from __future__ import annotations

import numpy as np

from cosmos_policy.datasets.egoverse_gr1_ret_v2_dataset import EgoVerseGr1RetV2Dataset

IDENT9 = np.array([0, 0, 0, 1, 0, 0, 0, 1, 0], np.float64)


def _sixd_to_R(c1: np.ndarray, c2: np.ndarray) -> np.ndarray:
    b1 = c1 / np.linalg.norm(c1, axis=-1, keepdims=True)
    b2 = c2 - np.sum(b1 * c2, axis=-1, keepdims=True) * b1
    b2 /= np.linalg.norm(b2, axis=-1, keepdims=True)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=-1)


def cmd_delta_block(blk: np.ndarray) -> np.ndarray:
    """(n,9) abs command eef9 -> (n-1,9) local-frame delta [dpos3|dR 6D]."""
    p = blk[:, :3]
    Rm = _sixd_to_R(blk[:, 3:6], blk[:, 6:9])
    RtT = np.transpose(Rm[:-1], (0, 2, 1))
    dp = np.einsum("nij,nj->ni", RtT, p[1:] - p[:-1])
    dR = np.einsum("nij,njk->nik", RtT, Rm[1:])
    return np.concatenate([dp, dR[:, :, 0], dR[:, :, 1]], axis=1)


def cmd_delta57(rows: np.ndarray) -> np.ndarray:
    """(n,57) abs_action57 command rows -> (n-1,57) cmd-delta target."""
    n = len(rows) - 1
    out = np.empty((n, 57), np.float64)
    out[:, 0:9] = IDENT9
    out[:, 9:18] = cmd_delta_block(rows[:, 9:18])
    out[:, 18:33] = rows[1:, 18:33]
    out[:, 33:42] = cmd_delta_block(rows[:, 33:42])
    out[:, 42:57] = rows[1:, 42:57]
    return out


class EgoVerseGr1RetV2CmdDeltaDataset(EgoVerseGr1RetV2Dataset):
    """v2 layout, action target = COMMAND-space deltas from abs_action57."""

    def _act_chunk(self, g, w0: int) -> np.ndarray:
        rows = g.action(self.action_group)
        n = len(rows)
        sel = np.stack([rows[min(max(w0 + k, 0), n - 1)] for k in range(self.chunk_size + 1)])
        return cmd_delta57(sel).astype(np.float32)


# ---------------------------------------------------------------------------
# HYBRID delta ("cdh", 2026-07-19): per-task target —
#   PnPSinkToRightCounterPlate* episodes -> LEGACY state-space delta (action57;
#     측정 리앵커의 자기보정이 운반·배치에 유리 — 구델타 Plate 2/3 vs cd ~13%)
#   everything else -> cmd-delta (Faucet 80%/DoorL2 d3 부활의 그 타깃)
# Normalization happens INSIDE _act_chunk (task-dependent stats), so the config
# must NOT rely on __getitem__'s act normalization (we force normalize_actions
# False after init). Episode task is detected from the zarr root path (queries
# "<Task>_...", demos "dthdemo_<Task>_..." / "taskdemo_<Task>_...").
# ---------------------------------------------------------------------------
PLATE_TAG = "PnPSinkToRightCounterPlate"


class EgoVerseGr1RetV2HybridDeltaDataset(EgoVerseGr1RetV2Dataset):
    def __init__(self, *args, cmd_delta_stats_path: str = "", state_delta_stats_path: str = "", **kw):
        super().__init__(*args, **kw)
        import json as _json
        def _load(pth):
            d = _json.load(open(pth))
            return {k: {"lo": np.asarray(v["lo"], np.float32), "hi": np.asarray(v["hi"], np.float32)}
                    for k, v in d.items()}
        self._cd_stats = _load(cmd_delta_stats_path)
        self._sd_stats = _load(state_delta_stats_path)
        self.normalize_actions = False          # act norm은 _act_chunk 내부에서 태스크별로

    @staticmethod
    def _ep_emb(g) -> str:
        base = g.root.rsplit("/", 1)[-1]
        return "mimicdroid" if (base.startswith("dthdemo_") or base.startswith("taskdemo_")) else "gr1"

    def _act_chunk(self, g, w0: int) -> np.ndarray:
        is_plate = PLATE_TAG in g.root
        emb = self._ep_emb(g)
        if is_plate:                             # 구델타 (state-space, action57)
            raw = self._chunk(g.action("action57"), w0)
            return self._norm(raw, self._sd_stats, emb).astype(np.float32)
        rows = g.action("abs_action57")          # cd (command-space)
        n = len(rows)
        sel = np.stack([rows[min(max(w0 + k, 0), n - 1)] for k in range(self.chunk_size + 1)])
        return self._norm(cmd_delta57(sel), self._cd_stats, emb).astype(np.float32)
