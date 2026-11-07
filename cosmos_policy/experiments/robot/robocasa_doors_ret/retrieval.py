# -----------------------------------------------------------------------------
# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# -----------------------------------------------------------------------------

"""Online retrieval for RoboCasa→MG: pool loading + FAISS + per-chunk query.

Per episode
-----------
    retrieval.reset()
    retrieval.initialize_episode(left_t0, right_t0)         # picks top-M pool
    for t in range(max_steps):
        retrieval.push_observation(left_t, right_t, proprio_t)
        if t % num_open_loop_steps == 0:
            result = retrieval.retrieve()                    # dict, see docstring

Pool source
-----------
    pool_features_root / <folder> / features_shard_*.hdf5     — DINO CLS + proprio + action
    pool_source_root   / <folder> / demos.hdf5                — JPEG wrist/L/R (pulled at match time)

At eval we don't know the future action chunk or the final episode
length, so those two groups in the query feature are zero-filled. Zero
contributes a *constant* distance for every pool entry so top-K ranking
is unaffected (see ``feature_assembly.py`` docstring).
"""

from __future__ import annotations

import glob
import io
import os
from collections import deque
from pathlib import Path
from typing import Optional

import faiss
import h5py
import numpy as np
import timm
import torch
from PIL import Image

from .feature_assembly import (
    DEFAULT_WEIGHTS,
    DIM_PROPRIO,
    FEATURE_DIM,
    H_CHUNK,
    N_HIST,
    build_initial_scene_vec,
    build_query_feature,
)

DINO_MODEL = "vit_base_patch14_dinov2.lvd142m"
DINO_INPUT = 224
DINO_DIM = 768

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def _decode_jpeg(buf) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(bytes(buf))).convert("RGB"))


class RoboCasaDoorsRetrieval:
    """Online retrieval against the MG pool for RoboCasa door tasks.

    Pool-features tensor lives in RAM; FAISS IndexFlatL2 is built once at
    init. Per episode, ``initialize_episode`` pre-filters to the top-M
    candidate pool episodes by initial-scene cosine, and all subsequent
    ``retrieve()`` calls search within that restricted row set.

    DINO forward happens only at ``retrieve()`` time (once per chunk).
    Env-step observations are buffered raw in ``push_observation()``.
    """

    def __init__(
        self,
        pool_features_root: str,   # .../robocasa-retrieval/kitchen_doors
        pool_source_root: str,     # .../robocasa-MG_3000_hdf5/kitchen_doors
        task_split: list[str],
        pool_stride: int = 4,      # stride for pool feature rows (memory vs coverage)
        weights: dict[str, float] = DEFAULT_WEIGHTS,
        pre_top_m: int = 200,
        retrieval_top_k_choice: int = 1,
        device: torch.device = torch.device("cuda"),
    ):
        self.pool_features_root = Path(pool_features_root)
        self.pool_source_root = Path(pool_source_root)
        self.task_split = list(task_split)
        self.pool_stride = int(pool_stride)
        self.weights = dict(weights)
        self.pre_top_m = int(pre_top_m)
        self.retrieval_top_k_choice = int(retrieval_top_k_choice)
        self.device = device

        # ── 1. DINO model for online feature extraction ────────────────────
        print("[retrieval] loading DINOv2 …")
        self.dino = (
            timm.create_model(
                DINO_MODEL, pretrained=True, num_classes=0, img_size=DINO_INPUT
            )
            .eval()
            .to(device)
        )
        self._mean = IMAGENET_MEAN.to(device)
        self._std = IMAGENET_STD.to(device)

        # ── 2. Assemble pool features in memory ─────────────────────────────
        self._load_pool()

        # ── 3. FAISS indexes ────────────────────────────────────────────────
        print(f"[retrieval] building FAISS — {self.pool_feats.shape[0]:,} rows × {FEATURE_DIM}d")
        self.query_index = faiss.IndexFlatL2(FEATURE_DIM)
        self.query_index.add(self.pool_feats)
        self.init_index = faiss.IndexFlatIP(self.pool_init_feats.shape[1])
        self.init_index.add(self.pool_init_feats)
        print(f"[retrieval] pool ready — {len(self.pool_ep_keys):,} episodes, "
              f"{self.pool_feats.shape[0]:,} query rows")

        # ── 4. Source HDF5 handles (kept open for fast JPEG pulls) ─────────
        self._source_files: dict[str, h5py.File] = {}
        for folder in self.task_split:
            p = self.pool_source_root / folder / "demos.hdf5"
            assert p.exists(), f"missing source: {p}"
            self._source_files[folder] = h5py.File(p, "r")

        # ── 5. Per-episode state ────────────────────────────────────────────
        self._cand_row_ids: Optional[np.ndarray] = None    # (pre_top_m × pool_stride-capped,)
        self._cand_feats: Optional[np.ndarray] = None      # cached candidate feature subset
        self._cand_row_meta: Optional[list] = None
        self._buf_left: deque[np.ndarray] = deque(maxlen=N_HIST + 1)
        self._buf_right: deque[np.ndarray] = deque(maxlen=N_HIST + 1)
        self._buf_proprio: deque[np.ndarray] = deque(maxlen=N_HIST + 1)
        self._step: int = 0

    # ---------------------------------------------------------------- #
    # Pool loading
    # ---------------------------------------------------------------- #

    def _load_pool(self):
        pool_rows: list[np.ndarray] = []
        pool_row_meta: list[tuple[str, str, int]] = []
        init_vecs: list[np.ndarray] = []
        pool_ep_keys: list[tuple[str, str]] = []

        for folder in self.task_split:
            shards = sorted((self.pool_features_root / folder).glob("features_shard_*.hdf5"))
            if not shards:
                raise FileNotFoundError(
                    f"no feature shards under {self.pool_features_root / folder}"
                )
            for sp in shards:
                with h5py.File(sp, "r") as f:
                    demo_keys = sorted(f["data"].keys(), key=lambda k: int(k.split("_")[-1]))
                    for demo_key in demo_keys:
                        g = f["data"][demo_key]
                        init_vecs.append(
                            build_initial_scene_vec(
                                g["dino_patch_left_init"][:],
                                g["dino_patch_right_init"][:],
                            )
                        )
                        pool_ep_keys.append((folder, demo_key))

                        T = int(g.attrs["T"])
                        if T < H_CHUNK + 1:
                            continue
                        proprio = g["proprio"][:].astype(np.float32)       # (T, 11)
                        action = g["action"][:].astype(np.float32)         # (T, 10)
                        cls_l = g["dino_cls_left"][:].astype(np.float32)   # (T+1, 768)
                        cls_r = g["dino_cls_right"][:].astype(np.float32)

                        for t in range(0, T - H_CHUNK + 1, self.pool_stride):
                            past_l = cls_l[t - N_HIST] if t >= N_HIST else None
                            past_r = cls_r[t - N_HIST] if t >= N_HIST else None
                            start = max(0, t - N_HIST)
                            pr_win = np.zeros((N_HIST + 1, DIM_PROPRIO), dtype=np.float32)
                            pr = proprio[start : t + 1]
                            pr_win[-pr.shape[0] :] = pr
                            feat = build_query_feature(
                                cur_cls_left=cls_l[t],
                                cur_cls_right=cls_r[t],
                                past_cls_left=past_l,
                                past_cls_right=past_r,
                                cur_proprio=proprio[t],
                                proprio_window=pr_win,
                                weights=self.weights,
                                include_action=True,
                                action_chunk=action[t : t + H_CHUNK],
                                include_time=True,
                                progress=t / max(T - 1, 1),
                            )
                            pool_rows.append(feat)
                            pool_row_meta.append((folder, demo_key, t))

        self.pool_feats = np.stack(pool_rows).astype(np.float32)
        self.pool_init_feats = np.stack(init_vecs).astype(np.float32)
        self.pool_ep_keys = pool_ep_keys
        self.pool_row_meta = pool_row_meta
        # Fast (folder, demo) → episode row lookup (for pre-filter)
        self._ep_to_pool_row_idx = {k: i for i, k in enumerate(pool_ep_keys)}
        # Per-row episode row idx (for fast membership mask during search).
        self._row_ep_idx = np.array(
            [self._ep_to_pool_row_idx[(f, d)] for f, d, _ in pool_row_meta],
            dtype=np.int64,
        )

    # ---------------------------------------------------------------- #
    # DINO inference (batched)
    # ---------------------------------------------------------------- #

    @torch.no_grad()
    def _dino_cls(self, images_uint8: np.ndarray) -> np.ndarray:
        """Return CLS (N, 768) for (N, H, W, 3) uint8 images. Auto-resizes
        to 224×224 if needed (RoboCasa sim is typically 128 or 224)."""
        if images_uint8.shape[1] != DINO_INPUT or images_uint8.shape[2] != DINO_INPUT:
            resized = np.stack(
                [
                    np.asarray(
                        Image.fromarray(img).resize((DINO_INPUT, DINO_INPUT), Image.BILINEAR)
                    )
                    for img in images_uint8
                ]
            )
        else:
            resized = images_uint8
        t = torch.from_numpy(resized).to(self.device, non_blocking=True)
        t = t.permute(0, 3, 1, 2).float().div_(255.0)
        t = (t - self._mean) / self._std
        feats = self.dino.forward_features(t)    # (N, 257, 768)
        return feats[:, 0].cpu().numpy().astype(np.float32)

    @torch.no_grad()
    def _dino_patch_tokens(self, image_uint8: np.ndarray) -> np.ndarray:
        """(256, 768) patch tokens for a single image."""
        if image_uint8.shape[0] != DINO_INPUT or image_uint8.shape[1] != DINO_INPUT:
            image_uint8 = np.asarray(
                Image.fromarray(image_uint8).resize((DINO_INPUT, DINO_INPUT), Image.BILINEAR)
            )
        t = torch.from_numpy(image_uint8[None]).to(self.device, non_blocking=True)
        t = t.permute(0, 3, 1, 2).float().div_(255.0)
        t = (t - self._mean) / self._std
        feats = self.dino.forward_features(t)    # (1, 257, 768)
        return feats[0, 1:].cpu().numpy().astype(np.float32)

    # ---------------------------------------------------------------- #
    # Episode control
    # ---------------------------------------------------------------- #

    def reset(self):
        self._buf_left.clear()
        self._buf_right.clear()
        self._buf_proprio.clear()
        self._step = 0
        self._cand_row_ids = None
        self._cand_feats = None
        self._cand_row_meta = None

    def initialize_episode(self, left_t0: np.ndarray, right_t0: np.ndarray):
        """Compute initial-scene query + pre-filter top-M pool episodes."""
        left_patch = self._dino_patch_tokens(left_t0)
        right_patch = self._dino_patch_tokens(right_t0)
        q_init = build_initial_scene_vec(left_patch, right_patch).reshape(1, -1)
        _, top_ep_ids = self.init_index.search(q_init, self.pre_top_m)
        cand_ep_set = set(int(e) for e in top_ep_ids[0])
        mask = np.isin(self._row_ep_idx, list(cand_ep_set))
        self._cand_row_ids = np.where(mask)[0]
        self._cand_feats = self.pool_feats[self._cand_row_ids]
        self._cand_row_meta = [self.pool_row_meta[i] for i in self._cand_row_ids]
        print(f"[retrieval] episode init: pre-filter top-{self.pre_top_m} episodes → "
              f"{len(self._cand_row_ids):,} candidate rows")

    def push_observation(
        self, left: np.ndarray, right: np.ndarray, proprio: np.ndarray
    ):
        """Called every env step. Only buffers; does not run DINO."""
        self._buf_left.append(left.copy())
        self._buf_right.append(right.copy())
        self._buf_proprio.append(proprio.astype(np.float32).copy())
        self._step += 1

    # ---------------------------------------------------------------- #
    # Retrieval (called per chunk)
    # ---------------------------------------------------------------- #

    def retrieve(self) -> dict:
        """Run online DINO + FAISS search, return the ret-condition payload.

        Returned dict keys:
            ret_wrist, ret_left, ret_right : (H, W, 3) uint8
            ret_proprio                     : (9,) float32 (raw, un-normalized)
            ret_actions                     : (H_CHUNK, 7) float32 (raw)
            match_folder, match_demo, match_t
        """
        assert self._cand_row_ids is not None, (
            "call initialize_episode() first"
        )
        assert len(self._buf_left) > 0, "push_observation at least once"

        # Current (rightmost of buffer)
        cur_left = self._buf_left[-1]
        cur_right = self._buf_right[-1]
        cur_proprio = self._buf_proprio[-1]

        # Past at t-N (if history is long enough)
        if len(self._buf_left) >= N_HIST + 1:
            past_left = self._buf_left[-N_HIST - 1]
            past_right = self._buf_right[-N_HIST - 1]
            has_past = True
        else:
            past_left = past_right = None
            has_past = False

        # DINO forward — batch 4 (cur_L, cur_R, past_L, past_R) when we have past, else 2
        if has_past:
            batch = np.stack([cur_left, cur_right, past_left, past_right], axis=0)
            cls = self._dino_cls(batch)
            cur_cls_left, cur_cls_right, past_cls_left, past_cls_right = (
                cls[0], cls[1], cls[2], cls[3]
            )
        else:
            batch = np.stack([cur_left, cur_right], axis=0)
            cls = self._dino_cls(batch)
            cur_cls_left, cur_cls_right = cls[0], cls[1]
            past_cls_left = past_cls_right = None

        # Proprio window (right-aligned, zero-padded at front).
        pr_win = np.zeros((N_HIST + 1, DIM_PROPRIO), dtype=np.float32)
        buf_props = np.stack(list(self._buf_proprio), axis=0)
        pr_win[-buf_props.shape[0]:] = buf_props[-(N_HIST + 1):]

        # Query feature — ZERO action and ZERO time (unknown at eval).
        q = build_query_feature(
            cur_cls_left=cur_cls_left,
            cur_cls_right=cur_cls_right,
            past_cls_left=past_cls_left,
            past_cls_right=past_cls_right,
            cur_proprio=cur_proprio,
            proprio_window=pr_win,
            weights=self.weights,
            include_action=False,
            include_time=False,
        ).reshape(1, -1)

        # Stage-2 FAISS search restricted to candidate rows only.
        diff = self._cand_feats - q
        d2 = (diff * diff).sum(axis=1)
        order = np.argsort(d2)[: self.retrieval_top_k_choice]
        pick = int(np.random.choice(order))
        m_folder, m_demo, m_t = self._cand_row_meta[pick]

        # Pull the raw ret frames at (m_t + chunk_size), wrist + left + right.
        m_t_future = m_t + H_CHUNK
        ret_payload = self._pull_pool_frame(m_folder, m_demo, m_t_future)

        # Ret proprio (raw) + ret action chunk (raw, 7-dim).
        src = self._source_files[m_folder][f"data/{m_demo}"]
        T_pool = int(src["robot_states"].shape[0])
        m_t_future_clamped = min(m_t_future, T_pool - 1)
        ret_proprio = src["robot_states"][m_t_future_clamped].astype(np.float32)
        # Raw action chunk: slice + pad at tail if episode too short.
        actions_all = src["actions"]       # (T, 12)
        idxs = [min(m_t + i, T_pool - 1) for i in range(H_CHUNK)]
        ret_actions = actions_all[idxs, :7].astype(np.float32)

        return {
            "ret_wrist": ret_payload["wrist"],
            "ret_left":  ret_payload["left"],
            "ret_right": ret_payload["right"],
            "ret_proprio": ret_proprio,
            "ret_actions": ret_actions,
            "match_folder": m_folder,
            "match_demo": m_demo,
            "match_t": m_t,
        }

    # ---------------------------------------------------------------- #
    # Helpers
    # ---------------------------------------------------------------- #

    def _pull_pool_frame(
        self, folder: str, demo_key: str, t: int
    ) -> dict[str, np.ndarray]:
        src = self._source_files[folder][f"data/{demo_key}/obs"]
        T = src["robot0_agentview_left_rgb"].shape[0]
        t = max(0, min(t, T - 1))
        return dict(
            wrist=_decode_jpeg(src["robot0_eye_in_hand_rgb"][t]),
            left=_decode_jpeg(src["robot0_agentview_left_rgb"][t]),
            right=_decode_jpeg(src["robot0_agentview_right_rgb"][t]),
        )

    def close(self):
        for f in self._source_files.values():
            try:
                f.close()
            except Exception:
                pass
        self._source_files.clear()
