# SPDX-License-Identifier: Apache-2.0
"""Sprout real-robot post-train dataset: the Tabletop per-view (pv) layout on native 29-D data, one ego camera.

Store: ``WAM/dataset/mcap_episodes/to_sprout29.py`` writes the same HDF5 + manifest shape as
``convert_tabletop_to_egoverse57.py``, so ``TabletopEpisode`` reads it. What differs from Tabletop:

* **29-D**, not 57-D::

      head9(0:9) | Leef9(9:18) | Lgrip(18) | Reef9(19:28) | Rgrip(28)

  ``head9`` is the torso velocity command ``[vx, vy, vz, rot6d(Rz(vyaw / 15))]``. The groups are read as stored
  (``native20`` stays off). In the cup and drawer tasks the unused arm is written at the reset pose
  (``to_sprout29.py --idle-arm``).

* **15 Hz**, the camera's rate: four frames are 4/15 s per latent, a 32-frame chunk is 2.1 s.

* **One camera**, the ego view (ZED left, ``images/front_1``). ``images/center_top`` (ZED right) is in the store
  but not used.

With ``per_view=True, per_view_blank=0`` and one camera the sequence is::

    0 cur_state (blank, proprio in the latent) | 1 cur_image | 2 action (GEN) | 3 future image t+32 (GEN)
    state_t = 4, chunk_duration = 13, num_conditional_frames = 2

Smoke check::

    SPROUT_ROOT=... python -m cosmos_policy.datasets.sprout_dataset
"""
from __future__ import annotations

import os

import json

import numpy as np
import torch

from cosmos_policy.datasets.dataset_utils import preprocess_image
from cosmos_policy.datasets.egoverse_gr1_ret_dataset import _decode
from cosmos_policy.datasets.egoverse_gr1_ret_v2_dataset import NUM_DUPLICATES
from cosmos_policy.datasets.egoverse_ret_v2_dataset import (
    NOW_S,
    ROLE_ACTION,
    ROLE_CUR_IMAGE,
    ROLE_CUR_STATE,
    ROLE_FUTURE,
    ROLE_RET_ACTION,
    ROLE_RET_VIDEO,
)
from cosmos_policy.datasets.egoverse_ret_v3_state_dataset import load_stats_json
from cosmos_policy.datasets.egoverse_zread import FRONT
from cosmos_policy.datasets.tabletop_sim_dataset import TabletopEpisode, TabletopSimPlainEndImageDataset

SPROUT_ROOT = os.environ.get("SPROUT_ROOT", os.path.expanduser("~/WAM/dataset/sprout_ego29_3task"))
DIM29 = 29
FPS = 15.0


class SproutPlainEndDataset(TabletopSimPlainEndImageDataset):
    """Current state and ego image -> 32 x 29 ``abs_action29`` + the chunk-END ego image."""

    act_dim = DIM29
    st_dim = DIM29
    dt = 4.0 / FPS
    per_view_cams = (FRONT,)

    def __init__(self, *args, dataset_root: str = SPROUT_ROOT, state_group: str = "state29",
                 action_group: str = "abs_action29", **kw):
        super().__init__(*args, dataset_root=dataset_root, state_group=state_group, action_group=action_group, **kw)

    def bucket_key(self, r: int):
        return ("sprout", "plain_end")


ICL_TASKS = ("sprout_cup_to_drawer", "sprout_open_drawer")     # the tasks with Quest ghost demos (pnp has none)
HUMAN_EMB = "quest"
HUMAN_DIM = 57


class SproutIclDataset(SproutPlainEndDataset):
    """In-context post-train: a human demo of the same moment conditions the robot chunk (GR1 v6 layout, shrunk).

    The human demo is the Quest GHOST of the query's own robot episode (``quest_capture/pc/to_sprout_ghost.py``): a
    person followed that episode in VR at the robot's speed, trimmed to its length, so human frame t is robot frame t.
    At test time the person demonstrates in the same scene first and the robot follows, so the own ghost is the
    matching training demo. ``other_prob`` instead draws, with that probability, a retrieved ghost of ANOTHER episode
    (``manifest/human_pairs.npz``, 32-frame DTW windows) for robustness to a demo that differs from the scene.

    Layout (state_t = 7, chunk_duration = 25, num_conditional_frames = 5)::

        idx role         t            content
         0  CUR_STATE    NOW          blank (current state29 tiled in the latent; pv0 keeps one state, not a chunk)
         1  RET_ACTION   NOW          human state57 s_h[h+1 .. h+32] (quest-normalised, tiled)
         2  RET_VIDEO    NOW - 8 dt   human frame h        (demo window start)
         3  RET_VIDEO    NOW - dt     human frame h + 32   (demo window end)
         4  CUR_IMAGE    NOW          robot ego frame t
        -- conditioning ends here --
         5  ACTION       NOW          32 x 29 abs_action29 (GEN)
         6  FUTURE       NOW + dt     robot ego frame t + 32 (GEN)

    The demo times follow v6: its 8 demo latents sat at NOW - 8 dt .. NOW - dt, so the demo's first and last frames
    keep those two positions. ``h = t + jitter``, jitter uniform in [-jitter, jitter] frames during training (a test
    demo is never in lockstep with the robot), 0 without augmentation.
    """

    def __init__(self, *args, jitter: int = 6, other_prob: float = 0.0, task_filter=ICL_TASKS,
                 human_stats_path: str | None = None, **kw):
        self.jitter = int(jitter)
        self.other_prob = float(other_prob)
        super().__init__(*args, task_filter=list(task_filter), **kw)
        man = os.path.join(self.dataset_root, "manifest")
        self.human_stats = load_stats_json(human_stats_path or os.path.join(man, "human_state57_stats.json"))
        self._hcache = {}
        self._pairs = None
        if self.other_prob > 0:
            self._pairs = _OtherGhostPairs(os.path.join(man, "human_pairs.npz"), os.path.join(man, "human_ghost.json"))
        missing = sorted({str(e) for e in self.wep
                          if not os.path.exists(os.path.join(self.dataset_root, "human_ghost", str(e) + ".h5"))})
        if missing:
            raise FileNotFoundError(f"no ghost demo for {len(missing)} robot episodes, e.g. {missing[:3]}; "
                                    "run quest_capture/pc/to_sprout_ghost.py")
        print(f"[SproutIcl] windows={self.N} jitter=+-{self.jitter} other_prob={self.other_prob} "
              f"state_t={self.state_t} chunk_duration={self.chunk_duration}", flush=True)

    def __getstate__(self):
        s = super().__getstate__()
        s["_hcache"] = {}           # every dataloader worker opens its own human h5 handles
        return s

    @property
    def state_t(self) -> int:
        return 7

    def bucket_key(self, r: int):
        return ("sprout", "icl")

    def _human(self, ep: str) -> TabletopEpisode:
        g = self._hcache.get(ep)
        if g is None:
            g = TabletopEpisode(os.path.join(self.dataset_root, "human_ghost", ep + ".h5"))
            self._hcache[ep] = g
        return g

    def _demo(self, q: int, qw0: int, aug: bool) -> tuple[TabletopEpisode, int]:
        """(human episode, demo window start h) for query window q."""
        ep = str(self.wep[q])
        if aug and self._pairs is not None and np.random.rand() < self.other_prob:
            got = self._pairs.pick(ep, qw0)
            if got is not None:
                return self._human(got[0]), got[1]
        j = np.random.randint(-self.jitter, self.jitter + 1) if (aug and self.jitter > 0) else 0
        return self._human(ep), qw0 + j

    def _plain_build_per_view(self, q, idx, aug, gq, qw0, q_tgt, past, S, blank1, blank4) -> dict:
        gh, h = self._demo(q, qw0, aug)
        nh = gh.n_frames
        h = int(min(max(h, 0), nh - 1))
        C = self.chunk_size
        dup = lambda im: np.repeat(im[None], NUM_DUPLICATES, axis=0)
        rd = lambda g, i: _decode(g.frame(min(max(i, 0), g.n_frames - 1), FRONT), S)
        ret = self._norm(self._chunk(gh.state("state57"), h + 1), self.human_stats, HUMAN_EMB)

        images = np.concatenate([
            blank1,                      # 0 cur_state
            blank4.copy(),               # 1 ret_action (injected)
            dup(rd(gh, h)),              # 2 demo start
            dup(rd(gh, h + C)),          # 3 demo end
            dup(rd(gq, qw0)),            # 4 cur image
            blank4.copy(),               # 5 action (GEN)
            dup(rd(gq, qw0 + C)),        # 6 future (GEN)
        ], axis=0)
        assert images.shape[0] == self.chunk_duration, (images.shape, self.chunk_duration)
        images = preprocess_image(images, final_image_size=S, normalize_images=False,
                                  use_image_aug=aug, stronger_image_aug=aug)

        T, dt = self.state_t, self.dt
        roles = np.array([ROLE_CUR_STATE, ROLE_RET_ACTION, ROLE_RET_VIDEO, ROLE_RET_VIDEO, ROLE_CUR_IMAGE,
                          ROLE_ACTION, ROLE_FUTURE], np.int64)
        tpos = np.array([NOW_S, NOW_S, NOW_S - 8 * dt, NOW_S - dt, NOW_S, NOW_S, NOW_S + dt], np.float32)
        cls = self.wcls[q]
        t5 = (torch.squeeze(self.t5_text_embeddings[cls]) if cls in self.t5_text_embeddings
              else torch.zeros(512, 1024))
        q_emb = self.wemb[q]
        return {
            "video": images,
            "command": str(cls),
            "actions": q_tgt.astype(np.float32),
            "action_mask": np.ones(self.act_dim, np.float32),
            "t5_text_embeddings": t5,
            "t5_text_mask": torch.ones(512, dtype=torch.int64),
            "fps": 16,
            "padding_mask": torch.zeros(1, S, S),
            "image_size": S * torch.ones(4),
            "__key__": idx,
            "embodiment": q_emb, "retrieved_embodiment": HUMAN_EMB,
            "t_positions": tpos, "slot_roles": roles,
            "num_conditional_frames": 5,
            "retrieved_actions": ret.astype(np.float32),
            "retrieved_action_mask": np.ones(HUMAN_DIM, np.float32),
            "retrieved_video_start_latent_idx": 2,
            "retrieved_video_end_latent_idx": 4,
            "retrieved_action_latent_idx": 1,
            "retrieved_state_latent_idx": -1,
            "current_image_latent_idx": 4,
            "current_image2_latent_idx": -1,
            "current_wrist_image_latent_idx": -1,
            "current_wrist_image2_latent_idx": -1,
            "action_latent_idx": 5,
            "future_video_start_latent_idx": 6,
            "future_video_end_latent_idx": T,
            "future_image_latent_idx": 6,
            "future_image2_latent_idx": -1,
            "future_wrist_image_latent_idx": -1,
            "future_wrist_image2_latent_idx": -1,
            "proprio": past.reshape(-1).astype(np.float32),
            "future_proprio": np.zeros(self.st_dim, np.float32),
            "current_proprio_latent_idx": 0,
            "future_proprio_latent_idx": -1,
            "value_latent_idx": -1,
            "has_ret_data": 1, "has_ret_image": 1, "has_current_image": 1,
            "rollout_data_mask": 0, "rollout_data_success_mask": 0,
            "world_model_sample_mask": 0, "value_function_sample_mask": 0,
            "global_rollout_idx": -1, "value_function_return": float("-100"),
            "next_value_function_return": float("-100"),
            "next_action_chunk": q_tgt.copy().astype(np.float32),
        }


class _OtherGhostPairs:
    """Retrieved ghosts of OTHER robot episodes for a robot window (``other_prob``).

    pairs.npz rows are robot windows every 5 frames; the row at or before t is used and the human window is shifted
    by the same offset. The own ghost is skipped, so this is always a demo from a different scene.
    """

    def __init__(self, pairs_path: str, index_path: str):
        z = np.load(pairs_path)
        h2r = json.load(open(index_path))                         # human episode -> "<task>/<robot episode>"
        self.rows = {}
        for t, e, w0, me, mw in zip(z["robot_task"], z["robot_ep"], z["robot_w0"], z["match_ep"], z["match_w0"]):
            key = f"{t}/{e}"
            c = [(h2r[m], int(w)) for m, w in zip(me, mw) if m in h2r and h2r[m] != key]
            self.rows.setdefault(key, []).append((int(w0), c))
        for v in self.rows.values():
            v.sort()

    def pick(self, ep: str, t: int):
        rows = self.rows.get(ep)
        if not rows:
            return None
        k = max(0, np.searchsorted([w for w, _ in rows], t, side="right") - 1)
        w0, c = rows[k]
        if not c:
            return None
        hep, hw0 = c[np.random.randint(len(c))]
        return hep, hw0 + (t - w0)


def make(root: str = SPROUT_ROOT, cls=None, **kw) -> SproutPlainEndDataset:
    manifest = f"{root}/manifest"
    args = dict(
        manifest_dir=manifest, dataset_root=root, use_center_top=False,
        t5_text_embeddings_path=f"{manifest}/t5_embeddings.pkl",
        action_stats_path=f"{manifest}/abs_action29_stats_pctl.json",
        state_stats_path=f"{manifest}/state29_stats_pctl.json",
        target_group="abs_action29", target_stats_path=f"{manifest}/abs_action29_stats_pctl.json",
        robust_state_stats=False, chunk_size=32, per_frame_stride=1, final_image_size=224,
        num_duplicates_per_image=4, use_residual_actions=False, retrieval_dropout_prob=0.0, require_t5=True,
        per_view=True, per_view_blank=0,
    )
    args.update(kw)
    return (cls or SproutPlainEndDataset)(**args)


if __name__ == "__main__":
    import sys

    ds = make(require_t5=False, cls=SproutIclDataset if "icl" in sys.argv[1:] else None)
    s = ds[len(ds) // 2]
    print(len(ds), "windows; video", s["video"].shape, "actions", s["actions"].shape, "proprio", s["proprio"].shape)
    print("t_positions", s["t_positions"], "roles", s["slot_roles"], "n_cond", s["num_conditional_frames"])
    print("action range", float(s["actions"].min()), float(s["actions"].max()), "nan", bool(np.isnan(s["actions"]).any()))
