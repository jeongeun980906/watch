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
                 human_stats_path: str | None = None, demo29: bool = False, match_table: bool = False,
                 base_abs: bool = False, base_abs_range: float = 1.0, base_abs_out: bool = True,
                 base_rel: bool = False, base_rel_range: float = 0.6, **kw):
        self.jitter = int(jitter)
        # demo29: the demo is the RETARGETED human (quest_capture/pc/retarget.py -> human_ghost_r29/, "state29"),
        # normalised with the robot's state stats, i.e. the same representation as the robot state and target
        self.demo29 = str(demo29).lower() in ("1", "true")      # configs may hand it over as a string
        self.human_dir = "human_ghost_r29" if self.demo29 else "human_ghost"
        # match_table: the demo window start is the inference retrieval's h for this t (scripts/sprout_match_table.py,
        # manifest/ghost_match_r29.json: retargeted EEF + gripper, GR1 band, own ghost) instead of t + jitter
        self.match = None
        # base_abs: head9 = the torso pose in the episode's start frame [x, y, 0, rot6d(Rz(yaw))] (sprout_icl_aug
        # `poses` / `abs_head`) for the state, the target and the (retargeted) demo, instead of the velocity command
        self.base_abs = str(base_abs).lower() in ("1", "true")
        # base_abs_out False: the TARGET keeps the velocity command (no differencing noise); state + demo absolute
        self.base_abs_out = str(base_abs_out).lower() in ("1", "true")
        # base_rel: the TARGET (and the demo chunk) as the torso pose relative to the chunk's start pose (the robot's
        # current pose / the demo's pose at h): small range -> precise; the state stays the start-frame pose
        self.base_rel = str(base_rel).lower() in ("1", "true")
        self._abs = {}
        self.other_prob = float(other_prob)
        super().__init__(*args, task_filter=list(task_filter), **kw)
        if self.base_abs:                  # normalisation of the pose: +-range m, rot6d in [-1, 1] (mirror-symmetric)
            B = float(base_abs_range)
            lo9, hi9 = np.array([-B, -B, -1, -1, -1, -1, -1, -1, -1.0]), np.array([B, B, 1, 1, 1, 1, 1, 1, 1.0])
            sts = [self.state_stats] + ([self.target_stats, getattr(self, "action_stats", None)] if self.base_abs_out else [])
            if not self.base_abs_out:              # copy, so the target's (velocity) stats stay as they are
                import copy
                self.state_stats = copy.deepcopy(self.state_stats)
                sts = [self.state_stats]
            for st in {id(x): x for x in sts if x}.values():
                for k, v in st.items():
                    lo, hi = np.asarray(v["lo"], np.float64).copy(), np.asarray(v["hi"], np.float64).copy()
                    lo[:9], hi[:9] = lo9, hi9
                    v["lo"], v["hi"] = lo.astype(np.float32), hi.astype(np.float32)
            if self.base_rel:                  # the target (and demo chunk) range: +-base_rel_range m
                import copy
                self.target_stats = copy.deepcopy(self.target_stats)
                Rr = float(base_rel_range)
                for k, v in self.target_stats.items():
                    lo, hi = np.asarray(v["lo"], np.float64).copy(), np.asarray(v["hi"], np.float64).copy()
                    lo[:9], hi[:9] = [-Rr, -Rr, -1, -1, -1, -1, -1, -1, -1], [Rr, Rr, 1, 1, 1, 1, 1, 1, 1]
                    v["lo"], v["hi"] = lo.astype(np.float32), hi.astype(np.float32)
        man = os.path.join(self.dataset_root, "manifest")
        if str(match_table).lower() in ("1", "true"):
            self.match = {k: np.asarray(v, int) for k, v in json.load(open(os.path.join(man, "ghost_match_r29.json"))).items()}
        self.human_stats = load_stats_json(human_stats_path or os.path.join(man, "human_state57_stats.json"))
        self._hcache = {}
        self._pairs = None
        if self.other_prob > 0:
            self._pairs = _OtherGhostPairs(os.path.join(man, "human_pairs.npz"), os.path.join(man, "human_ghost.json"))
        missing = sorted({str(e) for e in self.wep
                          if not os.path.exists(os.path.join(self.dataset_root, self.human_dir, str(e) + ".h5"))})
        if missing:
            raise FileNotFoundError(f"no ghost demo for {len(missing)} robot episodes, e.g. {missing[:3]}; "
                                    "run quest_capture/pc/to_sprout_ghost.py")
        print(f"[SproutIcl] windows={self.N} jitter=+-{self.jitter} other_prob={self.other_prob} "
              f"state_t={self.state_t} chunk_duration={self.chunk_duration}", flush=True)

    def __getstate__(self):
        s = super().__getstate__()
        s["_hcache"] = {}           # every dataloader worker opens its own human h5 handles
        s["_abs"] = {}
        return s

    @property
    def state_t(self) -> int:
        return 7

    def bucket_key(self, r: int):
        return ("sprout", "icl")

    @property
    def demo_key(self) -> str:
        if self.base_abs:
            return "state29abs"
        return "state29" if self.demo29 else "state57"

    def _grp(self, wi: int):
        g = super()._grp(wi)
        if not self.base_abs or not isinstance(g, TabletopEpisode):     # the server's LiveEpisode is already absolute
            return g
        ep = str(self.wep[wi])
        a = self._abs.get(ep)
        if a is None:
            a = self._abs[ep] = AbsEpisode(g, self.base_abs_out)
        return a

    def _demo_norm(self, x, q_emb, clip: bool = True):
        if self.base_rel:                      # the demo chunk is relative too: normalised as the target
            return (self._norm if clip else self._norm_free)(x, self.target_stats, q_emb)
        if self.demo29:
            return (self._norm if clip else self._norm_free)(x, self.state_stats, q_emb)
        return (self._norm if clip else self._norm_free)(x, self.human_stats, HUMAN_EMB)

    def _human(self, ep: str) -> TabletopEpisode:
        g = self._hcache.get(ep)
        if g is None:
            g = TabletopEpisode(os.path.join(self.dataset_root, self.human_dir, ep + ".h5"))
            self._hcache[ep] = g
        return g

    def _demo(self, q: int, qw0: int, aug: bool) -> tuple[TabletopEpisode, int]:
        """(human episode, demo window start h) for query window q."""
        ep = str(self.wep[q])
        if aug and self._pairs is not None and np.random.rand() < self.other_prob:
            got = self._pairs.pick(ep, qw0)
            if got is not None:
                return self._human(got[0]), got[1]
        if self.match is not None and ep in self.match:
            m = self.match[ep]
            return self._human(ep), int(m[min(qw0, len(m) - 1)])
        j = np.random.randint(-self.jitter, self.jitter + 1) if (aug and self.jitter > 0) else 0
        return self._human(ep), qw0 + j

    def _plain_build_per_view(self, q, idx, aug, gq, qw0, q_tgt, past, S, blank1, blank4) -> dict:
        gh, h = self._demo(q, qw0, aug)
        nh = gh.n_frames
        h = int(min(max(h, 0), nh - 1))
        C = self.chunk_size
        dup = lambda im: np.repeat(im[None], NUM_DUPLICATES, axis=0)
        rd = lambda g, i: _decode(g.frame(min(max(i, 0), g.n_frames - 1), FRONT), S)
        x = self._chunk(gh.state(self.demo_key), h + 1)
        if self.base_rel:                      # the demo chunk relative to the demo's pose at h
            from cosmos_policy.datasets import sprout_icl_aug as _A
            x = np.asarray(x, np.float64).copy()
            ref = np.asarray(gh.state(self.demo_key), np.float64)[min(h, gh.n_frames - 1), :9]
            x[:, :9] = _A.rel_from_abs(x[:, :9], ref)
        ret = self._demo_norm(x, self.wemb[q], clip=True)

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
        return self._per_view_sample(q, idx, aug, images, q_tgt, past, ret)

    def _per_view_sample(self, q, idx, aug, images, q_tgt, past, ret) -> dict:
        S = self.final_image_size
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
            "retrieved_action_mask": np.ones(np.asarray(ret).shape[-1], np.float32),
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



class AbsEpisode:
    """A robot episode with head9 as the torso pose in its start frame (integrated abs_action29 commands): state row t
    = pose(t), action row t = pose(t + 1). `raw` is the episode as stored."""

    def __init__(self, g, abs_out: bool = True):
        from cosmos_policy.datasets import sprout_icl_aug as A
        self.raw, self.abs_out = g, abs_out
        cmd = np.asarray(g.action("abs_action29"), np.float64)
        self.xy, self.q = A.poses(cmd)                       # T + 1 poses
        self._st, self._ac = {}, {}
        self._A = A

    def __getattr__(self, k):
        return getattr(self.raw, k)

    def state(self, group: str):
        if group not in self._st:
            x = np.asarray(self.raw.state(group), np.float64).copy()
            if x.shape[-1] == 29:
                x[:, :9] = self._A.abs_head(self.xy[:len(x)], self.q[:len(x)])
            self._st[group] = x
        return self._st[group]

    def action(self, group: str):
        if group not in self._ac:
            x = np.asarray(self.raw.action(group), np.float64).copy()
            if x.shape[-1] == 29 and self.abs_out:
                x[:, :9] = self._A.abs_head(self.xy[1:len(x) + 1], self.q[1:len(x) + 1])
            self._ac[group] = x
        return self._ac[group]


class SproutIclAugDataset(SproutIclDataset):
    """SproutIclDataset + demo-consistent augmentation (``sprout_icl_aug.py``): the demo and the robot target are
    transformed together (tempo, yaw, wrist rotation, arm mirror, full mirror), so that the demo decides what the
    robot does. ``cf_aug``: factor probabilities / ranges (sprout_icl_aug.DEFAULT), None or {} off -> exactly the
    parent's sample."""

    def __init__(self, *args, cf_aug: dict | None = None, **kw):
        super().__init__(*args, **kw)
        from cosmos_policy.datasets import sprout_icl_aug as A

        self.A = A
        self.cf_aug = dict(cf_aug) if cf_aug else {}
        self.idle = {}                              # the reset pose of each arm, where the data holds the idle arm
        for task, side in (("sprout_cup_to_drawer", "left"), ("sprout_open_drawer", "right")):
            q = int(np.flatnonzero(np.array([str(e).split("/")[0] == task for e in self.wep]))[0])
            st = np.asarray(self._grp(q).state(self.state_group), np.float64)
            at = A.EEF[side]
            self.idle[side] = st[0, at:at + 10].copy()
        print(f"[SproutIclAug] cf_aug={self.cf_aug or 'off'}", flush=True)

    def _norm_free(self, a, stats, emb):
        """_norm without its clip to [-1, 1] (same values inside the range). The stats come from the original data
        (e.g. the base yaw only ever turned right in cup), so a mirrored or reversed sample lies outside; clipping
        would corrupt its target. Capped at +-cf_aug["zcap"] (4)."""
        s = stats.get(str(emb), stats["_global"])
        lo, hi = np.asarray(s["lo"]), np.asarray(s["hi"])
        z = self.cf_aug.get("zcap", 4.0)
        return np.clip(2.0 * (a - lo) / (hi - lo + 1e-8) - 1.0, -z, z).astype(np.float32)

    def _plain_build(self, q: int, idx: int, aug: bool, fut_frames: np.ndarray) -> dict:
        if not (aug and self.cf_aug):
            s = super()._plain_build(q, idx, aug, fut_frames)
            s["future_loss_weight"] = np.float32(1.0)
            return s
        if self.base_abs:
            return self._build_abs(q, idx)
        A = self.A
        gq = self._grp(q)
        qw0 = int(self.ww0[q])
        q_emb = self.wemb[q]
        C = self.chunk_size
        side = A.TASK_ARM[str(self.wep[q]).split("/")[0]]
        gh, h = self._demo(q, qw0, aug)
        h = int(min(max(h, 0), gh.n_frames - 1))
        rng = np.random.default_rng(int(torch.randint(0, 2**31 - 1, (1,)).item()))
        f = A.draw(self.cf_aug, rng)
        tgt, cur, dem, dend, fend, wfut, flip = A.apply(
            f, np.asarray(gq.action(self.target_group), np.float64), np.asarray(gq.state(self.state_group), np.float64),
            qw0, np.asarray(gh.state(self.demo_key), np.float64), h, C, side, self.idle, demo29=self.demo29)
        _tstats = self.target_stats if (self.target_group and self.target_stats) else self.state_stats
        q_tgt = self._norm_free(tgt, _tstats, q_emb)
        past = self._norm_free(cur[None], self.state_stats, q_emb).astype(np.float32)
        ret = self._demo_norm(dem, q_emb, clip=False)

        S = self.final_image_size
        blank1 = np.zeros((1, S, S, 3), np.uint8)
        blank4 = np.repeat(blank1, NUM_DUPLICATES, axis=0)

        def rd(g, i):
            im = _decode(g.frame(min(max(int(i), 0), g.n_frames - 1), FRONT), S)
            return np.ascontiguousarray(im[:, ::-1]) if flip else im

        dup = lambda im: np.repeat(im[None], NUM_DUPLICATES, axis=0)  # noqa: E731
        images = np.concatenate([blank1, blank4.copy(), dup(rd(gh, h)), dup(rd(gh, h + dend)), dup(rd(gq, qw0)),
                                 blank4.copy(), dup(rd(gq, qw0 + fend))], axis=0)
        s = self._per_view_sample(q, idx, aug, images, q_tgt, past, ret)
        s["future_loss_weight"] = np.float32(wfut)
        s["cf_aug"] = str(sorted(f))
        return s


def _build_abs_impl(self, q: int, idx: int) -> dict:
    """SproutIclAugDataset._plain_build for base_abs: the factors act on the velocity commands (as stored), then the
    torso poses are integrated from the window's start pose (mirrored with the sample when a mirror factor is on)."""
    A = self.A
    gq = self._grp(q)
    raw = gq.raw
    qw0 = int(self.ww0[q])
    q_emb = self.wemb[q]
    C = self.chunk_size
    side = A.TASK_ARM[str(self.wep[q]).split("/")[0]]
    gh, h = self._demo(q, qw0, True)
    h = int(min(max(h, 0), gh.n_frames - 1))
    rng = np.random.default_rng(int(torch.randint(0, 2**31 - 1, (1,)).item()))
    f = A.draw(self.cf_aug, rng)
    Sv = np.asarray(gh.state("state29"), np.float64)
    tgt, cur, dem, dend, fend, wfut, flip = A.apply(
        f, np.asarray(raw.action(self.target_group), np.float64), np.asarray(raw.state(self.state_group), np.float64),
        qw0, Sv, h, C, side, self.idle, demo29=True, full_demo=True)
    sgn = -1.0 if f.get("mirror") else 1.0
    xyR, qR = gq.xy[min(qw0, len(gq.q) - 1)] * np.array([1.0, sgn]), gq.q[min(qw0, len(gq.q) - 1)] * sgn
    dxy, dq = A.poses(Sv[:h])
    xyD, qD = dxy[-1] * np.array([1.0, sgn]), dq[-1] * sgn
    tgt = tgt.copy()
    if self.base_rel:                          # relative to the current pose: integrate from the origin
        tgt[:, :9] = A.window_abs(tgt, np.zeros(2), 0.0)
    elif self.base_abs_out:
        tgt[:, :9] = A.window_abs(tgt, xyR, qR)
    cur = cur.copy(); cur[:9] = A.abs_head(xyR[None], np.array([qR]))[0]
    d = dem[1:].copy()
    d[:, :9] = A.window_abs(dem[:-1], np.zeros(2), 0.0) if self.base_rel else A.window_abs(dem[:-1], xyD, qD)
    _tstats = self.target_stats if (self.target_group and self.target_stats) else self.state_stats
    q_tgt = self._norm_free(tgt, _tstats, q_emb)
    past = self._norm_free(cur[None], self.state_stats, q_emb).astype(np.float32)
    ret = self._demo_norm(d, q_emb, clip=False)
    S = self.final_image_size
    blank1 = np.zeros((1, S, S, 3), np.uint8)
    blank4 = np.repeat(blank1, NUM_DUPLICATES, axis=0)

    def rd(g, i):
        im = _decode(g.frame(min(max(int(i), 0), g.n_frames - 1), FRONT), S)
        return np.ascontiguousarray(im[:, ::-1]) if flip else im

    dup = lambda im: np.repeat(im[None], NUM_DUPLICATES, axis=0)  # noqa: E731
    images = np.concatenate([blank1, blank4.copy(), dup(rd(gh, h)), dup(rd(gh, h + dend)), dup(rd(gq, qw0)),
                             blank4.copy(), dup(rd(gq, qw0 + fend))], axis=0)
    s = self._per_view_sample(q, idx, True, images, q_tgt, past, ret)
    s["future_loss_weight"] = np.float32(wfut)
    s["cf_aug"] = str(sorted(f))
    return s


SproutIclAugDataset._build_abs = _build_abs_impl


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
