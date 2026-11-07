# SPDX-License-Identifier: Apache-2.0
"""Tabletop-Sim (TwinVLA bimanual ALOHA) post-train dataset — GR1 SHORT layout.

Thin subclass of :class:`EgoVerseGr1RetShortStateDataset`. The 57-D representation, the
latent layout, the chunk size and the normalization all stay exactly as the GR1 SHORT
state stack defines them, so ``egoverse_ret_short_state_v11`` warm-starts with no shape
gaps. Only two things are swapped:

* **Storage.** GR1 episodes are zarr-v3 stores read by ``EgoZarrEpisode``; ours are one
  HDF5 per episode written by ``convert_tabletop_to_egoverse57.py``. :class:`TabletopEpisode`
  is a duck type of the zarr reader — same four methods the loader touches
  (``n_frames`` / ``frame`` / ``state`` / ``action``) — so ``_grid_frame``, ``_chunk``,
  ``build_sample`` and everything above them are inherited untouched.

* **Retrieval.** There is no pair pool yet, so ``pairs.npz`` carries ``match_idx = -1``
  everywhere and every sample takes the ``has_ret = 0`` branch (``retrieved_actions``
  zeroed, layout unchanged). ``blank_retrieval=True`` makes that explicit rather than
  incidental — it stays blank even if someone later fills the manifest in. Precedent:
  ``cosmos_predict2p5_2b_egoverse_ret_v3_448_state_blankret_v11``.

Layout (inherited, ``with_proprio=True``, ``state_t=4``, 13 pixel frames)::

    0  cur_state   past state57 chunk, tiled          cond
    1  ret_action  BLANK                              cond
    2  cur_image   448 2x2 grid x4 -> 1 latent        cond
    3  action      32 x 57 abs_action57               GEN
    -- num_conditional_frames = 3 --

Grid slots (``convert_tabletop_to_egoverse57.py`` writes the same names the zarr reader
uses, so ``_grid_frame`` needs no override)::

    front_1 = back(teleoperator_pov) TL | sideleft = wrist_left  TR
    sideright = wrist_right          BL | center_top = front     BR (use_center_top only)

Smoke check::

    python -m cosmos_policy.datasets.tabletop_sim_dataset
"""
from __future__ import annotations

import os

import h5py
import numpy as np

from cosmos_policy.datasets.egoverse_gr1_ret_dataset import FRONT, SIDELEFT, SIDERIGHT, _decode
import torch

from cosmos_policy.datasets.dataset_utils import preprocess_image
from cosmos_policy.datasets.egoverse_gr1_plain_dataset import EgoVerseGr1PlainStateDataset
from cosmos_policy.datasets.egoverse_gr1_ret_dataset import ACTION_DIM, STATE_DIM
from cosmos_policy.datasets.tabletop_convert import DIM20, decode_57_to_20
from cosmos_policy.datasets.egoverse_gr1_ret_v2_dataset import DT, NATIVE, NUM_DUPLICATES

# proprio 슬롯에 넣는 과거 프레임 수. 1 = 현재 상태만 (아래 _plain_build 주석 참고).
PROPRIO_HIST = 1
from cosmos_policy.datasets.egoverse_ret_v2_dataset import (
    NOW_S,
    ROLE_ACTION,
    ROLE_CUR_IMAGE,
    ROLE_CUR_STATE,
    ROLE_FUTURE,
    ROLE_RET_ACTION,
    ROLE_RET_VIDEO,
)
from cosmos_policy.datasets.egoverse_gr1_ret_short_dataset import (
    EgoVerseGr1RetShortStateDataset,
)

# Tabletop-Sim data root (data/ + manifest/, HF Jeongeun/tabletopsim-watch); TABLETOP_ROOT overrides it.
TABLETOP_ROOT = os.environ.get("TABLETOP_ROOT", "/mnt/ddn/dataset/tabletop_ego57")
DEFAULT_MANIFEST = f"{TABLETOP_ROOT}/manifest"


class TabletopEpisode:
    """HDF5-backed stand-in for ``EgoZarrEpisode`` (one converted Tabletop episode).

    The file handle is opened lazily and dropped on pickling: the dataset's ``_zcache``
    is cleared in ``__getstate__``, so every dataloader worker opens its own handles.
    """

    def __init__(self, path: str):
        self.path = path
        self._f: h5py.File | None = None
        self._cache: dict[str, np.ndarray] = {}
        with h5py.File(path, "r") as f:
            self._n = int(f.attrs["n_frames"])

    # ---- pickling ----
    def __getstate__(self):
        s = self.__dict__.copy()
        s["_f"] = None
        s["_cache"] = {}
        return s

    def __setstate__(self, s):
        self.__dict__.update(s)

    # ---- reader surface used by the loader ----
    @property
    def n_frames(self) -> int:
        return self._n

    def _file(self) -> h5py.File:
        if self._f is None:
            self._f = h5py.File(self.path, "r")
        return self._f

    def frame(self, i: int, cam: str = FRONT) -> bytes:
        """Raw JPEG bytes for frame ``i`` of ``cam``; ``b''`` when absent.

        ``cam`` arrives in the zarr array-dir spelling (``images.front_1``); our HDF5
        nests it as ``images/front_1``.
        """
        key = cam.replace(".", "/")
        f = self._file()
        if key not in f:
            return b""
        return f[key][min(max(int(i), 0), self._n - 1)].tobytes()

    def _read_2d(self, name: str) -> np.ndarray:
        arr = self._cache.get(name)
        if arr is None:
            f = self._file()
            if name not in f:
                raise KeyError(f"{self.path}: no group '{name}'")
            arr = np.asarray(f[name][()], np.float64)
            self._cache[name] = arr
        return arr

    def state(self, group: str) -> np.ndarray:
        return self._read_2d(group)

    def action(self, group: str) -> np.ndarray:
        return self._read_2d(group)

    def has_state(self, group: str) -> bool:
        return group in self._file()


class _TabletopStoreMixin:
    """Swaps the egoverse zarr store for the Tabletop HDF5 store, and guards T5 coverage.

    Everything else — layout, roles, t_positions, normalization, target — is inherited
    from the GR1 dataset being subclassed, so the two Tabletop variants stay in lockstep
    with their GR1 counterparts by construction.
    """

    def _tabletop_init(self, dataset_root: str, require_t5: bool) -> None:
        self._apply_task_filter()
        self._check_t5_coverage(require_t5)
        print(
            f"[TabletopSim] {type(self).__name__} root={dataset_root} windows={self.N} "
            f"target={self.target_group} state_t={self.state_t} "
            f"chunk_duration={self.chunk_duration} n_cond={getattr(self, '_n_cond', 'see layout')}",
            flush=True,
        )

    @property
    def act_dim(self) -> int:
        return DIM20 if getattr(self, "native20", False) else ACTION_DIM

    @property
    def st_dim(self) -> int:
        return DIM20 if getattr(self, "native20", False) else STATE_DIM

    def _apply_task_filter(self) -> None:
        """task_filter 가 있으면 그 task 의 윈도우만 남긴다 (task 전용 모델 학습용).

        TwinVLA 는 task 당 전용 모델을 학습한다 (train_twinvla.sh 가 data_mix 로 단일
        데이터셋 이름을 받고, mixtures.py 에 aloha mixture 가 없어 [(name, 1.0)] 으로 폴백).
        5 task 공유 모델과 비교하려면 같은 축이 필요해서 넣었다. wep 은
        "<task>/episode_N" 형식이므로 접두어로 거른다.
        """
        tf = getattr(self, "task_filter", None)
        if not tf:
            return
        keep = np.array([str(e).split("/")[0] in tf for e in self.wep], bool)
        n0 = len(self.wep)
        for a in ("wep", "ww0", "wcls", "wemb", "query_idx"):
            if hasattr(self, a):
                setattr(self, a, np.asarray(getattr(self, a))[keep])
        if hasattr(self, "match_idx"):
            self.match_idx = np.asarray(self.match_idx)[keep]
        self.N = int(keep.sum())
        # query_idx 는 원본 윈도우 번호를 담고 있어 필터 후 재색인해야 build_sample(q) 가 맞는다.
        self.query_idx = np.arange(self.N, dtype=np.int64)
        print(f"[TabletopSim] task_filter={sorted(tf)} -> windows {n0} -> {self.N}", flush=True)

    def _grp(self, wi: int) -> "TabletopEpisode":
        ep = str(self.wep[wi])
        g = self._zcache.get(ep)
        if g is None:
            g = TabletopEpisode(os.path.join(self.dataset_root, ep + ".h5"))
            self._zcache[ep] = g
        return g

    def _check_t5_coverage(self, require: bool) -> None:
        """Every ``cls`` (= the episode's instruction) must have a T5 embedding.

        The base falls back to ``zeros(512, 1024)`` on a miss, which trains silently on
        no language at all. ``aloha_box_into_pot_easy`` alone carries 6 instructions
        (3 box colours x 2 pots), so a stale pickle is easy to end up with — see
        docs/robocasa_all24_t5.md §0 for what that costs.
        """
        need = {str(c) for c in self.wcls}
        missing = sorted(need - set(self.t5_text_embeddings))
        msg = (
            f"[TabletopSim] {len(missing)}/{len(need)} instructions missing from the T5 "
            f"pickle -> they would train on ZERO text embeddings: {missing[:6]}"
        )
        if missing:
            if require:
                raise KeyError(msg)
            print(msg + "  (require_t5=False, continuing)", flush=True)
        else:
            print(f"[TabletopSim] T5 coverage OK ({len(need)} instructions)", flush=True)


class TabletopSimShortStateDataset(_TabletopStoreMixin, EgoVerseGr1RetShortStateDataset):
    """Tabletop-Sim 5-task post-train, SHORT layout, ``abs_action57`` target."""

    def __init__(
        self,
        *args,
        dataset_root: str = TABLETOP_ROOT,
        blank_retrieval: bool = True,
        require_t5: bool = True,
        **kw,
    ):
        # set before super(): _grp may be reached from the base ctor's stats pass
        self.dataset_root = dataset_root
        self.blank_retrieval = bool(blank_retrieval)
        super().__init__(*args, **kw)
        self._check_t5_coverage(require_t5)
        print(
            f"[TabletopSim] root={dataset_root} windows={self.N} "
            f"blank_ret={self.blank_retrieval} target={self.target_group} "
            f"state_t={self.state_t} chunk_duration={self.chunk_duration}",
            flush=True,
        )

    def bucket_key(self, r: int):
        """Single fixed shape -> one bucket. Overridden only to stop the inherited
        ``("gr1", "md")`` label showing up in this dataset's sampler logs."""
        return ("tabletop", "blank")

    def __getitem__(self, idx: int) -> dict:
        if not self.blank_retrieval:
            return super().__getitem__(idx)
        if self.debug:
            idx = 0
        q = int(self.query_idx[idx % self.N])
        # m = q only to keep the signature happy; has_ret=0 zeroes retrieved_actions,
        # so nothing from the query leaks into its own conditioning slot.
        return self.build_sample(q, q, idx=idx, has_ret=0, aug=True, rand_delta=True)


class _PlainSampleMixin:
    """Shared PLAIN sample assembly for both future-block variants.

    The two arms differ ONLY in which frames fill the future block, so they build the
    sample through one code path — a divergence between them would otherwise be invisible
    and would silently confound the comparison.

    Future block convention (both arms): the frames the action chunk actually PRODUCES,
    i.e. ``s[t+1 .. t+32]``. The action chunk is ``act[t .. t+31]`` and ``act[k]`` lands on
    ``s[k+1]``, so the block starts at ``t+1``. NOTE this differs from GR1's plain dataset,
    which reuses ``_grid_clip(g, w0)`` = ``[t .. t+31]`` and therefore re-shows the current
    frame as its first future latent.
    """

    def _plain_build_nofp(self, q, idx, aug, gq, qw0, q_tgt, past, S, blank1, blank4) -> dict:
        """RoboCasa ``..._t5_nofp`` 레이아웃 이식 (value 슬롯 제외).

        stock RoboCasa 슬롯표는 run_robocasa_eval.py:106 에 그대로 있다::

            0 blank | 1 curr proprio | 2 curr wrist | 3 curr primary | 4 curr secondary
            | 5 action | 6 future proprio | 7 future wrist | 8 future primary
            | 9 future secondary | 10 value                     -> state_t 11, cd 41

        nofp 는 6번(future proprio)만 뺀다 -> state_t 10, cd 37. 여기서 10번(value)까지
        빼서 9 슬롯으로 만든다::

            0 blank      (복제 안 하는 토크나이저 첫 프레임)
            1 proprio    (blank x4, 자기 슬롯)
            2 front_1    (t)
            3 sideleft   (t)
            4 sideright  (t)
            -- num_conditional_frames = 5 --
            5 action     (GEN)
            6 front_1    (t+32)
            7 sideleft   (t+32)
            8 sideright  (t+32)

            state_t = 9, chunk_duration = 1 + 4*8 = 33

        per_view 판과의 차이는 두 가지다:
          * proprio 가 latent 0 에 얹히지 않고 자기 슬롯(1)을 갖는다 — stock RoboCasa 방식
          * slot_roles / t_positions 를 내보내지 않는다. stock 은 bridge_layout=False 라
            bridge_pe 가 빈 dict 이고 모델의 기본 PE 를 쓴다 (= "PE 기본").

        카메라->슬롯 매핑: RoboCasa 는 wrist / primary / secondary 순서다. Tabletop 에는
        wrist 가 없으므로 슬롯 순서는 그대로 두고 front_1 / sideleft / sideright 를 채운다
        (기존 per-view 빌더와 같은 카메라 순서). 키 이름은 RoboCasa 슬롯 위치를 따른다.
        """
        n = gq.n_frames
        rd = lambda i, cam: _decode(gq.frame(min(max(i, 0), n - 1), cam), S)
        fut_i = qw0 + self.chunk_size                            # t+32
        dup = lambda im: np.repeat(im[None], NUM_DUPLICATES, axis=0)

        images = np.concatenate([
            blank1,                                              # 0 토크나이저 blank
            blank4.copy(),                                       # 1 proprio (자기 슬롯)
            dup(rd(qw0, FRONT)),                                 # 2
            dup(rd(qw0, SIDELEFT)),                              # 3
            dup(rd(qw0, SIDERIGHT)),                             # 4
            blank4.copy(),                                       # 5 action (GEN)
            dup(rd(fut_i, FRONT)),                               # 6
            dup(rd(fut_i, SIDELEFT)),                            # 7
            dup(rd(fut_i, SIDERIGHT)),                           # 8
        ], axis=0)
        assert images.shape[0] == self.chunk_duration, (images.shape, self.chunk_duration)
        assert self.state_t == 9, f"nofp layout is state_t=9, got {self.state_t}"
        images = preprocess_image(images, final_image_size=S, normalize_images=False,
                                  use_image_aug=aug, stronger_image_aug=aug)

        cls = self.wcls[q]
        t5 = (torch.squeeze(self.t5_text_embeddings[cls]) if cls in self.t5_text_embeddings
              else torch.zeros(512, 1024))
        q_emb = self.wemb[q]
        # slot_roles / t_positions 는 일부러 넣지 않는다 (PE 기본).
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
            "embodiment": q_emb, "retrieved_embodiment": q_emb,
            "num_conditional_frames": 5,
            "retrieved_video_start_latent_idx": -1,
            "retrieved_video_end_latent_idx": -1,
            "retrieved_action_latent_idx": -1,
            "retrieved_state_latent_idx": -1,
            "current_proprio_latent_idx": 1,
            "current_wrist_image_latent_idx": 2,
            "current_image_latent_idx": 3,
            "current_image2_latent_idx": 4,
            "current_wrist_image2_latent_idx": -1,
            "action_latent_idx": 5,
            "future_video_start_latent_idx": 6,
            "future_video_end_latent_idx": 9,
            "future_wrist_image_latent_idx": 6,
            "future_image_latent_idx": 7,
            "future_image2_latent_idx": 8,
            "future_wrist_image2_latent_idx": -1,
            "future_proprio_latent_idx": -1,
            "value_latent_idx": -1,
            "proprio": past.reshape(-1).astype(np.float32),
            "future_proprio": np.zeros(self.st_dim, np.float32),
            "has_ret_data": 0, "has_ret_image": 0, "has_current_image": 1,
            "rollout_data_mask": 0, "rollout_data_success_mask": 0,
            "world_model_sample_mask": 0, "value_function_sample_mask": 0,
            "global_rollout_idx": -1, "value_function_return": float("-100"),
            "next_value_function_return": float("-100"),
            "next_action_chunk": q_tgt.copy().astype(np.float32),
        }

    def _plain_build_per_view(self, q, idx, aug, gq, qw0, q_tgt, past, S, blank1, blank4) -> dict:
        """b2 스타일 뷰 분리. 그리드 한 장 대신 카메라 3대가 각자 latent 를 차지한다.

        RoboCasa ``bridge_b2`` (bridge_layout + ret_action + ret_video=1 + future_stagger)
        의 PE 를 그대로 옮긴 것이다::

            idx role            t          내용
             0  CUR_STATE       NOW        blank (proprio 는 latent 안에 실림)
             1  RET_ACTION      NOW        zero-pixel
             2  RET_VIDEO       NOW - dt   zero-pixel
             3  CUR_IMAGE       NOW        front      (t)
             4  CUR_IMAGE       NOW        sideleft   (t)
             5  CUR_IMAGE       NOW        sideright  (t)
            -- num_conditional_frames = 6 (액션 슬롯 직전) --
             6  ACTION          NOW        GEN
             7  FUTURE          NOW + dt   front      (t+32)
             8  FUTURE          NOW + 2dt  sideleft
             9  FUTURE          NOW + 3dt  sideright

        state_t = 10, chunk_duration = 1 + 4*9 = 37.

        b2 가 고친 버그를 그대로 반영한다: n_cond 가 액션 슬롯 직전이 아니면 마지막 카메라가
        경계 밖으로 밀려 "줘야 할 뷰를 생성"하게 된다. 초기 loss 로는 안 드러나고 20K 에서
        future_image 가 0.097 -> 0.33 으로 벌어졌다.
        """
        n = gq.n_frames
        rd = lambda i, cam: _decode(gq.frame(min(max(i, 0), n - 1), cam), S)
        fut_i = qw0 + self.chunk_size                            # t+32
        dup = lambda im: np.repeat(im[None], NUM_DUPLICATES, axis=0)
        nb = self.per_view_blank                                 # prefix 슬롯 수 (b2 = 2)

        nf = bool(getattr(self, "no_future", False))              # FUTURE 슬롯 자체를 뺀다
        images = np.concatenate([
            blank1,                                              # 0 cur_state
            *[blank4.copy() for _ in range(nb)],                 # ret_action / ret_video
            dup(rd(qw0, FRONT)),                                 # cur front
            dup(rd(qw0, SIDELEFT)),                              # cur sideleft
            dup(rd(qw0, SIDERIGHT)),                             # cur sideright
            blank4.copy(),                                       # action  (GEN)
        ] + ([] if nf else [
            dup(rd(fut_i, FRONT)),                               # future front
            dup(rd(fut_i, SIDELEFT)),                            # future sideleft
            dup(rd(fut_i, SIDERIGHT)),                           # future sideright
        ]), axis=0)
        assert images.shape[0] == self.chunk_duration, (images.shape, self.chunk_duration)
        images = preprocess_image(images, final_image_size=S, normalize_images=False,
                                  use_image_aug=aug, stronger_image_aug=aug)

        T = self.state_t
        assert T == (5 if nf else 8) + nb, \
            f"per_view(nb={nb}, no_future={nf}) layout is state_t={(5 if nf else 8)+nb}, got {T}"
        c0 = 1 + nb                                              # 첫 카메라 슬롯
        a0 = c0 + 3                                              # action 슬롯
        f0 = a0 + 1                                              # 첫 future 슬롯
        roles = np.empty(T, np.int64); tpos = np.empty(T, np.float32)
        roles[0], tpos[0] = ROLE_CUR_STATE, NOW_S
        if nb >= 1: roles[1], tpos[1] = ROLE_RET_ACTION, NOW_S
        if nb >= 2: roles[2], tpos[2] = ROLE_RET_VIDEO, NOW_S - DT
        roles[c0:c0+3], tpos[c0:c0+3] = ROLE_CUR_IMAGE, NOW_S    # 세 뷰가 role/t 를 공유 (b2)
        roles[a0], tpos[a0] = ROLE_ACTION, NOW_S
        if not nf:
            roles[f0:f0+3] = ROLE_FUTURE
            tpos[f0:f0+3] = NOW_S + DT * np.arange(1, 4)         # stagger (b2: True)

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
            "embodiment": q_emb, "retrieved_embodiment": q_emb,
            "t_positions": tpos, "slot_roles": roles,
            "num_conditional_frames": a0,
            "retrieved_video_start_latent_idx": 2 if nb >= 2 else -1,
            "retrieved_video_end_latent_idx": 3 if nb >= 2 else -1,
            "retrieved_action_latent_idx": 1 if nb >= 1 else -1,
            "retrieved_state_latent_idx": -1,
            "current_image_latent_idx": c0,
            "current_image2_latent_idx": c0 + 1,
            "current_wrist_image_latent_idx": c0 + 2,
            "current_wrist_image2_latent_idx": -1,
            "action_latent_idx": a0,
            "future_video_start_latent_idx": -1 if nf else f0,
            "future_video_end_latent_idx": -1 if nf else T,
            "future_image_latent_idx": -1 if nf else f0,
            "future_image2_latent_idx": -1 if nf else f0 + 1,
            "future_wrist_image_latent_idx": -1 if nf else f0 + 2,
            "future_wrist_image2_latent_idx": -1,
            "proprio": past.reshape(-1).astype(np.float32),
            "future_proprio": np.zeros(self.st_dim, np.float32),
            "current_proprio_latent_idx": 0,
            "future_proprio_latent_idx": -1,
            "value_latent_idx": -1,
            "has_ret_data": 0, "has_ret_image": 0, "has_current_image": 1,
            "rollout_data_mask": 0, "rollout_data_success_mask": 0,
            "world_model_sample_mask": 0, "value_function_sample_mask": 0,
            "global_rollout_idx": -1, "value_function_return": float("-100"),
            "next_value_function_return": float("-100"),
            "next_action_chunk": q_tgt.copy().astype(np.float32),
        }

    def _plain_build(self, q: int, idx: int, aug: bool, fut_frames: np.ndarray) -> dict:
        gq = self._grp(q)
        qw0 = int(self.ww0[q])
        q_emb = self.wemb[q]
        q_state = gq.state(self.state_group)

        if self.target_group is not None:
            q_tgt = self._chunk(np.asarray(gq.action(self.target_group), np.float64), qw0)
        else:
            q_tgt = self._chunk(q_state, qw0 + 1)
        # 2026-08-23: NATIVE(32) -> PROPRIO_HIST(1). proprio 슬롯은 CURRENT state 만 받는다.
        #
        # 과거 청크는 egoverse STATE-CHUNK 계약에서 온 것인데 Tabletop 롤아웃에서는 재현이
        # 불가능했다: 클라이언트가 /predict 를 청크당 한 번만 부르므로 서버가 쌓는 "과거 32
        # 프레임"은 실제로 32 스텝 간격으로 떨어진 샘플이고, 학습의 연속 s[t-31..t] 와
        # 정규화 [-1,1] 공간에서 L1 0.62~0.78 만큼 어긋났다 (실측, query 1 이후 전 구간).
        # 첫 청크만 일치하고 그 뒤 조건화가 통째로 분포 밖으로 나가므로 teacher forcing 은
        # 정확한데 롤아웃 성공률이 2/100 에 붙어 있었다.
        past = self._past_chunk(q_state, qw0, PROPRIO_HIST).astype(np.float32)
        if self.native20:
            # 정규화 직전에 20-D 로 내린다 — 통계도 20-D 짜리를 쓴다.
            q_tgt = decode_57_to_20(q_tgt)
            past = decode_57_to_20(past).astype(np.float32)
        if self.normalize_state:
            _tstats = self.target_stats if (self.target_group and self.target_stats) else self.state_stats
            q_tgt = self._norm(q_tgt, _tstats, q_emb)
            past = self._norm(past, self.state_stats, q_emb)

        S = self.final_image_size
        blank1 = np.zeros((1, S, S, 3), np.uint8)
        blank4 = np.repeat(blank1, NUM_DUPLICATES, axis=0)
        if getattr(self, "nofp", False):
            return self._plain_build_nofp(q, idx, aug, gq, qw0, q_tgt, past, S, blank1, blank4)
        if self.per_view:
            return self._plain_build_per_view(q, idx, aug, gq, qw0, q_tgt, past, S, blank1, blank4)
        cur = self._grid_frame(gq, qw0)[None]                    # cur_image = frame t
        # BRIDGE prefix (bridge_ret=True): egoverse mid 계약의 zero-pixel RET 슬롯 2개를
        # cur_image 앞에 넣는다. warm-start 체크포인트(egoverse_ret_v3_448_state)가 본 구조가
        #   0 cur_state | 1 ret_action | 2..1+k ret_video | 2+k cur_image | action | future
        # 이고 n_cond = k+3 이다 (egoverse_ret_v2_dataset.py 모듈 docstring). RoboCasa 가
        # 같은 이유로 bridge_ret_action / bridge_ret_video 를 넣었다. 픽셀은 0 이고 역할과
        # rope 시각만 남는다 — 백본이 학습 때 본 prefix 구조를 그대로 보게 하는 것이 목적.
        pre = [blank4.copy(), blank4.copy()] if self.bridge_ret else []
        images = np.concatenate([
            blank1,                                              # cur_state
            *pre,                                                # ret_action, ret_video (BRIDGE)
            np.repeat(cur, NUM_DUPLICATES, axis=0),              # cur_image -> 1 latent
            blank4,                                              # action    (GEN)
            fut_frames,                                          # future    (GEN)
        ], axis=0)
        assert images.shape[0] == self.chunk_duration, (images.shape, self.chunk_duration)
        images = preprocess_image(images, final_image_size=S, normalize_images=False,
                                  use_image_aug=aug, stronger_image_aug=aug)

        T = self.state_t
        nb = 2 if self.bridge_ret else 0        # bridge prefix latents
        K = T - 3 - nb                          # future latents
        roles = np.empty(T, np.int64)
        tpos = np.empty(T, np.float32)
        roles[0], tpos[0] = ROLE_CUR_STATE, NOW_S
        if self.bridge_ret:
            roles[1], tpos[1] = ROLE_RET_ACTION, NOW_S          # 데모 액션 자리 (zero)
            roles[2], tpos[2] = ROLE_RET_VIDEO, NOW_S - DT      # 데모 프레임 자리, PAST 시각
        roles[1+nb], tpos[1+nb] = ROLE_CUR_IMAGE, NOW_S
        roles[2+nb], tpos[2+nb] = ROLE_ACTION, NOW_S
        roles[3+nb:], tpos[3+nb:] = ROLE_FUTURE, NOW_S + DT * np.arange(1, K + 1)

        cls = self.wcls[q]
        t5 = (torch.squeeze(self.t5_text_embeddings[cls]) if cls in self.t5_text_embeddings
              else torch.zeros(512, 1024))
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
            "embodiment": q_emb, "retrieved_embodiment": q_emb,
            "t_positions": tpos, "slot_roles": roles,
            # n_cond = k+3 (mid 계약). plain: 2 (cur_state+cur_image), bridge: 4.
            "num_conditional_frames": 2 + nb,
            "retrieved_video_start_latent_idx": 2 if self.bridge_ret else -1,
            "retrieved_video_end_latent_idx": 3 if self.bridge_ret else -1,
            "retrieved_action_latent_idx": 1 if self.bridge_ret else -1,
            "retrieved_state_latent_idx": -1,
            "current_image_latent_idx": 1 + nb,
            "future_video_start_latent_idx": 3 + nb,
            "future_video_end_latent_idx": T,
            "action_latent_idx": 2 + nb,
            "proprio": past.reshape(-1).astype(np.float32),
            "future_proprio": np.zeros(self.st_dim, np.float32),
            "current_proprio_latent_idx": 0,
            "current_wrist_image_latent_idx": -1, "current_wrist_image2_latent_idx": -1,
            "current_image2_latent_idx": -1, "future_proprio_latent_idx": -1,
            "future_wrist_image_latent_idx": -1, "future_wrist_image2_latent_idx": -1,
            "future_image_latent_idx": 3 + nb, "future_image2_latent_idx": -1,
            "value_latent_idx": -1,
            "has_ret_data": 0, "has_ret_image": 0, "has_current_image": 1,
            "rollout_data_mask": 0, "rollout_data_success_mask": 0,
            "world_model_sample_mask": 0, "value_function_sample_mask": 0,
            "global_rollout_idx": -1, "value_function_return": float("-100"),
            "next_value_function_return": float("-100"),
            "next_action_chunk": q_tgt.copy().astype(np.float32),
        }


class TabletopSimPlainStateDataset(_PlainSampleMixin, _TabletopStoreMixin,
                                   EgoVerseGr1PlainStateDataset):
    """Tabletop-Sim 5-task post-train in the GR1 PLAIN (BC) layout — FULL future video.

    Structural twin of ``cosmos_predict2p5_2b_egoverse_gr1_plain_state_v6``::

        slot:  cur_state | cur_image | action | future x8
        role:      0     |     3     |   4    |   5
        time:     NOW    |    NOW    |  NOW   | NOW+dt .. NOW+8*dt
        cond:     yes    |    yes    |  GEN   |  GEN
        state_t = 11, chunk_duration = 41, num_conditional_frames = 2

    Future block = frames ``t+1 .. t+32`` (see ``_PlainSampleMixin``), the 32 observations
    the predicted action chunk produces. The retrieval channel is removed outright rather
    than zeroed. This is the arm that makes PLAIN ~4x slower per iter than PLAIN-END.
    """

    def __init__(self, *args, dataset_root: str = TABLETOP_ROOT, require_t5: bool = True,
                 bridge_ret: bool = False, task_filter: list | None = None,
                 native20: bool = False,
        per_view: bool = False,
        per_view_blank: int = 2,
        nofp: bool = False, **kw):
        self.dataset_root = dataset_root
        # native20: 57-D egoverse 인코딩을 거치지 않고 Tabletop 원본 20-D
        # (팔당 xyz3 + rot6d6 + grip1) 로 학습/추론한다. 57-D 는 egoverse warm-start 계약을
        # 맞추려던 것이라 warm start 가 없는 arm 에는 이득이 없고, 11/57 이 상수 채널
        # (head9 + hand15 의 dim 24/48) 이라 그만큼이 낭비다. abs_action20 은 상수 채널이 0 이다.
        # 저장소에 20-D 는 없지만 decode_57_to_20 이 정확한 역변환이라 온더플라이로 만든다
        # (test_tabletop_action57_roundtrip.py 14/14 통과).
        self.native20 = bool(native20)
        # 뷰 분리 (b2 스타일). 2x2 그리드 한 장 대신 카메라 3대를 각자 latent 슬롯에 넣는다.
        # PE 규약은 RoboCasa b2 (bridge_layout + bridge_ret_action + bridge_ret_video=1 +
        # bridge_future_stagger) 와 글자 그대로 맞춘다:
        #   * 카메라 3슬롯은 전부 ROLE_CUR_IMAGE / t = NOW  (뷰 정체성은 role 이 아니라 순서)
        #   * future 3슬롯은 전부 ROLE_FUTURE, t 는 NOW+dt, NOW+2dt, NOW+3dt 로 stagger
        #   * RET_ACTION(NOW) + RET_VIDEO(NOW-dt) zero-pixel prefix 를 그대로 둔다
        #   * num_conditional_frames 는 액션 슬롯 직전에서 끊는다 (b2 가 5->6 으로 고친 그 버그)
        self.per_view = bool(per_view)
        # zero-pixel prefix 슬롯 수. 2 = b2 (RET_ACTION + RET_VIDEO), 1 = RET_ACTION 만,
        # 0 = prefix 없음. 픽셀이 0 이라 정보는 없고 role/rope 구조만 남는다 — 백본이
        # mid 학습 때 본 구조를 얼마나 되찾는지를 재려고 눈금으로 뺐다.
        self.per_view_blank = int(per_view_blank)
        # nofp: RoboCasa 의 stock 레이아웃에서 future proprio 슬롯을 뺀 판
        # (cosmos_predict2p5_2b_480p_robocasa_all_24_tasks_t5_nofp) 를 그대로 옮긴 것.
        # per_view 와 달리 (a) proprio 가 자기 슬롯을 갖고 (b) slot_roles / t_positions 를
        # 아예 내보내지 않는다 — stock RoboCasa 는 bridge_layout=False 라 bridge_pe 가 빈
        # dict 이고 모델의 기본 PE 를 쓴다. value 슬롯도 없다 (사용자 확정).
        #   0 blank | 1 proprio | 2,3,4 cur cam x3 | 5 action | 6,7,8 future cam x3
        #   state_t = 9, chunk_duration = 33, num_conditional_frames = 5
        self.nofp = bool(nofp)
        # ⚠ no_future 는 **여기서 받지 않는다**. 부모 EgoVerseGr1RetV2StateDataset 가 이미
        # 같은 이름의 파라미터를 갖고 self.no_future 를 세팅하므로, 여기서 시그니처로
        # 가로채면 부모에게 안 넘어가고 부모가 기본값 False 로 **덮어쓴다**.
        # 2026-09-17 실제 사고: ap_actonly / ap_mid 가 state_t=5 여야 하는데 8 로 학습됐다
        # (ap_base 만 우연히 정상). **kw 로 흘려보내면 부모가 super().__init__() 앞에서
        # self.no_future 를 세팅하므로 state_t 프로퍼티가 제때 올바른 값을 본다.
        # per_view 판에서의 의미: FUTURE 카메라 3슬롯을 시퀀스에서 통째로 뺀다
        # (state_t 8+nb -> 5+nb, chunk_duration 29 -> 17, future 프레임 디코드 안 함).
        self.task_filter = list(task_filter) if task_filter else None
        # bridge_ret: zero-pixel RET_ACTION + RET_VIDEO 슬롯을 cur_image 앞에 넣어
        # egoverse mid 계약(k=1)을 맞춘다. warm-start arm 에서만 의미가 있다.
        self.bridge_ret = bool(bridge_ret)
        super().__init__(*args, **kw)
        self._tabletop_init(dataset_root, require_t5)

    @property
    def state_t(self) -> int:
        # 부모(EgoVerseGr1PlainStateDataset)의 K+3 에 bridge prefix 2 를 더한다.
        # chunk_duration 은 부모가 1 + (state_t-1)*NUM_DUPLICATES 로 유도하므로 자동으로 따라온다.
        base = EgoVerseGr1PlainStateDataset.state_t.fget(self)
        return base + (2 if getattr(self, "bridge_ret", False) else 0)

    def bucket_key(self, r: int):
        return ("tabletop", "plain")

    def build_sample(self, q: int, m: int = -1, idx: int = 0, has_ret: int = 0,
                     aug: bool = False, rand_delta: bool = False) -> dict:
        del m, has_ret, rand_delta
        gq = self._grp(q)
        fut = self._grid_clip(gq, int(self.ww0[q]) + 1)          # frames t+1 .. t+32
        return self._plain_build(q, idx, aug, fut)


class TabletopSimPlainEndImageDataset(TabletopSimPlainStateDataset):
    """PLAIN layout, but the future block is the SINGLE chunk-END frame ``t+32``.

    Same generation targets as the GR1 v1 post-train ("GENERATE = action chunk + chunk-END
    image"), with one camera grid instead of three::

        slot:  cur_state | cur_image | action | future(t+32)
        role:      0     |     3     |   4    |   5
        time:     NOW    |    NOW    |  NOW   | NOW + dt
        state_t = 4, chunk_duration = 13, num_conditional_frames = 2

    Why it exists: the full PLAIN arm generates all 32 future frames (8 latents), which is
    what makes it ~4x slower. Predicting only where the chunk LANDS keeps the "imagine the
    outcome" auxiliary signal at the SHORT arm's sequence length. Its target ``t+32`` is
    the last frame of PLAIN's ``t+1 .. t+32`` block, so the two arms are nested rather
    than offset.
    """

    FUTURE_K = 1

    @property
    def state_t(self) -> int:            # cur_state [+ret_action +ret_video] + cur_image + action + future
        if getattr(self, "nofp", False):
            # blank + proprio + cam3 + action + fut3 (RoboCasa nofp 에서 value 만 뺀 것)
            return 9
        if getattr(self, "per_view", False):
            # state + nb prefix + cam3 + action [+ fut3]
            nb = int(getattr(self, "per_view_blank", 2))
            return (5 if getattr(self, "no_future", False) else 8) + nb
        return self.FUTURE_K + 3 + (2 if getattr(self, "bridge_ret", False) else 0)

    def bucket_key(self, r: int):
        return ("tabletop", "plain_end")

    def build_sample(self, q: int, m: int = -1, idx: int = 0, has_ret: int = 0,
                     aug: bool = False, rand_delta: bool = False) -> dict:
        del m, has_ret, rand_delta
        gq = self._grp(q)
        end = self._grid_frame(gq, int(self.ww0[q]) + self.chunk_size)[None]   # t+32
        return self._plain_build(q, idx, aug, np.repeat(end, NUM_DUPLICATES, axis=0))
