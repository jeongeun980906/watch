# SPDX-License-Identifier: Apache-2.0
"""HTTP inference server for the Tabletop-Sim post-train policy (closed-loop sim eval).

Stripped sibling of ``eval_gr1_server.py``: same model path, same SHORT state layout,
but no retrieval pool and no delta targets — Tabletop trains on ``abs_action57`` with a
BLANK ret slot, so the server only has to build one sample shape.

**Why a server at all.** Tabletop-Sim pins ``mujoco==3.2.0`` / ``dm_control==1.0.21`` /
``numpy<2``; this venv is on numpy 2.x. The two cannot share a process, so the sim runs
in its own venv and talks HTTP (the same split ALOHA.md uses).

**Where the split is drawn.** The client sends *raw* Tabletop quantities — four JPEG
camera tiles and the 20-D ``ee_6d_pos`` proprio — and gets a 20-D action chunk back,
ready to hand to ``env.step``. Every bit of 57-D knowledge (frame transform, hand15 lerp,
normalization) stays here and comes from ``tabletop_convert``, the same module the
converter used, so train and serve cannot drift apart.

Protocol::

    POST /reset    {}                                  -> {"ok": true}
    POST /predict  {"images": {cam: <b64 jpeg>, ...},   -> {"actions": [[20] x chunk],
                    "proprio20": [20],                       "chunk": int}
                    "instruction": str}

Launch (this venv, GPU)::

    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. uv run --extra cu128 --group robocasa --python 3.10 \
      python -m cosmos_policy.scripts.tabletop_sim_eval_server \
        --checkpoint /mnt/ddn/tmp/tabletop_sim/.../model_000008000.pt --port 8010
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import torch
from PIL import Image

from cosmos_policy._src.imaginaire.utils import distributed
from cosmos_policy._src.imaginaire.utils.context_managers import distributed_init
from cosmos_policy._src.predict2.utils.model_loader import load_model_from_checkpoint
from cosmos_policy.datasets.dataset_utils import preprocess_image
from cosmos_policy.datasets.egoverse_gr1_ret_v2_dataset import DT
from cosmos_policy.datasets.egoverse_ret_v2_dataset import (
    NOW_S,
    ROLE_ACTION,
    ROLE_CUR_IMAGE,
    ROLE_CUR_STATE,
    ROLE_FUTURE,
    ROLE_RET_ACTION,
    ROLE_RET_VIDEO,
)
from cosmos_policy.datasets.tabletop_convert import DIM20, DIM57, decode_57_to_20, encode_20_to_57
from cosmos_policy.datasets.tabletop_sim_dataset import TABLETOP_ROOT
from cosmos_policy.models.policy_text2world_model import extract_action_chunk_from_latent

CHUNK = 32
NUM_DUP = 4
EMB = "tabletop"
# Grid quadrants, in the order convert_tabletop_to_egoverse57.py writes them.
# TL=back(teleoperator_pov) TR=wrist_left BL=wrist_right BR=front(collaborator_pov).
TILES = (("front_1", 0, 0), ("sideleft", 0, 1), ("sideright", 1, 0), ("center_top", 1, 1))
# per-view 레이아웃의 카메라 순서. 학습(_plain_build_per_view)의 FRONT/SIDELEFT/SIDERIGHT
# = egoverse_zread 의 "images.front_1" / "images.sideleft" / "images.sideright" 와 짝이다.
# 순서가 곧 뷰 정체성이다 (세 슬롯이 role/t 를 공유하므로 순서 말고는 구분 근거가 없다).
PV_CAMS = ("front_1", "sideleft", "sideright")


def _decode_jpeg(buf: bytes, size: int) -> np.ndarray:
    if not buf:
        return np.zeros((size, size, 3), np.uint8)
    img = Image.open(io.BytesIO(buf)).convert("RGB")
    if img.size != (size, size):
        img = img.resize((size, size), Image.BILINEAR)
    return np.asarray(img)


def _lohi(path: str, emb: str = EMB):
    d = json.load(open(path))
    e = d if ("lo" in d and "hi" in d) else d.get(emb, d["_global"])
    return np.asarray(e["lo"], np.float32), np.asarray(e["hi"], np.float32)


def _to_cuda(sample: dict, tk: dict) -> dict:
    from torch.utils.data import default_collate

    batch = default_collate([sample])
    return {
        k: (v.to(**tk) if (torch.is_tensor(v) and v.dtype.is_floating_point)
            else v.cuda() if torch.is_tensor(v) else v)
        for k, v in batch.items()
    }


class TabletopPolicy:
    def __init__(self, args) -> None:
        with distributed_init():
            distributed.init()
        self.model, config = load_model_from_checkpoint(
            experiment_name=args.experiment,
            s3_checkpoint_dir=args.checkpoint,
            config_file=args.config,
            enable_fsdp=False,
            load_ema_to_reg=False,
        )
        self.model = self.model.cuda().eval()
        self.tk = self.model.tensor_kwargs
        self.chunk_size = int(getattr(config.model.config, "chunk_size", CHUNK))
        # shift defaults to the TRAINING value; a hand-picked shift here is a silent
        # train/eval mismatch (the mistake called out in eval_gr1_server's --shift help).
        self.shift = args.shift if args.shift is not None else float(config.model.config.shift)
        self.guidance = args.guidance
        self.num_steps = args.num_steps
        self.S = int(args.resolution)
        self.use_center_top = bool(args.grid4)

        # ---- latent 계약: 실험 이름이 유일한 근거 -------------------------------------
        # run_robocasa_eval.validate_config 와 같은 관례. 플래그로 받으면 빠뜨렸을 때
        # "에러 없이 점수만 낮은" 실패가 난다 — 서버가 short 레이아웃(state_t=4)을 만들고
        # 체크포인트는 plain(state_t=11)이면 latent 가 통째로 어긋난다.
        #   *_plain_state*     -> PLAIN,     future 8 latents (frames t+1..t+32), state_t 11
        #   *_plain_end*       -> PLAIN_END, future 1 latent  (frame  t+32),      state_t 4
        #   그 외              -> SHORT,     blank ret slot,                      state_t 4
        _exp = str(args.experiment)
        if "_plain_end" in _exp:
            self.layout, self.future_k = "PLAIN_END", 1
        elif "_plain_state" in _exp or "_plain" in _exp:
            self.layout, self.future_k = "PLAIN", 8
        else:
            self.layout, self.future_k = "SHORT", 0
        # "_bridge" -> zero-pixel RET_ACTION + RET_VIDEO 슬롯 2개 (egoverse mid 계약 k=1).
        self.bridge_ret = ("_bridge" in _exp) and self.layout != "SHORT"
        # "_pv{N}" -> RoboCasa b2 스타일 뷰 분리. 2x2 그리드 한 장 대신 카메라 3대가 각자
        # latent 슬롯을 갖고, N 은 zero-pixel prefix 슬롯 수다 (b2 = 2, 이 arm 은 0).
        # 학습쪽 짝: TabletopSim...(per_view=True, per_view_blank=N) 의 _plain_build_per_view.
        # "_nofp" -> RoboCasa stock 레이아웃에서 future proprio(+value) 를 뺀 판.
        # per_view 와 슬롯 인벤토리는 같지만 proprio 가 자기 슬롯(1)을 갖고, slot_roles /
        # t_positions 를 아예 내보내지 않는다 (stock RoboCasa 는 bridge_layout=False 라
        # bridge_pe 가 빈 dict — 모델 기본 PE). 학습쪽 짝: TabletopSim...(nofp=True).
        self.nofp = ("_nofp" in _exp) and self.layout != "SHORT"
        _pv = re.search(r"_pv(\d+)", _exp)
        self.per_view = (_pv is not None) and self.layout != "SHORT" and not self.nofp
        _cd = int(getattr(config.model.config.tokenizer, "chunk_duration", -1))
        _st = int(getattr(config.model.config, "state_t", -1))
        # no_future: per_view 레이아웃에서 FUTURE 카메라 3슬롯이 시퀀스에 아예 없는 arm
        # (학습쪽 TabletopSim...(per_view=True, no_future=True), state_t 8+nb -> 5+nb).
        # 이름 규칙이 아니라 **체크포인트 config 의 state_t** 로 판정한다 — 이름은 언제든
        # 갈리지만 state_t 는 학습이 실제로 쓴 값이라 어긋날 수가 없다.
        self.no_future = False
        if self.nofp:
            self.nb = 0
            self.state_t = 9        # blank + proprio + cam3 + action + fut3
        elif self.per_view:
            self.nb = int(_pv.group(1))
            self.no_future = (_st > 0 and _st == 5 + self.nb)
            # state + nb + cam3 + action [+ future3]
            self.state_t = (5 if self.no_future else 8) + self.nb
        else:
            self.nb = 2 if self.bridge_ret else 0
            self.state_t = (self.future_k + 3 + self.nb) if self.layout != "SHORT" else 4
        if self.no_future:
            self.future_k = 0
        _tag = "+NOFP" if self.nofp else (
            ("+PER_VIEW(nb=%d%s)" % (self.nb, ",NO_FUTURE" if self.no_future else ""))
            if self.per_view else ("+BRIDGE" if self.bridge_ret else ""))
        print(f"latent contract: {self.layout}{_tag} "
              f"future_k={self.future_k} state_t={self.state_t} "
              f"(config state_t={_st} chunk_duration={_cd}) exp={_exp}", flush=True)

        # 해상도는 학습 config 가 근거다. per-view arm 은 224 로 학습하는데 서버 기본값이
        # 448 이라, --resolution 을 빠뜨리면 state_t 검사는 통과하고 점수만 조용히 낮아진다.
        try:
            _cfg_S = int(config.dataloader_train.dataset.final_image_size)
        except Exception:
            _cfg_S = 0
        if _cfg_S > 0 and _cfg_S != self.S:
            if self.per_view or self.nofp:
                print(f"[TabletopPolicy] resolution {self.S} -> {_cfg_S} "
                      f"(학습 config 값으로 강제)", flush=True)
                self.S = _cfg_S
            else:
                print(f"[TabletopPolicy] WARNING: --resolution {self.S} 가 학습 config 의 "
                      f"final_image_size={_cfg_S} 와 다르다", flush=True)
        # 학습 config 와 어긋나면 즉시 죽는다. 조용히 낮은 점수가 나오는 것보다 낫다.
        if _st > 0 and _st != self.state_t:
            raise SystemExit(
                f"ERROR: state_t mismatch — 서버가 만드는 레이아웃 {self.state_t} vs "
                f"체크포인트 config {_st}. 실험 이름({_exp})에서 유도한 계약이 틀렸다.")

        man = args.manifest_dir
        # native 20-D arm ("_a20"): 57-D egoverse 인코딩을 거치지 않고 Tabletop 원본
        # ee_6d_pos 20-D 를 그대로 쓴다. 학습 쪽 native20 플래그와 짝이다.
        self.a20 = "_a20" in _exp
        self.dim = DIM20 if self.a20 else DIM57
        _sf = "state20_stats_pctl.json" if self.a20 else "state57_stats_pctl.json"
        _tf = "abs_action20_stats_pctl.json" if self.a20 else "abs_action57_stats_pctl.json"
        self._slo, self._shi = _lohi(os.path.join(man, _sf))
        self._tlo, self._thi = _lohi(os.path.join(man, _tf))
        print(f"action space: {'NATIVE 20-D (ee_6d_pos)' if self.a20 else 'egoverse 57-D'} "
              f"stats={_tf}", flush=True)
        import pickle

        with open(os.path.join(man, "t5_embeddings.pkl"), "rb") as fp:
            self._t5 = pickle.load(fp)

        self._past: list[np.ndarray] = []   # proprio state57 history, for slot 0
        self._chunk_i = 0
        print(
            f"[TabletopPolicy] chunk={self.chunk_size} shift={self.shift} guidance={self.guidance} "
            f"steps={self.num_steps} S={self.S} grid4={self.use_center_top} "
            f"t5={len(self._t5)} manifest={man}",
            flush=True,
        )

    # ---- state ----
    def reset(self) -> None:
        self._past.clear()
        self._chunk_i = 0
        # Episode counter for the attention dump; without it every episode would overwrite
        # ep0. Mirrors GR1Policy._ep_seq.
        self._ep_seq = getattr(self, "_ep_seq", -1) + 1

    def _save_attn(self, out_dir: str, cap: dict, sample: dict) -> None:
        """One npz per /predict: the action-query attention maps plus the RGB frames they
        attend to, so the renderer needs nothing else.

        Unlike the GR1 grid-3cam layout, pv0 is PER-VIEW: front / sideleft / sideright each
        own a whole CUR_IMAGE latent (slots c0..c0+2), so no quadrant splitting applies and
        every image-bearing slot maps to exactly one camera. Pixel<->latent packing is the
        same causal-VAE contract: latent 0 is pixel frame 0, latent t>0 owns
        [1+(t-1)*NUM_DUP, 1+t*NUM_DUP)."""
        ep_dir = os.path.join(out_dir, f"ep{getattr(self, '_ep_seq', 0):04d}")
        os.makedirs(ep_dir, exist_ok=True)

        roles = np.asarray(sample["slot_roles"]).astype(int)
        video = sample["video"]
        video = video.numpy() if torch.is_tensor(video) else np.asarray(video)

        def frame_of(slot: int) -> np.ndarray:
            f = 0 if slot == 0 else 1 + (slot - 1) * NUM_DUP
            return np.transpose(video[:, min(f, video.shape[1] - 1)], (1, 2, 0))

        img_slots = [t for t in range(len(roles))
                     if int(roles[t]) in (ROLE_CUR_IMAGE, ROLE_RET_VIDEO)]
        frames = (np.stack([frame_of(t) for t in img_slots]) if img_slots
                  else np.zeros((0, 1, 1, 3), np.uint8))
        # PV_CAMS order is the view identity for the CUR_IMAGE slots (they share role and
        # timestamp, so slot order is the only thing distinguishing the cameras).
        cur = [t for t in img_slots if int(roles[t]) == ROLE_CUR_IMAGE]
        view_names = [PV_CAMS[i] if i < len(PV_CAMS) else f"cam{i}" for i in range(len(cur))]

        np.savez_compressed(
            os.path.join(ep_dir, f"chunk{self._chunk_i:04d}.npz"),
            attn_mean=cap["mean"].numpy().astype(np.float16),
            attn_last=cap["last"].numpy().astype(np.float16),
            layers=np.asarray(cap["layers"]),
            query_slot=np.asarray(cap["query_slot"]),
            denoise_steps=np.asarray(cap["steps"]),
            slot_roles=roles,
            t_positions=np.asarray(sample["t_positions"], np.float32),
            img_slot_idx=np.asarray(img_slots),
            cur_slot_idx=np.asarray(cur),
            view_names=np.asarray(view_names),
            frames=frames,
            task=np.asarray(str(sample.get("command", ""))),
            chunk=np.asarray(self._chunk_i),
            demo_ep=np.asarray(""), demo_w0=np.asarray(-1),
        )

    def _past_state_chunk(self) -> np.ndarray:
        """PAST state57 chunk s[t-31..t]; left-pad by repeating the first frame, exactly
        as ``EgoVerseGr1RetV2StateDataset._past_chunk`` does with s[0]."""
        arr = np.stack(self._past) if self._past else np.zeros((1, self.dim), np.float64)
        idx = np.arange(len(arr) - self.chunk_size, len(arr))
        return arr[np.clip(idx, 0, len(arr) - 1)]

    def _grid(self, tiles: dict[str, bytes]) -> np.ndarray:
        S = self.S
        h = S // 2
        canvas = np.zeros((S, S, 3), np.uint8)
        for cam, r, c in TILES:
            if cam == "center_top" and not self.use_center_top:
                continue          # 3-view: BR stays blank, matching the mid
            canvas[r * h:(r + 1) * h, c * h:(c + 1) * h] = _decode_jpeg(tiles.get(cam, b""), h)
        return canvas

    # ---- sample ----
    def _build_sample(self, tiles: dict[str, bytes], instruction: str) -> dict:
        if self.layout == "SHORT":
            return self._build_sample_short(tiles, instruction)
        if self.nofp:
            return self._build_sample_nofp(tiles, instruction)
        if self.per_view:
            return self._build_sample_per_view(tiles, instruction)
        return self._build_sample_plain(tiles, instruction)

    def _cur_state_and_t5(self, instruction: str):
        """PLAIN: 정규화된 CURRENT state (57-D) 와 T5 임베딩.

        2026-08-23: 과거 청크가 아니라 현재 상태만 쓴다 — 학습(_plain_build 의
        PROPRIO_HIST=1)과 같은 계약. 클라이언트가 /predict 를 청크당 한 번만 부르므로
        과거 청크는 애초에 재현 불가였다.
        """
        past = np.stack(self._past[-1:]) if self._past else np.zeros((1, self.dim), np.float64)
        cur_state = np.clip(
            2.0 * (past - self._slo) / (self._shi - self._slo + 1e-8) - 1.0, -1.0, 1.0
        ).astype(np.float32).reshape(-1)
        t5 = (torch.squeeze(torch.as_tensor(self._t5[instruction]))
              if instruction in self._t5 else torch.zeros(512, 1024))
        if instruction not in self._t5:
            print(f"[TabletopPolicy] WARNING: no T5 embedding for {instruction!r} -> zeros",
                  flush=True)
        return cur_state, t5

    def _build_sample_nofp(self, tiles: dict[str, bytes], instruction: str) -> dict:
        """Inference twin of ``_PlainSampleMixin._plain_build_nofp`` (RoboCasa nofp 레이아웃).

            0 blank      (복제 안 하는 토크나이저 첫 프레임)
            1 proprio    (blank x4, 자기 슬롯)
            2 front_1    (t)
            3 sideleft   (t)
            4 sideright  (t)
            -- num_conditional_frames = 5 --
            5 action     (GEN)
            6 front_1    (t+32)   추론 때는 알 수 없으므로 blank
            7 sideleft   (t+32)
            8 sideright  (t+32)

            state_t = 9, chunk_duration = 33

        slot_roles / t_positions 는 일부러 넣지 않는다 (PE 기본). 학습쪽과 같은 계약이다.
        """
        S = self.S
        T = self.state_t
        blank1 = np.zeros((1, S, S, 3), np.uint8)
        blank4 = np.repeat(blank1, NUM_DUP, axis=0)

        def dup(im: np.ndarray) -> np.ndarray:
            return np.repeat(im[None], NUM_DUP, axis=0)

        images = np.concatenate(
            [blank1,                                              # 0 토크나이저 blank
             blank4.copy(),                                       # 1 proprio (자기 슬롯)
             *[dup(_decode_jpeg(tiles.get(c, b""), S)) for c in PV_CAMS],   # 2,3,4
             blank4.copy(),                                       # 5 action   (GEN)
             *[blank4.copy() for _ in range(3)]],                 # 6,7,8 future (GEN)
            axis=0)
        expect = 1 + (T - 1) * NUM_DUP
        assert images.shape[0] == expect, (images.shape, expect)
        assert T == 9, f"nofp layout is state_t=9, got {T}"
        images = preprocess_image(images, final_image_size=S, normalize_images=False,
                                  use_image_aug=False, stronger_image_aug=False)

        cur_state, t5 = self._cur_state_and_t5(instruction)
        zeros_act = np.zeros((self.chunk_size, self.dim), np.float32)
        return {
            "video": images,
            "command": str(instruction),
            "actions": zeros_act,
            "action_mask": np.ones(self.dim, np.float32),
            "t5_text_embeddings": t5,
            "t5_text_mask": torch.ones(512, dtype=torch.int64),
            "fps": 16,
            "padding_mask": torch.zeros(1, S, S),
            "image_size": S * torch.ones(4),
            "__key__": 0,
            "embodiment": EMB, "retrieved_embodiment": EMB,
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
            "proprio": cur_state,
            "future_proprio": np.zeros(self.dim, np.float32),
            "has_ret_data": 0, "has_ret_image": 0, "has_current_image": 1,
            "rollout_data_mask": 0, "rollout_data_success_mask": 0,
            "world_model_sample_mask": 0, "value_function_sample_mask": 0,
            "global_rollout_idx": -1, "value_function_return": float("-100"),
            "next_value_function_return": float("-100"),
            "next_action_chunk": zeros_act.copy(),
        }

    def _build_sample_per_view(self, tiles: dict[str, bytes], instruction: str) -> dict:
        """Inference twin of ``_PlainSampleMixin._plain_build_per_view`` (RoboCasa b2 PE).

            idx role         t          내용
             0  CUR_STATE    NOW        blank 1프레임 (복제 안 함), proprio 가 안에 실림
            [1  RET_ACTION   NOW        zero-pixel]        nb >= 1
            [2  RET_VIDEO    NOW - dt   zero-pixel]        nb >= 2
            c0  CUR_IMAGE    NOW        front      (t)
            c0+1 CUR_IMAGE   NOW        sideleft   (t)
            c0+2 CUR_IMAGE   NOW        sideright  (t)
            -- num_conditional_frames = a0 (액션 슬롯 직전) --
            a0  ACTION       NOW        GEN
            f0  FUTURE       NOW + dt   front      (t+32)
            f0+1 FUTURE      NOW + 2dt  sideleft
            f0+2 FUTURE      NOW + 3dt  sideright

        b2 에서 그대로 가져온 것: 세 카메라가 role/t 를 공유하고 슬롯 순서로만 구분되는 것,
        future 의 뷰별 dt stagger, n_cond 가 액션 슬롯 직전에서 끊기는 것(경계가 밀리면
        마지막 카메라를 "줘야 할 뷰인데 생성"하게 되고 초기 loss 로는 안 드러난다).

        future 슬롯 픽셀은 학습 때 실제 t+32 프레임이지만 추론 때는 알 수 없다. GEN 슬롯이라
        내용은 안 쓰이므로 blank 로 채우고 개수·위치·시각만 정확히 맞춘다 (plain 빌더와 동일).
        """
        S = self.S
        T = self.state_t
        nb = self.nb
        blank1 = np.zeros((1, S, S, 3), np.uint8)
        blank4 = np.repeat(blank1, NUM_DUP, axis=0)

        def dup(im: np.ndarray) -> np.ndarray:
            return np.repeat(im[None], NUM_DUP, axis=0)

        images = np.concatenate(
            [blank1,                                              # 0 cur_state
             *[blank4.copy() for _ in range(nb)],                 # ret_action / ret_video
             *[dup(_decode_jpeg(tiles.get(c, b""), S)) for c in PV_CAMS],   # cur x3
             blank4.copy(),                                       # action   (GEN)
             *([] if self.no_future                               # future x3 (GEN)
               else [blank4.copy() for _ in range(3)])],
            axis=0)
        expect = 1 + (T - 1) * NUM_DUP
        assert images.shape[0] == expect, (images.shape, expect)
        images = preprocess_image(images, final_image_size=S, normalize_images=False,
                                  use_image_aug=False, stronger_image_aug=False)

        nf = self.no_future
        assert T == (5 if nf else 8) + nb, \
            f"per_view(nb={nb}, no_future={nf}) layout is state_t={(5 if nf else 8)+nb}, got {T}"
        c0 = 1 + nb
        a0 = c0 + 3
        f0 = a0 + 1
        roles = np.empty(T, np.int64)
        tpos = np.empty(T, np.float32)
        roles[0], tpos[0] = ROLE_CUR_STATE, NOW_S
        if nb >= 1:
            roles[1], tpos[1] = ROLE_RET_ACTION, NOW_S
        if nb >= 2:
            roles[2], tpos[2] = ROLE_RET_VIDEO, NOW_S - DT
        roles[c0:c0+3], tpos[c0:c0+3] = ROLE_CUR_IMAGE, NOW_S
        roles[a0], tpos[a0] = ROLE_ACTION, NOW_S
        if not nf:
            roles[f0:f0+3] = ROLE_FUTURE
            tpos[f0:f0+3] = NOW_S + DT * np.arange(1, 4)

        cur_state, t5 = self._cur_state_and_t5(instruction)
        zeros_act = np.zeros((self.chunk_size, self.dim), np.float32)
        return {
            "video": images,
            "command": str(instruction),
            "actions": zeros_act,
            "action_mask": np.ones(self.dim, np.float32),
            "t5_text_embeddings": t5,
            "t5_text_mask": torch.ones(512, dtype=torch.int64),
            "fps": 16,
            "padding_mask": torch.zeros(1, S, S),
            "image_size": S * torch.ones(4),
            "__key__": 0,
            "embodiment": EMB, "retrieved_embodiment": EMB,
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
            "proprio": cur_state,
            "future_proprio": np.zeros(self.dim, np.float32),
            "current_proprio_latent_idx": 0,
            "future_proprio_latent_idx": -1,
            "value_latent_idx": -1,
            "has_ret_data": 0, "has_ret_image": 0, "has_current_image": 1,
            "rollout_data_mask": 0, "rollout_data_success_mask": 0,
            "world_model_sample_mask": 0, "value_function_sample_mask": 0,
            "global_rollout_idx": -1, "value_function_return": float("-100"),
            "next_value_function_return": float("-100"),
            "next_action_chunk": zeros_act.copy(),
        }

    def _build_sample_plain(self, tiles: dict[str, bytes], instruction: str) -> dict:
        """Inference twin of ``_PlainSampleMixin._plain_build``.

            slot:  cur_state | cur_image | action | future x K
            role:      0     |     3     |   4    |   5
            time:     NOW    |    NOW    |  NOW   | NOW+dt .. NOW+K*dt
            cond:     yes    |    yes    |  GEN   |  GEN

        학습 때 future 슬롯에는 실제 미래 프레임(생성 타깃)이 들어가지만 추론 때는 알 수 없다.
        GEN 슬롯이라 픽셀 내용은 쓰이지 않으므로 blank 로 채우되 **개수와 위치는 정확히** 맞춘다.
        """
        S = self.S
        T = self.state_t
        K = self.future_k
        nb = self.nb
        blank1 = np.zeros((1, S, S, 3), np.uint8)
        blank4 = np.repeat(blank1, NUM_DUP, axis=0)
        cur4 = np.repeat(self._grid(tiles)[None], NUM_DUP, axis=0)
        pre = [blank4.copy(), blank4.copy()] if self.bridge_ret else []
        images = np.concatenate(
            [blank1,                                   # cur_state
             *pre,                                     # ret_action, ret_video (BRIDGE)
             cur4,                                     # cur_image
             blank4,                                   # action  (GEN)
             np.repeat(blank1, NUM_DUP * K, axis=0)],  # future  (GEN)
            axis=0)
        expect = 1 + (T - 1) * NUM_DUP
        assert images.shape[0] == expect, (images.shape, expect)
        images = preprocess_image(images, final_image_size=S, normalize_images=False,
                                  use_image_aug=False, stronger_image_aug=False)

        roles = np.empty(T, np.int64)
        tpos = np.empty(T, np.float32)
        roles[0], tpos[0] = ROLE_CUR_STATE, NOW_S
        if self.bridge_ret:
            roles[1], tpos[1] = ROLE_RET_ACTION, NOW_S
            roles[2], tpos[2] = ROLE_RET_VIDEO, NOW_S - DT
        roles[1+nb], tpos[1+nb] = ROLE_CUR_IMAGE, NOW_S
        roles[2+nb], tpos[2+nb] = ROLE_ACTION, NOW_S
        roles[3+nb:], tpos[3+nb:] = ROLE_FUTURE, NOW_S + DT * np.arange(1, K + 1)

        cur_state, t5 = self._cur_state_and_t5(instruction)
        zeros_act = np.zeros((self.chunk_size, self.dim), np.float32)
        return {
            "video": images,
            "command": str(instruction),
            "actions": zeros_act,
            "action_mask": np.ones(self.dim, np.float32),
            "t5_text_embeddings": t5,
            "t5_text_mask": torch.ones(512, dtype=torch.int64),
            "fps": 16,
            "padding_mask": torch.zeros(1, S, S),
            "image_size": S * torch.ones(4),
            "__key__": 0,
            "embodiment": EMB, "retrieved_embodiment": EMB,
            "t_positions": tpos, "slot_roles": roles,
            "num_conditional_frames": 2 + nb,            # cur_state [+ret x2] + cur_image
            "retrieved_video_start_latent_idx": 2 if self.bridge_ret else -1,
            "retrieved_video_end_latent_idx": 3 if self.bridge_ret else -1,
            "retrieved_action_latent_idx": 1 if self.bridge_ret else -1,
            "retrieved_state_latent_idx": -1,
            "current_image_latent_idx": 1 + nb,
            "future_video_start_latent_idx": 3 + nb,
            "future_video_end_latent_idx": T,
            "action_latent_idx": 2 + nb,
            "proprio": cur_state,
            "future_proprio": np.zeros(self.dim, np.float32),
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
            "next_action_chunk": zeros_act.copy(),
        }

    def _build_sample_short(self, tiles: dict[str, bytes], instruction: str) -> dict:
        """Inference twin of ``TabletopSimShortStateDataset`` (state_t=4, blank ret)."""
        S = self.S
        blank1 = np.zeros((1, S, S, 3), np.uint8)
        blank4 = np.repeat(blank1, NUM_DUP, axis=0)
        cur4 = np.repeat(self._grid(tiles)[None], NUM_DUP, axis=0)
        images = np.concatenate([blank1, blank4, cur4, blank4], axis=0)
        assert images.shape[0] == 1 + 3 * NUM_DUP, images.shape
        images = preprocess_image(images, final_image_size=S, normalize_images=False,
                                  use_image_aug=False, stronger_image_aug=False)

        past = self._past_state_chunk()
        cur_state = np.clip(
            2.0 * (past - self._slo) / (self._shi - self._slo + 1e-8) - 1.0, -1.0, 1.0
        ).astype(np.float32).reshape(-1)

        t5 = (torch.squeeze(torch.as_tensor(self._t5[instruction]))
              if instruction in self._t5 else torch.zeros(512, 1024))
        if instruction not in self._t5:
            print(f"[TabletopPolicy] WARNING: no T5 embedding for {instruction!r} -> zeros",
                  flush=True)
        zeros_act = np.zeros((self.chunk_size, self.dim), np.float32)
        return {
            "video": images,
            "command": str(instruction),
            "actions": zeros_act,
            "action_mask": np.ones(self.dim, np.float32),
            "retrieved_actions": zeros_act.copy(),      # blank ret
            "retrieved_action_mask": np.ones(self.dim, np.float32),
            "t5_text_embeddings": t5,
            "t5_text_mask": torch.ones(512, dtype=torch.int64),
            "fps": 16,
            "padding_mask": torch.zeros(1, S, S),
            "image_size": S * torch.ones(4),
            "__key__": 0,
            "embodiment": EMB, "retrieved_embodiment": EMB,
            "t_positions": np.full(4, NOW_S, np.float32),
            "slot_roles": np.array(
                [ROLE_CUR_STATE, ROLE_RET_ACTION, ROLE_CUR_IMAGE, ROLE_ACTION], np.int64),
            "num_conditional_frames": 3,
            "retrieved_video_start_latent_idx": -1,
            "retrieved_video_end_latent_idx": -1,
            "retrieved_action_latent_idx": 1,
            "retrieved_state_latent_idx": -1,
            "current_image_latent_idx": 2,
            "future_video_start_latent_idx": 4,
            "future_video_end_latent_idx": 4,
            "action_latent_idx": 3,
            "proprio": cur_state,
            "future_proprio": np.zeros(self.dim, np.float32),
            "current_proprio_latent_idx": 0,
            "current_wrist_image_latent_idx": -1, "current_wrist_image2_latent_idx": -1,
            "current_image2_latent_idx": -1, "future_proprio_latent_idx": -1,
            "future_wrist_image_latent_idx": -1, "future_wrist_image2_latent_idx": -1,
            "future_image_latent_idx": -1, "future_image2_latent_idx": -1,
            "value_latent_idx": -1,
            "has_ret_data": 0, "has_ret_image": 0, "has_current_image": 1,
            "rollout_data_mask": 0, "rollout_data_success_mask": 0,
            "world_model_sample_mask": 0, "value_function_sample_mask": 0,
            "global_rollout_idx": -1, "value_function_return": float("-100"),
            "next_value_function_return": float("-100"),
            "next_action_chunk": zeros_act.copy(),
        }

    # ---- inference ----
    def predict(self, req: dict) -> dict:
        proprio20 = np.asarray(req["proprio20"], np.float64).reshape(1, 20)
        # a20 은 20-D 를 그대로, 57-D arm 은 egoverse 인코딩을 거쳐 쌓는다.
        self._past.append(proprio20[0] if self.a20 else encode_20_to_57(proprio20)[0])
        tiles = {k: base64.b64decode(v) for k, v in req["images"].items()}
        sample = self._build_sample(tiles, str(req["instruction"]))

        data_batch = _to_cuda(sample, self.tk)
        # Action-query self-attention dump (TABLETOP_ATTN_DIR=<dir>). Off by default and
        # additive: unset, the sampler call below is the original one. Same capture as the
        # GR1 server — see analysis_tools/gr1_attn_capture.py and docs/gr1_attention_maps.md.
        _attn_dir = os.environ.get("TABLETOP_ATTN_DIR")
        with torch.autocast("cuda", dtype=self.tk.get("dtype", torch.bfloat16)):
            _raw, x0, _cond = self.model.get_data_and_condition(data_batch)
            a_idx = int(data_batch["action_latent_idx"][0]) if torch.is_tensor(
                data_batch["action_latent_idx"]) else int(data_batch["action_latent_idx"])
            _sample_fn = lambda: self.model.generate_samples_from_batch(
                data_batch, guidance=self.guidance, state_shape=x0.shape[1:],
                n_sample=x0.shape[0], num_steps=self.num_steps, shift=self.shift,
            )
            if _attn_dir:
                from analysis_tools.gr1_attn_capture import Gr1AttnCapture
                _lay = os.environ.get("TABLETOP_ATTN_LAYERS")
                with Gr1AttnCapture(
                    self.model.net, query_slot=a_idx,
                    layers=[int(x) for x in _lay.split(",")] if _lay else None,
                    head_reduce=os.environ.get("TABLETOP_ATTN_HEADS", "mean"),
                    step_stride=int(os.environ.get("TABLETOP_ATTN_STEP_STRIDE", "4")),
                ) as _cap:
                    latents = _sample_fn()
                self._save_attn(_attn_dir, _cap.result(), sample)
            else:
                latents = _sample_fn()
            act_norm = extract_action_chunk_from_latent(
                latents[:, :, a_idx], (self.chunk_size, self.dim),
                decoder=getattr(self.model, "action_decoder", None),
            )[0].float().cpu().numpy()

        # abs_action57: de-norm with the TARGET stats, then straight back to 20-D.
        # No integration, no re-anchoring — the prediction IS the command.
        act = (act_norm + 1.0) / 2.0 * (self._thi - self._tlo) + self._tlo
        act20 = act if self.a20 else decode_57_to_20(act)
        # 롤아웃이 실제로 내보낸 명령을 남긴다 (TABLETOP_ACT_DUMP=<dir>). 안 켜면 no-op.
        # 오프라인 프로브는 전부 데이터셋 입력을 쓰므로, 실패하는 궤적 자체는 여기서만 보인다.
        _act_dump = os.environ.get("TABLETOP_ACT_DUMP")
        if _act_dump:
            self._save_action(_act_dump, proprio20, act_norm, act20)
        self._chunk_i += 1
        return {"actions": act20.astype(np.float32).tolist(), "chunk": self._chunk_i}

    def _save_action(self, root: str, proprio20: np.ndarray,
                     act_norm: np.ndarray, act20: np.ndarray) -> None:
        """청크당 한 파일. 어텐션 덤프(_save_attn)와 같은 ep/chunk 레이아웃을 쓴다."""
        ep = getattr(self, "_ep_seq", 0)
        d = os.path.join(root, f"ep{ep:03d}")
        os.makedirs(d, exist_ok=True)
        p_raw = np.asarray(proprio20, np.float64).reshape(-1)
        p_norm = np.clip(2.0 * (p_raw - self._slo) / (self._shi - self._slo + 1e-8) - 1.0, -1.0, 1.0)
        np.savez_compressed(
            os.path.join(d, f"chunk{self._chunk_i:04d}.npz"),
            proprio_raw=p_raw.astype(np.float32),
            proprio_norm=p_norm.astype(np.float32),
            act_norm=np.asarray(act_norm, np.float32),     # [-1,1] 공간, 학습 타깃과 같은 단위
            act20=np.asarray(act20, np.float32),           # 실제 env 에 들어가는 명령
            chunk=np.asarray(self._chunk_i),
        )


def make_handler(policy: TabletopPolicy):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, obj, code=200):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):  # noqa: N802
            n = int(self.headers.get("Content-Length", 0))
            try:
                req = json.loads(self.rfile.read(n) or b"{}")
            except Exception as e:
                return self._send({"error": f"bad json: {e}"}, 400)
            try:
                if self.path.rstrip("/") == "/reset":
                    policy.reset()
                    return self._send({"ok": True})
                if self.path.rstrip("/") == "/predict":
                    return self._send(policy.predict(req))
                return self._send({"error": f"unknown path {self.path}"}, 404)
            except Exception as e:
                import traceback

                traceback.print_exc()
                return self._send({"error": f"{type(e).__name__}: {e}"}, 500)

        def log_message(self, *a):  # quiet
            pass

    return Handler


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="cosmos_policy/config/config.py")
    ap.add_argument("--experiment",
                    default="cosmos_predict2p5_2b_tabletop_sim_5task_short_state")
    ap.add_argument("--checkpoint", required=True, help=".pt from convert_dcp_to_pt.py")
    ap.add_argument("--manifest-dir", default=f"{TABLETOP_ROOT}/manifest")
    ap.add_argument("--resolution", type=int, default=448)
    ap.add_argument("--num-steps", type=int, default=35)
    ap.add_argument("--guidance", type=float, default=1.5)
    ap.add_argument("--shift", type=float, default=None,
                    help="flow sampling shift; default None = the training value from the config")
    ap.add_argument("--grid4", action="store_true",
                    help="4-view grid (BR = collaborator_pov). Must match how the model was trained")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8010)
    args = ap.parse_args()

    # Same SDPA guard as eval_gr1_server: force the math backend so a kernel always exists.
    try:
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        try:
            torch.backends.cuda.enable_cudnn_sdp(False)
        except Exception:
            pass
        torch.backends.cuda.enable_math_sdp(True)
    except Exception:
        pass

    policy = TabletopPolicy(args)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(policy))
    print(f"Tabletop-Sim policy server ready on {args.host}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
