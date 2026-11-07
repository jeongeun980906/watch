"""EgoDex retrieval-augmented dataset — OFFLINE EVAL of the trained EgoVerse policy
on the held-out EgoDex embodiment (Apple Vision Pro human-hand demos).

EgoDex uses the SAME egoverse zarr schema (images.front_1 + action57/state57, human
robot, 30 Hz, state57_abs) but ships a SELF-CONTAINED window/pair table
``egodex_pairs_top3.npz`` (within-egodex WINDOW-level top-3 retrieval) instead of the
v9 windows.npz/pairs.npz manifest. So we subclass EgoVerseRetDataset and only swap
the manifest source + zarr root; the sample dict, latent layout, per-emb normalization,
video/action reading, and image aug are all inherited unchanged.

Pairs schema (egodex_pairs_top3.npz):
    names   (E,)       episode dir names               win_ep (W,)  window -> ep index
    win_w0  (W,)       window start frame              win_wl (W,)  window length (frames)
    top_idx (W,3)      top-3 matched WINDOW indices    top_dtw (W,3) DTW score
Every window is embodiment "egodex" (human), cls "other" (egodex task vocab is not the
v9 lang_taxonomy). The query self-retrieves its top-3 egodex windows as the demo
condition — same query+demo paradigm the policy trained on.

Text: the model never saw an "[egodex]" tag, so we ALIAS the egodex text command to the
trained "[human]" embedding (in-distribution generic-human condition). Normalization is
egodex-native: action_stats_path defaults to egodex_action57_stats.json ({egodex,_global}).
"""
from __future__ import annotations

import os
import pickle

import numpy as np

from cosmos_policy.datasets.egoverse_ret_dataset import (
    CLASS_TO_COMMAND,
    EgoVerseRetDataset,
    NUM_DUPLICATES,
)
from cosmos_policy.datasets.egoverse_zread import EgoZarrEpisode

EGODEX_ROOT = "/mnt/ddn/dataset/egodex_egoverse"
DEFAULT_PAIRS = "/mnt/ddn/dataset/egodex_pairs_top3.npz"
DEFAULT_T5_EMB = "/mnt/ddn/dataset/egoverse_pairs_v9/t5_embeddings.pkl"
DEFAULT_ACT_STATS = "/mnt/ddn/dataset/egodex_action57_stats.json"

EGODEX_EMB = "egodex"        # per-emb normalization key (its own action range)
TEXT_EMB = "human"           # trained tag used for the text condition (egodex ~ generic human)
EGODEX_CLS = "other"         # egodex tasks are outside the v9 lang_taxonomy


class EgoDexRetDataset(EgoVerseRetDataset):
    def __init__(
        self,
        pairs_path: str = DEFAULT_PAIRS,
        t5_text_embeddings_path: str = DEFAULT_T5_EMB,
        action_stats_path: str = DEFAULT_ACT_STATS,
        num_frames: int = 92,
        final_image_size: int = 224,
        normalize_images: bool = False,
        use_image_aug: bool = False,       # eval: no aug
        use_stronger_image_aug: bool = False,
        retrieval_dropout_prob: float = 0.0,
        epoch_length=None,
        debug: bool = False,
        **_ignored,
    ):
        # NOTE: intentionally NOT calling super().__init__ (it loads v9 windows.npz/
        # pairs.npz which egodex doesn't have). We replicate the same attributes from
        # the egodex self-contained table, then reuse every inherited method.
        assert num_frames % NUM_DUPLICATES == 0, "num_frames must be divisible by 4"
        self.manifest_dir = pairs_path
        self.num_frames = int(num_frames)
        self.R = self.num_frames // NUM_DUPLICATES
        self.final_image_size = final_image_size
        self.normalize_images = normalize_images
        self.use_image_aug = use_image_aug
        self.use_stronger_image_aug = use_stronger_image_aug
        self.retrieval_dropout_prob = retrieval_dropout_prob
        self.debug = debug

        P = np.load(pairs_path, allow_pickle=True)
        names = P["names"].astype(str)          # (E,) ep dir names
        win_ep = P["win_ep"].astype(np.int64)   # (W,) window -> ep idx
        win_w0 = P["win_w0"].astype(np.int64)   # (W,) start frame
        win_wl = P["win_wl"].astype(np.int64)   # (W,) length (frames)
        W = len(win_ep)

        # window tables in the same shape EgoVerseRetDataset expects
        self.wep = names[win_ep]                        # (W,) episode dir name per window
        self.ww0 = win_w0
        self.ww1 = win_w0 + win_wl                      # native span (30 Hz)
        self.wisev = np.zeros(W, bool)                  # egodex is human (never eva)
        self.wemb = np.full(W, EGODEX_EMB, dtype=object).astype(str)
        self.wcls = np.full(W, EGODEX_CLS, dtype=object).astype(str)

        # pairs: query = every window; matches = its top-3 egodex windows (-1 padded)
        self.query_idx = np.arange(W, dtype=np.int64)
        self.match_idx = P["top_idx"].astype(np.int64)  # (W, 3)
        self.N = W
        self.epoch_length = epoch_length or self.N

        with open(t5_text_embeddings_path, "rb") as fp:
            self.t5_text_embeddings = pickle.load(fp)
        # alias "<cmd> [egodex]" -> the trained "<cmd> [human]" embedding (no T5 needed)
        for cmd in CLASS_TO_COMMAND.values():
            human_key = f"{cmd} [{TEXT_EMB}]"
            if human_key in self.t5_text_embeddings:
                self.t5_text_embeddings[f"{cmd} [{EGODEX_EMB}]"] = self.t5_text_embeddings[human_key]

        self._zcache = {}
        self.act_stats = self._load_or_compute_action_stats(action_stats_path)
        print(f"[EgoDexRetDataset] windows={self.N} num_frames={self.num_frames} "
              f"R={self.R} chunk_duration={2 * self.R + 4} "
              f"emb_stats_keys={list(self.act_stats.keys())}")

    # egodex episodes all live under one human root (no eva); reopen per worker.
    def _grp(self, wi: int) -> EgoZarrEpisode:
        key = "X|" + self.wep[wi]
        g = self._zcache.get(key)
        if g is None:
            g = EgoZarrEpisode(os.path.join(EGODEX_ROOT, self.wep[wi]))
            self._zcache[key] = g
        return g


if __name__ == "__main__":
    import sys
    nf = int(sys.argv[1]) if len(sys.argv) > 1 else 92
    ds = EgoDexRetDataset(num_frames=nf)
    print("len", len(ds))
    s = ds[0]
    print("video", s["video"].shape, s["video"].dtype)
    print("actions", s["actions"].shape, "mask sum", s["action_mask"].sum())
    print("retrieved_actions", s["retrieved_actions"].shape)
    print("emb", s["embodiment"], "ret_emb", s["retrieved_embodiment"], "cls->cmd via t5 OK")
    print("t5", s["t5_text_embeddings"].shape)
