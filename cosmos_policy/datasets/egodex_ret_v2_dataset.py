"""EgoDex offline-eval dataset in the mid-v2 COMPACT layout (role ids + continuous-time rope).

v2-format adapter over the same source the old ``EgoDexRetDataset`` reads
(egodex_pairs_top3*.npz window/pair table + egodex_egoverse zarr). MRO does the work:

    EgoDexRetV2Dataset -> EgoVerseRetV2Dataset -> EgoDexRetDataset -> EgoVerseRetDataset

* ``EgoVerseRetV2Dataset.__init__`` super()-chains into ``EgoDexRetDataset.__init__``
  (builds wep/ww0/ww1/wemb/wcls/query_idx/match_idx from the self-contained pairs npz,
  aliases the "[egodex]" text command to the trained "[human]" embedding), then adds the
  v2 extras (_m_emb demo-embodiment lock + state57 stats).
* The whole v2 ``__getitem__`` (compact packing, slot_roles, t_positions,
  num_conditional_frames=k+3, grid frames) is inherited unchanged.
* ``_grp`` resolves to EgoDexRetDataset's (egodex zarr root).

EgoDex is 30 Hz human-like -> ``_is_md`` False -> native 92f, k=g=23, T=50 (the human
bucket the v2 model trained on). Single-cam episodes -> the 2x2 grid keeps TL only
(side-cam reads raise and are skipped), exactly like v2's human-episode handling.

Defaults follow the v2 TRAINING config where the model's expectations matter (t5 pkl =
v10's, so the aliased [human] tag embedding is the trained one) and are egodex-native
where the data's own range matters (action57/state57 min-max stats), mirroring the old
egodex eval's egodex-native action stats.
"""
from __future__ import annotations

from cosmos_policy.datasets.egodex_ret_dataset import (
    DEFAULT_ACT_STATS as EGODEX_ACT_STATS,
    EgoDexRetDataset,
)
from cosmos_policy.datasets.egoverse_ret_v2_dataset import EgoVerseRetV2Dataset

EGODEX_PAIRS_3S = "/mnt/ddn/dataset/egodex_pairs_top3_3s.npz"
V10_T5_EMB = "/mnt/ddn/dataset/egoverse_pairs_v10/t5_embeddings.pkl"
EGODEX_STATE_STATS = "/mnt/ddn/dataset/egodex_state57_stats.json"


class EgoDexRetV2Dataset(EgoVerseRetV2Dataset, EgoDexRetDataset):
    def __init__(
        self,
        pairs_path: str = EGODEX_PAIRS_3S,
        t5_text_embeddings_path: str = V10_T5_EMB,
        action_stats_path: str = EGODEX_ACT_STATS,
        state_stats_path: str = EGODEX_STATE_STATS,
        **kw,
    ):
        # v2 __init__ consumes state_stats_path; the rest reaches EgoDexRetDataset.
        super().__init__(
            pairs_path=pairs_path,
            t5_text_embeddings_path=t5_text_embeddings_path,
            action_stats_path=action_stats_path,
            state_stats_path=state_stats_path,
            **kw,
        )


if __name__ == "__main__":
    import numpy as np

    ds = EgoDexRetV2Dataset()
    print("len", len(ds))
    np.random.seed(0)
    s = ds[0]
    T = len(s["slot_roles"])
    print("video", s["video"].shape, "actions", s["actions"].shape,
          "ret_actions", s["retrieved_actions"].shape)
    print("T", T, "ncond", s["num_conditional_frames"],
          "roles", s["slot_roles"].tolist())
    print("tpos", np.round(s["t_positions"], 3).tolist())
    print("emb", s["embodiment"], "ret", s["retrieved_embodiment"])
