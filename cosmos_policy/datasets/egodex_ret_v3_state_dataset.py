"""EgoDex offline-eval dataset in the mid-V3 layout — STATE-CHUNK prediction variant.

State twin of ``EgoDexRetV3Dataset`` for evaluating the ``..._ret_v3_448_state``
checkpoints on the held-out EgoDex embodiment: same egodex source plumbing
(egodex_pairs_top3*.npz table, egodex zarr root, [egodex]->[human] text alias),
same full-frame v3 pixels, but the STATE-chunk sample semantics of
``EgoVerseRetV3StateDataset``:
  * "actions"/"next_action_chunk" : FUTURE state57 chunk s[t+1 .. t+CHUNK]
  * "retrieved_actions"           : demo's FUTURE state chunk from its w0
  * "proprio" (slot 0)            : PAST state chunk s[t-CHUNK+1 .. t] (s[0] left-pad)

MRO: EgoDexRetV3StateDataset -> EgoVerseRetV3StateDataset -> EgoVerseRetV3Dataset
     -> EgoDexRetV2Dataset -> EgoVerseRetV2Dataset -> EgoDexRetDataset -> EgoVerseRetDataset
* ``__getitem__``/``build_sample``: EgoVerseRetV3StateDataset (state-chunk semantics).
* ``_grid_frame``: EgoVerseRetV3Dataset (full-frame, no grid).
* source/table/defaults: EgoDexRetV2Dataset (egodex-native stats).

Normalization: ROBUST (mean +/- 4*std) per the state datasets' saturation-gate
convention, computed over the egodex episodes and cached at
/mnt/ddn/dataset/egodex_state57_stats_staterobust.json.
"""
from __future__ import annotations

from cosmos_policy.datasets.egodex_ret_v2_dataset import (
    EGODEX_STATE_STATS,
    EGODEX_PAIRS_3S,
    V10_T5_EMB,
    EgoDexRetV2Dataset,
)
from cosmos_policy.datasets.egodex_ret_dataset import DEFAULT_ACT_STATS as EGODEX_ACT_STATS
from cosmos_policy.datasets.egoverse_ret_v3_state_dataset import EgoVerseRetV3StateDataset


class EgoDexRetV3StateDataset(EgoVerseRetV3StateDataset, EgoDexRetV2Dataset):
    """EgoDex eval in the v3 full-frame STATE-chunk layout — see module doc."""

    def __init__(
        self,
        pairs_path: str = EGODEX_PAIRS_3S,
        t5_text_embeddings_path: str = V10_T5_EMB,
        action_stats_path: str = EGODEX_ACT_STATS,
        state_stats_path: str = EGODEX_STATE_STATS,
        **kw,
    ):
        # EgoVerseRetV3StateDataset.__init__ resolves the ROBUST stats path from
        # state_stats_path (-> egodex_state57_stats_staterobust.json) after the
        # EgoDexRetV2Dataset chain builds the window table.
        super().__init__(
            pairs_path=pairs_path,
            t5_text_embeddings_path=t5_text_embeddings_path,
            action_stats_path=action_stats_path,
            state_stats_path=state_stats_path,
            **kw,
        )


if __name__ == "__main__":
    import numpy as np

    ds = EgoDexRetV3StateDataset(final_image_size=448)
    print("len", len(ds))
    np.random.seed(0)
    s = ds[0]
    print("video", s["video"].shape, "actions(states)", s["actions"].shape,
          "ret_actions(states)", s["retrieved_actions"].shape,
          "proprio(past chunk)", s["proprio"].shape)
    print("T", len(s["slot_roles"]), "ncond", s["num_conditional_frames"],
          "emb", s["embodiment"], "ret", s["retrieved_embodiment"])
