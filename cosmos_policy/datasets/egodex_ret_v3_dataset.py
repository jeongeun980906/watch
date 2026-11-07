"""EgoDex offline-eval dataset in the mid-V3 layout (v2 compact layout, FULL-FRAME).

V3 adapter over the same egodex source as ``EgoDexRetV2Dataset`` — identical
plumbing (egodex_pairs_top3*.npz table, egodex zarr root, [egodex]->[human] text
alias, egodex-native action57/state57 stats), but every video token is the primary
camera decoded FULL-FRAME at final_image_size (no 2x2 grid TL packing), matching
what ``EgoVerseRetV3Dataset`` fed the v3 mid-training.

MRO: EgoDexRetV3Dataset -> EgoVerseRetV3Dataset -> EgoDexRetV2Dataset
     -> EgoVerseRetV2Dataset -> EgoDexRetDataset -> EgoVerseRetDataset
* ``__init__`` resolves to EgoDexRetV2Dataset's (egodex defaults, state57 stats).
* ``_grid_frame`` resolves to EgoVerseRetV3Dataset's (full-frame, no grid).
Everything else (compact packing, slot_roles, t_positions, k/g buckets) is the
inherited v2 ``__getitem__`` unchanged.
"""
from __future__ import annotations

from cosmos_policy.datasets.egodex_ret_v2_dataset import EgoDexRetV2Dataset
from cosmos_policy.datasets.egoverse_ret_v3_dataset import EgoVerseRetV3Dataset


class EgoDexRetV3Dataset(EgoVerseRetV3Dataset, EgoDexRetV2Dataset):
    """EgoDex eval in the v3 full-frame compact layout — see module doc."""


if __name__ == "__main__":
    ds = EgoDexRetV3Dataset(final_image_size=448)
    s = ds[0]
    print({k: getattr(v, "shape", v) for k, v in s.items() if hasattr(v, "shape")})
    print("emb:", s["embodiment"], "ret:", s["retrieved_embodiment"],
          "k+3:", s["num_conditional_frames"])
