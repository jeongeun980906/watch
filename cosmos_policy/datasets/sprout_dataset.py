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

from cosmos_policy.datasets.egoverse_zread import FRONT
from cosmos_policy.datasets.tabletop_sim_dataset import TabletopSimPlainEndImageDataset

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


def make(root: str = SPROUT_ROOT, **kw) -> SproutPlainEndDataset:
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
    return SproutPlainEndDataset(**args)


if __name__ == "__main__":
    import numpy as np

    ds = make(require_t5=False)
    s = ds[len(ds) // 2]
    print(len(ds), "windows; video", s["video"].shape, "actions", s["actions"].shape, "proprio", s["proprio"].shape)
    print("t_positions", s["t_positions"], "roles", s["slot_roles"], "n_cond", s["num_conditional_frames"])
    print("action range", float(s["actions"].min()), float(s["actions"].max()), "nan", bool(np.isnan(s["actions"]).any()))
