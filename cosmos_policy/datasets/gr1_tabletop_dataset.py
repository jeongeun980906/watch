"""RoboCasa GR1 tabletop teleop dataset (LeRobot format) for Cosmos Policy.

Source: nvidia/PhysicalAI-Robotics-GR00T-Teleop-Sim — 24 tabletop tasks x 1000 human teleop
demos, ~4.5M frames, recorded on GR1ArmsAndWaistFourierHands.

Differences from RoboCasaDataset that drive the design here:

* **Storage.** LeRobot, not monolithic HDF5: low-dim state/action live in one parquet per
  episode, frames live in a matching mp4. Frames are decoded on demand via lerobot's
  `decode_video_frames`, so nothing is held in RAM between samples. RoboCasaDataset's
  `lazy_load_demos` path has no analogue — every read here is lazy.
* **One camera.** `observation.images.ego_view` only, so there is no wrist/secondary view and
  the latent sequence is correspondingly shorter (see LATENT LAYOUT below).
* **44-dim actions.** left_arm 7 | left_hand 6 | left_leg 6 | neck 3 | right_arm 7 |
  right_hand 6 | right_leg 6 | waist 3. Legs and neck are identically zero — the embodiment is
  "ArmsAndWaist", those joints are not actuated — but all 44 are kept: 44 x chunk_size 32 =
  1408 values still fit the 16x28x28 = 12544-element action slot, and slicing them out would
  add a mapping that has to stay in sync with eval.
* **Language is the task name.** meta/tasks.jsonl carries "PnPBottleToCabinetClose", not a
  natural-language instruction, so there are exactly 24 unique strings across the dataset and
  the open-vocabulary embedding gap that affects RoboCasa cannot occur here.

LATENT LAYOUT (one camera, matching the RoboCasa slot order):
    proprio | ego_view | action | future proprio | future ego_view
Callers must set the model's `state_t` / `tokenizer.chunk_duration` to match the number of
slots this yields; `latent_slot_count()` returns it so the config cannot drift from the loader.
"""

import json
import os
import pickle
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from cosmos_policy.datasets.dataset_utils import preprocess_image, rescale_episode_data
from cosmos_policy.utils.utils import duplicate_array

EGO_VIEW_KEY = "observation.images.ego_view"

# --- slot-role / continuous-time PE (EgoVerse "mid" trick) -------------------------------
# The EgoVerse mid checkpoints were trained with a per-latent role id and a per-latent
# TIMESTAMP in seconds (egoverse_ret_v2_dataset.py). Their role vocabulary and time unit are
# reused verbatim here so a warm-started backbone sees the same conventions it learned; the
# role embedding rows in those checkpoints are demonstrably trained (per-row norms
# 0.22/0.75/0.76/0.81/0.79/0.97), so dropping them throws away a learned slot prior.
#
# NOW_S is arbitrary (rope is shift-invariant); DT must match the real spacing, and it does:
# num_duplicates_per_image=4 frames per latent at 20 fps = 0.2 s, the same 4.0/20.0 the
# EgoVerse datasets use. The future slots sit future_offset frames ahead.
NOW_S = 4.0
DT = 4.0 / 20.0
ROLE_CUR_STATE, ROLE_RET_ACTION, ROLE_RET_VIDEO, ROLE_CUR_IMAGE, ROLE_ACTION, ROLE_FUTURE = 0, 1, 2, 3, 4, 5


def build_slot_pe(
    state_t: int,
    proprio_idx: int,
    image_idx: int,
    action_idx: int,
    future_proprio_idx: int,
    future_image_idx: int,
    future_offset_s: float,
    ret_action_idx: int = -1,
    ret_video_start: int = -1,
    ret_video_end: int = -1,
):
    """Per-latent (t_positions, slot_roles) for the GR1 tabletop layout.

    Shared by the training loader and the eval server so the two cannot drift apart — every
    train/eval mismatch found on this project so far came from two places computing the same
    thing independently.

    Latent 0 is the tokenizer's un-duplicated blank; EgoVerse gives its own latent 0 the
    CUR_STATE role, so it gets the same here.
    """
    import numpy as _np

    roles = _np.full(state_t, ROLE_CUR_STATE, _np.int64)
    tpos = _np.full(state_t, NOW_S, _np.float32)
    for idx, role in (
        (proprio_idx, ROLE_CUR_STATE),
        (image_idx, ROLE_CUR_IMAGE),
        (action_idx, ROLE_ACTION),
        (future_proprio_idx, ROLE_FUTURE),
        (future_image_idx, ROLE_FUTURE),
    ):
        if idx is not None and 0 <= idx < state_t:
            roles[idx] = role
            if role == ROLE_FUTURE:
                tpos[idx] = NOW_S + future_offset_s
    # Blank retrieval prefix: RET_VIDEO latents sit at a PAST time, one DT apart, which is what
    # gives the backbone the non-NOW rope positions it was mid-trained on.
    if 0 <= ret_action_idx < state_t:
        roles[ret_action_idx] = ROLE_RET_ACTION
        tpos[ret_action_idx] = NOW_S
    if 0 <= ret_video_start < ret_video_end <= state_t:
        n = ret_video_end - ret_video_start
        roles[ret_video_start:ret_video_end] = ROLE_RET_VIDEO
        tpos[ret_video_start:ret_video_end] = NOW_S - DT * _np.arange(n, 0, -1)
    return tpos, roles


ACTION_DIM = 44
# left_arm 7 | left_hand 6 | left_leg 6 | neck 3 | right_arm 7 | right_hand 6 | right_leg 6 | waist 3
ACTION_GROUPS = {
    "left_arm": (0, 7),
    "left_hand": (7, 13),
    "left_leg": (13, 19),
    "neck": (19, 22),
    "right_arm": (22, 29),
    "right_hand": (29, 35),
    "right_leg": (35, 41),
    "waist": (41, 44),
}


def latent_slot_count(use_proprio: bool = True, predict_future_states: bool = True) -> int:
    """Number of duplicated content slots per sample (excludes the leading blank frame)."""
    slots = 1  # ego_view
    slots += 1 if use_proprio else 0
    slots += 1  # action
    if predict_future_states:
        slots += 1  # future ego_view
        slots += 1 if use_proprio else 0
    return slots


def model_shape(
    use_proprio: bool = True,
    predict_future_states: bool = True,
    num_duplicates_per_image: int = 4,
    plain_bc_layout: bool = False,
    n_blank_ret_video: int = 0,
    use_blank_ret_action: bool = False,
) -> tuple[int, int]:
    """Return (state_t, chunk_duration) the model config must use for this layout.

    Frames = 1 leading blank + slots * duplicates; the WAN2.1 tokenizer turns that into
    1 + (frames - 1) // 4 latents. Verified against RoboCasaDataset, which emits 37 frames for
    9 slots and pairs with chunk_duration=37 / state_t=10 in the paper's Predict2 config.
    """
    if plain_bc_layout:
        # blank1 (cur_state) + [ret_action] + [ret_video]*n + cur_image + action + future image;
        # proprio rides in latent 0.
        slots = 3 + n_blank_ret_video + (1 if use_blank_ret_action else 0)
    else:
        slots = latent_slot_count(use_proprio, predict_future_states)
    frames = 1 + slots * num_duplicates_per_image
    return 1 + (frames - 1) // 4, frames


class GR1TabletopDataset(Dataset):
    """LeRobot-backed dataset over the GR1 tabletop teleop demos.

    Args:
        data_dir: directory holding the `gr1_unified.<Task>` task directories.
        t5_text_embeddings_path: pickle of {task string: (1, 512, 1024) tensor}.
        chunk_size: number of action steps predicted per sample.
        tasks: task directory names to include (default: every `gr1_unified.*` found).
        max_episodes_per_task: cap for smoke tests; None uses every episode.
        future_offset: frames between the current step and the predicted future state.
        video_reader_cache_size: open decord readers kept per worker (episodes are read in runs).
    """

    def __init__(
        self,
        data_dir: str,
        t5_text_embeddings_path: str,
        chunk_size: int = 32,
        tasks: list[str] | None = None,
        max_episodes_per_task: int | None = None,
        allow_incomplete_tasks: bool = False,
        future_offset: int = 16,
        use_proprio: bool = True,
        normalize_proprio: bool = True,
        normalize_actions: bool = True,
        predict_future_states: bool = True,
        use_image_aug: bool = True,
        use_stronger_image_aug: bool = True,
        num_duplicates_per_image: int = 4,
        final_image_size: int = 224,
        # False to match RoboCasaDataset: preprocess_image returns (C, T, H, W), so torchvision's
        # Normalize would read the T axis as channels. The model normalises instead.
        normalize_images: bool = False,
        dataset_stats_path: str | None = None,
        video_reader_cache_size: int = 8,
        # Off by default: the base arm trains without it, so enabling this must be an explicit
        # per-experiment choice rather than something that silently changes every run.
        emit_slot_pe: bool = False,
        # Mirror the mimicdroid-GR1 plain-BC post-train layout exactly
        # (datasets/egoverse_gr1_plain_dataset.py):
        #     blank1 (cur_state, proprio injected) | cur_image | action | future image
        #     roles [0, 3, 4, 5]   n_cond = 2   state_t = 4
        # The default layout copies RoboCasaDataset instead: it spends a separate duplicated
        # slot on proprio AND keeps the un-duplicated blank, so role 0 lands on two latents (one
        # of them all zeros) and there is an extra future-proprio slot the reference does not
        # have. Warm-starting a checkpoint trained under the reference contract into that layout
        # asks it to read a slot arrangement it never saw.
        plain_bc_layout: bool = False,
        # Keep the STRUCTURE of the retrieval prefix the ret-mid checkpoints were trained with,
        # filled with zero pixels: a RET_ACTION latent and n_blank_video RET_VIDEO latents at a
        # PAST time. egoverse_ret_experiment_configs.py calls this the "BLANK-VIDEO bridge" and
        # reports it restores warm-start transfer that a prefix-less layout loses — roles 1 and 2
        # are otherwise never seen in post-train and arrive as zero-init role_embedder rows.
        # Only meaningful together with plain_bc_layout.
        n_blank_ret_video: int = 0,
        use_blank_ret_action: bool = False,
        # --- accepted only because the experiment inherits its dataloader from the LIBERO
        # config, whose dataset kwargs merge into this one. None of these have meaning here:
        # there is one camera (no wrist / second third-person view), and the dataset ships no
        # rollouts or value-function returns. They are declared rather than swallowed by
        # **kwargs so that a genuine typo still raises, and are rejected if set to a value that
        # would silently change behaviour.
        use_wrist_images: bool = False,
        use_third_person_images: bool = False,
        rollout_data_dir: str | None = None,
        return_value_function_returns: bool = False,
        treat_success_rollouts_as_demos: bool = False,
        demonstration_sampling_prob: float = 1.0,
        success_rollout_sampling_prob: float = 0.0,
        gamma: float = 0.99,
    ):
        if use_wrist_images:
            raise ValueError("GR1 tabletop has a single ego_view camera; use_wrist_images is not supported")
        if use_third_person_images:
            raise ValueError("GR1 tabletop has a single ego_view camera; use_third_person_images is not supported")
        if rollout_data_dir:
            raise ValueError("GR1 tabletop ships demos only; rollout_data_dir is not supported")
        if return_value_function_returns:
            raise ValueError("GR1 tabletop has no reward-to-go; return_value_function_returns is not supported")
        self.data_dir = Path(data_dir)
        self.chunk_size = chunk_size
        self.future_offset = future_offset
        self.use_proprio = use_proprio
        self.normalize_proprio = normalize_proprio
        self.normalize_actions = normalize_actions
        self.predict_future_states = predict_future_states
        self.use_image_aug = use_image_aug
        self.use_stronger_image_aug = use_stronger_image_aug
        self.num_duplicates_per_image = num_duplicates_per_image
        self.final_image_size = final_image_size
        self.normalize_images = normalize_images
        # Populated per DataLoader worker; decord readers are not picklable so this must not be
        # filled before fork.
        self._video_readers: dict = {}
        self._video_reader_cache_size = video_reader_cache_size
        self.emit_slot_pe = emit_slot_pe
        self.plain_bc_layout = plain_bc_layout
        self.n_blank_ret_video = int(n_blank_ret_video)
        self.use_blank_ret_action = bool(use_blank_ret_action)

        task_dirs = (
            [self.data_dir / t for t in tasks]
            if tasks
            else sorted(p for p in self.data_dir.glob("gr1_unified.*") if p.is_dir())
        )
        if not task_dirs:
            raise FileNotFoundError(f"No gr1_unified.* task directories under {self.data_dir}")

        # A download in flight leaves task directories that have data/ but no meta/. Training on
        # a silently reduced task set is worse than not starting, so this is fatal unless the
        # caller opts in (smoke tests against a partial download).
        incomplete = [d for d in task_dirs if not (d / "meta" / "info.json").exists()]
        if incomplete:
            names = ", ".join(d.name for d in incomplete[:3]) + (" ..." if len(incomplete) > 3 else "")
            if not allow_incomplete_tasks:
                raise FileNotFoundError(
                    f"{len(incomplete)}/{len(task_dirs)} task directories under {self.data_dir} have no "
                    f"meta/info.json ({names}). The download is probably still running. Pass "
                    "allow_incomplete_tasks=True to train on the complete ones anyway."
                )
            print(f"WARNING: skipping {len(incomplete)} incomplete task directories: {names}")
            task_dirs = [d for d in task_dirs if d not in set(incomplete)]
            if not task_dirs:
                raise FileNotFoundError(f"Every task directory under {self.data_dir} is incomplete")

        # Index episodes without opening any parquet or mp4: meta/episodes.jsonl already carries
        # the per-episode length, which is all that is needed to build the flat sample index.
        self.episodes: list[dict] = []
        self.unique_commands: set[str] = set()
        missing_files: list[str] = []
        for task_dir in task_dirs:
            info = json.loads((task_dir / "meta" / "info.json").read_text())
            chunks_size = info.get("chunks_size", 1000)
            data_tmpl = info["data_path"]
            video_tmpl = info["video_path"]
            with open(task_dir / "meta" / "episodes.jsonl") as f:
                records = [json.loads(line) for line in f if line.strip()]
            if max_episodes_per_task is not None:
                records = records[:max_episodes_per_task]
            for rec in records:
                ep_idx = rec["episode_index"]
                # Prefer the per-episode `remarks` sentence over the task id. For the 18
                # Posttrain*Novel* tasks the object is randomised per episode and `remarks` names
                # it ("pick the croissant from the plate and place it in the bowl") — 10 distinct
                # objects per task, 186 sentences across the dataset. `tasks[0]` collapses all of
                # them to one task id, so the policy is told "the object" while several sit on
                # the table (the env even scores `grasp_distractor_obj` separately). The eval env
                # publishes the same sentence under `annotation.human.coarse_action`, prefixed
                # with "unlocked_waist: ", so train and eval share one vocabulary.
                command = rec.get("remarks") or (
                    rec["tasks"][0] if rec.get("tasks") else rec.get("description", "")
                )
                chunk = ep_idx // chunks_size
                parquet = task_dir / data_tmpl.format(episode_chunk=chunk, episode_index=ep_idx)
                video = task_dir / video_tmpl.format(
                    episode_chunk=chunk, video_key=EGO_VIEW_KEY, episode_index=ep_idx
                )
                # episodes.jsonl lists every episode in the source dataset, including ones whose
                # files have not arrived yet. Indexing them anyway defers the failure to a
                # DataLoader worker mid-training, where it surfaces as an opaque decode error.
                if not (parquet.exists() and video.exists()):
                    missing_files.append(f"{task_dir.name}/episode_{ep_idx:06d}")
                    continue
                self.episodes.append(
                    {
                        "task_dir": task_dir,
                        "episode_index": ep_idx,
                        "length": rec["length"],
                        "command": command,
                        "parquet": parquet,
                        "video": video,
                        "fps": info["fps"],
                    }
                )
                self.unique_commands.add(command)

        if missing_files:
            shown = ", ".join(missing_files[:3]) + (" ..." if len(missing_files) > 3 else "")
            if not allow_incomplete_tasks:
                raise FileNotFoundError(
                    f"{len(missing_files)} episodes listed in episodes.jsonl have no parquet or mp4 "
                    f"on disk ({shown}). The download is probably still running. Pass "
                    "allow_incomplete_tasks=True to train on what is present."
                )
            print(f"WARNING: skipping {len(missing_files)} episodes with missing files: {shown}")

        # Flat (episode, step) index. The last `chunk_size` steps are dropped so every sample has
        # a full action chunk without padding.
        self.index: list[tuple[int, int]] = []
        for ep_i, ep in enumerate(self.episodes):
            usable = ep["length"] - self.chunk_size
            self.index.extend((ep_i, s) for s in range(max(usable, 0)))
        if not self.index:
            raise ValueError(f"No usable samples: every episode is shorter than chunk_size={chunk_size}")

        with open(t5_text_embeddings_path, "rb") as f:
            self.t5_text_embeddings = pickle.load(f)
        missing = sorted(c for c in self.unique_commands if c not in self.t5_text_embeddings)
        if missing:
            raise KeyError(
                f"{len(missing)} task strings have no text embedding in {t5_text_embeddings_path}: {missing[:5]}"
            )

        self.dataset_stats = None
        if dataset_stats_path and os.path.exists(dataset_stats_path):
            with open(dataset_stats_path) as f:
                self.dataset_stats = {k: np.array(v) for k, v in json.load(f).items()}
        elif normalize_actions or normalize_proprio:
            # Every task ships meta/stats.json, so derive the pooled stats rather than making
            # the caller precompute anything.
            self.dataset_stats = self._pooled_stats(task_dirs)

        print(
            f"GR1TabletopDataset: {len(task_dirs)} tasks, {len(self.episodes)} episodes, "
            f"{len(self.index)} samples, {len(self.unique_commands)} unique commands"
        )

    @staticmethod
    def _pooled_stats(task_dirs: list[Path]) -> dict[str, np.ndarray]:
        """Pool the per-task meta/stats.json files into one set of normalisation stats.

        Each task ships its own stats, but training mixes all 24, so min/max are taken across
        tasks and mean/std are pooled by trajectory length (a plain average of per-task means
        would over-weight short tasks). std is combined via the law of total variance, so the
        spread *between* task means is not thrown away.

        Legs and neck are constant zero on this embodiment, giving std = 0 and max - min = 0,
        so the range has to be widened to keep the rescale denominator finite. It is widened
        SYMMETRICALLY (v +/- 0.5) rather than upward (max = min + 1): the upward form puts the
        constant value at the bottom of the range, so `2*(x-min)/(max-min) - 1` maps it to -1 and
        the model has to emit the extreme of the output range for 15 of 44 dimensions that carry
        no information at all — a cost that action_loss_multiplier then scales up.
        """
        keymap = {"actions": "action", "proprio": "observation.state"}
        pooled: dict[str, np.ndarray] = {}
        for out_key, in_key in keymap.items():
            mins, maxs, means, varis, weights = [], [], [], [], []
            for task_dir in task_dirs:
                s = json.loads((task_dir / "meta" / "stats.json").read_text())
                st = s[in_key]
                n = float(s.get("total_trajectory_length") or 1)
                mins.append(np.asarray(st["min"], dtype=np.float64))
                maxs.append(np.asarray(st["max"], dtype=np.float64))
                means.append(np.asarray(st["mean"], dtype=np.float64))
                varis.append(np.asarray(st["std"], dtype=np.float64) ** 2)
                weights.append(n)
            w = np.asarray(weights)[:, None]
            total = w.sum()
            mean = (np.stack(means) * w).sum(0) / total
            within = (np.stack(varis) * w).sum(0) / total
            between = ((np.stack(means) - mean) ** 2 * w).sum(0) / total
            std = np.sqrt(within + between)
            lo, hi = np.stack(mins).min(0), np.stack(maxs).max(0)
            rng = hi - lo
            constant = np.abs(rng) < 1e-6
            # Symmetric widening: a constant v gets [v-0.5, v+0.5] so it normalises to 0.
            lo = np.where(constant, lo - 0.5, lo)
            hi = np.where(constant, hi + 0.5, hi)
            std = np.where(std < 1e-6, 1.0, std)
            pooled[f"{out_key}_min"] = lo
            pooled[f"{out_key}_max"] = hi
            pooled[f"{out_key}_mean"] = mean
            pooled[f"{out_key}_std"] = std
            pooled[f"{out_key}_median"] = mean  # stats.json has no median; mean is the closest proxy
            if constant.any():
                print(f"  {out_key}: {int(constant.sum())}/{len(lo)} constant dims (legs/neck) clamped")
        return pooled

    def __len__(self) -> int:
        return len(self.index)

    def _read_lowdim(self, ep: dict) -> tuple[np.ndarray, np.ndarray]:
        df = pd.read_parquet(ep["parquet"], columns=["observation.state", "action"])
        state = np.stack(df["observation.state"].to_numpy()).astype(np.float32)
        action = np.stack(df["action"].to_numpy()).astype(np.float32)
        return state, action

    def _read_frames(self, ep: dict, frame_indices: list[int]) -> np.ndarray:
        """Decode the requested frames as uint8 (N, H, W, C).

        decord rather than lerobot's `decode_video_frames`: decord is already a dependency of
        this repo (the cu128 extra), indexes by frame number so there is no timestamp/tolerance
        round-trip, and keeps the training pods from needing lerobot (whose video_utils pulls in
        `datasets`) on PYTHONPATH.

        Readers are cached per worker because opening an mp4 costs far more than seeking inside
        one, and consecutive samples usually come from the same episode.
        """
        import decord

        path = str(ep["video"])
        reader = self._video_readers.get(path)
        if reader is None:
            reader = decord.VideoReader(path, num_threads=1)
            if len(self._video_readers) >= self._video_reader_cache_size:
                self._video_readers.pop(next(iter(self._video_readers)))
            self._video_readers[path] = reader
        n = len(reader)
        safe = [min(max(i, 0), n - 1) for i in frame_indices]
        return reader.get_batch(safe).asnumpy()

    def __getitem__(self, idx: int) -> dict:
        ep_i, step = self.index[idx]
        ep = self.episodes[ep_i]
        state, action = self._read_lowdim(ep)

        last = len(state) - 1
        future_idx = min(step + self.future_offset, last)
        frames = self._read_frames(ep, [step, future_idx])
        current_image, future_image = frames[0], frames[1]

        action_chunk = action[step : step + self.chunk_size]
        proprio = state[step]
        future_proprio = state[future_idx]
        # rescale_episode_data returns the rescaled array itself, not a dict.
        if self.normalize_actions and self.dataset_stats is not None:
            action_chunk = rescale_episode_data({"actions": action_chunk}, self.dataset_stats, "actions")
        if self.normalize_proprio and self.dataset_stats is not None:
            proprio = rescale_episode_data({"proprio": proprio[None]}, self.dataset_stats, "proprio")[0]
            future_proprio = rescale_episode_data({"proprio": future_proprio[None]}, self.dataset_stats, "proprio")[0]

        # Latent slots, in the same order RoboCasaDataset uses so the model config carries over:
        #   proprio | ego_view | action | future proprio | future ego_view
        blank = np.zeros_like(current_image)
        if self.plain_bc_layout:
            # blank1 doubles as the proprio slot (the value is injected into that latent), so
            # there is no separate proprio slot and no future-proprio slot. The optional blank
            # retrieval prefix sits between them, matching the ret-mid contract.
            dup = self.num_duplicates_per_image
            nbv = self.n_blank_ret_video
            image_list = [np.expand_dims(blank, axis=0)]
            seq = 1
            retrieved_action_latent_idx = -1
            if self.use_blank_ret_action:
                image_list.append(duplicate_array(blank.copy(), total_num_copies=dup))
                retrieved_action_latent_idx = seq
                seq += 1
            retrieved_video_start_latent_idx = retrieved_video_end_latent_idx = -1
            if nbv:
                retrieved_video_start_latent_idx = seq
                for _ in range(nbv):
                    image_list.append(duplicate_array(blank.copy(), total_num_copies=dup))
                    seq += 1
                retrieved_video_end_latent_idx = seq
            image_list.append(duplicate_array(current_image, total_num_copies=dup))
            current_image_latent_idx = seq
            seq += 1
            image_list.append(duplicate_array(blank.copy(), total_num_copies=dup))
            action_latent_idx = seq
            seq += 1
            image_list.append(duplicate_array(future_image, total_num_copies=dup))
            future_image_latent_idx = seq
            current_proprio_latent_idx = 0
            future_proprio_latent_idx = -1
        else:
            image_list = []
            seq = 0
            # Slot 0 is a single un-duplicated blank frame the tokenizer requires; every real slot is
            # duplicated num_duplicates_per_image times. Total frames = 1 + slots * duplicates, and
            # the tokenizer maps that to 1 + (frames - 1) // 4 latents. Omitting it silently shifts
            # every latent index by one.
            image_list.append(np.expand_dims(blank, axis=0))
            seq += 1
            current_proprio_latent_idx = -1
            if self.use_proprio:
                image_list.append(duplicate_array(blank, total_num_copies=self.num_duplicates_per_image))
                current_proprio_latent_idx = seq
                seq += 1
            image_list.append(duplicate_array(current_image, total_num_copies=self.num_duplicates_per_image))
            current_image_latent_idx = seq
            seq += 1
            image_list.append(duplicate_array(blank.copy(), total_num_copies=self.num_duplicates_per_image))
            action_latent_idx = seq
            seq += 1

            future_proprio_latent_idx = future_image_latent_idx = -1
            if self.predict_future_states:
                if self.use_proprio:
                    image_list.append(duplicate_array(blank.copy(), total_num_copies=self.num_duplicates_per_image))
                    future_proprio_latent_idx = seq
                    seq += 1
                image_list.append(duplicate_array(future_image, total_num_copies=self.num_duplicates_per_image))
                future_image_latent_idx = seq
                seq += 1

        images = np.concatenate(image_list, axis=0)
        if np.isnan(images).any() or np.isinf(images).any():
            raise ValueError(f"Invalid image data for sample {idx} (episode {ep['episode_index']})")
        images = preprocess_image(
            images,
            final_image_size=self.final_image_size,
            normalize_images=self.normalize_images,
            use_image_aug=self.use_image_aug,
            stronger_image_aug=self.use_stronger_image_aug,
        )

        command = ep["command"]
        extra = {}
        if self.emit_slot_pe:
            tpos, roles = build_slot_pe(
                state_t=len(image_list),  # one latent per appended slot (blank included)
                proprio_idx=current_proprio_latent_idx,
                image_idx=current_image_latent_idx,
                action_idx=action_latent_idx,
                future_proprio_idx=future_proprio_latent_idx,
                future_image_idx=future_image_latent_idx,
                future_offset_s=self.future_offset / float(ep["fps"]),
                ret_action_idx=locals().get("retrieved_action_latent_idx", -1),
                ret_video_start=locals().get("retrieved_video_start_latent_idx", -1),
                ret_video_end=locals().get("retrieved_video_end_latent_idx", -1),
            )
            extra = {"t_positions": tpos, "slot_roles": roles}
        return {
            **extra,
            "video": images,
            "command": command,
            "actions": action_chunk,
            "t5_text_embeddings": torch.squeeze(self.t5_text_embeddings[command]),
            "t5_text_mask": torch.ones(512, dtype=torch.int64),
            "fps": 16,
            "padding_mask": torch.zeros(1, self.final_image_size, self.final_image_size),
            "image_size": self.final_image_size * torch.ones(4),
            "proprio": proprio,
            "future_proprio": future_proprio,
            "__key__": idx,
            # This dataset has no rollout/value-function split; keep the keys the model expects.
            "rollout_data_mask": 0,
            "rollout_data_success_mask": 0,
            "world_model_sample_mask": 0,
            "value_function_sample_mask": 0,
            "global_rollout_idx": -1,
            "action_latent_idx": action_latent_idx,
            "value_latent_idx": -1,
            "current_proprio_latent_idx": current_proprio_latent_idx,
            "current_wrist_image_latent_idx": -1,
            "current_image_latent_idx": current_image_latent_idx,
            "current_image2_latent_idx": -1,
            "future_proprio_latent_idx": future_proprio_latent_idx,
            "future_wrist_image_latent_idx": -1,
            "future_image_latent_idx": future_image_latent_idx,
            "future_image2_latent_idx": -1,
            "value_function_return": float("-100"),
        }
