"""Object-aware initial-scene filter for RoboTwin RAG inference.

Subclass of :class:`RoboTwinDinoProprioRetrieval` that adds a landmark-L2
term to the per-rollout initial-scene episode filter. Generalizes over
arbitrary task object schemas via the JSON config maintained in
``robotwin_rag_tools/robotwin_rag/configs/object_init_tasks.json`` —
prompts, expected counts, and IOU disambiguation rules all come from there.

Pool object positions come from the offline cache produced by
``robotwin_rag_tools/scripts/run_object_extract.py`` at:
    {obj_init_root}/{task}/{split}/episode{N}.npz
        {name}_xy       (n, 2)   float32  abs px       (n: number kept after filtering)
        {name}_box      (n, 4)   float32
        {name}_score    (n,)     float32
        ...
        image_size      (2,)     int32    [H, W]
        _object_names   (k,)     |U64     names listed in this NPZ — preferred over config

Query positions are detected ONCE per rollout — on the first ``act`` after
``reset()`` — by running SAM3 on the live head image with the same prompts
+ IOU rules as the offline extractor.

Distance metric per (query_scene, pool_scene): mean L2 over all matched
landmarks across all object classes (each landmark contributes equally,
regardless of which class). Episodes whose pool scene is missing or
incomplete are excluded outright.

Score change vs. parent:
    ep_score = w_dino * ep_dino_cos
             - w_prop * ep_start_proprio_L2
             - w_obj  * ep_landmark_L2          # ← new term
"""
from __future__ import annotations

import base64
import json
import os
import pickle
import subprocess
import sys
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Progression-prefilter helpers (DINO + T5 + area-weighted greedy SAM3).
# Module-level so the override below can stay short.
# ---------------------------------------------------------------------------

# Per-pair sub-weights for the greedy distance (xy + log-area + log-aspect).
_PROG_SAM3_W_XY = 1.0
_PROG_SAM3_W_SIZE = 0.5
_PROG_SAM3_W_SHAPE = 0.5
_PROG_SAM3_UNMATCHED_DIST = 2.0


def _quadrant_rgb_sig(image_rgb: np.ndarray, box_xyxy) -> np.ndarray:
    """4 quadrants × mean RGB → (12,) float32 in [0, 255].

    Mirrors run_object_extract.py:_quadrant_rgb_sig so live-query and pool
    signatures share the same computation. Captures bbox-internal asymmetry
    (handle vs body of a roller, etc.) that bbox (w, h) discards.
    """
    H, W = image_rgb.shape[:2]
    x0 = max(0, int(round(float(box_xyxy[0]))))
    y0 = max(0, int(round(float(box_xyxy[1]))))
    x1 = min(W, int(round(float(box_xyxy[2]))))
    y1 = min(H, int(round(float(box_xyxy[3]))))
    if x1 <= x0 or y1 <= y0:
        return np.zeros(12, dtype=np.float32)
    crop = image_rgb[y0:y1, x0:x1].astype(np.float32)
    h, w = crop.shape[:2]
    cy, cx = h // 2, w // 2
    quads = (
        crop[:cy, :cx],   # TL
        crop[:cy, cx:],   # TR
        crop[cy:, :cx],   # BL
        crop[cy:, cx:],   # BR
    )
    return np.concatenate(
        [q.reshape(-1, 3).mean(axis=0) if q.size else np.zeros(3, dtype=np.float32) for q in quads]
    ).astype(np.float32)


def _scene_to_object_list(scene: dict) -> list[dict]:
    """Convert RoboTwinObjectAwareRetrieval's scene dict (one normalized
    [cx, cy, w, h] row per object name) into the per-object descriptor list
    used by the greedy matcher: {name, cx, cy, area, aspect}.

    Scene rows here are [cx, cy, w, h] normalized to [0, 1] by image size,
    matching what _detect_query_scene / _load_pool_object_scenes produces.
    """
    out: list[dict] = []
    for name, row in scene.items():
        if name.startswith("_"):
            continue
        arr = np.asarray(row, dtype=np.float32).reshape(-1)
        if arr.size < 4:
            continue
        cx, cy, w, h = float(arr[0]), float(arr[1]), float(arr[2]), float(arr[3])
        w = max(w, 1e-6)
        h = max(h, 1e-6)
        out.append({
            "name": str(name),
            "cx": cx,
            "cy": cy,
            "area": w * h,
            "aspect": w / h,
        })
    return out


def _greedy_pair_dist(a: dict, b: dict) -> float:
    """Per-object-pair distance: xy L2 + |Δlog(area)| + |Δlog(aspect)|."""
    dxy = float(np.hypot(a["cx"] - b["cx"], a["cy"] - b["cy"]))
    ds = abs(float(np.log(a["area"] + 1e-6) - np.log(b["area"] + 1e-6)))
    da = abs(float(np.log(a["aspect"]) - np.log(b["aspect"])))
    return _PROG_SAM3_W_XY * dxy + _PROG_SAM3_W_SIZE * ds + _PROG_SAM3_W_SHAPE * da


def _greedy_pair_dist_weighted(
    a: dict, b: dict, *, w_xy: float, w_size: float, w_shape: float,
) -> float:
    """Per-object-pair distance with custom sub-weights — used by the
    chunk-0 xy-emphasized rescore."""
    dxy = float(np.hypot(a["cx"] - b["cx"], a["cy"] - b["cy"]))
    ds = abs(float(np.log(a["area"] + 1e-6) - np.log(b["area"] + 1e-6)))
    da = abs(float(np.log(a["aspect"]) - np.log(b["aspect"])))
    return w_xy * dxy + w_size * ds + w_shape * da


def _greedy_sim_weighted(
    q_objs: list[dict], p_objs: list[dict], *,
    w_xy: float, w_size: float, w_shape: float, tau: float = 0.4,
) -> float:
    """Greedy bipartite sim with custom sub-weights (xy-emphasis variant).
    Mirrors render_progression_runtime_video._sam3_score_xy_weighted."""
    if not q_objs or not p_objs:
        return 0.0
    pairs = sorted(
        (_greedy_pair_dist_weighted(a, b, w_xy=w_xy, w_size=w_size, w_shape=w_shape), i, j)
        for i, a in enumerate(q_objs)
        for j, b in enumerate(p_objs)
    )
    used_a, used_b = set(), set()
    distances: list[float] = []
    for d, i, j in pairs:
        if i in used_a or j in used_b:
            continue
        used_a.add(i); used_b.add(j); distances.append(d)
    if not distances:
        return 0.0
    return float(np.exp(-float(np.mean(distances)) / max(tau, 1e-6)))


def _greedy_area_weighted_sim(
    q_objs: list[dict], p_objs: list[dict], *, tau: float = 0.4
) -> float:
    """Uniform-weighted greedy bipartite matching similarity, in [0, 1].

    Score = exp(-mean(d_pair) / tau), with each greedy-matched pair counted
    equally regardless of object area. Mirrors progression_prefilter.py's
    sam3_score. (Function name retains ``area_weighted`` for backward compat
    with import sites — the weighting is now uniform.)

    Why uniform: area-weighted averaging hides small-object xy asymmetries
    behind a large matched anchor (e.g. basket pair dominates → small can
    left↔right mismatch invisible). For our cross-embodiment retrieval, the
    SAM3 prefilter should surface that asymmetry so we don't pick mirrored
    pool eps as top candidates.
    """
    if not q_objs or not p_objs:
        return 0.0
    pairs = sorted(
        (_greedy_pair_dist(a, b), i, j)
        for i, a in enumerate(q_objs)
        for j, b in enumerate(p_objs)
    )
    used_a, used_b = set(), set()
    distances: list[float] = []
    for d, i, j in pairs:
        if i in used_a or j in used_b:
            continue
        used_a.add(i)
        used_b.add(j)
        distances.append(d)
    # Object-count mismatches (one side has leftover objects) are NOT
    # penalised — see progression_prefilter.sam3_score docstring for the
    # rationale. Pool-side bookkeeping objects (caps, lids, accessories)
    # shouldn't tank cross-task similarity when the primary manipuland matches.
    if not distances:
        return 0.0
    d_mean = float(np.mean(distances))
    return float(np.exp(-d_mean / max(tau, 1e-6)))


def _load_t5_task_unit_vec(task: str, t5_path: str) -> Optional[np.ndarray]:
    """Mean-pool valid tokens of a task's T5 embedding and L2-normalize.
    Returns None on failure so callers can disable the T5 term gracefully."""
    if not t5_path or not os.path.isfile(t5_path):
        return None
    try:
        with open(t5_path, "rb") as f:
            d = pickle.load(f)
        # Keys typically use spaces ("place cans plasticbox") not underscores.
        key = task.replace("_", " ")
        if key not in d and task in d:
            key = task
        if key not in d:
            return None
        raw = d[key]
        arr = torch.as_tensor(raw).to(torch.float32).cpu().numpy().squeeze(0)  # (T, D)
        norms = np.linalg.norm(arr, axis=1)
        mask = norms > 1e-6
        v = arr[mask].mean(axis=0) if mask.sum() > 0 else arr.mean(axis=0)
        n = np.linalg.norm(v)
        return (v / max(n, 1e-8)).astype(np.float32)
    except Exception as exc:  # noqa: BLE001
        print(f"[progression-prefilter] WARN: T5 load failed for {task!r}: {exc!r}", flush=True)
        return None

from cosmos_policy.experiments.robot.robotwin_ret.retrieval import (
    RoboTwinDinoProprioRetrieval,
)


# ---------------------------------------------------------------------------
# Inline copy of the object_init task spec loader. We keep this co-located
# (rather than importing from the sibling robotwin_rag_tools repo) so the
# cosmos_policy server has no extra path dependency — same convention as the
# inline _DinoCLSExtractor mirror of robotwin_rag.features.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _ObjectSpec:
    name: str
    prompts: tuple[str, ...]
    expected_count: int
    max_keep: int
    threshold: float = 0.5
    mask_threshold: float = 0.5
    exclude_overlap_with: Optional[str] = None
    iou_threshold: float = 0.3
    # Mirror robotwin_rag.object_init_config.ObjectSpec extension fields so
    # online query-side detection applies the same gates as the offline
    # extractor (run_object_extract.py).
    y_min: float = 0.0
    y_max: float = 1.0
    exclude_xyxy_norm: tuple[tuple[float, float, float, float], ...] = ()

    @property
    def prompt(self) -> str:
        """Legacy single-prompt accessor — returns first entry of ``prompts``."""
        return self.prompts[0]


@dataclass(frozen=True)
class _TaskSpec:
    task: str
    objects: tuple[_ObjectSpec, ...]


_DEFAULT_CONFIG_PATH = "/mnt/ddn/robotwin_rag_tools/robotwin_rag/configs/object_init_tasks.json"


def _load_task_spec(task: str, config_path: str) -> _TaskSpec:
    with open(config_path, "r") as f:
        raw = json.load(f)
    if task not in raw:
        raise KeyError(
            f"task {task!r} not in {config_path}. Known tasks: {sorted(raw.keys())}"
        )
    objs: list[_ObjectSpec] = []
    seen: set[str] = set()
    for entry in raw[task]["objects"]:
        if "prompts" in entry:
            prompts = tuple(str(p) for p in entry["prompts"])
        elif "prompt" in entry:
            prompts = (str(entry["prompt"]),)
        else:
            raise ValueError(
                f"object entry {entry.get('name', '?')!r} in task {task!r} "
                f"must have either 'prompt' (str) or 'prompts' (list[str])"
            )
        if not prompts:
            raise ValueError(
                f"object {entry.get('name', '?')!r} in task {task!r}: "
                f"prompts list must be non-empty"
            )
        spec = _ObjectSpec(
            name=entry["name"],
            prompts=prompts,
            expected_count=int(entry["expected_count"]),
            max_keep=int(entry["max_keep"]),
            threshold=float(entry.get("threshold", 0.5)),
            mask_threshold=float(entry.get("mask_threshold", 0.5)),
            exclude_overlap_with=entry.get("exclude_overlap_with"),
            iou_threshold=float(entry.get("iou_threshold", 0.3)),
            y_min=float(entry.get("y_min", 0.0)),
            y_max=float(entry.get("y_max", 1.0)),
            exclude_xyxy_norm=tuple(
                tuple(float(v) for v in box) for box in entry.get("exclude_xyxy_norm", [])
            ),
        )
        if spec.exclude_overlap_with is not None and spec.exclude_overlap_with not in seen:
            raise ValueError(
                f"object {spec.name!r}: exclude_overlap_with={spec.exclude_overlap_with!r} "
                f"must reference an object listed earlier in the task"
            )
        seen.add(spec.name)
        objs.append(spec)
    return _TaskSpec(task=task, objects=tuple(objs))


# ---------------------------------------------------------------------------
# Pool side: load per-episode landmark positions from the object_init cache
# ---------------------------------------------------------------------------


def _load_pool_object_scenes(
    obj_init_root: str,
    task: str,
    pool_data: list[dict],
    expected_counts: dict[str, int],
    # Optional per-task overrides for multi-task pool: when a pool episode has
    # ``ep_meta["task"]`` set to one of these keys, use that task's
    # obj_init/expected_counts instead of the top-level ``task`` argument.
    expected_counts_by_task: Optional[dict[str, dict[str, int]]] = None,
) -> list[Optional[dict]]:
    """For each pool episode in ``pool_data`` order, return either a scene
    dict or ``None`` when the cache is missing / incomplete (any expected
    object short of its required count).

    Scene keys:
      ``<name>``           (n, 2) float32 normalized [0,1] xy — used by distance
      ``_pixel_centroids`` dict[name -> (n, 2) float32] raw pixel centroids
      ``_pixel_boxes``     dict[name -> (n, 4) float32] raw pixel xyxy
      ``_scores``          dict[name -> list[float]]
      ``_image_hw``        (H, W) the pool frame's pixel size
    """
    out: list[Optional[dict]] = []
    n_missing_file = 0
    n_incomplete = 0
    for ep_meta in pool_data:
        # Multi-task pool: per-ep task picks its own obj_init folder + the
        # expected_counts of that task's spec. Single-task pool degenerates
        # to the legacy path (ep_meta["task"] == task; same counts).
        ep_task = ep_meta.get("task", task)
        ep_expected = expected_counts
        if expected_counts_by_task is not None and ep_task in expected_counts_by_task:
            ep_expected = expected_counts_by_task[ep_task]
        npz = os.path.join(obj_init_root, ep_task, ep_meta["split"], f"episode{ep_meta['ep_num']}.npz")
        if not os.path.isfile(npz):
            out.append(None)
            n_missing_file += 1
            continue
        with np.load(npz) as d:
            img_hw = d["image_size"]
            objects: dict[str, np.ndarray] = {}
            boxes_by_name: dict[str, np.ndarray] = {}
            scores_by_name: dict[str, list[float]] = {}
            ok = True
            for name, expected in ep_expected.items():
                key = f"{name}_xy"
                if key not in d.files:
                    ok = False
                    break
                xy = d[key]
                if xy.shape[0] < expected:
                    ok = False
                    break
                objects[name] = xy.astype(np.float32)
                box_key = f"{name}_box"
                score_key = f"{name}_score"
                boxes_by_name[name] = (
                    d[box_key].astype(np.float32) if box_key in d.files else np.zeros((0, 4), dtype=np.float32)
                )
                scores_by_name[name] = (
                    d[score_key].astype(np.float32).tolist() if score_key in d.files else []
                )
        if not ok:
            out.append(None)
            n_incomplete += 1
            continue
        H, W = float(img_hw[0]), float(img_hw[1])
        norm = np.array([W, H], dtype=np.float32)
        normed: dict[str, np.ndarray] = {}
        sizes: dict[str, np.ndarray] = {}
        wh: dict[str, np.ndarray] = {}
        quad_sigs: dict[str, np.ndarray] = {}
        # Pull quad_sig fields (one (n, 12) per object) from the same npz when
        # present. Absent fields fall back to a zero matrix so downstream
        # checks `q_sig is not None and q_sig.shape[0] >= n` simply skip them.
        with np.load(npz) as d:
            for name in objects:
                qs_key = f"{name}_quad_sig"
                if qs_key in d.files:
                    quad_sigs[name] = d[qs_key].astype(np.float32)
        for name, xy in objects.items():
            xy_n = (xy / norm).astype(np.float32)
            # Per-box aspect ratio scalar = max(w_px, h_px) / min(w_px, h_px)
            # in [1, inf). Pure shape signal, area-/normalization-invariant.
            # SAME row order as centroids.
            box = boxes_by_name.get(name, np.zeros((0, 4), dtype=np.float32))
            if box.shape[0] >= xy_n.shape[0]:
                w_px = np.maximum(box[: xy_n.shape[0], 2] - box[: xy_n.shape[0], 0], 1e-3)
                h_px = np.maximum(box[: xy_n.shape[0], 3] - box[: xy_n.shape[0], 1], 1e-3)
                long_side = np.maximum(w_px, h_px)
                short_side = np.maximum(np.minimum(w_px, h_px), 1e-3)
                sz = (long_side / short_side).astype(np.float32)
                # Normalized (w, h) pair. Captures orientation that aspect (max/min)
                # discards — horizontal vs vertical roller has same aspect but
                # different (w, h). Used by the optional ``w_obj_wh_l1`` term.
                wh_pair = np.stack([w_px / W, h_px / H], axis=1).astype(np.float32)
            else:
                sz = np.zeros(xy_n.shape[0], dtype=np.float32)
                wh_pair = np.zeros((xy_n.shape[0], 2), dtype=np.float32)
            qs = quad_sigs.get(name)
            if qs is None or qs.shape[0] < xy_n.shape[0]:
                qs = np.zeros((xy_n.shape[0], 12), dtype=np.float32)
            else:
                qs = qs[: xy_n.shape[0]].astype(np.float32)
            if xy_n.shape[0] > 1:
                order = np.argsort(xy_n[:, 0])
                xy_n = xy_n[order]
                sz = sz[order]
                wh_pair = wh_pair[order]
                qs = qs[order]
            normed[name] = xy_n
            sizes[name] = sz
            wh[name] = wh_pair
            quad_sigs[name] = qs
        normed["_norm_sizes"] = sizes
        normed["_norm_wh"] = wh
        normed["_norm_quad_sig"] = quad_sigs
        normed["_pixel_centroids"] = objects
        normed["_pixel_boxes"] = boxes_by_name
        normed["_scores"] = scores_by_name
        normed["_image_hw"] = (int(img_hw[0]), int(img_hw[1]))
        out.append(normed)

    n_ok = sum(1 for s in out if s is not None)
    print(
        f"[obj-prefilter] pool object scenes: {n_ok}/{len(out)} usable "
        f"(missing_file={n_missing_file}, incomplete_detection={n_incomplete})",
        flush=True,
    )
    return out


# ---------------------------------------------------------------------------
# Query side: lazy SAM3 detector
# ---------------------------------------------------------------------------


_DEFAULT_SAM3_DAEMON_PYTHON = "/mnt/ddn/robotwin_rag_tools/.venv/bin/python"
_DEFAULT_SAM3_DAEMON_SCRIPT = "/mnt/ddn/robotwin_rag_tools/scripts/sam3_daemon.py"
_DEFAULT_SAM3_DAEMON_PYTHONPATH = "/mnt/ddn/robotwin_rag_tools"


class _Sam3Detector:
    """SAM3 detector that runs in a sidecar subprocess.

    Why subprocess? cosmos_policy pins ``transformers==4.57.1`` which lacks
    the ``Sam3Model`` class — that one only landed in ``transformers>=5.x``.
    Upgrading transformers in cosmos_policy would risk breaking the Cosmos
    diffusion model loaders, so we instead spawn ``scripts/sam3_daemon.py``
    under a Python that already has the right ``transformers`` (the
    robotwin_rag_tools env). Communication is line-buffered JSON over the
    daemon's stdin/stdout — see ``sam3_daemon.py`` for the protocol.

    The daemon stays alive for the whole server lifetime, so the SAM3 model
    is loaded exactly once even though we only call it once per rollout.
    """

    def __init__(
        self,
        device: Optional[str] = None,  # accepted for API compatibility; daemon owns its own GPU placement
        daemon_python: str = _DEFAULT_SAM3_DAEMON_PYTHON,
        daemon_script: str = _DEFAULT_SAM3_DAEMON_SCRIPT,
        daemon_pythonpath: str = _DEFAULT_SAM3_DAEMON_PYTHONPATH,
        startup_timeout_s: float = 120.0,
    ) -> None:
        self._device_hint = device
        self._proc: Optional[subprocess.Popen] = None
        self._daemon_python = daemon_python
        self._daemon_script = daemon_script
        self._daemon_pythonpath = daemon_pythonpath
        self._startup_timeout_s = float(startup_timeout_s)
        self._spawn()

    # ------------------------------------------------------------------
    def _spawn(self) -> None:
        env = os.environ.copy()
        env["PYTHONPATH"] = self._daemon_pythonpath + (
            os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
        )
        # Force unbuffered stdout from the daemon so our readline() doesn't
        # wait for a full pipe block.
        env["PYTHONUNBUFFERED"] = "1"
        if not os.path.isfile(self._daemon_python):
            raise RuntimeError(
                f"SAM3 daemon python not found: {self._daemon_python}. "
                "Override via cfg.sam3_daemon_python."
            )
        if not os.path.isfile(self._daemon_script):
            raise RuntimeError(
                f"SAM3 daemon script not found: {self._daemon_script}. "
                "Override via cfg.sam3_daemon_script."
            )
        print(
            f"[sam3-sidecar] spawning daemon: {self._daemon_python} {self._daemon_script}",
            flush=True,
        )
        self._proc = subprocess.Popen(
            [self._daemon_python, self._daemon_script],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=sys.stderr,
            env=env,
            text=True,
            bufsize=1,  # line-buffered
        )
        # Wait for the "ready" handshake (or a fatal error) before returning.
        ready_line = self._proc.stdout.readline()
        if not ready_line:
            raise RuntimeError("SAM3 daemon exited before sending ready handshake")
        try:
            ready = json.loads(ready_line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"SAM3 daemon sent non-JSON handshake: {ready_line!r}"
            ) from exc
        if not ready.get("ok") or not ready.get("ready"):
            raise RuntimeError(f"SAM3 daemon handshake failed: {ready!r}")
        print("[sam3-sidecar] daemon ready", flush=True)

    def _ensure_alive(self) -> None:
        if self._proc is None or self._proc.poll() is not None:
            raise RuntimeError("SAM3 daemon is not running (exited unexpectedly)")

    def _round_trip(self, request: dict) -> dict:
        self._ensure_alive()
        assert self._proc is not None and self._proc.stdin is not None and self._proc.stdout is not None
        line = json.dumps(request) + "\n"
        self._proc.stdin.write(line)
        self._proc.stdin.flush()
        response_line = self._proc.stdout.readline()
        if not response_line:
            self._ensure_alive()
            raise RuntimeError("SAM3 daemon closed stdout without a response")
        return json.loads(response_line)

    def __del__(self) -> None:
        try:
            if self._proc is not None and self._proc.poll() is None:
                self._proc.stdin.close() if self._proc.stdin else None
                self._proc.wait(timeout=5)
        except Exception:
            try:
                self._proc.kill()  # type: ignore[union-attr]
            except Exception:
                pass

    # ------------------------------------------------------------------
    def detect(
        self,
        image: np.ndarray,
        text: str,
        threshold: float = 0.5,
        mask_threshold: float = 0.5,
        max_keep: int = 4,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(centroids[n,2], boxes[n,4], scores[n])`` in absolute pixels."""
        if image.ndim != 3 or image.shape[-1] != 3 or image.dtype != np.uint8:
            raise ValueError(f"expected (H,W,3) uint8 RGB, got {image.shape} {image.dtype}")
        H, W, _ = image.shape
        request = {
            "image_shape": [int(H), int(W), 3],
            "image_b64": base64.b64encode(np.ascontiguousarray(image).tobytes()).decode("ascii"),
            "queries": [{
                "prompt": text,
                "threshold": float(threshold),
                "mask_threshold": float(mask_threshold),
                "max_keep": int(max_keep),
            }],
        }
        resp = self._round_trip(request)
        if not resp.get("ok"):
            raise RuntimeError(f"SAM3 daemon error: {resp.get('error')!r}")
        result = resp["results"][0]
        cents = np.asarray(result["centroids"], dtype=np.float32).reshape(-1, 2)
        boxes = np.asarray(result["boxes"], dtype=np.float32).reshape(-1, 4)
        scores = np.asarray(result["scores"], dtype=np.float32).reshape(-1)
        return cents, boxes, scores


def _box_iou(a: np.ndarray, b: np.ndarray) -> float:
    x0 = max(a[0], b[0]); y0 = max(a[1], b[1])
    x1 = min(a[2], b[2]); y1 = min(a[3], b[3])
    if x1 <= x0 or y1 <= y0:
        return 0.0
    inter = (x1 - x0) * (y1 - y0)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / union)


# Dynamic threshold ladder — when the configured threshold doesn't yield
# ``expected_count`` detections, retry at progressively lower thresholds.
# Mirrors robotwin_rag_tools/scripts/run_object_extract.py so offline
# extraction and online query detection behave consistently.
_FALLBACK_THRESHOLD_LADDER: tuple[float, ...] = (0.3, 0.2, 0.1)


# ---------------------------------------------------------------------------
# Object-aware retrieval
# ---------------------------------------------------------------------------


class RoboTwinObjectAwareRetrieval(RoboTwinDinoProprioRetrieval):
    """Adds an object-position L2 term to the parent's initial-scene filter.

    Per-step scoring (DINO + proprio history + current proprio) is unchanged —
    only the *episode subset* selected on the first act of each rollout differs.
    Object schema (which prompts to detect, expected counts, IOU rules) is
    pulled from the shared JSON config so this stays in lockstep with the
    offline ``run_object_extract.py`` pipeline.
    """

    def __init__(
        self,
        *args,
        obj_init_root: str = "/mnt/ddn/robotwin_rag_tools/cache/object_init",
        obj_init_config_path: str = _DEFAULT_CONFIG_PATH,
        w_obj: float = 1.0,
        # Categorical shape-class step penalty inside the prefilter distance:
        #   aspect = max(w, h) / max(min(w, h), eps)
        #   class = "long" if aspect > aspect_threshold else "square"
        #   d_pair = ||xy_q - xy_p||₂ + w_obj_shape * (class_q != class_p)
        # When two objects fall in the same shape class (both square or both
        # long), the term is 0 and ranking is pure centroid-xy. When they
        # differ (e.g. tall thin gripper detected as burger vs the actual
        # square burger), shape_mismatch=1 → w_obj_shape penalty pushes the
        # mismatched candidate down.
        w_obj_shape: float = 0.0,
        aspect_threshold: float = 1.5,
        # Restrict the shape term to a subset of object names. Empty tuple =
        # apply to all (legacy). e.g. ``("burger",)`` makes only the burger
        # class' shape mismatch contribute, while tray/fries stay xy-only.
        w_obj_size_objects: tuple[str, ...] = (),
        # Continuous (w, h) L1 penalty: ``w_obj_wh_l1 * (|Δw_norm| + |Δh_norm|)``
        # added to per-pair distance. Captures orientation (a horizontal roller
        # vs vertical one have the same aspect but different (w, h) pair).
        # 0 = disabled (legacy). Apply per-object via ``w_obj_wh_objects``.
        w_obj_wh_l1: float = 0.0,
        w_obj_wh_objects: tuple[str, ...] = (),
        # 4-quadrant bbox-RGB signature L2 (normalized to [0, 1]) added to
        # per-pair distance. Captures bbox-internal asymmetry that bbox (w,
        # h) discards — handle-vs-body of a roller, 180° flip on the long
        # axis, etc. ``w_obj_quad_sig_objects`` (if non-empty) restricts the
        # term to specific object names (e.g. roller, mic). 0 = disabled.
        w_obj_quad_sig: float = 0.0,
        w_obj_quad_sig_objects: tuple[str, ...] = (),
        # Restrict the xy ranking term to a subset of object names. Empty
        # tuple = average over all detected objects (legacy). e.g.
        # ``("burger", "fries")`` makes ranking ignore tray's centroid
        # (treats it as existence-only filter) and average xy distance over
        # just burger and fries — focuses pick decisions on the manipulands
        # rather than diluting through stable scene anchors.
        obj_xy_objects: tuple[str, ...] = (),
        # Optional per-object weight applied INSIDE the xy ranking average.
        # Dict of ``object_name -> weight``. Names not in the dict use 1.0.
        # Useful when one allowed-list object varies far more than another
        # and dominates the mean (e.g. burger ranges 0.05-0.5 while fries
        # stays ~0.05 — boost fries' weight to keep it from being ignored).
        obj_xy_weights: Optional[dict] = None,
        sam3_device: Optional[str] = None,
        sam3_daemon_python: str = _DEFAULT_SAM3_DAEMON_PYTHON,
        sam3_daemon_script: str = _DEFAULT_SAM3_DAEMON_SCRIPT,
        sam3_daemon_pythonpath: str = _DEFAULT_SAM3_DAEMON_PYTHONPATH,
        # ── progression-prefilter knobs ────────────────────────────────────
        # Replaces the legacy DINO+proprio+landmark_L2 prefilter scoring.
        # Per-frame retrieval (downstream of prefilter) is unchanged — only
        # the *initial-scene episode subset* selection moves to:
        #   score = w_dino_prefilter · DINO_cos
        #         + w_t5_prefilter   · (1.0 if task_q==task_p else cos(T5))
        #         + w_sam3_prefilter · exp(-d_greedy / sam3_tau)
        # Single-task pool (task_q == task_p for all eps) → T5 term degenerates
        # to a constant 1.0 and behaviour mirrors the legacy SAM3-aware path
        # except for the SAM3 formula (now area-weighted greedy).
        w_dino_prefilter: float = 5.0,
        w_t5_prefilter: float = 0.5,
        w_sam3_prefilter: float = 5.0,
        sam3_tau: float = 0.4,
        # Prefilter hard constraints (match render_progression_runtime_video.py).
        # - ``max_per_task``: cap how many eps from one pool task can survive
        #   prefilter top-N. Prevents a single task family from monopolising
        #   the candidate slate.
        # - ``exact_task_only_when_available``: when at least one pool ep has
        #   ``task_p == task_q``, drop all non-exact candidates outright.
        # - ``diversify_when_no_exact``: legacy "best-2-tasks" split. The
        #   renderer disables this (it interferes with task-family preference);
        #   set ``True`` here only for backward-compat experiments.
        max_per_task: int = 3,
        exact_task_only_when_available: bool = True,
        diversify_when_no_exact: bool = False,
        # Chunk-0 (first retrieval call) re-ranks prefilter candidates with
        # xy-emphasized SAM3: re-runs the greedy matcher with custom sub-
        # weights so object image-position dominates over size/shape match.
        # Picked ep's local_idx is hard-locked to 0 by ``_force_initial_pool_fr0_n_acts``.
        chunk0_w_sam3_xy: float = 5.0,
        chunk0_sam3_w_xy: float = 2.0,
        chunk0_sam3_w_size: float = 0.2,
        chunk0_sam3_w_shape: float = 0.2,
        # Per-chunk (k>=1) ep-level biases added on top of DINO + proprio.
        # Mirror render_progression_runtime_video.py weights["w_t5"]/["w_sam3"].
        w_t5: float = 5.0,
        w_sam3: float = 2.0,
        t5_path: str = "/mnt/ddn/dataset/RoboTwin2.0/dataset/t5_embeddings.pkl",
        query_task_name: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.obj_init_root = obj_init_root
        self.obj_init_config_path = obj_init_config_path
        self.w_obj = float(w_obj)
        self.w_obj_shape = float(max(0.0, w_obj_shape))
        self.aspect_threshold = float(max(1.0, aspect_threshold))
        self.w_obj_size_objects = tuple(w_obj_size_objects) if w_obj_size_objects else ()
        self.w_obj_wh_l1 = float(max(0.0, w_obj_wh_l1))
        self.w_obj_wh_objects = tuple(w_obj_wh_objects) if w_obj_wh_objects else ()
        self.w_obj_quad_sig = float(max(0.0, w_obj_quad_sig))
        self.w_obj_quad_sig_objects = tuple(w_obj_quad_sig_objects) if w_obj_quad_sig_objects else ()
        self.obj_xy_objects = tuple(obj_xy_objects) if obj_xy_objects else ()
        self.obj_xy_weights: dict[str, float] = (
            {str(k): float(v) for k, v in obj_xy_weights.items()} if obj_xy_weights else {}
        )
        self._sam3_device = sam3_device
        self._sam3_daemon_python = sam3_daemon_python
        self._sam3_daemon_script = sam3_daemon_script
        self._sam3_daemon_pythonpath = sam3_daemon_pythonpath

        # Query-side task spec (used for the SAM3 daemon prompts on the query
        # head image). For single-task pool this is the only spec needed; for
        # multi-task pool each pool task may have its own spec loaded below.
        self._task_spec: _TaskSpec = _load_task_spec(self.task, obj_init_config_path)
        expected_counts = {o.name: o.expected_count for o in self._task_spec.objects}

        # Per-pool-task spec lookup (multi-task pool): each pool ep loads its
        # own object_init NPZ under ``obj_init_root/{ep_task}/...`` and is
        # validated against THAT task's expected counts. Falls back to the
        # single-task path when pool_tasks == [self.task].
        self._task_spec_by_task: dict[str, _TaskSpec] = {self.task: self._task_spec}
        expected_counts_by_task: dict[str, dict[str, int]] = {self.task: expected_counts}
        for ptask in getattr(self, "pool_tasks", [self.task]):
            if ptask in self._task_spec_by_task:
                continue
            spec = _load_task_spec(ptask, obj_init_config_path)
            self._task_spec_by_task[ptask] = spec
            expected_counts_by_task[ptask] = {
                o.name: o.expected_count for o in spec.objects
            }

        self._pool_obj_scenes: list[Optional[dict]] = _load_pool_object_scenes(
            obj_init_root=self.obj_init_root,
            task=self.task,
            pool_data=self._pool_data,
            expected_counts=expected_counts,
            expected_counts_by_task=expected_counts_by_task,
        )
        self._sam3: Optional[_Sam3Detector] = None
        self._pending_query_head: Optional[np.ndarray] = None

        # Optional bridge to the server-side _RetrievalLogger so per-rollout
        # SAM3 detection artefacts land in the same per-episode log dir as
        # retrieved.mp4 / retrieval_meta.jsonl. Set via attach_logger().
        self._logger: Optional[object] = None

        # ── Progression-prefilter state ───────────────────────────────────
        self.w_dino_prefilter = float(w_dino_prefilter)
        self.w_t5_prefilter = float(w_t5_prefilter)
        self.w_sam3_prefilter = float(w_sam3_prefilter)
        self.sam3_tau = float(max(1e-6, sam3_tau))
        self.max_per_task = int(max(1, max_per_task))
        self.exact_task_only_when_available = bool(exact_task_only_when_available)
        self.diversify_when_no_exact = bool(diversify_when_no_exact)
        self.chunk0_w_sam3_xy = float(chunk0_w_sam3_xy)
        self.chunk0_sam3_w_xy = float(chunk0_sam3_w_xy)
        self.chunk0_sam3_w_size = float(chunk0_sam3_w_size)
        self.chunk0_sam3_w_shape = float(chunk0_sam3_w_shape)
        self.w_t5 = float(w_t5)
        self.w_sam3 = float(w_sam3)
        self._t5_path = t5_path
        self._query_task_name = query_task_name or self.task
        # Per-pool-episode task tag — pulled from pool_data (set by retrieval.py
        # at cache load time). Multi-task pool: each ep has its own task; the
        # progression prefilter uses this for T5 fuzzy / exact-match scoring.
        self._pool_task_per_ep = [ep.get("task", self.task) for ep in self._pool_data]
        # Per-task L2-normalized T5 vector cache (mean-pooled valid tokens).
        # Loaded for every distinct pool_task + the query task so the
        # prefilter can do task-pair cosine lookups without re-reading the pkl.
        self._t5_task_unit: dict[str, np.ndarray] = {}
        for task_name in (
            {self._query_task_name}
            | set(getattr(self, "pool_tasks", [self.task]))
            | set(self._pool_task_per_ep)
        ):
            v = _load_t5_task_unit_vec(task_name, self._t5_path)
            if v is not None:
                self._t5_task_unit[task_name] = v
        self._query_t5_unit: Optional[np.ndarray] = self._t5_task_unit.get(
            self._query_task_name
        )
        if self._query_t5_unit is None:
            print(
                f"[progression-prefilter] WARN: no T5 for query task "
                f"{self._query_task_name!r}; T5 term disabled.",
                flush=True,
            )

        n_ok = sum(1 for s in self._pool_obj_scenes if s is not None)
        if n_ok == 0:
            raise RuntimeError(
                f"[obj-prefilter] no pool episodes have usable object_init data under "
                f"{self.obj_init_root}/{self.task}/. Run robotwin_rag_tools/"
                f"scripts/run_object_extract.py first."
            )
        prompt_summary = ", ".join(
            f"{o.name!r}={list(o.prompts)!r}" if len(o.prompts) > 1 else f"{o.name!r}={o.prompts[0]!r}"
            for o in self._task_spec.objects
        )
        print(
            f"[RoboTwinObjectAwareRetrieval] task={self.task} w_obj={self.w_obj} "
            f"w_obj_shape={self.w_obj_shape} (aspect_thr={self.aspect_threshold}, "
            f"objs={self.w_obj_size_objects or 'all'}) "
            f"w_obj_wh_l1={self.w_obj_wh_l1} (objs={self.w_obj_wh_objects or 'all'}) "
            f"w_obj_quad_sig={self.w_obj_quad_sig} (objs={self.w_obj_quad_sig_objects or 'all'}) "
            f"prompts=[{prompt_summary}] pool_with_objects={n_ok}/{len(self._pool_obj_scenes)}",
            flush=True,
        )

    # ------------------------------------------------------------------ API
    def attach_logger(self, logger: object) -> None:
        """Wire in the server's ``_RetrievalLogger`` so SAM3 previews land in
        the same per-episode dir as the rest of the retrieval log. The logger
        only needs ``root_dir``, ``run_tag``, and ``episode_idx`` attributes."""
        self._logger = logger

    def get_retrieved(
        self,
        query_head_image: np.ndarray,
        query_proprio: np.ndarray,
        dataset_stats: dict,
        query_proprio_history: Optional[np.ndarray] = None,
    ) -> dict:
        # Stash for _compute_initial_filter (called by super().get_retrieved
        # only on the first act after reset).
        self._pending_query_head = query_head_image
        return super().get_retrieved(
            query_head_image=query_head_image,
            query_proprio=query_proprio,
            dataset_stats=dataset_stats,
            query_proprio_history=query_proprio_history,
        )

    def reset(self) -> None:
        super().reset()
        self._pending_query_head = None

    # ----------------------------------------------------------------- impl
    def _detect_query_scene(self, query_head_image: np.ndarray) -> Optional[dict]:
        """Run SAM3 on the head image, applying the same prompt/IOU pipeline
        as the offline extractor. Two nested fallback ladders mirror
        ``run_object_extract.py``:

        1. **Prompt ladder** — outer loop over ``spec.prompts``. First prompt
           that meets ``expected_count`` wins.
        2. **Threshold ladder** — for each prompt, retry at progressively lower
           thresholds (``spec.threshold → 0.3 → 0.2 → 0.1``).

        Detections are capped at ``max_keep`` (not ``expected_count``) so a
        scene that genuinely has more than the gate count still records every
        hit — important when a class can have 1 or 2 instances per scene
        (e.g. ``bread`` in ``place_bread_basket``).

        Returns ``None`` only when even the relaxed prompt+threshold ladder
        can't satisfy ``expected_count`` for some object class.
        """
        if self._sam3 is None:
            self._sam3 = _Sam3Detector(
                device=self._sam3_device,
                daemon_python=self._sam3_daemon_python,
                daemon_script=self._sam3_daemon_script,
                daemon_pythonpath=self._sam3_daemon_pythonpath,
            )
        H, W = query_head_image.shape[:2]
        norm = np.array([W, H], dtype=np.float32)

        per_obj_centroids: dict[str, np.ndarray] = {}
        per_obj_boxes: dict[str, np.ndarray] = {}
        per_obj_scores: dict[str, np.ndarray] = {}
        per_obj_threshold_used: dict[str, float] = {}
        per_obj_prompt_used: dict[str, str] = {}
        for spec in self._task_spec.objects:
            thresholds = [spec.threshold]
            for t in _FALLBACK_THRESHOLD_LADDER:
                if t < thresholds[-1]:
                    thresholds.append(t)

            best_cents = np.zeros((0, 2), dtype=np.float32)
            best_boxes = np.zeros((0, 4), dtype=np.float32)
            best_scores = np.zeros((0,), dtype=np.float32)
            used_thr = thresholds[0]
            used_prompt = spec.prompts[0]
            target = max(spec.expected_count, 0)

            done = False
            for prompt in spec.prompts:
                for thr in thresholds:
                    cents, boxes, scores = self._sam3.detect(
                        query_head_image, prompt,
                        threshold=thr, mask_threshold=spec.mask_threshold,
                        max_keep=spec.max_keep,
                    )
                    if spec.exclude_overlap_with is not None and per_obj_boxes.get(spec.exclude_overlap_with, np.zeros((0, 4))).size:
                        ref_boxes = per_obj_boxes[spec.exclude_overlap_with]
                        keep_idx = [
                            i for i in range(boxes.shape[0])
                            if max(_box_iou(boxes[i], rb) for rb in ref_boxes) < spec.iou_threshold
                        ]
                        cents = cents[keep_idx]
                        boxes = boxes[keep_idx]
                        scores = scores[keep_idx]
                    # y-band gate: drop detections whose box y-center falls
                    # outside [y_min*H, y_max*H]. Mirrors run_object_extract.py.
                    if spec.y_min > 0.0 or spec.y_max < 1.0:
                        y_min_px = spec.y_min * H
                        y_max_px = spec.y_max * H
                        keep = []
                        for i in range(boxes.shape[0]):
                            yc = 0.5 * (boxes[i, 1] + boxes[i, 3])
                            if y_min_px <= yc <= y_max_px:
                                keep.append(i)
                        cents = cents[keep]
                        boxes = boxes[keep]
                        scores = scores[keep]
                    # Static exclude regions (normalized [x0,y0,x1,y1] boxes).
                    # Drop detections whose box-center sits inside any.
                    if spec.exclude_xyxy_norm:
                        excl_px = [
                            (x0 * W, y0 * H, x1 * W, y1 * H)
                            for (x0, y0, x1, y1) in spec.exclude_xyxy_norm
                        ]
                        keep = []
                        for i in range(boxes.shape[0]):
                            cx = 0.5 * (boxes[i, 0] + boxes[i, 2])
                            cy = 0.5 * (boxes[i, 1] + boxes[i, 3])
                            inside = any(
                                ex0 <= cx <= ex1 and ey0 <= cy <= ey1
                                for (ex0, ey0, ex1, ey1) in excl_px
                            )
                            if not inside:
                                keep.append(i)
                        cents = cents[keep]
                        boxes = boxes[keep]
                        scores = scores[keep]
                    cents = cents[: spec.max_keep]
                    boxes = boxes[: spec.max_keep]
                    scores = scores[: spec.max_keep]
                    if cents.shape[0] > best_cents.shape[0]:
                        best_cents = cents
                        best_boxes = boxes
                        best_scores = scores
                        used_thr = thr
                        used_prompt = prompt
                    if best_cents.shape[0] >= target:
                        done = True
                        break
                if done:
                    break

            if best_cents.shape[0] < spec.expected_count:
                print(
                    f"[obj-prefilter] SAM3 detection insufficient on query head: "
                    f"{spec.name!r} got {best_cents.shape[0]} (need {spec.expected_count}) "
                    f"even at thr={used_thr} prompt={used_prompt!r}. "
                    f"Falling back to DINO+proprio-only initial filter.",
                    flush=True,
                )
                return None
            per_obj_centroids[spec.name] = best_cents
            per_obj_boxes[spec.name] = best_boxes
            per_obj_scores[spec.name] = best_scores
            per_obj_threshold_used[spec.name] = used_thr
            per_obj_prompt_used[spec.name] = used_prompt
            if used_thr < spec.threshold or used_prompt != spec.prompts[0]:
                print(
                    f"[obj-prefilter] {spec.name!r}: succeeded with "
                    f"prompt={used_prompt!r} thr={used_thr} "
                    f"(configured prompt={spec.prompts[0]!r} thr={spec.threshold}).",
                    flush=True,
                )

        scene: dict = {}
        sizes: dict[str, np.ndarray] = {}
        wh: dict[str, np.ndarray] = {}
        quad_sigs: dict[str, np.ndarray] = {}
        for name, cents in per_obj_centroids.items():
            xy = (cents / norm).astype(np.float32)
            box = per_obj_boxes.get(name, np.zeros((0, 4), dtype=np.float32))
            n = xy.shape[0]
            if box.shape[0] >= n:
                w_px = np.maximum(box[:n, 2] - box[:n, 0], 1e-3)
                h_px = np.maximum(box[:n, 3] - box[:n, 1], 1e-3)
                long_side = np.maximum(w_px, h_px)
                short_side = np.maximum(np.minimum(w_px, h_px), 1e-3)
                sz = (long_side / short_side).astype(np.float32)
                wh_pair = np.stack([w_px / float(norm[0]), h_px / float(norm[1])], axis=1).astype(np.float32)
                qs = np.stack(
                    [_quadrant_rgb_sig(query_head_image, box[i]) for i in range(n)],
                    axis=0,
                ) if n > 0 else np.zeros((0, 12), dtype=np.float32)
            else:
                sz = np.zeros(n, dtype=np.float32)
                wh_pair = np.zeros((n, 2), dtype=np.float32)
                qs = np.zeros((n, 12), dtype=np.float32)
            if xy.shape[0] > 1:
                order = np.argsort(xy[:, 0])
                xy = xy[order]
                sz = sz[order]
                wh_pair = wh_pair[order]
                qs = qs[order]
            scene[name] = xy
            sizes[name] = sz
            wh[name] = wh_pair
            quad_sigs[name] = qs
        scene["_norm_sizes"] = sizes
        scene["_norm_wh"] = wh
        scene["_norm_quad_sig"] = quad_sigs
        scene["_scores"] = {n: per_obj_scores[n].tolist() for n in per_obj_centroids}
        # Stash raw pixel-coord boxes/centroids/scores for visualization later.
        scene["_pixel_boxes"] = {n: per_obj_boxes[n] for n in per_obj_boxes}
        scene["_pixel_centroids"] = {n: per_obj_centroids[n] for n in per_obj_centroids}
        return scene

    _PREVIEW_PALETTE: tuple[tuple[int, int, int], ...] = (
        (0, 255, 0),     # lime
        (255, 0, 0),     # red
        (0, 255, 255),   # cyan
        (255, 0, 255),   # magenta
        (255, 255, 0),   # yellow
        (255, 165, 0),   # orange
    )
    _PREVIEW_TOP_K: int = 3

    def _draw_scene_overlays(
        self,
        canvas_rgb: np.ndarray,
        boxes_by_name: dict,
        cents_by_name: dict,
        scores_by_name: dict,
    ) -> np.ndarray:
        """Draw bbox + centroid + score directly on an RGB canvas.

        cv2 drawing functions don't interpret the channel semantics — they
        write values into channels by index. So if we pass our color tuples
        in RGB order (e.g. (255, 0, 0) for red) and the canvas is RGB-ordered,
        the output is correctly colored without any cvtColor round-trip.
        """
        import cv2

        for i, spec in enumerate(self._task_spec.objects):
            color = self._PREVIEW_PALETTE[i % len(self._PREVIEW_PALETTE)]
            boxes = boxes_by_name.get(spec.name, np.zeros((0, 4), dtype=np.float32))
            cents = cents_by_name.get(spec.name, np.zeros((0, 2), dtype=np.float32))
            scores = scores_by_name.get(spec.name, [])
            for j in range(boxes.shape[0]):
                x0, y0, x1, y1 = (int(round(v)) for v in boxes[j].tolist())
                cv2.rectangle(canvas_rgb, (x0, y0), (x1, y1), color, 2)
                cx, cy = int(round(cents[j, 0])), int(round(cents[j, 1]))
                cv2.drawMarker(canvas_rgb, (cx, cy), color,
                               markerType=cv2.MARKER_CROSS, markerSize=8, thickness=2)
                label = f"{spec.name} {scores[j]:.2f}" if j < len(scores) else spec.name
                cv2.putText(canvas_rgb, label, (x0, max(0, y0 - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(canvas_rgb, label, (x0, max(0, y0 - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
        return canvas_rgb

    def _add_panel_title(self, panel_rgb: np.ndarray, title: str) -> np.ndarray:
        """Prepend a black bar with a title above the panel."""
        import cv2

        h, w = panel_rgb.shape[:2]
        bar = np.zeros((20, w, 3), dtype=np.uint8)
        cv2.putText(bar, title, (4, 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
        return np.concatenate([bar, panel_rgb], axis=0)

    def _load_pool_first_frame_raw(self, ep_index: int) -> Optional[np.ndarray]:
        """Read the un-resized head frame 0 from the pool episode's HDF5.

        RoboTwin's HDF5 JPEGs were encoded via cv2.imencode and round-trip
        through cv2.imdecode WITHOUT a BGR↔RGB reorder, so the decoded array
        is already in RGB order (matches what the live sim sends as
        ``obs['primary_image']``). See robotwin_rag/io_hdf5.decode_rgb_dataset
        for the canonical reference — adding a cvtColor here was a long-
        standing source of channel-swap bugs.
        """
        import cv2
        import h5py

        ep_meta = self._pool_data[ep_index]
        hdf5_path = os.path.join(
            self.dataset_root, self.task, ep_meta["split"], "data",
            f"episode{ep_meta['ep_num']}.hdf5",
        )
        try:
            with h5py.File(hdf5_path, "r") as f:
                raw_bytes = f["/observation/head_camera/rgb"][0]
        except Exception as exc:  # noqa: BLE001
            print(f"[obj-prefilter] failed to open {hdf5_path}: {exc!r}", flush=True)
            return None
        arr = np.frombuffer(raw_bytes, dtype=np.uint8)
        rgb = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return rgb

    def _save_sam3_preview(
        self,
        image: np.ndarray,
        q_scene: Optional[dict],
        keep_eps: Optional[np.ndarray] = None,
        ep_obj_pen: Optional[np.ndarray] = None,
    ) -> None:
        """Write a multi-panel PNG: ``query | top1_pool | top2_pool | top3_pool``.

        Each panel shows the head image with the offline-cached bboxes drawn.
        Lands in the same per-episode dir the server's _RetrievalLogger uses.
        Silently no-ops when no logger is attached.

        When ``q_scene is None`` (SAM3 detection failed on the live query head)
        the raw head image is still saved with an explicit "SAM3 FAIL" title so
        we can visually inspect what the detector was looking at.
        """
        if self._logger is None:
            return
        try:
            import cv2
            import pathlib as _pathlib
        except ImportError:
            return

        root_dir = getattr(self._logger, "root_dir", None)
        run_tag = getattr(self._logger, "run_tag", None)
        episode_idx = getattr(self._logger, "episode_idx", None)
        if root_dir is None or run_tag is None or episode_idx is None:
            return

        ep_dir = _pathlib.Path(root_dir) / run_tag / f"episode_{int(episode_idx):04d}"
        ep_dir.mkdir(parents=True, exist_ok=True)

        # ---- Query panel (RGB throughout — cv2 only used for drawing) ----
        # Must copy: the caller passes obs["primary_image"] straight through
        # and cv2.rectangle / drawMarker / putText mutate in place. Without
        # the copy, the very same array is later fed to the policy as the
        # current-step head image, with SAM3 boxes/labels burned into it.
        q_canvas = np.ascontiguousarray(image).copy()
        if q_scene is not None:
            self._draw_scene_overlays(
                q_canvas,
                q_scene.get("_pixel_boxes") or {},
                q_scene.get("_pixel_centroids") or {},
                q_scene.get("_scores") or {},
            )
            q_title = "query (live)"
        else:
            q_title = "query (live, SAM3 FAIL)"
        q_h, q_w = q_canvas.shape[:2]
        panels: list[np.ndarray] = [self._add_panel_title(q_canvas, q_title)]

        # ---- Top-K retrieved pool panels ---------------------------------
        if keep_eps is not None:
            for rank, ei in enumerate(keep_eps[: self._PREVIEW_TOP_K]):
                ei_int = int(ei)
                pool_img_rgb = self._load_pool_first_frame_raw(ei_int)
                pool_scene = self._pool_obj_scenes[ei_int]
                if pool_img_rgb is None or pool_scene is None:
                    continue
                p_canvas = np.ascontiguousarray(pool_img_rgb)
                self._draw_scene_overlays(
                    p_canvas,
                    pool_scene.get("_pixel_boxes") or {},
                    pool_scene.get("_pixel_centroids") or {},
                    pool_scene.get("_scores") or {},
                )
                # Resize to match query height for clean horizontal concat.
                if p_canvas.shape[0] != q_h:
                    new_w = int(round(p_canvas.shape[1] * q_h / p_canvas.shape[0]))
                    p_canvas = cv2.resize(p_canvas, (new_w, q_h))
                ep_meta = self._pool_data[ei_int]
                obj_l2 = (
                    f"{float(ep_obj_pen[ei_int]):.4f}"
                    if ep_obj_pen is not None and np.isfinite(ep_obj_pen[ei_int])
                    else "n/a"
                )
                title = f"top{rank+1}: {ep_meta['split']}/ep{ep_meta['ep_num']} obj={obj_l2}"
                panels.append(self._add_panel_title(p_canvas, title))

        canvas_full = np.concatenate(panels, axis=1)
        out_path = ep_dir / "sam3_query.png"
        # cv2.imwrite expects BGR — flip channels at the very last step rather
        # than carrying BGR-ordered arrays through the whole pipeline.
        cv2.imwrite(str(out_path), canvas_full[..., ::-1])
        print(
            f"[obj-prefilter] saved SAM3 preview ({len(panels)} panels) → {out_path}",
            flush=True,
        )

    def _scene_distance(self, q_scene: dict, p_scene: dict) -> float:
        """Weighted mean over selected landmarks of
        ``xy_L2 + w_obj_shape * (shape_class_q != shape_class_p)``.

        ``obj_xy_objects`` (if non-empty) restricts the averaging to a subset
        of object classes — others are still required to exist in both
        scenes (existence filter) but their xy doesn't dilute the ranking.
        ``obj_xy_weights`` (if non-empty) applies per-class weights inside
        the average so a positionally-stable class (e.g. fries) won't be
        drowned out by a wider-varying one (e.g. burger).
        """
        distances: list[float] = []
        weights: list[float] = []
        q_sizes_all = q_scene.get("_norm_sizes", {})
        p_sizes_all = p_scene.get("_norm_sizes", {})
        q_wh_all = q_scene.get("_norm_wh", {})
        p_wh_all = p_scene.get("_norm_wh", {})
        for spec in self._task_spec.objects:
            q_xy = q_scene.get(spec.name)
            p_xy = p_scene.get(spec.name)
            if q_xy is None or p_xy is None:
                return float("inf")
            n = min(q_xy.shape[0], p_xy.shape[0])
            if n == 0:
                return float("inf")
            d_xy = np.linalg.norm(q_xy[:n] - p_xy[:n], axis=1)
            d = d_xy
            # Apply shape-class step penalty only when (a) w_obj_shape > 0
            # AND (b) per-object allowlist is empty OR this object's name is
            # in it. Categorizes each box as "long" (aspect > threshold) vs
            # "square" (aspect ≤ threshold) and adds a constant penalty when
            # query and pool fall in different categories — keeps centroid xy
            # as the primary ranking signal within the same shape class.
            apply_shape = (
                self.w_obj_shape > 0
                and (not self.w_obj_size_objects or spec.name in self.w_obj_size_objects)
            )
            if apply_shape:
                q_sz = q_sizes_all.get(spec.name)
                p_sz = p_sizes_all.get(spec.name)
                # q_sz / p_sz are (n,) arrays of aspect ratios in [1, inf).
                if (
                    q_sz is not None and p_sz is not None
                    and q_sz.ndim == 1 and p_sz.ndim == 1
                    and q_sz.shape[0] >= n and p_sz.shape[0] >= n
                ):
                    q_long = q_sz[:n] > self.aspect_threshold
                    p_long = p_sz[:n] > self.aspect_threshold
                    mismatch = (q_long != p_long).astype(np.float32)
                    d = d_xy + self.w_obj_shape * mismatch
            # Continuous (w, h) L1 penalty — captures orientation that the
            # rotation-invariant aspect (max/min) discards. Useful for elongated
            # objects whose pose rotates in-plane (e.g. roller).
            apply_wh = (
                self.w_obj_wh_l1 > 0
                and (not self.w_obj_wh_objects or spec.name in self.w_obj_wh_objects)
            )
            if apply_wh:
                q_wh = q_wh_all.get(spec.name)
                p_wh = p_wh_all.get(spec.name)
                if (
                    q_wh is not None and p_wh is not None
                    and q_wh.ndim == 2 and q_wh.shape[1] == 2
                    and q_wh.shape[0] >= n and p_wh.shape[0] >= n
                ):
                    d_wh = np.abs(q_wh[:n] - p_wh[:n]).sum(axis=1).astype(np.float32)
                    d = d + self.w_obj_wh_l1 * d_wh
            # Bbox-crop 4-quadrant mean-RGB signature L2 (normalized to [0, 1]).
            # Picks up bbox-internal asymmetry — handle vs body, 180° flip —
            # that (w, h) discards. ``w_obj_quad_sig_objects`` (if non-empty)
            # restricts the term to specific classes (e.g. roller, mic).
            apply_quad_sig = (
                self.w_obj_quad_sig > 0
                and (not self.w_obj_quad_sig_objects or spec.name in self.w_obj_quad_sig_objects)
            )
            if apply_quad_sig:
                q_sig_all = q_scene.get("_norm_quad_sig", {})
                p_sig_all = p_scene.get("_norm_quad_sig", {})
                q_sig = q_sig_all.get(spec.name)
                p_sig = p_sig_all.get(spec.name)
                if (
                    q_sig is not None and p_sig is not None
                    and q_sig.ndim == 2 and q_sig.shape[1] == 12
                    and q_sig.shape[0] >= n and p_sig.shape[0] >= n
                ):
                    d_sig = (np.linalg.norm(q_sig[:n] - p_sig[:n], axis=1) / 255.0).astype(np.float32)
                    d = d + self.w_obj_quad_sig * d_sig
            # Object existence is required for all classes (early returns
            # above kick in if any class is missing). But xy ranking only
            # averages over the obj_xy_objects allowlist when set — focuses
            # the per-pair distance on chosen manipulands rather than
            # diluting through stable scene anchors like a fixed tray.
            if not self.obj_xy_objects or spec.name in self.obj_xy_objects:
                w = self.obj_xy_weights.get(spec.name, 1.0)
                distances.extend(d.tolist())
                weights.extend([w] * d.shape[0])
        if not distances:
            return float("inf")
        if not weights or all(w == 1.0 for w in weights):
            return float(np.mean(distances))
        d_arr = np.asarray(distances, dtype=np.float32)
        w_arr = np.asarray(weights, dtype=np.float32)
        denom = float(w_arr.sum())
        return float((d_arr * w_arr).sum() / denom) if denom > 0 else float("inf")

    def _compute_legacy_sam3_filter(
        self,
        q_dino_unit: np.ndarray,
        query_proprio_raw: np.ndarray,
        dataset_stats: dict,
    ) -> None:
        """Pre-progression-refactor prefilter — DINO + proprio + SAM3 landmark-L2.

            ep_score = w_dino * ep_dino_cos
                     - w_prop * ep_start_proprio_L2
                     - w_obj  * ep_landmark_L2

        Used by :meth:`_compute_initial_filter`'s exact-match short-circuit so
        the SAM3 object-position signal keeps contributing on same-task pools
        (the progression formula bypasses SAM3 by design in that regime).
        Falls back to bare DINO+proprio when SAM3 detection on the query head
        produces no usable scene; the rollout still runs.
        """
        p_min = np.asarray(dataset_stats["proprio_min"], dtype=np.float32)
        p_max = np.asarray(dataset_stats["proprio_max"], dtype=np.float32)
        q_prop_norm = (
            2.0 * ((np.asarray(query_proprio_raw, dtype=np.float32) - p_min) / (p_max - p_min + 1e-8)) - 1.0
        )
        ep0_prop_norm = (
            2.0 * ((self._ep_start_proprio_raw - p_min) / (p_max - p_min + 1e-8)) - 1.0
        ).astype(np.float32)

        ep_dino_score = self._ep_start_dino_unit @ q_dino_unit                        # (n_ep,)
        ep_prop_pen = np.linalg.norm(ep0_prop_norm - q_prop_norm[None, :], axis=1)    # (n_ep,)

        # SAM3 landmark-L2 per pool ep. `_scene_distance` returns +inf when a
        # required object class is missing in either side → those eps drop out
        # of the top-N. When SAM3 fails on the query (q_scene is None) we
        # skip the SAM3 term entirely so the rollout still runs on DINO+proprio.
        n_ep = ep_dino_score.shape[0]
        ep_obj_pen = np.full(n_ep, np.inf, dtype=np.float32)
        q_scene: Optional[dict] = None
        if self._pending_query_head is not None:
            q_scene = self._detect_query_scene(self._pending_query_head)
        if q_scene is not None:
            for ei, p_scene in enumerate(self._pool_obj_scenes):
                if p_scene is None:
                    continue
                ep_obj_pen[ei] = self._scene_distance(q_scene, p_scene)

        valid_obj = np.isfinite(ep_obj_pen)
        if q_scene is None or not valid_obj.any():
            ep_score = self.w_dino * ep_dino_score - self.w_prop * ep_prop_pen
        else:
            ep_score = (
                self.w_dino * ep_dino_score
                - self.w_prop * ep_prop_pen
                - self.w_obj * ep_obj_pen
            )

        finite_count = int(np.isfinite(ep_score).sum())
        n_keep = min(self.n_initial_filter, finite_count if finite_count > 0 else n_ep)
        n_keep = max(1, min(n_keep, n_ep))
        keep = np.argpartition(-ep_score, n_keep - 1)[:n_keep]
        keep = keep[np.argsort(-ep_score[keep])].astype(np.int32)                     # ranked

        self._allowed_episodes = keep
        ep_set = set(int(e) for e in keep.tolist())
        mask = np.fromiter(
            (int(e) in ep_set for e in self._frame_episode), dtype=bool, count=len(self._frame_episode)
        )
        self._allowed_frame_idx = np.flatnonzero(mask).astype(np.int32)

        print(
            f"[obj-prefilter] legacy SAM3 filter: w_dino={self.w_dino} "
            f"w_prop={self.w_prop} w_obj={self.w_obj}  kept {n_keep}/{n_ep} "
            f"episodes ({len(self._allowed_frame_idx)} frames). top-3:",
            flush=True,
        )
        for ei in keep[:3]:
            ep = self._pool_data[int(ei)]
            obj = float(ep_obj_pen[int(ei)])
            obj_s = f"{obj:.4f}" if np.isfinite(obj) else "n/a"
            print(
                f"  - {ep['split']}/episode{ep['ep_num']}  "
                f"dino={ep_dino_score[int(ei)]:+.3f}  "
                f"prop-L2={ep_prop_pen[int(ei)]:.3f}  "
                f"obj-L2={obj_s}",
                flush=True,
            )

        # Save SAM3 preview image so exact-match (train-task) evals also
        # produce sam3_query.png alongside retrieved.mp4 / generated.mp4.
        # Save even when SAM3 fails on the query head (q_scene is None) so we
        # can visually inspect what the detector was looking at.
        if self._pending_query_head is not None:
            self._save_sam3_preview(
                self._pending_query_head,
                q_scene,
                keep_eps=keep,
                ep_obj_pen=ep_obj_pen,
            )

    def _compute_initial_filter(
        self,
        q_dino_unit: np.ndarray,
        query_proprio_raw: np.ndarray,    # accepted for API compat; not used
        dataset_stats: dict,
    ) -> None:
        """Progression-style prefilter: DINO scene + T5 task (with exact-lang
        bonus) + area-weighted greedy SAM3 object match.

        EXACT-MATCH SHORT-CIRCUIT: when the pool contains any ep whose task
        equals the query task, the "progression" tricks (max_per_task cap,
        exact_task_only filter, xy-SAM3 chunk-0 rescore, per-chunk T5/SAM3
        biases) are bypassed and the legacy DINO+proprio+SAM3-landmark-L2
        prefilter runs (see :meth:`_compute_legacy_sam3_filter`). Rationale:
        progression weights are tuned for cross-task retrieval where the pool
        has no same-task example; layering them on a known-good single-task
        path only adds risk. The runtime sets ``self._exact_pool`` accordingly
        so per-chunk biases stay off too.

        Proprio is intentionally dropped from prefilter (per-frame scoring
        still uses it). When ``self._t5_task_vecs`` is unavailable (no T5
        pkl loaded) the T5 term is silently skipped. When SAM3 detection
        fails on the query head, the greedy term falls back to 0 for that
        pool ep — DINO still drives the ranking.

        Top-N episodes are pushed into ``self._allowed_episodes`` /
        ``self._allowed_frame_idx`` exactly as before, so the downstream
        per-frame scoring chain is unaffected.
        """
        # Detect exact-match availability. If the pool has at least one ep with
        # ``task_p == query_task``, fall back to the legacy DINO+proprio+SAM3
        # initial filter — the original working path for single-task pools.
        pool_task_per_ep = getattr(self, "_pool_task_per_ep", None)
        self._exact_pool = bool(
            pool_task_per_ep is not None
            and any(t == self._query_task_name for t in pool_task_per_ep)
        )
        if self._exact_pool:
            print(
                f"[progression-prefilter] EXACT-MATCH detected ({self._query_task_name} "
                f"in pool) → using legacy DINO+proprio+SAM3-landmark prefilter",
                flush=True,
            )
            return self._compute_legacy_sam3_filter(
                q_dino_unit, query_proprio_raw, dataset_stats
            )
        # -- DINO scene cosine (per-episode start frame) -------------------
        ep_dino_score = self._ep_start_dino_unit @ q_dino_unit                  # (n_ep,)

        # -- T5 fuzzy match (per-pool-episode task name vs query task) -----
        # When pool & query tasks match, score is 1.0; else cosine on the
        # per-task mean-pooled T5 embedding. If no T5 cache, score is 0.
        ep_t5_score = np.zeros_like(ep_dino_score, dtype=np.float32)
        q_t5_unit = self._query_t5_unit  # populated by parent class if loaded
        pool_task_per_ep = getattr(self, "_pool_task_per_ep", None)
        t5_by_task = getattr(self, "_t5_task_unit", None)
        if pool_task_per_ep is not None:
            for ei, ptask in enumerate(pool_task_per_ep):
                if ptask == self._query_task_name:
                    ep_t5_score[ei] = 1.0
                elif q_t5_unit is not None and t5_by_task is not None:
                    pv = t5_by_task.get(ptask)
                    if pv is not None:
                        ep_t5_score[ei] = float(q_t5_unit @ pv)

        # -- SAM3 greedy area-weighted match (progression formula) ---------
        q_scene: Optional[dict] = None
        if self._pending_query_head is not None:
            q_scene = self._detect_query_scene(self._pending_query_head)

        ep_sam3_score = np.zeros_like(ep_dino_score, dtype=np.float32)
        excluded_mask = np.zeros_like(ep_dino_score, dtype=bool)
        if q_scene is not None:
            q_objs = _scene_to_object_list(q_scene)
            for ei, p_scene in enumerate(self._pool_obj_scenes):
                if p_scene is None:
                    excluded_mask[ei] = True
                    continue
                p_objs = _scene_to_object_list(p_scene)
                ep_sam3_score[ei] = _greedy_area_weighted_sim(
                    q_objs, p_objs, tau=self.sam3_tau
                )

        # -- Composite score (higher = better) -----------------------------
        ep_score = (
            self.w_dino_prefilter * ep_dino_score
            + self.w_t5_prefilter * ep_t5_score
            + self.w_sam3_prefilter * ep_sam3_score
        )
        ep_score[excluded_mask] = -np.inf

        # Stash per-ep T5 / SAM3 for downstream per-chunk biases (retrieval.py
        # base class reads ``self._t5_per_ep`` / ``self._sam3_per_ep`` and adds
        # ``w_t5 * t5 + w_sam3 * sam3`` per frame). Only used when
        # ``self._exact_pool`` is False (exact-match path returned earlier).
        self._t5_per_ep = ep_t5_score.astype(np.float32, copy=True)
        self._sam3_per_ep = ep_sam3_score.astype(np.float32, copy=True)

        # Chunk-0 xy-emphasized SAM3 (per-ep). Recomputes the greedy match
        # with sub-weights (xy=2.0, size=0.2, shape=0.2 by default) so the
        # first action chunk's pool-ep choice is dominated by object image-
        # position rather than size/shape match (renderer parity).
        ep_sam3_xy = np.zeros_like(ep_dino_score, dtype=np.float32)
        if q_scene is not None:
            q_objs = _scene_to_object_list(q_scene)
            for ei, p_scene in enumerate(self._pool_obj_scenes):
                if p_scene is None:
                    continue
                p_objs = _scene_to_object_list(p_scene)
                ep_sam3_xy[ei] = _greedy_sim_weighted(
                    q_objs, p_objs,
                    w_xy=self.chunk0_sam3_w_xy,
                    w_size=self.chunk0_sam3_w_size,
                    w_shape=self.chunk0_sam3_w_shape,
                    tau=self.sam3_tau,
                )
        self._sam3_xy_per_ep = ep_sam3_xy

        # -- Optional hard filter: exact-task-only when an exact-language match
        #    exists in the pool. Drops cross-task candidates entirely so the
        #    retrieved frames stay within the query's task family.
        if (
            self.exact_task_only_when_available
            and pool_task_per_ep is not None
        ):
            pool_has_exact = any(t == self._query_task_name for t in pool_task_per_ep)
            if pool_has_exact:
                non_exact_mask = np.array(
                    [t != self._query_task_name for t in pool_task_per_ep],
                    dtype=bool,
                )
                ep_score = ep_score.copy()
                ep_score[non_exact_mask] = -np.inf

        # -- Pick top-N (with optional per-task cap) ------------------------
        finite_count = int(np.isfinite(ep_score).sum())
        n_keep = min(self.n_initial_filter, finite_count)
        if n_keep <= 0:
            # Degenerate: nothing scored. Keep first episode to avoid crash.
            keep = np.array([0], dtype=np.int32)
        else:
            # Apply max_per_task cap by walking the full sorted order and
            # taking the first N entries that respect the cap per task.
            if pool_task_per_ep is not None and self.max_per_task > 0:
                all_sorted = np.argsort(-ep_score).astype(np.int32)
                finite_sorted = all_sorted[np.isfinite(ep_score[all_sorted])]
                per_task_count: dict[str, int] = {}
                capped: list[int] = []
                for ei in finite_sorted:
                    t = pool_task_per_ep[int(ei)]
                    if per_task_count.get(t, 0) >= self.max_per_task:
                        continue
                    capped.append(int(ei))
                    per_task_count[t] = per_task_count.get(t, 0) + 1
                    if len(capped) >= n_keep:
                        break
                keep = np.array(capped[:n_keep], dtype=np.int32)
            else:
                keep = np.argpartition(-ep_score, n_keep - 1)[:n_keep]
                keep = keep[np.argsort(-ep_score[keep])].astype(np.int32)

            # Diversify: when NO exact-lang match exists in the kept set,
            # force the top-N to span the best-2 tasks (3+2 for K=5 in
            # ceil/floor split). This prevents one geometrically similar
            # task (e.g. grab_roller for adjust_bottle) from monopolising
            # the candidate pool. Disabled by default — the renderer turns
            # this off because it conflicts with task-family preference.
            if self.diversify_when_no_exact and pool_task_per_ep is not None:
                kept_tasks = [pool_task_per_ep[int(ei)] for ei in keep]
                has_exact = any(t == self._query_task_name for t in kept_tasks)
                if (not has_exact) and len(set(kept_tasks)) > 1 and n_keep > 1:
                    all_sorted = np.argsort(-ep_score).astype(np.int32)
                    finite_sorted = all_sorted[np.isfinite(ep_score[all_sorted])]

                    first_task = pool_task_per_ep[int(finite_sorted[0])]
                    second_task = None
                    for ei in finite_sorted:
                        if pool_task_per_ep[int(ei)] != first_task:
                            second_task = pool_task_per_ep[int(ei)]
                            break
                    if second_task is not None:
                        n_first = (n_keep + 1) // 2
                        n_second = n_keep - n_first
                        first_eps = [int(ei) for ei in finite_sorted
                                     if pool_task_per_ep[int(ei)] == first_task][:n_first]
                        second_eps = [int(ei) for ei in finite_sorted
                                      if pool_task_per_ep[int(ei)] == second_task][:n_second]
                        merged = first_eps + second_eps
                        if len(merged) < n_keep:
                            seen = set(merged)
                            for ei in finite_sorted:
                                if int(ei) in seen:
                                    continue
                                merged.append(int(ei))
                                if len(merged) >= n_keep:
                                    break
                        merged.sort(key=lambda x: -ep_score[x])
                        keep = np.array(merged[:n_keep], dtype=np.int32)
                        print(
                            f"[progression-prefilter] diversified (no exact): "
                            f"{n_first}×{first_task} + {n_second}×{second_task}",
                            flush=True,
                        )

        self._allowed_episodes = keep
        ep_set = set(int(e) for e in keep.tolist())
        mask = np.fromiter(
            (int(e) in ep_set for e in self._frame_episode), dtype=bool, count=len(self._frame_episode)
        )
        self._allowed_frame_idx = np.flatnonzero(mask).astype(np.int32)

        # -- Logging --------------------------------------------------------
        if q_scene is not None:
            scene_summary = ", ".join(
                f"{name}={q_scene[name].tolist()}" for name in (s.name for s in self._task_spec.objects)
            )
            print(f"[progression-prefilter] query scene: {scene_summary}  scores={q_scene.get('_scores')}", flush=True)
        else:
            print("[progression-prefilter] query scene: SAM3 FAIL (no detections on live query head)", flush=True)
        # Save sam3_query.png even when SAM3 fails so we can inspect what the
        # detector was looking at.
        if self._pending_query_head is not None:
            self._save_sam3_preview(
                self._pending_query_head,
                q_scene,
                keep_eps=keep,
                ep_obj_pen=None,
            )
        print(
            f"[progression-prefilter] weights  dino={self.w_dino_prefilter}  "
            f"t5={self.w_t5_prefilter}  sam3={self.w_sam3_prefilter}  "
            f"τ={self.sam3_tau}",
            flush=True,
        )
        print(
            f"[progression-prefilter] kept {len(keep)}/{len(self._pool_data)} "
            f"episodes ({len(self._allowed_frame_idx)} frames). top-3:",
            flush=True,
        )
        for ei in keep[:3]:
            ep = self._pool_data[int(ei)]
            ptask = pool_task_per_ep[int(ei)] if pool_task_per_ep is not None else "?"
            print(
                f"  - task={ptask}  {ep['split']}/episode{ep['ep_num']}  "
                f"dino={ep_dino_score[int(ei)]:+.3f}  "
                f"t5={ep_t5_score[int(ei)]:+.3f}  "
                f"sam3={ep_sam3_score[int(ei)]:.3f}  "
                f"total={ep_score[int(ei)]:+.3f}",
                flush=True,
            )
