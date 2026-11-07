"""Minimal zarr-v3 reader for EgoVerse episodes — no `zarr` dependency.

The cosmos training env is Python 3.10; zarr>=3 (needed for these v3 stores)
requires 3.11. The stores use simple, stable layouts, so we read the chunk
files directly with battle-tested ``numcodecs`` codecs (Zstd + VLenBytes),
verified byte-identical to the zarr reader.

Image dirs (``images.front_1`` / ``images.sideleft`` / ``images.sideright``)
appear in two on-disk formats (both ``variable_length_bytes``, inner codecs
vlen-bytes -> zstd):
  * simple   — 1 frame per chunk, file ``c/<i>``
  * sharded  — ``sharding_indexed`` codec, all frames packed into one shard
               file ``c/<outer>`` with a trailing index of (offset, nbytes)
               uint64 pairs (+ optional crc32c).
``action57/action`` / ``action20/action`` / ``state57/state`` are uniformly
non-sharded float64 (codecs bytes -> zstd), row-chunked; zarr omits all-fill
(all-zero) chunks. Multi-cam + state reading added for the GR1 multimodal
(image+state+action) retrieval loader.
"""
from __future__ import annotations

import json
import os

import numpy as np
from numcodecs import VLenBytes, Zstd

_ZSTD = Zstd()
_VLB = VLenBytes()
_EMPTY = (1 << 64) - 1   # sharding index sentinel for a missing inner chunk

FRONT = "images.front_1"
SIDELEFT = "images.sideleft"
SIDERIGHT = "images.sideright"
CENTERTOP = "images.center_top"   # 4th grid view (2026-07-16): near-horizontal cam, upper-cabinet/scene context


def _meta(array_dir: str) -> dict:
    with open(os.path.join(array_dir, "zarr.json")) as fp:
        return json.load(fp)


class _CamMeta:
    """Per-camera image-array metadata (format + shard layout)."""

    def __init__(self, img_dir: str):
        self.img_dir = img_dir
        m = _meta(img_dir)
        self.n_frames = int(m["shape"][0])
        self.ocs = int(m["chunk_grid"]["configuration"]["chunk_shape"][0])
        c0 = m["codecs"][0]
        self.sharded = c0["name"] == "sharding_indexed"
        self.inner = 1
        self.idx_crc = False
        if self.sharded:
            cfg = c0["configuration"]
            self.inner = int(cfg["chunk_shape"][0])
            self.idx_crc = any(c["name"] == "crc32c" for c in cfg.get("index_codecs", []))
            assert cfg.get("index_location", "end") == "end", "only index_location=end supported"
        self.shard_idx: dict[int, np.ndarray] = {}


class EgoZarrEpisode:
    """Direct-chunk reader for one EgoVerse episode directory (multi-cam + state)."""

    def __init__(self, root: str):
        self.root = root
        self._cams: dict[str, _CamMeta] = {}
        self._get_cam(FRONT)                       # default cam; also defines n_frames
        self._action_cache: dict[str, np.ndarray] = {}
        self._state_cache: dict[str, np.ndarray] = {}

    # ---- frames (per-camera) ----
    def _get_cam(self, cam: str) -> _CamMeta:
        cm = self._cams.get(cam)
        if cm is None:
            cm = _CamMeta(os.path.join(self.root, cam))
            self._cams[cam] = cm
        return cm

    @property
    def n_frames(self) -> int:
        return self._cams[FRONT].n_frames

    def _shard_index(self, cm: _CamMeta, shard_path: str, outer_idx: int, n_inner: int) -> np.ndarray:
        ent = cm.shard_idx.get(outer_idx)
        if ent is None:
            idx_bytes = n_inner * 16 + (4 if cm.idx_crc else 0)
            with open(shard_path, "rb") as fp:
                fp.seek(-idx_bytes, os.SEEK_END)
                raw = fp.read(idx_bytes)
            if cm.idx_crc:
                raw = raw[:-4]
            ent = np.frombuffer(raw, dtype="<u8").reshape(n_inner, 2)
            cm.shard_idx[outer_idx] = ent
        return ent

    def frame(self, i: int, cam: str = FRONT) -> bytes:
        """Raw JPEG bytes for frame i of camera `cam` (b'' if fill/empty)."""
        cm = self._get_cam(cam)
        if not cm.sharded:
            path = os.path.join(cm.img_dir, "c", str(int(i)))
            if not os.path.exists(path):
                return b""
            with open(path, "rb") as fp:
                arr = _VLB.decode(_ZSTD.decode(fp.read()))
            return bytes(arr[0])
        ocs = cm.ocs
        outer, inner = int(i) // ocs, int(i) % ocs
        n_inner = ocs // cm.inner
        shard_path = os.path.join(cm.img_dir, "c", str(outer))
        if not os.path.exists(shard_path):
            return b""
        ent = self._shard_index(cm, shard_path, outer, n_inner)
        off, nb = int(ent[inner, 0]), int(ent[inner, 1])
        if off == _EMPTY or nb == 0:
            return b""
        with open(shard_path, "rb") as fp:
            fp.seek(off)
            blob = fp.read(nb)
        arr = _VLB.decode(_ZSTD.decode(blob))
        return bytes(arr[0])

    # ---- 2-D float arrays: action{57,20}/action and state57/state ----
    def _read_2d(self, group: str, sub: str, cache: dict) -> np.ndarray:
        key = f"{group}/{sub}"
        if key in cache:
            return cache[key]
        adir = os.path.join(self.root, group, sub)
        meta = _meta(adir)
        assert meta["codecs"][0]["name"] != "sharding_indexed", f"sharded {sub} not supported"
        rows, dim = int(meta["shape"][0]), int(meta["shape"][1])
        crows = int(meta["chunk_grid"]["configuration"]["chunk_shape"][0])
        fill = float(meta.get("fill_value", 0.0) or 0.0)
        n_chunks = (rows + crows - 1) // crows
        parts = []
        for ci in range(n_chunks):
            cpath = os.path.join(adir, "c", str(ci), "0")
            if os.path.exists(cpath):
                with open(cpath, "rb") as fp:
                    dec = _ZSTD.decode(fp.read())
                parts.append(np.frombuffer(dec, dtype="<f8").reshape(crows, dim))
            else:
                parts.append(np.full((crows, dim), fill, dtype="<f8"))   # omitted all-fill chunk
        out = np.concatenate(parts, axis=0)[:rows].copy()
        cache[key] = out
        return out

    # ---- actions ----
    def has_action(self, group: str) -> bool:
        return os.path.exists(os.path.join(self.root, group, "action", "zarr.json"))

    def action(self, group: str) -> np.ndarray:
        """Full (M, D) float64 action array for 'action57'/'action20'."""
        return self._read_2d(group, "action", self._action_cache)

    def action_rows(self, group: str) -> int:
        return int(_meta(os.path.join(self.root, group, "action"))["shape"][0])

    # ---- state ----
    def has_state(self, group: str) -> bool:
        return os.path.exists(os.path.join(self.root, group, "state", "zarr.json"))

    def state(self, group: str) -> np.ndarray:
        """Full (N, D) float64 absolute state array for 'state57'/'state20'."""
        return self._read_2d(group, "state", self._state_cache)

    def state_rows(self, group: str) -> int:
        return int(_meta(os.path.join(self.root, group, "state"))["shape"][0])
