# SPDX-License-Identifier: Apache-2.0
"""Tabletop-Sim (TwinVLA bimanual ALOHA) <-> egoverse 57-D convention.

Single source of truth for the representation. Imported by:
  * ``convert_tabletop_to_egoverse57.py``      (offline dataset build)
  * ``test_tabletop_action57_roundtrip.py``    (round-trip proof)
  * the eval client (decode 57 -> 20 -> ``env.step``)

Source representation (Tabletop ``ee_6d_pos``, 20-D)::

    [ L_xyz(3) | L_rot6d(6) | L_grip(1) | R_xyz(3) | R_rot6d(6) | R_grip(1) ]

``rot6d`` = first two COLUMNS of the rotation matrix (``tabletop/utils.py:quat_to_6d``),
which is exactly the egoverse convention (``action57_state57_convention.md`` §3b).
Poses are **per-arm base-relative** (``get_relative_pose(gripper, {arm}/actuation_center)``;
the ``actuation_center`` site pose equals the arm's ``base_link`` pose in ``aloha.xml``).

Target representation (egoverse 57-D, §1)::

    head9(0:9) | Leef9(9:18) | Lhand15(18:33) | Reef9(33:42) | Rhand15(42:57)

Two things change on the way in:

1. **Frame.** Both arms are re-expressed in a single ROOT frame at the midpoint of the
   two arm bases, with ``x = front`` / ``y = left`` / ``z = up`` — the same axis
   convention as the egoverse waist frame (``R0 = [heading, UP x heading, UP]``, §3e).
   The left base sits at world ``x = -0.469`` and the right at ``+0.469`` with a 180 deg
   yaw, so the transform is a constant rigid motion per arm and is exactly invertible.

2. **Gripper sign.** egoverse uses ``g = 1 -> CLOSED``; Tabletop uses ``+1 -> OPEN``
   (``ALOHA_GRIPPER_UNNORMALIZE_FN(+1) = 0.037`` = open). Hence ``g = (1 - grip) / 2``.
   Getting this backwards trains the gripper exactly inverted, so both directions live
   here and are covered by the round-trip test.

``head9`` is the real ``teleoperator_pov`` camera pose expressed in the root frame (the
camera is fixed, so it is a constant — the GR1 rigs use the same trick with their own
``HEAD9``).
"""
from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# 57-D layout
# ---------------------------------------------------------------------------
HEAD_SLICE = slice(0, 9)
EEF_SLICES = (slice(9, 18), slice(33, 42))     # Leef9 | Reef9
HAND_SLICES = (slice(18, 33), slice(42, 57))   # Lhand15 | Rhand15
DIM57 = 57
DIM20 = 20

# 20-D layout
_L20 = (slice(0, 3), slice(3, 9), 9)           # pos | rot6d | grip index
_R20 = (slice(10, 13), slice(13, 19), 19)

# ---------------------------------------------------------------------------
# Frames (numbers read straight out of tabletop/assets/aloha/aloha.xml)
# ---------------------------------------------------------------------------
_RZ180 = np.array([[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]])

# arm base pose in world: (R_base, t_base)
ARM_BASE = {
    "left": (np.eye(3), np.array([-0.469, -0.019, 0.02])),
    "right": (_RZ180.copy(), np.array([0.469, -0.019, 0.02])),
}

# root = midpoint of the two bases; axes x=front(+y_world), y=left(-x_world), z=up
T_ROOT = np.array([0.0, -0.019, 0.02])
R_RW = np.array([[0.0, 1.0, 0.0],              # = Rz(-90 deg), world -> root
                 [-1.0, 0.0, 0.0],
                 [0.0, 0.0, 1.0]])

# `teleoperator_pov` camera (pos="0 -1.354 0.685" xyaxes="1 0 0  0 0.2 .8") expressed in
# the root frame as abs9 = [pos, R[:,0], R[:,1]]. Constant: the camera never moves.
# View direction (-z) is (0.970, 0, -0.243) -> looks forward and slightly down, the same
# "behind and above" geometry as the GR1 rigs' HEAD9.
TABLETOP_HEAD9 = np.array(
    [-1.335, 0.0, 0.665, 0.0, -1.0, 0.0, 0.24254, 0.0, 0.97014]
)

# Zero-motion identity delta, for the head block of a DELTA action57 (§3d).
IDENT9 = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0])

# ---------------------------------------------------------------------------
# hand15 anchors — copied verbatim from
# cosmos-framework/cosmos_framework/scripts/gr1_online_retrieval.py
# ---------------------------------------------------------------------------
OPEN15 = np.array([0.0005, 0.0958, -0.034, 0.0285, 0.0222, -0.1263, 0.0085, 0.0217, -0.1285,
                   -0.0116, 0.0222, -0.1282, -0.0314, 0.0229, -0.1259], float)
CLOSED15 = np.array([0.0052, 0.0775, -0.0701, 0.0266, 0.044, -0.0916, 0.0085, 0.0441, -0.1016,
                     -0.0097, 0.044, -0.0937, -0.0278, 0.044, -0.0919], float)
_GRIP_AXIS = CLOSED15 - OPEN15
_GRIP_AXIS_SQ = float(_GRIP_AXIS @ _GRIP_AXIS)


# ---------------------------------------------------------------------------
# rot6d <-> matrix
# ---------------------------------------------------------------------------
def rot6d_to_mat(d6: np.ndarray) -> np.ndarray:
    """``[col0(3), col1(3)] -> (...,3,3)`` via Gram-Schmidt. Same math as
    ``state_action_convert.rot6d_to_mat`` / ``eval_gr1_server._sixd_to_R``."""
    d6 = np.asarray(d6, np.float64)
    c0, c1 = d6[..., :3], d6[..., 3:6]
    b0 = c0 / (np.linalg.norm(c0, axis=-1, keepdims=True) + 1e-8)
    c1p = c1 - (b0 * c1).sum(-1, keepdims=True) * b0
    b1 = c1p / (np.linalg.norm(c1p, axis=-1, keepdims=True) + 1e-8)
    return np.stack([b0, b1, np.cross(b0, b1)], axis=-1)


def mat_to_rot6d(R: np.ndarray) -> np.ndarray:
    """``(...,3,3) -> [col0, col1]``."""
    R = np.asarray(R, np.float64)
    return np.concatenate([R[..., :, 0], R[..., :, 1]], axis=-1)


# ---------------------------------------------------------------------------
# frame transforms
# ---------------------------------------------------------------------------
def arm_to_root(p_arm: np.ndarray, R_arm: np.ndarray, side: str):
    """Per-arm base frame -> shared root frame. Batched over leading axes."""
    Rb, tb = ARM_BASE[side]
    p_world = p_arm @ Rb.T + tb
    p_root = (p_world - T_ROOT) @ R_RW.T
    R_root = np.einsum("ij,jk,...kl->...il", R_RW, Rb, R_arm)
    return p_root, R_root


def root_to_arm(p_root: np.ndarray, R_root: np.ndarray, side: str):
    """Shared root frame -> per-arm base frame. Exact inverse of ``arm_to_root``."""
    Rb, tb = ARM_BASE[side]
    p_world = p_root @ R_RW + T_ROOT
    p_arm = (p_world - tb) @ Rb
    R_arm = np.einsum("ji,kj,...kl->...il", Rb, R_RW, R_root)
    return p_arm, R_arm


# ---------------------------------------------------------------------------
# gripper <-> hand15
# ---------------------------------------------------------------------------
def grip_to_hand15(grip: np.ndarray) -> np.ndarray:
    """Tabletop grip in ``[-1, 1]`` (+1 = OPEN) -> hand15 lerp (§3c).

    NOTE the sign flip: egoverse ``g = 1`` means CLOSED.
    """
    g = (1.0 - np.clip(np.asarray(grip, np.float64), -1.0, 1.0)) / 2.0
    return (1.0 - g)[..., None] * OPEN15 + g[..., None] * CLOSED15


def hand15_to_grip(h15: np.ndarray) -> np.ndarray:
    """hand15 -> Tabletop grip in ``[-1, 1]`` (+1 = OPEN). Inverse of ``grip_to_hand15``."""
    g = np.clip((np.asarray(h15, np.float64) - OPEN15) @ _GRIP_AXIS / _GRIP_AXIS_SQ, 0.0, 1.0)
    return 1.0 - 2.0 * g


# ---------------------------------------------------------------------------
# 20 <-> 57
# ---------------------------------------------------------------------------
def encode_20_to_57(a20: np.ndarray, head9: np.ndarray | None = None) -> np.ndarray:
    """Tabletop ``ee_6d_pos`` (T,20) -> ABSOLUTE 57-D (state57 / abs_action57).

    Works for both the action stream (``actions/ee_6d_pos``, the recorded GELLO command)
    and the proprio stream (``observations/states/ee_6d_pos``) — they share the layout.
    """
    a20 = np.atleast_2d(np.asarray(a20, np.float64))
    out = np.zeros((a20.shape[0], DIM57), np.float64)
    out[:, HEAD_SLICE] = TABLETOP_HEAD9 if head9 is None else head9
    for side, (ps, rs, gi), eef, hand in (
        ("left", _L20, EEF_SLICES[0], HAND_SLICES[0]),
        ("right", _R20, EEF_SLICES[1], HAND_SLICES[1]),
    ):
        p_root, R_root = arm_to_root(a20[:, ps], rot6d_to_mat(a20[:, rs]), side)
        out[:, eef] = np.concatenate([p_root, mat_to_rot6d(R_root)], axis=-1)
        out[:, hand] = grip_to_hand15(a20[:, gi])
    return out


def decode_57_to_20(a57: np.ndarray) -> np.ndarray:
    """ABSOLUTE 57-D -> Tabletop ``ee_6d_pos`` (T,20), ready for ``env.step``.

    Exact inverse of :func:`encode_20_to_57` (the ``head9`` block is dropped — it is a
    constant and carries no arm information).
    """
    a57 = np.atleast_2d(np.asarray(a57, np.float64))
    out = np.zeros((a57.shape[0], DIM20), np.float64)
    for side, (ps, rs, gi), eef, hand in (
        ("left", _L20, EEF_SLICES[0], HAND_SLICES[0]),
        ("right", _R20, EEF_SLICES[1], HAND_SLICES[1]),
    ):
        blk = a57[:, eef]
        p_arm, R_arm = root_to_arm(blk[:, :3], rot6d_to_mat(blk[:, 3:9]), side)
        out[:, ps] = p_arm
        out[:, rs] = mat_to_rot6d(R_arm)
        out[:, gi] = hand15_to_grip(a57[:, hand])
    return out


def delta_action57(state57: np.ndarray) -> np.ndarray:
    """ABSOLUTE state57 -> legacy DELTA action57 (§3a), length ``T-1``.

    ``head9 = IDENT9``; per arm ``[R_t^T (p_{t+1} - p_t), (R_t^T R_{t+1})[:,0:2]]``;
    ``hand15`` is the absolute lerp at ``t+1``. Only needed if a delta-target ablation
    is ever run — the primary Tabletop target is ``abs_action57``.
    """
    s = np.asarray(state57, np.float64)
    out = np.zeros((s.shape[0] - 1, DIM57), np.float64)
    out[:, HEAD_SLICE] = IDENT9
    for eef, hand in zip(EEF_SLICES, HAND_SLICES):
        blk = s[:, eef]
        p, R = blk[:, :3], rot6d_to_mat(blk[:, 3:9])
        dp = np.einsum("tji,tj->ti", R[:-1], p[1:] - p[:-1])
        dR = np.einsum("tji,tjk->tik", R[:-1], R[1:])
        out[:, eef] = np.concatenate([dp, mat_to_rot6d(dR)], axis=-1)
        out[:, hand] = s[1:, hand]
    return out
