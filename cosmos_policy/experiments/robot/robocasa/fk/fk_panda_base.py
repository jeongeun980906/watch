"""Minimal Panda FK — qpos(7) → EEF pose in arm-base frame.

Uses the stock robosuite Panda XML (`base → link0 → ... → right_hand`).
`base` body sits at the origin, so world-frame xpos/xmat of `right_hand`
after `mj_forward` **is** the base-frame pose directly.
"""

from pathlib import Path

import mujoco
import numpy as np
import robosuite
from scipy.spatial.transform import Rotation as R

_DEFAULT_XML = Path(robosuite.__file__).parent / "models/assets/robots/panda/robot.xml"


class PandaFK:
    def __init__(self, xml_path: Path = _DEFAULT_XML, eef_body: str = "right_hand"):
        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.data = mujoco.MjData(self.model)

        self.eef_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, eef_body
        )
        if self.eef_body_id < 0:
            raise ValueError(f"body '{eef_body}' not in {xml_path}")

        self.arm_qpos_addr = np.array([
            self.model.jnt_qposadr[
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"joint{i}")
            ]
            for i in range(1, 8)
        ])

    def fk(self, qpos):
        qpos = np.asarray(qpos, dtype=np.float64).reshape(-1)
        if qpos.size != 7:
            raise ValueError(f"expected qpos of length 7, got {qpos.size}")
        self.data.qpos[self.arm_qpos_addr] = qpos
        mujoco.mj_forward(self.model, self.data)
        pos = self.data.xpos[self.eef_body_id].copy()
        mat = self.data.xmat[self.eef_body_id].reshape(3, 3).copy()
        quat_xyzw = R.from_matrix(mat).as_quat()
        return pos, quat_xyzw

    def fk_batch(self, qpos_arr):
        qpos_arr = np.asarray(qpos_arr, dtype=np.float64)
        assert qpos_arr.ndim == 2 and qpos_arr.shape[1] == 7
        T = qpos_arr.shape[0]
        pos = np.zeros((T, 3))
        quat = np.zeros((T, 4))
        for t in range(T):
            pos[t], quat[t] = self.fk(qpos_arr[t])
        return pos, quat


if __name__ == "__main__":
    fk = PandaFK()
    p, q = fk.fk(np.zeros(7))
    print(f"qpos=0 → pos={p.round(4)} quat_xyzw={q.round(4)}")
    p, q = fk.fk(np.array([0, -0.3, 0, -1.57, 0, 1.57, 0.78]))
    print(f"home    → pos={p.round(4)} quat_xyzw={q.round(4)}")
