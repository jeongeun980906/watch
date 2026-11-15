"""Sprout ICL server for the retargeted-demo arms (``..._sprout_icl_c32_r29*``, docs/real-robot-icl-r29.md).

``sprout_icl_server.py`` with the r29 settings on; that file keeps serving the earlier arms (``..._sprout_icl_c32``,
``_base``, ``_mix30*``) unchanged. ``sprout_serve.sh`` routes an EXP containing ``_r29`` here.

  * demo pools: ``human_ghost_r29/`` and ``human_context_r29/`` (retargeted state29 / state29abs)
  * retrieval: the EEF path + gripper x12 + base command x12 + EEF position x64, band +8..+96 demo frames per chunk,
    the demo held while the robot is still (truth for the tuning: robot-robot DTW to each demo's guide episode)
  * ``_r29abs`` / ``_r29rel``: the model's state is the start-frame torso pose integrated from the client's per-frame
    base commands (sprout_real/policy_r29.py sends them); the predicted poses go out as base velocity commands (a
    constant velocity to the chunk's end pose; none below SPROUT_ABS_DEAD_M / _DEG)

    EXP=cosmos_predict2p5_2b_sprout_icl_c32_r29rel_cfaug ./sprout_serve.sh <checkpoint.pt>
"""
import sys

from cosmos_policy.scripts import sprout_icl_server as I

I.DEMO29 = I.MATCH29 = True
I.IDLE_HOLD = True
R29_BAND = 96


def main() -> None:
    if not any(a.startswith("--mono-band") for a in sys.argv[1:]):
        sys.argv += ["--mono-band", str(R29_BAND)]
    exp = next((sys.argv[i + 1] for i, a in enumerate(sys.argv[:-1]) if a == "--experiment"), "")
    if exp and "_r29" not in exp:
        raise SystemExit(f"{exp}: not an r29 arm; serve it with sprout_icl_server.py")
    I.main()


if __name__ == "__main__":
    main()
