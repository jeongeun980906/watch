"""Sprout ICL server for the retargeted-demo arms (``..._sprout_icl_c32_r29*``, docs/real-robot-icl-r29.md).

``sprout_icl_server.py`` with the r29 settings on; that file keeps serving the earlier arms (``..._sprout_icl_c32``,
``_base``, ``_mix30*``) unchanged. ``sprout_serve.sh`` routes an EXP containing ``_r29`` here.

  * demo pools: ``human_ghost_r29/`` and ``human_context_r29/`` (retargeted state29 / state29abs)
  * retrieval: the EEF path + gripper x12 + base command x12 + EEF position x64, band +8..+96 demo frames per chunk,
    the demo held while the robot is still (truth for the tuning: robot-robot DTW to each demo's guide episode)
  * ``_r29abs`` / ``_r29rel``: the model's state is the start-frame torso pose integrated from the client's per-frame
    base commands (sprout_real/policy_icl_abs.py / policy_icl_rel.py send them); the predicted poses go out as base velocity commands (a
    constant velocity to the chunk's end pose; none below SPROUT_ABS_DEAD_M / _DEG)

    EXP=cosmos_predict2p5_2b_sprout_icl_c32_r29rel_cfaug ./sprout_serve.sh <checkpoint.pt> [flags]

Flags of this server (on top of every sprout_icl_server.py flag, e.g. --mono-floor / --mono-band / --lock / --steps):
    --grip-w 12 --base-w 12 --pos-w 64        retrieval channel weights (0 turns a channel off)
    --idle-hold / --no-idle-hold, --idle-m 0.02   hold the demo while the robot's EEF moved less than this (m)
    --abs-smooth -1                           base out: -1 = robot-speed bursts (--burst-v 0.38 m/s, --burst-deg-s 60,
                                              inside --burst-exec 24 frames: the client's --replan); 0 = constant
                                              velocity to the chunk end; N > 1 = N-frame mean; 1 = the raw poses
    --abs-dead-m 0.05 --abs-dead-deg 5        base out: no command when the chunk moves less
    --burst-start 6                           bursts start at this frame or later (the client's 0.4 s crossfade fades
                                              the base velocity too; 0 with the client's --blend-s 0)
    --base-xy-gain 1                          bursts: the predicted step x this (e.g. 1.5 when the base moves too little)
"""
import argparse
import sys

from cosmos_policy.scripts import sprout_icl_server as I

I.DEMO29 = I.MATCH29 = True
I.IDLE_HOLD = True
R29_BAND = 96


def main() -> None:
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--grip-w", type=float, default=I.GRIP_W)
    ap.add_argument("--base-w", type=float, default=I.BASE_W)
    ap.add_argument("--pos-w", type=float, default=I.POS_W)
    ap.add_argument("--idle-hold", dest="idle_hold", action="store_true", default=True)
    ap.add_argument("--no-idle-hold", dest="idle_hold", action="store_false")
    ap.add_argument("--idle-m", type=float, default=I.IDLE_M)
    ap.add_argument("--abs-smooth", type=int, default=I.ABS_SMOOTH)
    ap.add_argument("--abs-dead-m", type=float, default=I.ABS_DEAD_M)
    ap.add_argument("--burst-v", type=float, default=I.BURST_V)
    ap.add_argument("--burst-deg-s", type=float, default=I.BURST_W)
    ap.add_argument("--burst-exec", type=int, default=I.BURST_EXEC)
    ap.add_argument("--burst-start", type=int, default=I.BURST_START)
    ap.add_argument("--base-xy-gain", type=float, default=I.BASE_XY_GAIN)
    ap.add_argument("--abs-dead-deg", type=float, default=I.ABS_DEAD_DEG)
    a, rest = ap.parse_known_args(sys.argv[1:])
    I.GRIP_W, I.BASE_W, I.POS_W = a.grip_w, a.base_w, a.pos_w
    I.IDLE_HOLD, I.IDLE_M = a.idle_hold, a.idle_m
    I.ABS_SMOOTH, I.ABS_DEAD_M, I.ABS_DEAD_DEG = a.abs_smooth, a.abs_dead_m, a.abs_dead_deg
    I.BURST_V, I.BURST_W, I.BURST_EXEC = a.burst_v, a.burst_deg_s, a.burst_exec
    I.BURST_START, I.BASE_XY_GAIN = a.burst_start, a.base_xy_gain
    sys.argv = sys.argv[:1] + rest
    mode = {-1: f"bursts {I.BURST_V:g} m/s, {I.BURST_W:g} deg/s from frame {I.BURST_START} inside {I.BURST_EXEC} frames, "
                f"step x{I.BASE_XY_GAIN:g}", 0: "constant velocity",
            1: "raw poses"}.get(I.ABS_SMOOTH, f"{I.ABS_SMOOTH}-frame mean")
    print(f"[sprout-icl-r29] retrieval: grip x{I.GRIP_W:g} base x{I.BASE_W:g} position x{I.POS_W:g}, idle hold "
          f"{I.IDLE_HOLD} ({I.IDLE_M:g} m) | base out: {mode}, deadband {I.ABS_DEAD_M:g} m / {I.ABS_DEAD_DEG:g} deg",
          flush=True)
    if "-h" in rest or "--help" in rest:
        print(__doc__)
    if not any(x.startswith("--mono-band") for x in sys.argv[1:]):
        sys.argv += ["--mono-band", str(R29_BAND)]
    exp = next((sys.argv[i + 1] for i, a in enumerate(sys.argv[:-1]) if a == "--experiment"), "")
    if exp and "_r29" not in exp:
        raise SystemExit(f"{exp}: not an r29 arm; serve it with sprout_icl_server.py")
    I.main()


if __name__ == "__main__":
    main()
