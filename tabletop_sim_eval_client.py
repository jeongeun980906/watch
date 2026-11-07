#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Closed-loop Tabletop-Sim eval client — dm_control rollout against the policy server.

Runs in the **Tabletop-Sim venv** (mujoco 3.2.0 / dm_control 1.0.21 / numpy<2), which
cannot coexist with cosmos_policy's numpy 2.x — hence the HTTP split. Only stdlib +
numpy + cv2 + the ``tabletop`` package are needed here; every bit of 57-D knowledge lives
server-side (``cosmos_policy/scripts/tabletop_sim_eval_server.py``).

Per step the client sends four JPEG camera tiles and the 20-D ``ee_6d_pos`` proprio, and
gets a 20-D absolute action chunk back that goes straight into ``env.step`` — the target
is ``abs_action57``, so there is no integration or re-anchoring anywhere on this side.

Two things this script exists to get right:

1. **IK guard.** ``tabletop/aloha_ik.py`` retries an unreachable target for 5000 steps x
   11 resets — measured **3.1 s per call**, and it then returns a wildly different arm
   configuration (waist snapping to -3.14). A policy that emits one off-manifold pose
   stalls the whole eval and teleports the arm. ``--ik-max-steps`` / ``--ik-max-resets``
   tighten the solver and a failed solve HOLDS the previous joint angles instead.

2. **Instruction.** The policy is keyed on the instruction string. For
   ``aloha_lift_box`` the sim emits "Lift the box with the front facing the camera" while
   the training data says "Put ball top of the box and lift the box" — feeding the env
   string would condition on something never trained. The override below fixes that;
   ``--no-instruction-override`` disables it to measure the gap. Tasks whose instruction
   legitimately varies per episode (``aloha_box_into_pot_easy`` cycles 3 boxes x 2 pots)
   take the env string, and all 6 are in the T5 pickle.

Usage::

    MUJOCO_GL=egl python tabletop_sim_eval_client.py \
        --server http://127.0.0.1:8010 \
        --tasks aloha_lift_box aloha_dish_drainer \
        --episodes 50 --out-dir /mnt/ddn/tmp/tabletop_eval
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import time
import urllib.request

import cv2
import numpy as np

# Camera name in the converted store -> Tabletop observation key.
# Must stay in lockstep with GRID in convert_tabletop_to_egoverse57.py.
GRID = {
    "front_1": "back",
    "sideleft": "wrist_left",
    "sideright": "wrist_right",
    "center_top": "front",           # collaborator_pov; only sent with --grid4
}
TILE = 224            # 448 canvas / 2
JPEG_QUALITY = 92     # same as the converter, so the model sees the same compression

TRAIN_TASKS = [
    "aloha_dish_drainer",
    "aloha_handover_box",
    "aloha_shoes_table",
    "aloha_lift_box",
    "aloha_box_into_pot_easy",
]
OOD_TASKS = [
    "aloha_dish_drainer_new",
    "aloha_handover_box_new",
    "aloha_shoes_table_new",
    "aloha_lift_box_new",
]
# task -> the string the TRAINING data used, where the sim disagrees.
INSTRUCTION_OVERRIDE = {
    "aloha_lift_box": "Put ball top of the box and lift the box",
    "aloha_lift_box_new": "Put ball top of the box and lift the box",
}


# ---------------------------------------------------------------------------
def install_ik_guard(max_steps: int, max_resets: int) -> dict:
    """Replace ``AlohaIK.get_joint_pos`` with a bounded version that holds on failure.

    Returns a stats dict the caller can read after the rollout.
    """
    from dm_control.utils import inverse_kinematics as ik

    import tabletop.aloha_ik as A

    stats = {"calls": 0, "failures": 0, "worst_s": 0.0}

    def guarded(self, target_pos, target_quat, curr_qpos=None):
        t0 = time.time()
        physics2 = self.ik_physics.copy(share_model=True)
        self.resetter(physics2, curr_qpos)
        result = ik.qpos_from_site_pose(
            physics=physics2, site_name=A._SITE_NAME,
            target_pos=target_pos, target_quat=target_quat,
            joint_names=A._JOINTS, tol=A._TOL, max_steps=max_steps, inplace=True,
        )
        n = 0
        while (not result.success) and n < max_resets:
            self.resetter(physics2)
            result = ik.qpos_from_site_pose(
                physics=physics2, site_name=A._SITE_NAME,
                target_pos=target_pos, target_quat=target_quat,
                joint_names=A._JOINTS, tol=A._TOL, max_steps=max_steps, inplace=True,
            )
            n += 1
        stats["calls"] += 1
        stats["worst_s"] = max(stats["worst_s"], time.time() - t0)
        if not result.success:
            stats["failures"] += 1
            # HOLD: an unconverged solve returns an arbitrary configuration, which
            # teleports the arm. Repeating the current angles is a no-op command.
            if curr_qpos is not None:
                return np.asarray(curr_qpos, np.float64)[:6].copy()
        return result.qpos[:6]

    A.AlohaIK.get_joint_pos = guarded
    return stats


def encode_tiles(env, obs, grid4: bool) -> dict[str, str]:
    """Tabletop observation -> {cam: base64 jpeg} at the training tile size/quality."""
    out = {}
    for cam, key in GRID.items():
        if cam == "center_top":
            if not grid4:
                continue
            # `front` is commented out of AlohaTask.get_observation (aloha_env_base.py),
            # so render collaborator_pov directly rather than patching the env.
            img = env.physics.render(height=480, width=640, camera_id="collaborator_pov")
        else:
            img = obs["images"][key]
        img = cv2.resize(np.asarray(img), (TILE, TILE), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", img[:, :, ::-1],
                               [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
        if not ok:
            raise RuntimeError(f"jpeg encode failed for {cam}")
        out[cam] = base64.b64encode(buf.tobytes()).decode()
    return out


def post(server: str, path: str, payload: dict, timeout: float = 600.0) -> dict:
    req = urllib.request.Request(
        server.rstrip("/") + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        out = json.loads(r.read())
    if "error" in out:
        raise RuntimeError(f"server error: {out['error']}")
    return out


def save_video(frames: list[np.ndarray], path: str, fps: int = 20) -> None:
    if not frames:
        return
    h, w = frames[0].shape[:2]
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in frames:
        vw.write(np.ascontiguousarray(f[:, :, ::-1]))
    vw.release()


# ---------------------------------------------------------------------------
def run_episode(env, task: str, ep_idx: int, args, ik_stats: dict) -> dict:
    import tabletop  # noqa: F401  (env already built by the caller)

    # Two different benchmark_init behaviours, and the difference decides reproducibility:
    #   * dish_drainer / handover_box / shoes_table / lift_box (+ their _new twins) use
    #     AlohaTask.benchmark_init -> restores RECORDED object qpos from the npy. Fully
    #     deterministic, 50 stored initial states.
    #   * box_into_pot* overrides it -> picks combination_list[idx % 6] (target box x pot)
    #     and then calls initialize_episode(), which RE-RANDOMIZES object poses with
    #     np.random. Only the language/target is pinned by idx; the scene is not.
    # Seeding here makes that second class reproducible across runs.
    np.random.seed(args.seed * 100003 + ep_idx)
    env.reset()
    ts = env.task.benchmark_init(env.physics, ep_idx)

    env_instr = ts.observation["language_instruction"]
    instr = env_instr
    if not args.no_instruction_override and task in INSTRUCTION_OVERRIDE:
        instr = INSTRUCTION_OVERRIDE[task]
    if args.force_instruction:            # 명시 지정이 최우선
        instr = args.force_instruction

    post(args.server, "/reset", {})
    horizon = int(round(env.task.time_limit / 0.04))     # episode_len seconds @ 20 Hz
    max_reward = env.task.max_reward
    frames, step, n_infer = [], 0, 0
    reward = 0
    t0 = time.time()

    while step < horizon:
        obs = ts.observation
        resp = post(args.server, "/predict", {
            "images": encode_tiles(env, obs, args.grid4),
            "proprio20": np.asarray(obs["ee_6d_pos"], float).tolist(),
            "instruction": instr,
        })
        n_infer += 1
        chunk = np.asarray(resp["actions"], np.float64)
        n_exec = min(args.exec_steps or len(chunk), len(chunk), horizon - step)
        for a in chunk[:n_exec]:
            ts = env.step(a)
            step += 1
            reward = int(ts.reward or 0)
            max_reward = env.task.max_reward
            if args.save_video:
                frames.append(np.asarray(ts.observation["images"]["back"]))
            if reward >= max_reward:
                break
        if reward >= max_reward:
            break

    dt = time.time() - t0
    success = reward >= max_reward
    if args.save_video:
        vd = os.path.join(args.out_dir, task)
        os.makedirs(vd, exist_ok=True)
        save_video(frames, os.path.join(vd, f"ep{ep_idx:03d}_{'S' if success else 'F'}.mp4"))
    return {
        "task": task, "episode": ep_idx, "success": bool(success),
        "reward": reward, "max_reward": int(max_reward),
        "steps": step, "inferences": n_infer, "seconds": round(dt, 1),
        "instruction": instr,
        "instruction_env": env_instr,
        "instruction_overridden": instr != env_instr,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--server", default="http://127.0.0.1:8010")
    ap.add_argument("--tasks", nargs="*", default=TRAIN_TASKS,
                    help=f"default = the 5 training tasks; OOD set: {' '.join(OOD_TASKS)}")
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--start-episode", type=int, default=0)
    ap.add_argument("--exec-steps", type=int, default=0,
                    help="actions to execute per inference (0 = the whole 32-step chunk, "
                         "the GR1 client convention)")
    ap.add_argument("--grid4", action="store_true",
                    help="send the 4th view (collaborator_pov). Must match the server/model")
    ap.add_argument("--out-dir", default="/mnt/ddn/tmp/tabletop_eval")
    ap.add_argument("--save-video", action="store_true")
    ap.add_argument("--force-instruction", default=None,
                    help="이 문장을 env/override 대신 그대로 보낸다. hard 변형이 학습과 다른 "
                         "문장을 쓸 때(shoes_table_new: brown vs 학습 black) 언어 변수만 "
                         "떼어내 재보려고 쓴다. 기본값 None 이면 기존 동작 그대로.")
    ap.add_argument("--no-instruction-override", action="store_true",
                    help="send the sim's own instruction even where training used another "
                         "string (measures the language mismatch instead of avoiding it)")
    ap.add_argument("--ik-max-steps", type=int, default=300)
    ap.add_argument("--ik-max-resets", type=int, default=1)
    ap.add_argument("--seed", type=int, default=195,
                    help="only affects box_into_pot*, whose benchmark_init re-randomizes "
                         "object poses; the other tasks restore recorded states")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    ik_stats = install_ik_guard(args.ik_max_steps, args.ik_max_resets)
    import tabletop

    results = []
    for task in args.tasks:
        env = tabletop.env(task, "ee_6d_pos")
        for i in range(args.start_episode, args.start_episode + args.episodes):
            r = run_episode(env, task, i, args, ik_stats)
            results.append(r)
            done = [x for x in results if x["task"] == task]
            print(f"[{task}] ep{i:03d} {'SUCCESS' if r['success'] else 'fail   '} "
                  f"{r['reward']}/{r['max_reward']} steps={r['steps']:4d} "
                  f"infer={r['inferences']:3d} {r['seconds']:5.1f}s | "
                  f"running SR {sum(x['success'] for x in done)}/{len(done)}", flush=True)
            with open(os.path.join(args.out_dir, "results.json"), "w") as fp:
                json.dump({"results": results, "ik": ik_stats}, fp, indent=2)

    print("\n=== summary ===")
    for task in args.tasks:
        d = [x for x in results if x["task"] == task]
        if d:
            print(f"  {task:28s} {sum(x['success'] for x in d):3d}/{len(d):<3d} "
                  f"= {sum(x['success'] for x in d) / len(d):.2%}")
    print(f"  {'TOTAL':28s} {sum(x['success'] for x in results):3d}/{len(results):<3d} "
          f"= {sum(x['success'] for x in results) / max(1, len(results)):.2%}")
    print(f"\nIK: {ik_stats['calls']} calls, {ik_stats['failures']} failed "
          f"({ik_stats['failures'] / max(1, ik_stats['calls']):.1%}), "
          f"worst {ik_stats['worst_s']:.2f}s")
    print(f"results -> {os.path.join(args.out_dir, 'results.json')}")


if __name__ == "__main__":
    main()
