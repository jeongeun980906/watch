# HaWoR post-trains: GR1 + Tabletop eval on 4 nodes (after training)

For the HaWoR post-trains that were launched from [`egoverse_hawor_rgb.md`](egoverse_hawor_rgb.md) §6 (`hawor_full`)
or [`egoverse_hawor_rgb_md.md`](egoverse_hawor_rgb_md.md) §5 (`hawor_full_md`) **without** `EVAL=1`. This replaces
their eval sections (one-node GR1 eval, "send the pv0 checkpoint"): each finished post-train is evaluated on 4 nodes
with `EVAL_ONLY=1` (no training; `scripts/eval_after_post_4node.sh`). The training commands of those guides are
unchanged.

| post-train | eval at | eval | 4 nodes |
|---|---|---|---|
| GR1 v6 ICL | 4K | 37 L1-L3 atoms x 10 episodes, split over the 4 nodes | ~3 h |
| Tabletop pv0 | 8K | 10 tasks x 50 trials (EASY 5, OOD 4, full `aloha_box_into_pot`), seed 195, `EXEC=16`, 30 jobs | ~1 h |

The two evals are independent: run each as soon as its post-train has finished, on any 4 nodes.

## 1. Setup (once)

- GR1: [`egoverse_vid2robot_4node.md`](egoverse_vid2robot_4node.md) §1.2 (= [`gr1_eval_rebuttal_4node.md`](gr1_eval_rebuttal_4node.md) §1;
  reuse it if it is already set up).
- Tabletop-Sim: [`egoverse_vid2robot_4node.md`](egoverse_vid2robot_4node.md) §1.3 (`scripts/setup_tabletop_sim.sh`;
  reuse it if `$SIM_PY` exists).

Both must be on the shared filesystem: every node runs eval workers.

## 2. Run

On every node of the eval group (`NODE_RANK=0..3`), with the environment of the HaWoR guide (§0.3 / §0.1) and:

```bash
cd $WATCH && source .venv/bin/activate && export HF_HOME
export EGOVERSE_GR1_V6_MANIFEST=$ROOT/mimicdroid-gr1/manifest EGOVERSE_GR1_HUM_ROOT=$ROOT/mimicdroid-gr1/egoverse
export TABLETOP_RANDREF_DATA=$ROOT/tabletopsim-watch/data TABLETOP_RANDREF_MANIFEST=$ROOT/tabletopsim-watch/manifest
export SIM_PY=$ROOT/Tabletop-Sim-backup/.venv/bin/python
# + the GR1 eval variables of gr1_eval_rebuttal_4node.md §1.4 (CLIENT_PY, FW, MIMICDROID_TASKDEMOS, POOL, PLAY_POOL,
#   PAIRS_DIR; EGOVERSE_GR1_HUM_ROOT is the one above)
export M=hawor_full            # or hawor_full_md
```

The launcher still checks the training inputs of the stage, so `$MIDPT` (the mid `.pt` the post-train started from)
must be set as in the HaWoR guide.

```bash
# GR1 L1-L3 (after the GR1 post-train has finished)
EVAL_ONLY=1 MASTER_ADDR=<node 0> GR1_BASE_CKPT=$MIDPT \
  IMAGINAIRE_OUTPUT_ROOT=/shared/runs/gr1_posttrain_v6_448_state_${M}_4node \
  STAGE=gr1 MID=$M NODE_RANK=<0..3> bash scripts/train_egoverse_rebuttal_4node.sh
# Tabletop (after the pv0 post-train has finished)
EVAL_ONLY=1 MASTER_ADDR=<node 0> TABLETOP_RANDREF_CKPT=$MIDPT \
  IMAGINAIRE_OUTPUT_ROOT=/shared/runs/tabletop_pv0_${M}_4node \
  STAGE=pv0 MID=$M NODE_RANK=<0..3> bash scripts/train_egoverse_rebuttal_4node.sh
```

`IMAGINAIRE_OUTPUT_ROOT` must be the one the post-train used. Node 0 converts the final checkpoint to `.pt` (skipped
if the guide's convert step already wrote it); the other nodes wait for it.

- First line: `[eval] gr1 eval environment OK` / `[eval] pv0 eval environment OK`, then `[eval] node <r>/4 ... ckpt
  .../model_000004000.pt` (GR1) / `model_000008000.pt` (pv0).
- Logs:
  - `$IMAGINAIRE_OUTPUT_ROOT/pod_logs/<stage>_${M}_eval_node<r>.log`
  - GR1: `$IMAGINAIRE_OUTPUT_ROOT/eval/shard<r>/` (as `gr1_local_eval.sh`)
  - Tabletop: `$IMAGINAIRE_OUTPUT_ROOT/eval/<task>/ep<start>/` (as `tabletop_sim_eval.sh`)
- Node 0 ends with the merged `$IMAGINAIRE_OUTPUT_ROOT/eval/summary.txt`:
  - GR1: one line per atom, `atoms 37/37`, `TOTAL s/n`
  - Tabletop: per task `s/50`, EASY / OOD / FULL totals
- After a crash, run the same command again. To redo single GR1 atoms or Tabletop jobs, run them by hand as in
  [`gr1_eval_rebuttal_4node.md`](gr1_eval_rebuttal_4node.md) §2.2 / `tabletop_sim_eval.sh` (`TASKS`, `START`, `TRIALS`).

## 3. What to check, and what to send back

| check |
|---|
| GR1 `atoms 37/37`, no `ATOM FAILED`; `[eval] exp=` / `[eval] ckpt=` are the `${M}` ones |
| Tabletop every task `/50` (no `INCOMPLETE`), no `job FAILED`; every Tabletop `server.log` prints `latent contract: PLAIN_END+PER_VIEW(nb=0) ... state_t=8` |

**Send back:**
- `/shared/runs/gr1_posttrain_v6_448_state_${M}_4node/eval/` (`summary.txt`, `shard*/logs/`, `shard*.log`)
- `/shared/runs/tabletop_pv0_${M}_4node/eval/` (`summary.txt`, `*/ep*/results.json`, `*/ep*/config.txt`, `*_ep*.log`)
