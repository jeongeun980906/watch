#!/bin/bash
# Tabletop-Sim simulator client for tabletop_sim_eval.sh, from scratch. Idempotent: finished steps are skipped.
#
#   bash scripts/setup_tabletop_sim.sh <dir>        # e.g. $ROOT  -> <dir>/Tabletop-Sim-backup
#   export SIM_PY=<dir>/Tabletop-Sim-backup/.venv/bin/python
#
# The client runs in its own venv: Tabletop-Sim pins numpy<2 / mujoco 3.2.0, the policy server (training venv)
# is on numpy 2.x; the two talk over HTTP. The fork Tabletop-Sim-backup renders the `front` camera the training
# data has (the original repository does not), so it is pinned to that commit.
# Needs: git, uv, network (GitHub + PyPI), EGL on the node (headless rendering). On a shared filesystem, run it on
# ONE node; every node then uses the same SIM_PY.
set -euo pipefail

DIR=${1:?usage: setup_tabletop_sim.sh <dir>}
URL=https://github.com/jeongeun980906/Tabletop-Sim-backup.git
COMMIT=606319a                       # "backup: local edits for cosmos-policy experiments" (front camera)
SIM=$DIR/Tabletop-Sim-backup
mkdir -p "$DIR"

if [ ! -d "$SIM/.git" ]; then
  git clone "$URL" "$SIM"
fi
git -C "$SIM" fetch -q origin
git -C "$SIM" checkout -q "$COMMIT"
# object meshes (~2 GB)
git -C "$SIM" submodule update --init --recursive

# Every uv command runs INSIDE the Tabletop-Sim checkout: started from the WATCH repo, uv would read its pyproject
# ([tool.uv] override-dependencies = numpy>=2) and silently keep numpy 2.
cd "$SIM"
if [ ! -x .venv/bin/python ]; then
  uv venv .venv --python 3.10
fi
uv pip install --python .venv/bin/python -r requirements.txt     # includes `-e .`
uv pip install --python .venv/bin/python 'numpy==1.26.4'          # numpy < 2 (the version the evals ran with)

echo "[setup] check (EGL rendering, 4 cameras):"
MUJOCO_GL=egl .venv/bin/python - <<'PY'
import numpy, mujoco, tabletop
e = tabletop.env("aloha_lift_box", "ee_6d_pos")
cams = sorted(e.reset().observation["images"])
print(f"numpy {numpy.__version__}  mujoco {mujoco.__version__}  cameras {cams}")
assert numpy.__version__.startswith("1."), "numpy must be < 2"
assert cams == ["back", "front", "wrist_left", "wrist_right"], cams
print("Tabletop-Sim OK")
PY
echo "export SIM_PY=$SIM/.venv/bin/python"
