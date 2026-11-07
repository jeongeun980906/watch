#!/usr/bin/env bash
# GR1 one-shot (ICL) closed-loop eval on THIS box — the 8-GPU fulleval pod, shrunk to
# however many local GPUs you give it. Server (cosmos venv) + robocasa client
# (mimicdroid-robocasa venv) over HTTP, flock atom queue, one server per GPU.
#
# Protocol is the CERTIFIED v2 flag set (cosmos-gr1-fulleval2-6k-8gpu.yaml + runbook §4),
# NOT the older v1 6k yaml — v1 put --approach-demo-t0 on Plate L2/L3 and left FaucetL3
# without its floor-16 / forced bridge, and its own successor yaml calls those numbers
# "bridge-contaminated". Comparing against /mnt/ddn/tmp/gr1_v6_6k_fulleval requires v2.
#
#   CK=<model_000006000.pt> EXP=<experiment> OUT=<dir> ./gr1_local_eval.sh
#   MODE=cert  ...    # 4 seen tasks x 3 demos x 10 ep = 12 atoms  (quick certification)
#   MODE=full  ...    # 12 tasks / 37 atoms x 10 ep              (baseline-comparable)
#   GPUS="0 1" PORT0=8700 MPORT0=29600 ...
#
# ⚠️ Two arms at once (e.g. q25 vs baseline) MUST get different PORT0 *and* MPORT0 —
#    torch.distributed binds MASTER_PORT even for WORLD_SIZE=1, so a shared MPORT0 makes
#    both runs die with EADDRINUSE.
set -uo pipefail

CK=${CK:?set CK to a .pt (convert_dcp_to_pt.py output), not a DCP dir}
EXP=${EXP:-cosmos_predict2p5_2b_egoverse_gr1_ret_v6_448_state}
OUT=${OUT:?set OUT to a result dir}
MODE=${MODE:-full}
GPUS=${GPUS:-0 1}
PORT0=${PORT0:-8700}
MPORT0=${MPORT0:-29600}
EPISODES=${EPISODES:-10}
REPO=${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}   # this repo
FW=${FW:-/mnt/ddn/cosmos-framework}          # cosmos-framework checkout (GR1 sim client)
CLIENT_PY=${CLIENT_PY:-/mnt/ddn/mimicdroid-robocasa/.venv/bin/python}   # MimicDroid robocasa venv
# 모든 atom 의 클라이언트 호출에 덧붙일 플래그 (기본 빈 값 = 기존 동작 그대로).
# 용도 예: EXTRA_FLAGS=--raw-spawn 으로 PnPSinkToCabinet* 의 f3r2 기본 앵커를 끄고
# 원본 데모 스폰으로 돌려 "앵커가 L2 를 깨뜨리는가" 를 분리한다.
EXTRA_FLAGS="${EXTRA_FLAGS:-}"

case "$EXP" in
  *_plain*) echo "ERROR: $EXP is a BC/plain experiment. This runner passes --oneshot-pool3," >&2
            echo "       which silently produces garbage on a plain checkpoint. Use the BC runner." >&2
            exit 1;;
esac
[ -f "$CK" ] || { echo "ERROR: checkpoint not found: $CK" >&2; exit 1; }

LOGS=$OUT/logs; mkdir -p "$LOGS"
QF=$LOGS/queue.txt

# ---- atom queue (task + per-task client flags, runbook §4) ----------------------------
#   bridge  : Close*Door L2/L3, TurnOnFaucetL3, PnPSinkToMicrowaveTopL3   (--approach-demo-t0)
#             FaucetL3 + Microwave force it with --approach-min-steps 0
#   floor   : TurnOnFaucetL3 = 16, everything else the server default 8
#   spawn   : PnPSinkToCabinet* keep the client's f3r2 default (F+3/R+2 +-1cm) -> pass NOTHING
#             Plate* are raw spawn (the client only auto-anchors PnPSinkToCabinet*, so
#             passing nothing is already raw for them); Microwave has no anchor.
#   Plate*  : NO bridge. This is the v1-vs-v2 difference that invalidated the old numbers.
if [ "$MODE" = "cert" ]; then
  TASKS_SPEC=(
    "TurnOnFaucet|3|"
    "CloseLeftCabinetDoor|3|"
    "PnPSinkToCabinet|3|"
    "PnPSinkToRightCounterPlate|3|"
  )
elif [ "$MODE" = "l2l3" ]; then
  # L2 + L3 만 (seen 제외). Cabinet* 은 클라이언트 f3r2 기본 앵커를 그대로 쓴다 -> 플래그 없음.
  TASKS_SPEC=(
    "PnPSinkToCabinetL2|3|"
    "PnPSinkToRightCounterPlateL2|4|"
    "CloseLeftCabinetDoorL2|3|--approach-demo-t0"
    "CloseRightCabinetDoorL2|3|--approach-demo-t0"
    "PnPSinkToRightCounterPlateL3|3|"
    "CloseLeftCabinetDoorL3|3|--approach-demo-t0"
    "TurnOnFaucetL3|3|--approach-demo-t0 --approach-min-steps 0 --mono-floor 16"
    "PnPSinkToMicrowaveTopL3|3|--approach-demo-t0 --approach-min-steps 0"
  )
else
  TASKS_SPEC=(
    "TurnOnFaucet|3|"
    "CloseLeftCabinetDoor|3|"
    "PnPSinkToCabinet|3|"
    "PnPSinkToRightCounterPlate|3|"
    "PnPSinkToCabinetL2|3|"
    "PnPSinkToRightCounterPlateL2|4|"
    "PnPSinkToRightCounterPlateL3|3|"
    "CloseLeftCabinetDoorL2|3|--approach-demo-t0"
    "CloseLeftCabinetDoorL3|3|--approach-demo-t0"
    "CloseRightCabinetDoorL2|3|--approach-demo-t0"
    "TurnOnFaucetL3|3|--approach-demo-t0 --approach-min-steps 0 --mono-floor 16"
    "PnPSinkToMicrowaveTopL3|3|--approach-demo-t0 --approach-min-steps 0"
  )
fi

: > "$QF"
for spec in "${TASKS_SPEC[@]}"; do
  IFS='|' read -r task ndemo flags <<< "$spec"
  # ONLY_TASKS: space-separated task names to keep (default: all). Lets a single task be
  # re-run in isolation without hand-writing the server/client pair and its §4 flags,
  # which is exactly where a protocol mistake creeps in.
  if [ -n "${ONLY_TASKS:-}" ]; then
    case " $ONLY_TASKS " in *" $task "*) ;; *) continue;; esac
  fi
  for d in $(seq 1 "$ndemo"); do
    # ONLY_DEMOS: space-separated demo numbers to keep (default: all). ATOM FAILED 로
    # 소실된 특정 데모만 골라 다시 채울 때 쓴다. 나머지 조건은 그대로라 기존 판과
    # 합쳐서 하나의 arm 으로 읽을 수 있다.
    if [ -n "${ONLY_DEMOS:-}" ]; then
      case " $ONLY_DEMOS " in *" $d "*) ;; *) continue;; esac
    fi
    echo "$task --demo-only demo_$d --episodes $EPISODES $flags" >> "$QF"
  done
done
[ -s "$QF" ] || { echo "ERROR: 큐가 비었다 — ONLY_TASKS='${ONLY_TASKS:-}' ONLY_DEMOS='${ONLY_DEMOS:-}' 가 MODE=$MODE 와 겹치지 않는다" >&2; exit 1; }
echo "[eval] exp=$EXP"
echo "[eval] ckpt=$CK"
echo "[eval] mode=$MODE  atoms=$(wc -l < "$QF")  gpus='$GPUS'  out=$OUT"

next_atom() { ( flock 9; line=$(head -n1 "$QF" 2>/dev/null); [ -n "$line" ] && sed -i '1d' "$QF"; echo "$line" ) 9>"$QF.lock"; }
port_up()   { (exec 3<>/dev/tcp/127.0.0.1/"$1") 2>/dev/null && exec 3<&- && return 0; return 1; }

# ---- 시작 전 포트/서버 선점 검사 -------------------------------------------------------
# 이전 실행의 서버가 살아남아 PORT0 를 잡고 있으면, 새 서버는 MASTER_PORT 충돌로 조용히
# 죽고 클라이언트는 그 살아있는 OLD 서버(= 다른 체크포인트)에 붙는다. 결과는 정상적으로
# 나오지만 잰 것은 요청한 모델이 아니다 (2026-09-07 에 4K/6K 비교 4건 중 3건이 이렇게
# 오염됐다). 그래서 여기서 먼저 죽인다 — 조용한 오측정보다 즉시 실패가 낫다.
for g in $GPUS; do
  for pt in $((PORT0 + g)) $((MPORT0 + g)); do
    if (exec 3<>/dev/tcp/127.0.0.1/$pt) 2>/dev/null; then
      exec 3<&-
      echo "ERROR: 포트 $pt 가 이미 사용 중이다. 이전 실행의 서버가 남아 있으면" >&2
      echo "       그 서버(다른 체크포인트)에 붙어 조용히 잘못된 결과가 나온다." >&2
      echo "       확인:  pgrep -af eval_gr1_server" >&2
      echo "       정리 후 재실행하거나 PORT0/MPORT0 를 다른 값으로 줄 것." >&2
      exit 1
    fi
  done
done

# Server PIDs go to FILES, not a shell array. `worker "$g" &` runs in a SUBSHELL, so an
# array appended there never reaches this shell — the old `SERVER_PIDS+=(...)` left the
# parent's array empty and the EXIT trap killed nothing. Every finished run then leaked
# two servers that kept holding PORT0/MPORT0; the next run's servers died on EADDRINUSE
# and its clients silently talked to the OLD checkpoint (2026-09-07: three of four
# 4K-vs-6K comparisons were measured against the wrong model this way).
PIDDIR="$LOGS/server_pids"; rm -rf "$PIDDIR"; mkdir -p "$PIDDIR"
# Kill only the servers THIS script started. Never pkill by pattern here: the pattern
# matches this shell's own command line and takes the script down with it (exit 144).
cleanup() {
  for pf in "$PIDDIR"/*.pid; do
    [ -e "$pf" ] || continue
    p=$(cat "$pf" 2>/dev/null) || continue
    [ -n "$p" ] && [ -d "/proc/$p" ] || continue
    case "$(tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null)" in
      *eval_gr1_server*) kill -9 "$p" 2>/dev/null || true;;   # PID 재사용 방어
    esac
  done
}
trap cleanup EXIT INT TERM

worker() {
  local g=$1 port=$((PORT0 + $1)) mport=$((MPORT0 + $1))
  cd "$REPO"
  HF_HOME=${HF_HOME:-/mnt/ddn/hf_cache} HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  CUDA_VISIBLE_DEVICES=$g RANK=0 WORLD_SIZE=1 LOCAL_RANK=0 \
  MASTER_ADDR=127.0.0.1 MASTER_PORT=$mport \
  .venv/bin/python -m cosmos_policy.scripts.eval_gr1_server \
    --checkpoint "$CK" --experiment "$EXP" \
    --pred-target abs_action57 --resolution 448 \
    --pinned-advance mono --pin-lock-ep --mono-floor 8 \
    --port $port > "$LOGS/server_gpu$g.log" 2>&1 &
  local spid=$!; echo "$spid" > "$PIDDIR/gpu$g.pid"

  for _ in $(seq 1 180); do            # VAE + policy load is ~2-3 min
      port_up $port && break
      kill -0 $spid 2>/dev/null || { echo "[gpu$g] SERVER DIED"; tail -20 "$LOGS/server_gpu$g.log"; return 1; }
      sleep 10
  done
  port_up $port || { echo "[gpu$g] server never came up"; kill -9 $spid 2>/dev/null; return 1; }
  # A client launched the instant the port opens can die silently at ep0 t=0 while a
  # second server is still warming up (runbook §1.6).
  sleep 30
      # 서버가 실제로 로드한 체크포인트가 요청한 것인지 확인한다. 위 포트 가드를 우회한
      # 경우(예: 서버가 늦게 뜨는 사이 남의 서버가 끼어든 경우)의 마지막 방어선이다.
      if ! grep -q "$(basename "$CK")" "$LOGS/server_gpu$g.log" 2>/dev/null; then
          echo "[gpu$g] ERROR: 서버 로그에 $(basename "$CK") 가 없다 — 다른 체크포인트를 재고 있다" >&2
          tail -5 "$LOGS/server_gpu$g.log" >&2
          kill -9 $spid 2>/dev/null; return 1
      fi
  echo "[gpu$g] server ready $(date -u +%T)"

  while true; do
      local line; line=$(next_atom); [ -z "$line" ] && break
      local task=${line%% *} flags=${line#* }
      local demo; demo=$(echo "$flags" | grep -oE "demo_[0-9]+" | head -1)
      echo "[gpu$g] START ${task}_${demo} $(date -u +%T)"
      ( cd "$FW" && MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=$g PYOPENGL_PLATFORM=egl \
        CUDA_VISIBLE_DEVICES=$g PYTHONPATH="$FW" \
        "$CLIENT_PY" cosmos_framework/scripts/eval_gr1_client.py \
          --server "http://127.0.0.1:$port" --oneshot-pool3 \
          --task "$task" --max-steps 400 $EXTRA_FLAGS \
          --video-3cam --video-dir "$OUT/videos_$task" $flags \
      ) > "$LOGS/task_${task}_${demo}.log" 2>&1 || echo "[gpu$g] ATOM FAILED ${task}_${demo}"
      echo "[gpu$g] DONE  ${task}_${demo} $(date -u +%T)"
  done
  # 정상 경로에서도 자기 서버를 직접 내린다. EXIT trap 은 비정상 종료용 안전망일 뿐이다.
  kill -9 "$spid" 2>/dev/null || true
  rm -f "$PIDDIR/gpu$g.pid"
  echo "[gpu$g] queue empty (server $spid down)"
}

for g in $GPUS; do worker "$g" & done
wait

# ---- summary. The video writer can leak NUL bytes into a client log, after which plain
#      grep treats it as binary and silently returns nothing (runbook §1.6) -> tr -d '\0'.
{
  echo "# $EXP"; echo "# $CK"; echo "# $(date -u +%FT%TZ)"
  tot_s=0; tot_j=0
  for f in "$LOGS"/task_*.log; do
      [ -e "$f" ] || continue
      s=$(tr -d '\0' < "$f" | grep -acE ' -> SUCCESS'); j=$(tr -d '\0' < "$f" | grep -acE ' -> (SUCCESS|fail)')
      tot_s=$((tot_s+s)); tot_j=$((tot_j+j))
      printf '%-46s %s/%s %s\n' "$(basename "$f" .log | sed 's/^task_//')" "$s" "$j" "$([ "$j" -ge "$EPISODES" ] && echo ✓)"
  done
  [ "$tot_j" -gt 0 ] && printf '\nTOTAL %s/%s (%.1f%%)\n' "$tot_s" "$tot_j" "$(echo "$tot_s $tot_j" | awk '{print 100*$1/$2}')"
} | tee "$OUT/summary.txt"
echo "[eval] done -> $OUT/summary.txt"
