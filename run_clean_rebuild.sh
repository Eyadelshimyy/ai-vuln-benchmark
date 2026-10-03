#!/usr/bin/env bash
# VULCAN -- one-shot CLEAN REBUILD (unattended).
#
# Regenerates every repo x model under ONE identical generation config (the
# fixed generate_batch.py with an explicit max_tokens), then runs the whole
# analysis pipeline so every number in the thesis comes from the same draw:
#
#   PHASE 1  generate   (--overwrite: fresh samples, clears old repair blocks)
#   PHASE 2  benchmark  (per repo; aggregates all models present -> vuln@k table)
#   PHASE 3  diagnose   (per repo x model; labels every crash MODEL/MODEL_SYNTAX/INFRA/HARNESS)
#   PHASE 4  repair arm (per repo x model; bounded crash-only repair -> RQ3)
#
# FREE: everything runs against local Ollama -- no paid API. Expect ~1.5-2h.
#
# USAGE (run detached so it survives a disconnect):
#   chmod +x run_clean_rebuild.sh
#   nohup ./run_clean_rebuild.sh > rebuild.out 2>&1 &
#   tail -f rebuild.out            # watch the heartbeat
#
# All output is also saved under logs/rebuild_<timestamp>/ with one file per
# phase. When it finishes, paste benchmark.log and diagnose.log back in.
set -u

# always run from the repo root (where this script lives)
cd "$(dirname "$0")" || exit 1

export OPENSOURCE_BASE_URL="${OPENSOURCE_BASE_URL:-http://localhost:11434/v1}"
K="${K:-5}"
MODELS=("qwen2.5-coder:7b" "deepseek-coder:6.7b")
REPOS=(sqlite-utils tinydb invoke fabric paramiko)
CAND=calibration/mined_prompts

TS="$(date +%Y%m%d_%H%M)"
LOGDIR="logs/rebuild_${TS}"
mkdir -p "$LOGDIR"

banner() { echo; echo "############################################################"; echo "## $*"; echo "############################################################"; }
stamp()  { date +"%H:%M:%S"; }
safe()   { echo "$1" | tr ':/' '__'; }

START=$(date +%s)
banner "VULCAN CLEAN REBUILD | start $(date) | base_url=$OPENSOURCE_BASE_URL | k=$K"
echo "models: ${MODELS[*]}"
echo "repos : ${REPOS[*]}"
echo "logs  : $LOGDIR"

# ---------------------------------------------------------------------------
# PREFLIGHT -- fail fast if Ollama isn't up or a model isn't pulled.
# ---------------------------------------------------------------------------
echo "[$(stamp)] preflight: checking Ollama + models ..."
if ! python3 - "${MODELS[@]}" <<'PY'
import os, sys
from openai import OpenAI
wanted = sys.argv[1:]
try:
    c = OpenAI(api_key="not-needed", base_url=os.environ["OPENSOURCE_BASE_URL"], timeout=15)
    have = {m.id for m in c.models.list().data}
except Exception as e:
    print(f"  Ollama not reachable at {os.environ['OPENSOURCE_BASE_URL']}: {e}")
    sys.exit(1)
missing = [m for m in wanted if m not in have and m.split(':')[0] not in {h.split(':')[0] for h in have}]
if missing:
    print(f"  models not pulled: {missing}")
    print(f"  available: {sorted(have)}")
    print(f"  fix: ollama pull {missing[0]}")
    sys.exit(1)
print(f"  OK -- reachable, models present: {wanted}")
PY
then
    echo "ABORT: start Ollama (ollama serve) and pull the models, then re-run."
    exit 1
fi

# ---------------------------------------------------------------------------
# PHASE 1 -- GENERATION (uniform config, overwrite)
# ---------------------------------------------------------------------------
banner "PHASE 1/4  GENERATION  (fixed max_tokens, --overwrite)"
for m in "${MODELS[@]}"; do
  for repo in "${REPOS[@]}"; do
    log="$LOGDIR/gen_${repo}_$(safe "$m").log"
    echo "[$(stamp)] generate  $repo  $m  ->  $(basename "$log")"
    t0=$(date +%s)
    python3 generate_batch.py --candidates "$CAND/$repo/" --model "$m" --k "$K" --overwrite \
        > "$log" 2>&1
    rc=$?
    echo "[$(stamp)]   done in $(( $(date +%s) - t0 ))s  (exit $rc)"
  done
done

# ---------------------------------------------------------------------------
# PHASE 2 -- BENCHMARK (per repo; one file, the headline table)
# ---------------------------------------------------------------------------
banner "PHASE 2/4  BENCHMARK"
: > "$LOGDIR/benchmark.log"
for repo in "${REPOS[@]}"; do
  echo "========= $repo =========" | tee -a "$LOGDIR/benchmark.log"
  python3 analysis/iast/run_benchmark.py --candidates "$CAND/$repo/" 2>&1 | tee -a "$LOGDIR/benchmark.log"
done

# ---------------------------------------------------------------------------
# PHASE 3 -- CRASH DIAGNOSTIC (validity check; expect HARNESS=0)
# ---------------------------------------------------------------------------
banner "PHASE 3/4  CRASH DIAGNOSTIC"
: > "$LOGDIR/diagnose.log"
for m in "${MODELS[@]}"; do
  for repo in "${REPOS[@]}"; do
    echo "========= $repo  ($m) =========" | tee -a "$LOGDIR/diagnose.log"
    python3 diagnose_crashes.py --candidates "$CAND/$repo/" --model "$m" 2>&1 | tee -a "$LOGDIR/diagnose.log"
  done
done

# ---------------------------------------------------------------------------
# PHASE 4 -- REPAIR ARM (RQ3)
# ---------------------------------------------------------------------------
banner "PHASE 4/4  REPAIR ARM (RQ3)"
: > "$LOGDIR/repair.log"
for m in "${MODELS[@]}"; do
  for repo in "${REPOS[@]}"; do
    echo "========= $repo  ($m) =========" | tee -a "$LOGDIR/repair.log"
    python3 repair_arm.py --candidates "$CAND/$repo/" --model "$m" 2>&1 | tee -a "$LOGDIR/repair.log"
  done
done

banner "REBUILD COMPLETE | total $(( ($(date +%s) - START) / 60 )) min | $(date)"
echo "Benchmark : $LOGDIR/benchmark.log"
echo "Diagnostic: $LOGDIR/diagnose.log"
echo "Repair    : $LOGDIR/repair.log"
echo
echo ">>> Paste benchmark.log and diagnose.log back into the chat for review."
