#!/bin/bash
#SBATCH --job-name=prep-ak8-ctrl
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=36:00:00
#SBATCH --output=logs/prep-ak8-ctrl-%j.out
#SBATCH --error=logs/prep-ak8-ctrl-%j.err

# Controller: submit shard arrays in waves under MaxSubmitPU=1000.
# Already-running WJets/TTbar (and any partial ZJets) are left alone.
# This job only submits remaining work as submit slots free up.

set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-"${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"}
cd "${PROJECT_DIR}"
mkdir -p logs

export LOWERPT=${LOWERPT:-150}
export OUTPUT_ROOT=${OUTPUT_ROOT:-"${PROJECT_DIR}/data/processed/ak8-v2-pt150plus/shards"}
export NTUPLE_ROOT=${NTUPLE_ROOT:-"/HEP/export/home/hgao50/jet-anomaly-data/ak8-v2"}
MAX_SUBMIT=${MAX_SUBMIT:-1000}
WAVE=${WAVE:-150}          # tasks per wave (leave headroom)
CONCUR=${CONCUR:-32}       # %N concurrency within each array
POLL_SEC=${POLL_SEC:-120}

submit_wave() {
    local sample="$1"
    local start="$2"
    local end="$3"
    local name="prep-${sample}-${start}-${end}"
    echo "[$(date)] Submitting SAMPLE=${sample} array=${start}-${end}%${CONCUR}"
    SAMPLE="${sample}" LOWERPT="${LOWERPT}" OUTPUT_ROOT="${OUTPUT_ROOT}" \
        sbatch --job-name="${name}" \
        --array="${start}-${end}%${CONCUR}" \
        oscar_batch_prepare_ak8_shards.sh
}

queued_tasks() {
    # Count this user's pending+running jobs (array tasks expanded)
    squeue -u "${USER}" -r -h -o '%i' 2>/dev/null | wc -l | tr -d ' '
}

# Plan: (sample, n_files)
# Submit missing ranges by checking how many shard pkls already exist for that sample.
remaining_for() {
    local sample="$1"
    local n_files="$2"
    local out_dir="${OUTPUT_ROOT}/${sample}"
    mkdir -p "${out_dir}"
    local have
    have=$(find "${out_dir}" -maxdepth 1 -name '*_unscaled.pkl' -size +0c 2>/dev/null | wc -l | tr -d ' ')
    # Map completed count → next start index assuming sorted file order matches array index.
    # Safer: find first missing index by probing expected names is hard; use have as lower bound
    # and always scan for holes via a python helper.
    python3 - <<PY
import os, glob
ntuple = os.environ["NTUPLE_ROOT"]
sample = "${sample}"
out = "${out_dir}"
n_files = ${n_files}
roots = sorted(
    f for f in glob.glob(os.path.join(ntuple, sample, "*.root"))
    if os.path.getsize(f) > 0
)
label = {"qcd":"QCD","wjets":"WJet","zjets":"ZJet","ttbar":"TTbar"}[sample]
missing = []
for i, path in enumerate(roots[:n_files]):
    base = os.path.splitext(os.path.basename(path))[0]
    pkl = os.path.join(out, f"{label}__{base}_unscaled.pkl")
    if not (os.path.isfile(pkl) and os.path.getsize(pkl) > 0):
        missing.append(i)
print(" ".join(map(str, missing)))
PY
}

submit_missing_waves() {
    local sample="$1"
    local n_files="$2"
    local missing
    missing=$(remaining_for "${sample}" "${n_files}")
    if [[ -z "${missing// }" ]]; then
        echo "[$(date)] ${sample}: all ${n_files} shards present"
        return 0
    fi
    # Collapse missing indices into contiguous ranges, submit WAVE-sized chunks
    python3 - <<PY
missing = list(map(int, """${missing}""".split()))
wave = ${WAVE}
ranges = []
if missing:
    a = b = missing[0]
    for x in missing[1:]:
        if x == b + 1:
            b = x
        else:
            ranges.append((a, b))
            a = b = x
    ranges.append((a, b))
# emit up to one wave worth of indices as start-end lines (may split long ranges)
emitted = 0
for a, b in ranges:
    i = a
    while i <= b and emitted < wave:
        j = min(b, i + (wave - emitted) - 1)
        print(f"{i} {j}")
        emitted += (j - i + 1)
        i = j + 1
PY
}

echo "[$(date)] Controller start. LOWERPT=${LOWERPT} OUTPUT_ROOT=${OUTPUT_ROOT}"
echo "Max submit ~${MAX_SUBMIT}, wave=${WAVE}, concur=%${CONCUR}"

# Ordered priority: finish smaller samples, then QCD
SAMPLES=("ttbar:200" "zjets:600" "wjets:600" "qcd:1400")

while true; do
    all_done=1
    for spec in "${SAMPLES[@]}"; do
        sample="${spec%%:*}"
        n_files="${spec##*:}"
        mapfile -t waves < <(submit_missing_waves "${sample}" "${n_files}" || true)
        if [[ ${#waves[@]} -eq 0 ]]; then
            continue
        fi
        all_done=0
        for line in "${waves[@]}"; do
            [[ -z "${line}" ]] && continue
            start="${line%% *}"
            end="${line##* }"
            # wait for submit headroom
            while true; do
                q=$(queued_tasks)
                need=$((end - start + 1))
                # leave ~20 headroom for this controller + noise
                if (( q + need <= MAX_SUBMIT - 20 )); then
                    break
                fi
                echo "[$(date)] Queue busy (${q} tasks); need ${need}. Sleep ${POLL_SEC}s"
                sleep "${POLL_SEC}"
            done
            submit_wave "${sample}" "${start}" "${end}"
            sleep 5
        done
    done

    if [[ "${all_done}" -eq 1 ]]; then
        # double-check after a short wait in case of races
        sleep 30
        still=0
        for spec in "${SAMPLES[@]}"; do
            sample="${spec%%:*}"
            n_files="${spec##*:}"
            miss=$(remaining_for "${sample}" "${n_files}")
            if [[ -n "${miss// }" ]]; then
                still=1
                break
            fi
        done
        if [[ "${still}" -eq 0 ]]; then
            echo "[$(date)] All shards complete."
            for s in qcd wjets zjets ttbar; do
                echo -n "  ${s}: "
                find "${OUTPUT_ROOT}/${s}" -name '*_unscaled.pkl' -size +0c 2>/dev/null | wc -l
            done
            exit 0
        fi
    fi
    sleep "${POLL_SEC}"
done
