#!/bin/bash
set -euo pipefail

###
# Batch start script for comparative experiments
# Supports three algorithms: IC, LET, and RTSS
# Usage: bash run_compare.sh [NUM_REPEATS] [NUM_EXPERIMENTS] [TYPE]
# Example: bash run_compare.sh 100 5 RTSS
###

###
# Specify number of concurrent instances.
###
INITIAL_SEED=1755016037         
NUM_REPEATS=${1:-100}          # Number of repetitions per experiment
NUM_EXPERIMENTS=${2:-5}        # Number of concurrent experiments
TYPE=${3:-our}                 # MET / LET / RTSS(our)

###
# TYPE determines the algorithm abbreviation and file suffix
###
case "$TYPE" in
    IC) ALG="IC" ; SUFFIX="_IC" ;;
    LET) ALG="LET"; SUFFIX="_LET" ;;
    RTSS|our) ALG="RTSS"; SUFFIX="_RTSS" ;;
    *) echo "Unknown TYPE: $TYPE"; exit 1 ;;
esac

PYTHON=${PYTHON:-$(command -v python3 || command -v python)}

###
# output files path
###
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUT_DIR="compare/${TIMESTAMP}${SUFFIX}"

if [[ -d "$OUT_DIR" ]]; then
    echo "Directory exists, reusing: $OUT_DIR"
else
    mkdir -p "$OUT_DIR"
fi

###
# Merged file
###
COMMON_CSV_PASSIVE="${OUT_DIR}/common_results_passive${SUFFIX}_${TIMESTAMP}.csv"
COMMON_CSV_ACTIVE="${OUT_DIR}/common_results_active${SUFFIX}_${TIMESTAMP}.csv"

echo "Running $NUM_EXPERIMENTS experiments (${TYPE}) in parallel, each with $NUM_REPEATS repeats"
echo "Output directory: $OUT_DIR"

###
# Concurrent experiments
###
pids=()

cleanup() {
    echo
    echo "Interrupt received, terminating all experiments..."
    for pid in "${pids[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid" 2>/dev/null || true
            wait "$pid" 2>/dev/null || true
        fi
    done
    
    e
    exit 0
}

trap cleanup SIGINT SIGTERM

# Calculate independent seeds for each experiment to avoid duplication
for i in $(seq 1 $NUM_EXPERIMENTS); do
    seed=$(( INITIAL_SEED + (i-1)*NUM_REPEATS ))
    tmp_csv_passive="${OUT_DIR}/tmp_${i}_passive${SUFFIX}.csv"
    tmp_csv_active="${OUT_DIR}/tmp_${i}_active${SUFFIX}.csv"

    echo "Starting experiment $i/$NUM_EXPERIMENTS, seed=$seed"
    $PYTHON main.py "$seed" "$NUM_REPEATS" \
        --common_csv_passive "$tmp_csv_passive" \
        --common_csv_active "$tmp_csv_active" \
        --suffix "$SUFFIX" \
        --alg "$ALG" &
    pids+=($!)
done


for pid in "${pids[@]}"; do wait "$pid"; done
echo "All experiments finished, starting merge..."

###
# Merge temporary result files
###
# Use the first file to write header
head -n 1 "${OUT_DIR}/tmp_1_passive${SUFFIX}.csv"     > "$COMMON_CSV_PASSIVE"
head -n 1 "${OUT_DIR}/tmp_1_active${SUFFIX}.csv"  > "$COMMON_CSV_ACTIVE"
# Append the rest of files without their headers
for i in $(seq 1 $NUM_EXPERIMENTS); do
    tail -n +2 "${OUT_DIR}/tmp_${i}_passive${SUFFIX}.csv"    >> "$COMMON_CSV_PASSIVE"
    tail -n +2 "${OUT_DIR}/tmp_${i}_active${SUFFIX}.csv" >> "$COMMON_CSV_ACTIVE"
done

rm -f "${OUT_DIR}"/tmp_*"${SUFFIX}".csv

echo "Processing final files ..."

# Split csv file and draw graph
if $PYTHON generate_comparison.py \
        --common_csv_passive "$COMMON_CSV_PASSIVE" \
        --common_csv_active "$COMMON_CSV_ACTIVE" \
        --suffix "$SUFFIX"; then
    echo "Final comparison plot (${TYPE}) generated successfully"
else
    echo "Error occurred while generating comparison plot!"
fi