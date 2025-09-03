
#!/bin/bash
set -euo pipefail

INITIAL_SEED=1755016037
NUM_REPEATS=${1:-100}          # Number of repetitions per experiment
NUM_EXPERIMENTS=${2:-5}        # Number of concurrent experiments
TYPE=${3:-our}                 # MET / LET / RTSS(our)


# different files name
case "$TYPE" in
    MRT) ALG="MRT" ; SUFFIX="_MRT" ;;
    LET) ALG="LET"; SUFFIX="_LET" ;;
    RTSS|our) ALG="RTSS"; SUFFIX="_RTSS" ;;
    *) echo "Unknown TYPE: $TYPE"; exit 1 ;;
esac



PYTHON=${PYTHON:-$(command -v python3 || command -v python)}

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUT_DIR="compare/${TIMESTAMP}${SUFFIX}"


if [[ -d "$OUT_DIR" ]]; then
    echo "Directory exists, reusing: $OUT_DIR"
else
    mkdir -p "$OUT_DIR"
fi

COMMON_CSV="${OUT_DIR}/common_results${SUFFIX}_${TIMESTAMP}.csv"
COMMON_CSV_adjust="${OUT_DIR}/common_results_adjust${SUFFIX}_${TIMESTAMP}.csv"

echo "Running $NUM_EXPERIMENTS experiments (${TYPE}) in parallel, each with $NUM_REPEATS repeats"
echo "Output directory: $OUT_DIR"

# ----------- Concurrent experiments  -----------
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


for i in $(seq 1 $NUM_EXPERIMENTS); do
    seed=$(( INITIAL_SEED + (i-1)*NUM_REPEATS ))
    tmp_csv="${OUT_DIR}/tmp_${i}${SUFFIX}.csv"
    tmp_csv_adjust="${OUT_DIR}/tmp_${i}_adjust${SUFFIX}.csv"

    echo "Starting experiment $i/$NUM_EXPERIMENTS, seed=$seed"
    $PYTHON main.py "$seed" "$NUM_REPEATS" \
        --common_csv "$tmp_csv" \
        --common_csv_adjust "$tmp_csv_adjust" \
        --suffix "$SUFFIX" \
        --alg "$ALG" &
    pids+=($!)
done


for pid in "${pids[@]}"; do wait "$pid"; done
echo "All experiments finished, starting merge..."

# Use the first file to write header
head -n 1 "${OUT_DIR}/tmp_1${SUFFIX}.csv"     > "$COMMON_CSV"
head -n 1 "${OUT_DIR}/tmp_1_adjust${SUFFIX}.csv"  > "$COMMON_CSV_adjust"
# Append the rest of files without their headers
for i in $(seq 1 $NUM_EXPERIMENTS); do
    tail -n +2 "${OUT_DIR}/tmp_${i}${SUFFIX}.csv"    >> "$COMMON_CSV"
    tail -n +2 "${OUT_DIR}/tmp_${i}_adjust${SUFFIX}.csv" >> "$COMMON_CSV_adjust"
done

rm -f "${OUT_DIR}"/tmp_*"${SUFFIX}".csv

echo "Processing final files ..."

if $PYTHON generate_comparison.py \
        --common_csv "$COMMON_CSV" \
        --common_csv_adjust "$COMMON_CSV_adjust" \
        --suffix "$SUFFIX"; then
    echo "Final comparison plot (${TYPE}) generated successfully"
else
    echo "Error occurred while generating comparison plot!"
fi