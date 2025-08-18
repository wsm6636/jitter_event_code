
#!/bin/bash
set -euo pipefail

INITIAL_SEED=1755016010
NUM_REPEATS=${1:-100}          # Number of repetitions per experiment
NUM_EXPERIMENTS=${2:-5}        # Number of concurrent experiments
PYTHON=${PYTHON:-$(command -v python3 || command -v python)}

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUT_DIR="compare/${TIMESTAMP}"


if [[ -d "$OUT_DIR" ]]; then
    echo "Directory exists, reusing: $OUT_DIR"
else
    mkdir -p "$OUT_DIR"
fi

COMMON_CSV="${OUT_DIR}/common_results_${TIMESTAMP}.csv"
COMMON_CSV_C1="${OUT_DIR}/common_results_c1_${TIMESTAMP}.csv"

echo "Running $NUM_EXPERIMENTS experiments in parallel, each with $NUM_REPEATS repeats"
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
    tmp_csv="${OUT_DIR}/tmp_${i}.csv"
    tmp_csv_c1="${OUT_DIR}/tmp_${i}_c1.csv"

    echo "Starting experiment $i/$NUM_EXPERIMENTS, seed=$seed"
    $PYTHON main.py "$seed" "$NUM_REPEATS" \
        --common_csv "$tmp_csv" \
        --common_csv_c1 "$tmp_csv_c1" &
    pids+=($!)
done

for pid in "${pids[@]}"; do wait "$pid"; done
echo "All experiments finished, starting merge..."

# Use the first file to write header
head -n 1 "${OUT_DIR}/tmp_1.csv"          >  "$COMMON_CSV"
head -n 1 "${OUT_DIR}/tmp_1_c1.csv"       >  "$COMMON_CSV_C1"

# Append the rest of files without their headers
for i in $(seq 1 $NUM_EXPERIMENTS); do
    tail -n +2 "${OUT_DIR}/tmp_${i}.csv"    >> "$COMMON_CSV"
    tail -n +2 "${OUT_DIR}/tmp_${i}_c1.csv" >> "$COMMON_CSV_C1"
done

rm "${OUT_DIR}"/tmp_*.csv

echo "Merge complete! Final files:"
echo "  $COMMON_CSV"
echo "  $COMMON_CSV_C1"

echo "Processing final files ..."
python3 generate_comparison.py --common_csv $COMMON_CSV --common_csv_c1 $COMMON_CSV_C1

if [ $? -eq 0 ]; then
    echo "Final comparison plot generated successfully"
else
    echo "Error occurred while generating comparison plot!"
fi