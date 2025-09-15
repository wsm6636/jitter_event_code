#!/bin/bash

INITIAL_SEED=1755016010     # random seed for the first experiment
NUM_REPEATS=100             # number of repeats for each experiment
NUM_EXPERIMENTS=5           # total number of experiments to run

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

COMMON_CSV="common_results_${TIMESTAMP}.csv"
COMMON_CSV_C1="common_results_c1_${TIMESTAMP}.csv"

# current seed starts from the initial seed
current_seed=$INITIAL_SEED

echo "starting $NUM_EXPERIMENTS experiments, each with $NUM_REPEATS repeats, initial seed: $INITIAL_SEED , common CSV files: $COMMON_CSV and $COMMON_CSV_C1"


for i in $(seq 1 $NUM_EXPERIMENTS); do
    echo "=================== experiment $i/$NUM_EXPERIMENTS ==================="
    echo "current seed: $current_seed"
    
    python main.py $current_seed $NUM_REPEATS --common_csv $COMMON_CSV --common_csv_c1 $COMMON_CSV_C1
    
    if [ $? -eq 0 ]; then
        echo "experiment $i executed successfully!"
    else
        echo "experiment $i executed with errors!"
        exit 1
    fi
    
    current_seed=$((current_seed + NUM_REPEATS))
    
    echo "next experiment seed: $current_seed"
    echo ""
done

echo "all $NUM_EXPERIMENTS experiments completed successfully!"
echo "output results $COMMON_CSV and $COMMON_CSV_C1"

if [ -f "$COMMON_CSV" ]; then
    total_rows=$(wc -l < "$COMMON_CSV")
    echo "rtss result total rows: $((total_rows - 1))"  # cut off the header row
fi

if [ -f "$COMMON_CSV_C1" ]; then
    total_rows_c1=$(wc -l < "$COMMON_CSV_C1")
    echo "C1 result total rows: $((total_rows_c1 - 1))"  # cut off the header row
fi

echo ""
echo " ..."
python generate_comparison.py --common_csv $COMMON_CSV --common_csv_c1 $COMMON_CSV_C1

if [ $? -eq 0 ]; then
    echo "Final comparison plot generated successfully"
else
    echo "Error occurred while generating comparison plot!"
fi