
#!/bin/bash
set -euo pipefail

# ----------- 参数/路径 -----------
INITIAL_SEED=1755016010
NUM_REPEATS=${1:-100}          # 第 1 个命令行参数决定 NUM_REPEATS
NUM_EXPERIMENTS=${2:-5}        # 第 2 个命令行参数决定实验数
PYTHON=${PYTHON:-$(command -v python3 || command -v python)}

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUT_DIR="compare/${TIMESTAMP}"

# ---------- 2. 如果目录已存在则复用 ----------
if [[ -d "$OUT_DIR" ]]; then
    echo "目录已存在，复用：$OUT_DIR"
else
    mkdir -p "$OUT_DIR"
fi

COMMON_CSV="${OUT_DIR}/common_results_${TIMESTAMP}.csv"
COMMON_CSV_C1="${OUT_DIR}/common_results_c1_${TIMESTAMP}.csv"

echo "并行运行 $NUM_EXPERIMENTS 个实验，每个重复 $NUM_REPEATS 次"
echo "结果目录：$OUT_DIR"

# ----------- 并发实验 -----------
pids=()

# 清理函数：杀掉所有子进程
cleanup() {
    echo
    echo "收到中断信号，正在终止所有实验..."
    for pid in "${pids[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid" 2>/dev/null || true
            wait "$pid" 2>/dev/null || true
        fi
    done
    echo "已终止全部实验"
    exit 0
}

# 捕获 Ctrl+C 和 kill
trap cleanup SIGINT SIGTERM


for i in $(seq 1 $NUM_EXPERIMENTS); do
    seed=$(( INITIAL_SEED + (i-1)*NUM_REPEATS ))
    tmp_csv="${OUT_DIR}/tmp_${i}.csv"
    tmp_csv_c1="${OUT_DIR}/tmp_${i}_c1.csv"

    echo "启动实验 $i/$NUM_EXPERIMENTS，seed=$seed"
    $PYTHON main.py "$seed" "$NUM_REPEATS" \
        --common_csv "$tmp_csv" \
        --common_csv_c1 "$tmp_csv_c1" &
    pids+=($!)
done

# 等待全部结束
for pid in "${pids[@]}"; do wait "$pid"; done
echo "所有实验已结束，开始合并…"

# ----------- 合并结果 -----------
# 取第一个文件写表头
head -n 1 "${OUT_DIR}/tmp_1.csv"          >  "$COMMON_CSV"
head -n 1 "${OUT_DIR}/tmp_1_c1.csv"       >  "$COMMON_CSV_C1"

# 其余文件去掉表头再追加
for i in $(seq 1 $NUM_EXPERIMENTS); do
    tail -n +2 "${OUT_DIR}/tmp_${i}.csv"    >> "$COMMON_CSV"
    tail -n +2 "${OUT_DIR}/tmp_${i}_c1.csv" >> "$COMMON_CSV_C1"
done

# 可选：清理临时文件
rm "${OUT_DIR}"/tmp_*.csv

echo "合并完成！最终文件："
echo "  $COMMON_CSV"
echo "  $COMMON_CSV_C1"

echo "处理最终文件 ..."
python3 generate_comparison.py --common_csv $COMMON_CSV --common_csv_c1 $COMMON_CSV_C1

if [ $? -eq 0 ]; then
    echo "Final comparison plot generated successfully"
else
    echo "Error occurred while generating comparison plot!"
fi