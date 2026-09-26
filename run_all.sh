#!/bin/bash
# run_all.sh — sekali jalan di Vast.ai. Resume-able: kalau crash, jalankan ulang perintah yang sama.
#
#   bash run_all.sh                 # mode default: orchestrator (loop seed×fold di luar, hemat disk)
#   bash run_all.sh --sequential    # part per part (butuh disk lebih: bobot teacher 25 fold ditahan)
#   GUAVA_PROFILE=smoke bash run_all.sh   # uji pipeline cepat (1 seed, 1 fold, 1 epoch)
set -e
set -o pipefail
cd "$(dirname "$0")"
export GUAVA_ROOT="${GUAVA_ROOT:-/workspace/guava}"
export PIP_NO_CACHE_DIR=1
mkdir -p "$GUAVA_ROOT"/{logs,state,results,artifacts}
LOG_DIR="$GUAVA_ROOT/logs"

python -c "from guava_core import ensure_deps; ensure_deps()"

if [ "$1" == "--sequential" ]; then
    for part in part1_dataset_baseline part2_transformer_cnn part3_edge_candidates \
                part4_compression_kd part5_compression_prune_quant \
                part6_champion_export part7_excel_reporter; do
        echo "=== RUN $part ==="
        # state/*.json membuat part yang sama melanjutkan dari unit terakhir bila di-restart
        python ${part}.py 2>&1 | tee "$LOG_DIR/${part}_$(date +%Y%m%d_%H%M).log"
    done
else
    python part8_orchestrator.py 2>&1 | tee "$LOG_DIR/orchestrator_$(date +%Y%m%d_%H%M).log"
fi

echo "=== DONE ==="
python -c "from part0_disk_guard import report_disk; report_disk()"
