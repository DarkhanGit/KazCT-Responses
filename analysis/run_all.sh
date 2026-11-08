#!/usr/bin/env bash
# run_all.sh — полный конвейер статьи 3 от меток до таблиц/рисунков. Запускать из папки analysis/.
# Использование: bash run_all.sh <папка clean> [labels_final.csv] [translations.csv]
# На синтетике: bash run_all.sh ../test_clean labels_final_SYNTHETIC.csv
set -e
DATA=${1:-../test_clean}; LABELS=${2:-labels_final.csv}; MT=${3:-}
echo "== 1. базлайны (LR n-grams, length-only)"; python baselines.py --data "$DATA" --labels "$LABELS"
echo "== 2. энкодеры"; for M in xlmr mbert; do python train_encoder.py --data "$DATA" --labels "$LABELS" --model $M --conditions mono zeroshot fewshot joint --save_dir ../analysis_out/checkpoints; done
python train_encoder.py --data "$DATA" --labels "$LABELS" --model kazroberta --conditions mono fewshot
if [ -n "$MT" ]; then for M in xlmr mbert; do python train_encoder.py --data "$DATA" --labels "$LABELS" --model $M --conditions translate_test translate_train --mt "$MT"; done; fi
echo "== 3. LLM"; python score_llm.py --data "$DATA" --labels "$LABELS" --repo issai/LLama-3.1-KazLLM-1.0-8B
echo "== 4. fairness"; python fairness.py --survey "$DATA/survey_raw.csv"
echo "== 5. интерпретируемость"; python interpret.py --data "$DATA" --labels "$LABELS" --checkpoint_dir ../analysis_out/checkpoints --model xlmr ${SPANS:+--spans $SPANS}
echo "== 6. отчёт"; python report.py --data "$DATA" --labels "$LABELS"
