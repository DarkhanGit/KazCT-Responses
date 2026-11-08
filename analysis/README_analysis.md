# analysis/ — конвейер анализа статьи 3 (KazCT-Responses)

Вход: папка `clean/` из `forms_to_protocol.py` (responses.csv, labels_final.csv, labels_expert.csv, survey_raw.csv) и, для translate-условий, `translations.csv` (response_id, mt_lang, text_mt, mt_system).
Выход: `analysis_out/predictions/*.csv` (предсказания по ответам), `analysis_out/tables/T2–T7*.md|csv`, `analysis_out/figures/F2–F5*.png|pdf`, `analysis_out/fairness.csv`, `analysis_out/REPORT.md`.

| Скрипт | Что делает | Где запускать |
|---|---|---|
| `common.py` | фолды GroupKFold по ученику (seeds 11/22/33), метрики, bootstrap, Холм, потолок, формат предсказаний | — |
| `make_synthetic_labels.py` | СИНТЕТИЧЕСКИЕ метки для отладки (никогда не для результатов) | где угодно |
| `baselines.py` | LR на n-граммах и length-only; mono/zeroshot/joint | где угодно (CPU, ~3 мин) |
| `train_encoder.py` | XLM-R / mBERT / Kaz-RoBERTa; mono, zeroshot, fewshot k=50/100/200 ×3, joint, translate_test, translate_train | Legion 5 (GPU) |
| `score_llm.py` | LLM zero/few-shot с рубрикой (prompts_llm.py) | Legion 5 (GPU, 4-бит) |
| `fairness.py` | регрессия machine ~ human × lang (+ log len), бутстреп по ученикам, ECE по языкам | где угодно |
| `interpret.py` | Integrated Gradients на 100 ответах, IoU с опорными фрагментами экспертов (spans.csv) | Legion 5 |
| `report.py` | все таблицы и рисунки, проверка гипотез H1–H6 | где угодно |
| `run_all.sh` | всё по порядку | Legion 5 |

Проверено на синтетике 09.10.2026: baselines → fairness → report полностью; train_encoder и score_llm — на крошечной случайной модели / «сухом» режиме (формат файлов и логика условий), реальные модели — только на GPU.
Оценка времени на Legion 5 (RTX 5060, 8 ГБ) для ~2 000 ответов: один энкодер × условие × промпт × фолд ≈ 1–2 мин → XLM-R base все условия ≈ 5 промптов × 5 фолдов × 3 seed × (4 + 9 few-shot) прогонов ≈ 1 000 обучений ≈ 15–30 часов. Сокращение при нехватке времени: сначала seed 11 для всех условий, затем seeds 22/33.
