# KazCT-Responses

**A bilingual Kazakh–Russian corpus of student reasoning responses and cross-lingual automated scoring for a low-resource language.**

Project AP25796141 "Zhas Galym", S. Seifullin Kazakh Agro-Technical Research University (Astana, Kazakhstan).
Pre-registration: OSF ⟨DOI⟩ (registered ⟨date⟩). Paper: ⟨citation / DOI when available⟩.

> Status (⟨date⟩): ⟨data collection in progress / corpus v1.0 released⟩. Nothing in `data/` is synthetic; synthetic files used to develop the pipeline live only in `synthetic/` and are never used for reported results.

## What is in the corpus

| File | Rows | Description |
|---|---|---|
| `data/responses.csv` | ⟨N⟩ | one written response per row: `response_id`, `student_id`, `form_lang` (kz/ru), `text_lang_detected`, `university`, `gender`, `school_lang`, `question`, `text`, `n_chars`, `n_words`, flags, `batch_id` |
| `data/labels_expert.csv` | ⟨3 × N⟩ | every rater score in long format: `response_id`, `rater_id` (R1/R2/R3), `score` (0–4), `flag`, `rated_at`, `batch_id` |
| `data/labels_final.csv` | ⟨N⟩ | `score_r1`, `score_r2`, `score_r3`, `score_final`, `adjudicated`, `split` (train/test, grouped by student) |
| `data/translations.csv` | ⟨N⟩ | machine translations used in the translate conditions: `response_id`, `mt_lang`, `text_mt`, `mt_system`, `mt_date` |
| `data/spans_calibration.csv` | ⟨…⟩ | rater-marked supporting spans from the calibration batch (for interpretability) |
| `docs/rubric_v1.1_ru_kz.pdf` | — | scoring rubric with prompt-specific criteria and anchor responses |
| `docs/annotation_guide_ru.pdf` | — | annotator guide (Russian) |
| `docs/prompts_ru_kz.md` | — | the five prompts in both languages |
| `DATASHEET.md` | — | datasheet for the dataset (Gebru et al., 2021) |

Personal information was removed before release (see DATASHEET § Collection). Responses that could not be de-identified were withheld and are counted in `DATASHEET.md`.

## Reproducing the paper

```bash
pip install -r requirements.txt            # Python 3.11; GPU with ≥ 8 GB for encoders / LLM
cd analysis
bash run_all.sh ../data labels_final.csv ../data/translations.csv
```
`analysis/README_analysis.md` describes each script. `analysis_out/` is regenerated from the released labels; every table and figure in the paper comes from `analysis_out/tables` and `analysis_out/figures`. Model revisions (Hugging Face commit hashes) and package versions are pinned in `requirements.txt` and recorded in `analysis_out/predictions/*.csv`.

## Repository layout

```
data/            released corpus (CC BY 4.0)
docs/            rubric, annotation guide, prompts, consent-form templates (RU/KZ)
analysis/        pipeline: baselines, encoders, LLM scoring, fairness, interpretability, report
tools/           data-collection tooling: forms_to_protocol.py (Google Forms export → protocol files), annotation_tool.html
synthetic/       synthetic data used ONLY to develop and test the pipeline (clearly labelled)
preregistration/ frozen copy of the OSF pre-registration text and the code tag it references
```

## Licence

Data (`data/`, `docs/`): [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Code (`analysis/`, `tools/`): MIT. See `LICENSE-DATA` and `LICENSE-CODE`.

## Citation

See `CITATION.cff`. Please cite both the paper and the dataset DOI (Zenodo ⟨DOI⟩).

## Contact

Darkhan Zhaxybayev, S. Seifullin Kazakh Agro-Technical Research University — ⟨institutional e-mail⟩.
