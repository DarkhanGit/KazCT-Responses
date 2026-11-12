# KazCT-Responses

**A bilingual Kazakh–Russian corpus of student reasoning responses and cross-lingual automated scoring for a low-resource language.**

Project AP25796141 "Zhas Galym", S. Seifullin Kazakh Agro-Technical Research University (KATRU), Astana, Kazakhstan.
Pre-registration: OSF https://doi.org/10.17605/OSF.IO/SDGYE (23 October 2026, after data collection and before annotation or analysis; three registration updates, see `preregistration/`). Dataset archive: Zenodo https://doi.org/10.5281/zenodo.24047377. Paper: under review (preprint/DOI to be added).

> Status (8 November 2026): collection, annotation and all pre-registered analyses are complete; this is the **open tier** of the corpus (release v1.0). Response texts are **not** included — see "Two tiers" below.

## Corpus in one paragraph

368 undergraduate students at two universities in Astana (KATRU 260, L. N. Gumilyov ENU 108) answered five open-ended reasoning prompts (E1–E5) in a Kazakh-language form (222 students, 1,110 responses) or a Russian-language form (146 students, 730 responses) between 10 and 17 October 2026 — 1,840 responses in total. Two raters scored every response on a 0–4 reasoning-quality rubric and a third rater scored a 200-response subset; disagreements of two or more points were adjudicated. Inter-rater agreement (R1–R2, quadratic weighted kappa) is 0.866 for Kazakh and 0.790 for Russian on the full corpus. 87 responses contain no letter or digit and receive score 0 by rule; 27 are verbatim copies of another participant's response; the modelling set of the paper is 1,726 responses (1,038 Kazakh, 688 Russian).

## Two tiers

| Tier | Where | Contents | Licence |
|---|---|---|---|
| **Open** (this repository, Zenodo) | GitHub + Zenodo | all scores and flags keyed to response IDs, student-level metadata, response-level metadata (language, prompt, length, ingest flags), rubric and annotator guide, prompts, consent text, pre-registration texts and deviations log, all code, all model predictions, result tables and figures, datasheet | CC BY 4.0 (data, docs), MIT (code) |
| **Restricted** | held by the authors (on request) | response texts (`responses.csv` with `text`), machine translations (`translations.csv`), rater comments (which quote response words), human reference translations of the MT pilot | data-use agreement: research use only, no re-identification, no redistribution |

Participants were told that their responses would be seen only by researchers and were not asked to consent to open publication; therefore no response text, translation or quoting comment is released openly. Requests for the restricted tier: by e-mail to the maintainer (below), stating the research purpose; the data-use agreement is sent in reply.

## Files in `data/`

| File | Rows | Columns |
|---|---|---|
| `responses_meta.csv` | 1,840 | `response_id`, `student_id`, `question` (E1–E5), `level`, `text_lang_detected`, `n_chars`, `n_words`, `form_lang` (kz/ru), `university` (KATRU/ENU), `school_lang`, `is_empty`, `short` (< 150 characters), `copied_text`, `copy_of`, `batch_id` — everything from `responses.csv` except the text |
| `students.csv` | 368 | `student_id`, `form_lang`, `university`, `gender` (1 = male, 2 = female, 0 = not stated), `school_lang`, `internet_hours`, `attention_fail`, `ct_closed` (six closed critical-thinking items), `n_empty`, `n_short`, `submitted_at` |
| `labels_expert.csv` | 3,856 | every rater score in long format: `response_id`, `rater_id` (R1/R2/R3), `score` (0–4), `flag` (empty / offtopic / ai_suspect / lang_switch / pii), `rated_at`, `batch_id`. The rater `comment` column is in the restricted tier only |
| `labels_final.csv` | 1,813 (no rows for the 27 verbatim copies) | `score_r1`, `score_r2`, `score_r3` (200 responses), `score_final`, `adjudicated`, `split` (train/test, 80/20, grouped by student) |
| `adjudication.csv` | 18 | responses adjudicated by the project lead (`score_final`, date) |
| `r3_subset.csv` | 200 | the stratified subset scored by R3 |
| `agreement_2026-11-01.json` | — | inter-rater statistics as computed at the end of annotation |

Student IDs are row-based and not derived from any personal identifier. The free-text class-group field, cohort year and course are not released (quasi-identifiers; many groups have a single participant).

## Reproducing the paper

```bash
pip install -r requirements.txt        # Python 3.11; GPU with ≥ 8 GB VRAM for the encoders and LLMs
```
The pipeline in `analysis/` needs the restricted tier (texts, translations) as input; with the open tier alone you can recompute every table and figure of the paper from `results/` and from the released predictions (`predictions/*.csv` on Zenodo: one row per response and condition with `y_true`, `y_pred`, fold, seed, model revision). Entry points: `analysis/results_5_1_5_2.py` (corpus and human agreement), `run_encoders.ps1` / `train_encoder.py` (encoder conditions; `--legacy` reproduces the registered schedule), `run_translate.ps1` / `mt_compare.py` (MT pilot and translate conditions), `score_llm.py` (LLM conditions), `report_full.py` (Tables 3–5, hypotheses H1–H6), `supplement.py` (S1–S4, Figures 2–3), `interpret.py` (Integrated Gradients). `analysis/README_analysis.md` describes each script (Russian). Hugging Face model revisions are recorded in every prediction file.

## Repository layout

```
data/             open tier of the corpus (see table above)
docs/             rubric v1.1 + annotator guide v1.2 (RU), prompts E1–E5 (RU/KZ), consent screen, ingest report, annotation-tool instruction
preregistration/  OSF registration text, registration updates 1–3, DEVIATIONS.md (master log of departures), code snapshot attached to the registration
tools/            data-collection tooling: Google Forms script, forms_to_protocol_v2.py (export → protocol files), annotation_tool.html, anchor responses (written by the authors) and guide builder
analysis/         analysis pipeline (see above); analysis/legacy/ keeps the versions replaced during the study
gpu/              environment setup and smoke test for the GPU laptop (60 SYNTHETIC responses, not field data)
results/          every table and figure in the paper and supplement, diagnostic of the training schedule (Appendix C), interpretability summaries, MT pilot scores
```

## Departures from the pre-registration

The code as attached to the registration is tag [`prereg-v1.0`](https://github.com/DarkhanGit/KazCT-Responses/releases/tag/prereg-v1.0) of this repository (identical to `preregistration/analysis_code_snapshot_2026-10-22.zip`). All departures are logged with dates in `preregistration/DEVIATIONS.md` and summarised in Appendix C of the paper. The only change to a registered analysis procedure is the encoder training schedule (registration update 3, 4 November 2026): the registered schedule degenerated to single-class predictions for XLM-RoBERTa; the predictions of that first run are released unchanged in `predictions_legacy_procedure/` (Zenodo).

## Licence

Data and documentation (`data/`, `docs/`, `results/`, `DATASHEET.md`): [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Code (`analysis/`, `tools/`, `gpu/`): MIT. See `LICENSE-DATA` and `LICENSE-CODE`. Restricted tier: data-use agreement (see above).

## Citation

See `CITATION.cff`. Please cite both the dataset (Zenodo DOI above) and the paper.

## Maintainer

Darkhan Zhaxybayev, S. Seifullin Kazakh Agro-Technical Research University — d.zhaksybaev@kazatu.edu.kz. Corrections and additional annotations can be proposed through GitHub issues; accepted changes are versioned on Zenodo.
