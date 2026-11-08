# OSF Preregistration — KazCT-Responses (draft v0.2, 20.10.2026 — undergraduate sample)

Template: "OSF Preregistration" (the standard OSF form; the section headings below are the form's own questions, in its order). Paste each answer into the corresponding field on osf.io → Registrations → New registration → OSF Preregistration. **Register before any rater scores a response and before any model is trained.** Text in ⟨brackets⟩ must be filled in by the supervisor before submission.

Change log vs. draft v0.1 (04.10.2026, never registered): population changed from secondary-school pupils to undergraduate students (reason: the school collection lacked parental consent and was discarded); registration now takes place after data collection but before any annotation or analysis, which is declared openly below; actual sample sizes replace targets; few-shot k = 200 replaced by k = 150 because the Kazakh training folds hold fewer than 200 responses per prompt; attention-check failures are retained with a sensitivity analysis instead of being excluded (rate 24%, see Data exclusion); fairness covariates updated (university, school language instead of school, grade).

---

## Study Information

### Title
KazCT-Responses: A Bilingual Kazakh–Russian Corpus of Student Reasoning Responses and Cross-Lingual Automated Scoring for a Low-Resource Language

### Authors
Darkhan Zhaxybayev (S. Seifullin Kazakh Agro-Technical Research University, Astana) — principal investigator; ⟨co-authors in the order to be used in the paper⟩.

### Description
Automated scoring of open-ended written responses has been developed and validated almost entirely on English data. Kazakhstan's schools and universities teach in two languages, Kazakh (a low-resource Turkic language) and Russian (a high-resource language), and no annotated corpus of students' written reasoning exists for either language. This study (1) builds a corpus of undergraduate students' responses to five reasoning prompts, written in Kazakh or Russian and scored by trained raters on a 0–4 rubric, and (2) uses it to measure how well automated scoring models trained on Russian responses transfer to Kazakh under zero-shot, few-shot, joint and translation-based conditions, compared with monolingual Kazakh models and with the human–human agreement ceiling. Secondary aims are fairness of automated scores by language of response and interpretability of the models' decisions. The study is a resource-and-methods study in educational NLP; it does not aim to measure the critical-thinking level of students, and no such conclusions will be drawn. Funding: grant AP25796141 "Zhas Galym" of the Science Committee, Ministry of Science and Higher Education of the Republic of Kazakhstan.

### Hypotheses
All hypotheses concern quadratic weighted kappa (QWK) against the final human score, estimated by 5-fold cross-validation grouped by student with three seeds (see Analysis Plan). "Primary model" means XLM-RoBERTa base.

H1 (reliability of the construct). Inter-rater agreement between the two main raters (R1, R2) on the full corpus, excluding calibration items, is QWK ≥ 0.60 in Kazakh and in Russian.

H2 (monolingual scoring). The primary model trained and tested on Kazakh (mono-KZ) reaches QWK ≥ 0.50 on at least three of the five prompts. Secondary threshold: QWK ≥ 0.60 on at least three prompts.

H3 (zero-shot transfer loss). Zero-shot transfer (train on Russian only, test on Kazakh) is at least 0.15 QWK below mono-KZ, averaged over prompts.

H4 (few-shot recovery). Few-shot transfer with 150 Kazakh examples (train on Russian, then on 150 Kazakh examples) recovers at least half of the gap between zero-shot and mono-KZ, averaged over prompts.

H5 (translation). Translate-test (train on Russian, test on Kazakh responses machine-translated into Russian) is within 0.05 QWK of few-shot-150, averaged over prompts.

H6 (fairness by language). For the best-performing model, the coefficient of response language in a linear model of the machine score on the human score, language, their interaction and log response length is not distinguishable from zero (95% bootstrap confidence interval includes zero).

Non-confirmation of any hypothesis is not a reason to withhold publication: all pre-registered analyses will be reported regardless of outcome.

---

## Design Plan

### Study type
Observational Study — Data are collected from study participants (students' written responses and raters' scores); the automated-scoring experiments are computational comparisons of models on these data. There is no intervention on participants.

### Blinding
No treatment groups (not applicable). Raters are blind to all student metadata (university, gender, school language) and to each other's scores; the adjudicator is blind to rater identity. Model evaluation is blind by construction (held-out folds).

### Is there any additional blinding in this study?
Responses are presented to each rater in a rater-specific random order within each batch. The language of the form is not displayed, although it is inferable from the response language.

### Study design
Cross-sectional corpus construction with a fully crossed computational experiment. Factor 1: response language (Kazakh, Russian). Factor 2: prompt (E1–E5). Factor 3: model family (feature-based logistic regression; multilingual encoders XLM-R base and mBERT; Kazakh-specific encoder Kaz-RoBERTa; one open-weight LLM in zero-shot and few-shot prompting). Factor 4 (transfer condition, Russian → Kazakh, evaluated on Kazakh test folds): mono-KZ; zero-shot; few-shot with k = 50, 100, 150; joint; translate-test; translate-train. The reverse direction (Kazakh → Russian) is run as a secondary analysis. Each response receives a score from two main raters; a third rater scores a 200-response subset.

### Randomization
Few-shot subsets of k Kazakh examples are drawn at random, stratified by final score, three draws per k (seeds 11, 22, 33). Cross-validation folds are assigned at random with the student as the grouping unit (seeds 11, 22, 33). Within-batch presentation order for raters is a seeded random permutation keyed on the rater ID. Calibration items and annotation batches were assigned by a seeded random draw (seed 42) at ingest. No other randomisation.

---

## Sampling Plan

### Existing data
Registration following data collection, but before analysis. The responses were collected between 10 and 17 October 2026 (see Data collection procedures) and exported on 20 October 2026. As of the registration date: no response has been scored by any rater; no model has been trained on any real response; no text has been read by the authors beyond spot checks of the first exported rows during pipeline testing. The only processing performed is an automated ingest script (`forms_to_protocol_v2.py`, committed in the repository before registration) that removed test rows and repeat submissions, computed character counts and the flags listed under Data exclusion, assigned calibration and annotation batches, and produced the counts reported under Sample size. No metric that depends on scores exists yet. An earlier draft of this registration (04.10.2026) targeted secondary-school pupils; that collection was discontinued because parental consent had not been obtained, and none of those responses are used. A pilot of an earlier instrument (November 2025, n = 41, different item format) was used only to revise the instrument. Synthetic responses (n = 300 synthetic questionnaires) were generated to develop and test the analysis code; they are labelled as such, released with the code, and never enter any reported result.

### Explanation of existing data
Registration before analysis rather than before collection was unavoidable once the population changed on short notice; the authors state above exactly what has and has not been done with the data. The analysis code was written and tested on synthetic data before any real response was collected; the pipeline is frozen (tagged commit ⟨hash⟩) at registration. The counts under Sample size are the only knowledge the authors have of the data at registration.

### Data collection procedures
Participants are undergraduate students aged 18 or older at two universities in Astana, Kazakhstan: S. Seifullin Kazakh Agro-Technical Research University (KATRU) and L. N. Gumilyov Eurasian National University (ENU), recruited through instructors of regular class groups across faculties (convenience sample; no selection on ability). Students completed an online form (Google Forms) in Kazakh or Russian, by their own choice of form language, during or after a class session, on their own devices or in a computer room; instructors distributed the link and did not supervise the content of responses. The first screen of the form states who conducts the study (the university and grant), its purpose, that participation is voluntary and anonymous, that no name is asked and that "responses are seen only by researchers" (literal wording), and requires (a) a consent checkbox worded "I understand that participation is voluntary, the questionnaire is anonymous, and my responses will be used only for scientific purposes" and (b) a "yes" to "Are you 18 or older?"; a "no" ends the form. The consent text does not mention publication of the response texts in an open dataset; the consequences for data release are stated under Other. The Russian form existed as two identical copies (the second created inadvertently and used by some groups); their exports were merged, and the identity of the two copies' wording is verified against the forms. The form collects: a self-chosen participant code (used only for consent withdrawal), university and faculty, class group (free text, entered inconsistently and therefore not usable as a clustering variable), gender (with "prefer not to say"), language of instruction at the student's secondary school, daily internet use, 12 interest ratings, 9 preferred-format checkboxes, 6 motivation items, 13 learning-style items including one attention check, 6 closed critical-thinking items, five open-ended reasoning prompts (E1–E5; a 150-character minimum was intended but not enforced by the form), and two closing items. Only the five open-ended responses and the metadata listed under Variables enter this study; the other blocks serve other tasks of the grant and are not analysed here. No names, e-mail addresses or student ID numbers were collected.

### Sample size
Achieved (after removal of 2 test rows and 3 repeat submissions, before any further exclusion): 368 students — 146 in the Russian form (KATRU 109, ENU 37) and 222 in the Kazakh form (KATRU 151, ENU 71) ; 1,840 responses (five per student, including empty ones). Language of instruction at school: Kazakh 194, Russian 126, mixed 48. Gender: 236 male, 119 female, 13 not stated. Responses of at least 150 characters per prompt: 181–190 in Kazakh and 122–126 in Russian. The pre-registered minimum of 100 students per language is met; the planning target of 400 students is not.

### Sample size rationale
Ariely et al. (2023, Int. J. Artif. Intell. Educ. 33) report from learning curves that about 500 labelled responses suffice for good machine–human agreement in roughly 60% of their scoring models and about 900 for roughly 80%; Horbach et al. (2024, Int. J. Artif. Intell. Educ. 34) obtained their cross-lingual results with about 300 responses per prompt and language, and their best condition fine-tuned on about 270 target-language responses. With 146–222 students per language the corpus yields 146–222 responses per prompt and language, which supports few-shot conditions up to k = 150 but is below the ~500 level at which monolingual performance saturates; this is acknowledged as a limitation in advance, and the Russian (source-language) training sets of about 117 responses per prompt and fold are expected to limit transfer performance. The sample was constrained by the number of class groups reachable within one week, not by a power calculation.

### Stopping rule
Collection ended on 17 October 2026. No further collection is planned. If additional groups are collected later (for example to balance the Russian sample), the additional data will be added only before annotation starts and reported as an update to this registration; data collected after annotation starts will not be added to the corpus used for the pre-registered analyses.

---

## Variables

### Manipulated variables
None at the participant level. At the modelling level: transfer condition (mono-KZ; zero-shot; few-shot-50/100/150; joint; translate-test; translate-train), model family, and amount of Kazakh training data (0, 50, 100, 150, all).

### Measured variables
Per response: text; form language (kz/ru); automatically detected text language; prompt (E1–E5); character and token length; rater scores R1, R2 (0–4) for all responses and R3 (0–4) for the 200-response subset; rater flags (off-topic, empty, suspected AI-generated, code-switching, personal data); ingest flags (empty, short < 150 characters, verbatim copy of another participant's response); final score (see Indices). Per student: university (KATRU/ENU), gender (m/f/not stated), language of instruction at school (kz/ru/mixed), attention-check pass/fail, consent. Per model run: predicted score and class-probability vector for every test response, fold, seed and condition.

### Indices
Final human score: if |R1 − R2| ≤ 1, the higher of the two scores; if |R1 − R2| ≥ 2, the score assigned by the adjudicator (project lead, blind to rater identity). Human–human ceiling: QWK between R1 and R2 on the same test responses as the model (and, on the 200-response subset, mean pairwise QWK among R1, R2, R3). Model metrics per condition: QWK (primary), mean absolute error, exact agreement, within-one agreement, macro-F1 over five levels, expected calibration error (encoders only). Fairness index: language coefficient (and language × human-score interaction) in the linear model described in H6. Interpretability index: token-level intersection-over-union between Integrated-Gradients attributions and rater-marked supporting spans on 100 responses.

---

## Analysis Plan

### Statistical models
Scoring models predict the final human score (five ordinal classes) from the response text, trained per prompt. (1) Feature-based: logistic regression on TF-IDF word 1–3-grams and character 2–5-grams; a length-only logistic regression (log characters) as a floor. (2) Multilingual encoders: XLM-RoBERTa base (primary) and multilingual BERT cased, fine-tuned end-to-end with a classification head and cross-entropy loss; learning rate 2e-5, batch size 16 (with gradient accumulation if needed), up to 10 epochs with early stopping on the validation split, maximum 256 tokens, three seeds. XLM-RoBERTa large is run only if the base model's QWK confidence interval overlaps the human–human interval on at least two prompts (decision rule fixed here). (3) Kazakh-specific encoder: Kaz-RoBERTa (kz-transformers/kaz-roberta-conversational), same procedure, used only in mono-KZ and few-shot-KZ conditions. (4) LLM: one open-weight model selected by a pre-specified smoke test (memory fit on an 8 GB GPU in 4-bit quantisation; ≥ 95% format-valid outputs on 60 synthetic responses; throughput), among KazLLM-1.0-8B, Qwen3.5-9B-Kazakh and Sherkala-Chat-8B; prompted with the rubric and prompt in the response language, zero-shot and five-shot per level, greedy decoding, reasoning mode disabled, constrained to a single digit 0–4. The smoke-test result and the chosen model will be added to this registration as an update before any real response is scored by the LLM.

Transfer conditions, per prompt, evaluated on Kazakh test folds: mono-KZ (train on Kazakh training fold); zero-shot (train on Russian training fold only); few-shot-k (train on Russian, then continue on k Kazakh examples from the Kazakh training fold, k ∈ {50, 100, 150}; if a prompt's Kazakh training fold holds fewer than k eligible examples, that k is reported as not estimable for that prompt); joint (Russian + Kazakh training folds pooled); translate-test (train on Russian; Kazakh test responses machine-translated to Russian); translate-train (Russian training responses machine-translated to Kazakh; test on original Kazakh). Machine-translation system: ⟨fixed before any model is trained, after a pilot on 50 Kazakh responses comparing Google Translate, NLLB-200-distilled-600M and Yandex Translate by chrF against a human translation; the system name, version and date will be added as an update to this registration⟩. Reverse direction (Kazakh → Russian) with the same conditions as a secondary analysis.

Fairness: for each model and condition, a linear regression of the machine score on the final human score, response language, their interaction and log character length, fitted on all test predictions pooled over folds; the language coefficient with a 95% bootstrap CI (2,000 resamples over students) is the pre-registered quantity for H6. The same model with gender, university and school language of instruction as secondary fairness checks. Calibration: expected calibration error (10 bins) by language for the encoders.

Interpretability: Integrated Gradients attributions for the best encoder on 100 stratified responses (50 per language, 10 per prompt), compared with rater-marked supporting spans collected during calibration; token-level IoU reported with a bootstrap CI.

### Transformations
Response length is log-transformed (natural log of character count) in the fairness model. Texts are not otherwise normalised before encoder input; for the n-gram baseline, texts are lower-cased and tokenised on whitespace and punctuation. Scores are used as ordinal integers 0–4 without rescaling.

### Inference criteria
All metrics are reported as the mean over 5 folds × 3 seeds with a 95% bootstrap confidence interval over test responses (2,000 resamples). Comparisons between conditions use the bootstrap CI of the paired difference in QWK computed on the same test responses. Within each results table, the Holm procedure at α = 0.05 controls the family-wise error rate over the pre-registered pairwise comparisons (mono-KZ vs. each transfer condition; few-shot-150 vs. translate-test). A model is called "at human level" only when the CI of its QWK overlaps the human–human CI on the same responses. H1–H5 are evaluated against the thresholds stated above using point estimates, with CIs reported; H6 is evaluated by whether the CI of the language coefficient includes zero. No one-sided tests; no p-values are used as the primary criterion.

### Data exclusion
Applied at ingest, before registration (counts given): test rows entered by the research team (2) and repeat submissions in which all five responses are identical to an earlier row (3) are removed. Rows without the consent checkbox or with "no" to the age question are removed (0 such rows). Applied before annotation: a response that is a verbatim copy (≥ 30 characters) of another participant's response is flagged `copied_text` and excluded from annotation and modelling (27 responses), the first-submitted copy being kept; empty responses (43) receive score 0 without annotation and are retained in descriptive counts but excluded from modelling. Retained with flags: responses shorter than 150 characters (253) are annotated and modelled like any other; students failing the attention check (item C1.7 ≠ 2; 90 of 368, i.e. 24%) are retained in the main analysis because the check sits in an unrelated Likert block and response quality is assessed directly by raters, and a sensitivity analysis excluding them is reported in the supplement (this is a change from draft v0.1, made before any score existed). Responses flagged "off-topic" by both main raters receive score 0 and are retained; responses flagged "contains personal data" are de-identified or, if impossible, excluded from the public release but kept in the counts; responses flagged "suspected AI-generated" are retained, with a sensitivity analysis excluding them. Responses in the Kazakh form written in Russian (28 by automatic detection) are scored normally and analysed by detected text language in the fairness model; a sensitivity analysis by form language is reported. Calibration responses (30) are excluded from all reliability estimates.

### Missing data
A response shorter than one sentence is scored 0, not treated as missing. A student with fewer than five non-empty responses contributes the responses they have (per-prompt models do not require complete sets). Missing gender is coded as a separate category in fairness checks. Missing rater scores (a rater skipping a response) are completed by that rater before adjudication; if a score remains missing, the response is excluded from inter-rater statistics but keeps the other rater's score as final.

### Exploratory analysis (optional)
Exploratory and explicitly labelled as such: (a) prompt-level generalisation (leave-one-prompt-out); (b) leave-one-university-out generalisation; (c) an ordinal-regression head instead of cross-entropy; (d) XLM-RoBERTa large if the decision rule above is not met but compute allows; (e) correlation between automated scores on E1–E5 and the six closed critical-thinking items, reported descriptively only; (f) effect of data augmentation (back-translation) in the few-shot conditions, following Nazaretsky et al. (2023); (g) differences by language of instruction at school within the Kazakh-form sample.

---

## Other

### Other (optional)
Ethics: participants are adults who gave electronic informed consent on the first screen of the form (wording quoted under Data collection procedures); no names, contact details or identifiers were collected; a self-chosen code allows withdrawal. ⟨Either: "The study was reviewed by the Ethics Committee of S. Seifullin Kazakh Agro-Technical Research University (protocol No. …, date …)" or: "Under the university's regulations, anonymous questionnaire research with adult participants does not require ethics-committee review (letter of exemption No. …, date …)"; one of the two must be in hand before submission to a journal.⟩ Data release: because the consent wording promised that responses are seen only by researchers and did not mention open publication, the response texts and their machine translations will NOT be published openly. They will be deposited on Zenodo under restricted access and made available to researchers on request, subject to a data-use agreement (research use only; no re-identification attempts; no redistribution), ⟨pending the decision of the university's ethics body on this access model; if restricted sharing is not approved, the texts remain with the research team⟩. Released openly (CC BY 4.0): all rater scores and final scores keyed to response IDs, response-level metadata (language, prompt, length, flags), student-level metadata (university, gender, school language), the rubric with anchor responses, the annotation guidelines, the prompts for the LLM, the datasheet (Gebru et al., 2021), and all code (MIT) on Zenodo and GitHub. Timeline: calibration and annotation to 5 December 2026; modelling to 19 December 2026; manuscript submission January 2027 (target journal: Computers and Education: Artificial Intelligence; alternative: Information (MDPI)). Deviations: any departure from this registration will be listed in an appendix of the paper with its reason and date. Related registrations: none. Conflicts of interest: none. The authors are aware of a Kazakhstani research programme (BR24993072, Satbayev University / KBTU) working on AI-generated Bloom-aligned questions and on profile-based prediction of critical thinking; the present study differs in object (scoring of written reasoning against expert labels), data (an expert-annotated corpus available to researchers) and question (cross-lingual transfer), and does not reuse any of its materials.

---

## Checklist before clicking "Register"
1. Fill every ⟨bracket⟩: co-authors; frozen commit hash; ethics statement variant.
2. Attach as files to the registration: rubric v1.1 (RU/KZ with anchor responses), the five prompts (RU/KZ), the annotation guidelines, the consent screen text (RU/KZ), `ingest_report.md` (counts and flags as of 20.10.2026), and `forms_to_protocol_v2.py`.
3. Choose "Make registration public immediately" (preferred) or an embargo of at most 4 years; the registration DOI is cited in the paper either way.
4. Save the registration DOI and date into `claude/статус_работ.md` and into Methods.
5. Only after step 3: hand the calibration batch to the raters.
