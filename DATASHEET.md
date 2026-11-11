# Datasheet for KazCT-Responses

Following Gebru et al. (2021), *Datasheets for Datasets*, Communications of the ACM 64(12). Version 1.0 (8 November 2026), completed after annotation and analysis; updated 11 November 2026 (ethics status based on national legislation; the confirming KATRU letter is pending); the remaining ⟨bracket⟩ concerns the compensation of the external raters. Statements here must match Methods § 3 of the paper and the OSF pre-registration.

## Motivation

**For what purpose was the dataset created?** To enable training and evaluation of automated scoring models for the quality of written reasoning in Kazakh and Russian, and specifically to measure cross-lingual transfer from Russian (high-resource) to Kazakh (low-resource). No comparable annotated corpus of student responses existed in either language. The dataset is not intended to measure or report the critical-thinking level of Kazakhstani students.

**Who created the dataset and on behalf of which entity?** The research team of grant AP25796141 "Zhas Galym" at S. Seifullin Kazakh Agro-Technical Research University (Astana), principal investigator Darkhan Zhaxybayev; annotation by three bilingual raters: a team member (computer scientist with prior annotation experience), a professor of foreign philology at L. N. Gumilyov Eurasian National University, and an external school teacher of Kazakh (Pavlodar) who scored a 200-response subset.

**Was the study pre-registered?** Yes: OSF registration https://doi.org/10.17605/OSF.IO/SDGYE (23 October 2026), made after data collection and before any annotation or analysis; the registration states what was known about the data at that time.

**Who funded the creation of the dataset?** Science Committee of the Ministry of Science and Higher Education of the Republic of Kazakhstan, grant AP25796141 (2025–2027).

## Composition

**What do the instances represent?** One instance is one written response by one undergraduate student to one of five open-ended reasoning prompts (E1–E5), with its expert scores. A student contributes five instances (empty responses are kept as instances with score 0).

**How many instances are there?** 1,840 responses from 368 students at two universities in Astana (KATRU 260, ENU 108); 1,110 in the Kazakh form (222 students) and 730 in the Russian form (146 students). Of these, 43 are empty and 44 more contain no letter or digit (87 receive score 0 by rule and are excluded from modelling), 253 are shorter than 150 characters, and 27 are verbatim copies of another participant's response (flagged; excluded from the paper's models). The modelling set is 1,726 responses (1,038 Kazakh, 688 Russian). No response was withheld at release: the raters' personal-data flag (`pii`) was not used on any response.

**Does the dataset contain all possible instances or a sample?** A convenience sample of undergraduate class groups at two universities, recruited through instructors during one week (10–17 October 2026); not representative of Kazakhstan's student population (two-thirds men; faculties unevenly represented). Participation within a group was voluntary and unsupervised; no participation counts per group are available.

**What data does each instance consist of?** Open tier: everything below except the text. Restricted tier: in addition, the raw text of the response (as typed; spelling not corrected), its machine translation and the rater comments; language of the form (kz/ru); automatically detected language of the text; prompt ID; character and word counts; scores from raters R1 and R2 (all instances) and R3 (200-instance subset); final score (0–4); adjudication flag; rater flags (empty 179, suspected AI-generated 56, off-topic 13, code-switching 8, personal data 0 — counts over 3,856 rater–response pairs); batch ID; ingest flags (empty, short, copied_text, attention-check fail of the student); student-level metadata: student ID (row-based, not derived from any personal identifier), university (KATRU/ENU), gender (m/f/not stated), language of instruction at the student's secondary school (kz/ru/mixed).

**Is there a label or target?** Yes: `score_final` (0–4), the quality of reasoning according to rubric v1.1; individual rater scores are also released.

**Is any information missing?** R3 scores exist for 200 instances only (by design; R3 also scored the 30 calibration responses). No response was withheld for de-identification reasons. Students who did not pass the attention check (90 of 368) are flagged, not removed. Response texts are absent from the open tier by design (see Distribution).

**Are relationships between instances made explicit?** Yes: `student_id` links the five responses of a student; `university` links students to institution. The class-group field was free text, entered inconsistently, and is not released. The released train/test split is grouped by student.

**Are there recommended data splits?** `split` column: 80/20 by student, stratified by language and prompt. The paper's results use 5-fold GroupKFold by student with seeds 11/22/33 (code provided), not this split; the split is provided for comparability of future work.

**Are there errors, noise or redundancies?** Human scores carry rater disagreement: QWK R1–R2 = 0.866 (Kazakh) and 0.790 (Russian) on the full corpus, 0.852 and 0.771 on the modelling set; exact agreement and the three-rater statistics are in `data/agreement_2026-11-01.json` and in the paper (Section 5.2). Some responses may be partly AI-generated (flagged by raters, not verified). Spelling and grammar errors are left as written.

**Is the dataset self-contained?** Yes within each tier. Machine translations used in the paper's translate conditions are included with the texts in the restricted tier (`translations.csv`) with the system name and date.

**Does the dataset contain confidential data?** No. Raters were instructed to flag any name, school or personal detail (`pii` flag); no response was flagged. Participants' self-chosen withdrawal codes were never linked to names.

**Does the dataset contain content that might be offensive or distressing?** Responses are students' opinions on information reliability, AI, jobs, and a personal episode of changing one's mind (E5). E5 responses may describe personal conflicts; raters were instructed to flag responses disclosing personal data; none was flagged. Texts are nevertheless distributed only under restricted access (consent wording).

**Does the dataset identify subpopulations?** University, gender, form language and school language of instruction are included to enable fairness analysis.

**Is it possible to identify individuals?** Not from the released fields. The combination university × gender × five texts could in principle narrow a student within a known cohort; users must not attempt re-identification (licence condition).

**Does the dataset contain sensitive data?** Language of instruction can be read as a proxy for ethnicity in Kazakhstan; it is released because it is the primary variable of the study. No health, religious, political-opinion or precise-location data are collected.

## Collection process

**How was the data acquired?** Students typed responses into an online form (Google Forms; Russian and Kazakh versions with parallel content; the Russian form existed as two identical copies whose exports were merged) during or after a class session, on their own devices or in a computer room, after instructors distributed the link. No instruction script was read; instructions were on the form. A 150-character minimum was intended but not enforced by the form.

**What mechanisms or procedures were used?** Google Forms exports (CSV) were converted to the protocol files by `tools/forms_to_protocol_v2.py` (removal of 2 test rows and 3 repeat submissions; flags; batch assignment with seed 42); raters scored responses in `tools/annotation_tool.html`, which hides student metadata and randomises order per rater.

**If a sample, what was the sampling strategy?** Convenience sampling of class groups reachable through instructors at two universities, with the aim of at least 100 students per form language; no constraint on faculty or year of study.

**Who was involved in the collection and how were they compensated?** Instructors distributed the link as a courtesy to the project; they were not paid. Students received no compensation, grade or credit. Raters: one team member (paid under the grant), one university professor and one external school teacher (both external to the grant; ⟨compensation of the external raters to be stated by the PI⟩).

**Over what timeframe?** 10–17 October 2026 (collection); 24–30 October 2026 (annotation); 1 November 2026 (adjudication); 1–7 November 2026 (modelling).

**Were any ethical review processes conducted?** No committee review. Under the legislation of the Republic of Kazakhstan, review by a bioethics commission is mandatory for biomedical research (Code of the Republic of Kazakhstan "On Public Health and the Healthcare System" No. 360-VI of 7 July 2020, Art. 1(20) and Art. 228; Typical Rules of Scientific Ethics, Order of the Minister of Science and Higher Education No. 529 of 20 November 2024, para. 6(16)); this anonymous, non-interventional questionnaire of adult students is not biomedical research. The participant-protection requirements of the same Rules (informed consent, safety and dignity of participants, personal-data protection, para. 6(15), 6(17), 6(27), 6(28)) and the Law "On Personal Data and Their Protection" No. 94-V were followed. A confirming letter from S. Seifullin Kazakh Agrotechnical Research University was requested on 8 November 2026; its number and date will be added here on receipt (OSF registration update 4; paper Appendix C.10).

**Did the individuals consent?** Yes. All participants were adults (confirmed by a mandatory age question). Electronic informed consent on the first screen of the form, which stated the purpose, voluntariness and anonymity of the study and that responses are seen only by researchers; the consent checkbox read "I understand that participation is voluntary, the questionnaire is anonymous, and my responses will be used only for scientific purposes". Open publication of texts was not mentioned, which is why texts are distributed under restricted access (see Distribution). Students could simply not fill in the form; the number who declined is unknown.

**Was a mechanism provided to revoke consent?** Yes: a participant may ask the research team to withdraw their responses by quoting university, form language and their self-chosen code; no withdrawal request was received before the release of version 1.0 (8 November 2026). After release, withdrawn responses are removed in the next dataset version.

**Was an analysis of the potential impact on data subjects conducted?** The team assessed re-identification risk and the sensitivity of E5; mitigations: no names, e-mails or ID numbers collected, self-chosen codes known only to participants, rater flagging of personal data, withholding of non-de-identifiable responses.

## Preprocessing / cleaning / labelling

**Was any preprocessing done?** Responses were left as written. Attention-check failures are flagged. Test rows (2) and repeat submissions (3) were removed; verbatim copies between participants (27) are flagged and excluded from annotation. No personal-data spans were flagged by the raters, so no placeholder replacement was needed.

**Was the raw data saved?** Raw form exports are kept by the principal investigator and are not released (they contain the pre-de-identification texts).

**Labelling.** Rubric v1.1 (0–4; generic descriptors + prompt-specific criteria for levels 3 and 4), published with the dataset. Raters: R1 and R2 scored all responses, R3 a stratified subset of 200; calibration on 30 responses (R1–R2 QWK 0.835), batches of 100 in identical order, blind to each other; the pre-registered calibration meeting did not take place (all batch QWKs exceeded the re-calibration threshold; see DEVIATIONS.md C.2); 18 disagreements of ≥ 2 points were adjudicated by the principal investigator blind to rater identity; disagreements of 1 point were resolved as the higher score. Inter-rater reliability (QWK R1–R2): 0.866 Kazakh, 0.790 Russian. Raters were asked to copy into a comment the words that justified a score of 3–4; because these comments quote the responses, the `comment` column is in the restricted tier only.

**Is the labelling software available?** Yes, `tools/annotation_tool.html` (MIT).

## Uses

**Has the dataset been used for any tasks already?** The accompanying paper (automated scoring; cross-lingual transfer RU→KZ; fairness by language; interpretability).

**What other tasks could it be used for?** Multilingual educational NLP; few-shot learning for Kazakh; studies of code-switching in student writing; development of formative-feedback tools; as evaluation data for Kazakh LLMs.

**Is there anything that might impact future uses?** The construct is "quality of reasoning as operationalised by rubric v1.1 on five specific prompts" — not a validated measure of critical thinking, and the dataset should not be used to rank students, institutions or languages. Length is correlated with score; models trained on the data may learn length. The sample is not representative.

**Are there tasks for which the dataset should not be used?** High-stakes assessment of individual students; any attempt at re-identification; claims about the critical-thinking level of Kazakhstani students, of either language group, or of school pupils (the sample is undergraduates).

## Distribution

**How will it be distributed?** Two tiers. Open tier (Zenodo, versioned, DOI 10.5281/zenodo.24047377; GitHub https://github.com/DarkhanGit/KazCT-Responses): scores, flags, metadata, rubric, guidelines, prompts, code, predictions, result tables, this datasheet. Restricted tier (Zenodo restricted-access record): response texts, machine translations, rater comments and the human reference translations of the MT pilot, available to researchers on request under a data-use agreement (research use only; no re-identification; no redistribution). Requests are handled by the corresponding authors.

**When?** Open tier v1.0: 8 November 2026 (at submission). Restricted tier: on request from the corresponding authors (data-use agreement).

**Licence?** Open tier: CC BY 4.0 for data and documentation, MIT for code. Restricted tier: data-use agreement, research use only.

**Have any third parties imposed restrictions?** No third party; the restriction follows from the consent wording given to participants.

## Maintenance

**Who maintains the dataset?** The AP25796141 team; contact Darkhan Zhaxybayev, d.zhaksybaev@kazatu.edu.kz.

**Will the dataset be updated?** Errata and consent withdrawals produce new minor versions on Zenodo; the paper cites the exact version used.

**Can others contribute?** Corrections and additional annotations may be proposed via GitHub issues; the team reviews and versions them.
