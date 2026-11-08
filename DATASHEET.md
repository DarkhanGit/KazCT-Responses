# Datasheet for KazCT-Responses

Following Gebru et al. (2021), *Datasheets for Datasets*, Communications of the ACM 64(12). Draft v0.2 (20.10.2026), updated after collection and ingest, before annotation; items in ⟨brackets⟩ are completed at release. Statements here must match Methods § 3 of the paper and the OSF pre-registration.

## Motivation

**For what purpose was the dataset created?** To enable training and evaluation of automated scoring models for the quality of written reasoning in Kazakh and Russian, and specifically to measure cross-lingual transfer from Russian (high-resource) to Kazakh (low-resource). No comparable annotated corpus of student responses existed in either language. The dataset is not intended to measure or report the critical-thinking level of Kazakhstani students.

**Who created the dataset and on behalf of which entity?** The research team of grant AP25796141 "Zhas Galym" at S. Seifullin Kazakh Agro-Technical Research University (Astana), principal investigator Darkhan Zhaxybayev; annotation by three raters (two team members and one external school teacher of Kazakh).

**Who funded the creation of the dataset?** Science Committee of the Ministry of Science and Higher Education of the Republic of Kazakhstan, grant AP25796141 (2025–2027).

## Composition

**What do the instances represent?** One instance is one written response by one undergraduate student to one of five open-ended reasoning prompts (E1–E5), with its expert scores. A student contributes five instances (empty responses are kept as instances with score 0).

**How many instances are there?** 1,840 responses from 368 students at two universities in Astana (KATRU 260, ENU 108); 1,110 in the Kazakh form (222 students) and 730 in the Russian form (146 students). Of these, 43 are empty, 253 shorter than 150 characters, and 27 verbatim copies of another participant's response (flagged; excluded from the paper's models). ⟨Counts after de-identification review at release.⟩

**Does the dataset contain all possible instances or a sample?** A convenience sample of undergraduate class groups at two universities, recruited through instructors during one week (10–17 October 2026); not representative of Kazakhstan's student population (two-thirds men; faculties unevenly represented). Participation within a group was voluntary and unsupervised; no participation counts per group are available.

**What data does each instance consist of?** Raw text of the response (as typed; spelling not corrected); language of the form (kz/ru); automatically detected language of the text; prompt ID; character and word counts; scores from raters R1 and R2 (all instances) and R3 (200-instance subset); final score (0–4); adjudication flag; rater flags (off-topic, empty, suspected AI-generated, code-switching, personal data); batch ID; ingest flags (empty, short, copied_text, attention-check fail of the student); student-level metadata: student ID (row-based, not derived from any personal identifier), university (KATRU/ENU), gender (m/f/not stated), language of instruction at the student's secondary school (kz/ru/mixed).

**Is there a label or target?** Yes: `score_final` (0–4), the quality of reasoning according to rubric v1.1; individual rater scores are also released.

**Is any information missing?** R3 scores exist for 200 instances only (by design). ⟨Number⟩ responses withheld because they could not be de-identified are counted here but not released. Students who did not pass the attention check (90 of 368) are flagged, not removed. ⟨Number⟩ responses withheld because they could not be de-identified are counted here but not released.

**Are relationships between instances made explicit?** Yes: `student_id` links the five responses of a student; `university` links students to institution. The class-group field was free text, entered inconsistently, and is not released. The released train/test split is grouped by student.

**Are there recommended data splits?** `split` column: 80/20 by student, stratified by language and prompt. The paper's results use 5-fold GroupKFold by student with seeds 11/22/33 (code provided), not this split; the split is provided for comparability of future work.

**Are there errors, noise or redundancies?** Human scores carry rater disagreement (reported: QWK R1–R2 = ⟨…⟩). Some responses may be partly AI-generated (flagged by raters, not verified). Spelling and grammar errors are left as written.

**Is the dataset self-contained?** Yes. Machine translations used in the paper's translate conditions are included (`translations.csv`) with the system name and date.

**Does the dataset contain confidential data?** No. Names, institution names, place names and other identifiers found by raters were replaced by placeholders before release; the code-to-name mapping was never transferred to the research team.

**Does the dataset contain content that might be offensive or distressing?** Responses are students' opinions on information reliability, AI, jobs, and a personal episode of changing one's mind (E5). E5 responses may describe personal conflicts; raters flagged responses that disclose sensitive personal circumstances, and ⟨these were paraphrased / withheld⟩.

**Does the dataset identify subpopulations?** University, gender, form language and school language of instruction are included to enable fairness analysis.

**Is it possible to identify individuals?** Not from the released fields. The combination university × gender × five texts could in principle narrow a student within a known cohort; users must not attempt re-identification (licence condition).

**Does the dataset contain sensitive data?** Language of instruction can be read as a proxy for ethnicity in Kazakhstan; it is released because it is the primary variable of the study. No health, religious, political-opinion or precise-location data are collected.

## Collection process

**How was the data acquired?** Students typed responses into an online form (Google Forms; Russian and Kazakh versions with parallel content; the Russian form existed as two identical copies whose exports were merged) during or after a class session, on their own devices or in a computer room, after instructors distributed the link. No instruction script was read; instructions were on the form. A 150-character minimum was intended but not enforced by the form.

**What mechanisms or procedures were used?** Google Forms exports (CSV) were converted to the protocol files by `tools/forms_to_protocol_v2.py` (removal of 2 test rows and 3 repeat submissions; flags; batch assignment with seed 42); raters scored responses in `tools/annotation_tool.html`, which hides student metadata and randomises order per rater.

**If a sample, what was the sampling strategy?** Convenience sampling of class groups reachable through instructors at two universities, with the aim of at least 100 students per form language; no constraint on faculty or year of study.

**Who was involved in the collection and how were they compensated?** Instructors distributed the link as a courtesy to the project; they were not paid. Students received no compensation, grade or credit. Raters: two team members (paid under the grant) and one external teacher (⟨paid / volunteer⟩).

**Over what timeframe?** 10–17 October 2026 (collection); annotation ⟨November – December 2026⟩.

**Were any ethical review processes conducted?** ⟨Approved by the Ethics Committee of S. Seifullin Kazakh Agro-Technical Research University, protocol number and date / Exempt under the university's regulations for anonymous questionnaire research with adults, letter number and date⟩.

**Did the individuals consent?** Yes. All participants were adults (confirmed by a mandatory age question). Electronic informed consent on the first screen of the form, stating purpose, voluntariness, anonymity and the intended publication of anonymised response texts in an open research dataset; the form could not be continued without the consent checkbox. Students could simply not fill in the form; the number who declined is unknown.

**Was a mechanism provided to revoke consent?** Yes: a participant may ask the research team to withdraw their responses by quoting university, form language and their self-chosen code; withdrawals received before ⟨date⟩ were removed before release. After release, withdrawn responses are removed in the next dataset version.

**Was an analysis of the potential impact on data subjects conducted?** The team assessed re-identification risk and the sensitivity of E5; mitigations: no names, e-mails or ID numbers collected, self-chosen codes known only to participants, rater flagging of personal data, withholding of non-de-identifiable responses.

## Preprocessing / cleaning / labelling

**Was any preprocessing done?** Responses were left as written. Attention-check failures are flagged. Test rows (2) and repeat submissions (3) were removed; verbatim copies between participants (27) are flagged and excluded from annotation. Personal-data spans flagged by raters were replaced by placeholders (e.g., ⟨NAME⟩, ⟨SCHOOL⟩).

**Was the raw data saved?** Raw form exports are kept by the principal investigator and are not released (they contain the pre-de-identification texts).

**Labelling.** Rubric v1.1 (0–4; generic descriptors + prompt-specific criteria for levels 3 and 4), published with the dataset. Raters: R1 and R2 scored all responses, R3 a stratified subset of 200; calibration on 30 responses, batches of 100 in identical order, blind to each other; disagreements of ≥ 2 points adjudicated by the principal investigator blind to rater identity; disagreements of 1 point resolved as the higher score. Inter-rater reliability: ⟨QWK values⟩.

**Is the labelling software available?** Yes, `tools/annotation_tool.html` (MIT).

## Uses

**Has the dataset been used for any tasks already?** The accompanying paper (automated scoring; cross-lingual transfer RU→KZ; fairness by language; interpretability).

**What other tasks could it be used for?** Multilingual educational NLP; few-shot learning for Kazakh; studies of code-switching in student writing; development of formative-feedback tools; as evaluation data for Kazakh LLMs.

**Is there anything that might impact future uses?** The construct is "quality of reasoning as operationalised by rubric v1.1 on five specific prompts" — not a validated measure of critical thinking, and the dataset should not be used to rank students, institutions or languages. Length is correlated with score; models trained on the data may learn length. The sample is not representative.

**Are there tasks for which the dataset should not be used?** High-stakes assessment of individual students; any attempt at re-identification; claims about the critical-thinking level of Kazakhstani students, of either language group, or of school pupils (the sample is undergraduates).

## Distribution

**How will it be distributed?** Zenodo (versioned, DOI ⟨…⟩) and GitHub (⟨URL⟩).

**When?** ⟨On acceptance of the paper / at submission⟩.

**Licence?** CC BY 4.0 for data and documentation; MIT for code.

**Have any third parties imposed restrictions?** No.

## Maintenance

**Who maintains the dataset?** The AP25796141 team; contact ⟨e-mail⟩.

**Will the dataset be updated?** Errata and consent withdrawals produce new minor versions on Zenodo; the paper cites the exact version used.

**Can others contribute?** Corrections and additional annotations may be proposed via GitHub issues; the team reviews and versions them.
