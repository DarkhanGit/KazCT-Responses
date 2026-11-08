#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
forms_to_protocol_v2.py — ingest для студенческой версии анкеты (форма v2.0, октябрь 2026).

Отличия от v1 (школьная форма): экран согласия + «18 лет», A2 = вуз/факультет, A3 = группа, A5 = язык обучения в школе;
нет класса, нет A7/A8. Идентификатор студента = порядковый номер строки выгрузки (у Google каждая отправка — своя строка),
самопридуманный код A1 хранится отдельно (self_code) и нужен только для отзыва согласия.

  python forms_to_protocol_v2.py ingest --ru "<выгрузка RU>.csv" --kz "<выгрузка KZ>.csv" --out clean_v2

Команда labels не менялась — используйте forms_to_protocol.py labels ... (ей нужны только responses.csv и экспорты экспертов).
"""
import argparse, os, re, sys, datetime as dt
from collections import Counter, defaultdict
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from forms_to_protocol import (QUESTIONS, D_KEYS, FORMATS_RU, FORMATS_KZ, GENDER, SCHOOL_LANG, INTERNET,
                               norm, group_columns, detect_lang, MIN_CHARS)

FORM_VERSION = "2.0"
TEST_CODES = {"99-99-01", "99-99-02", "99-99-03"}
UNI_PATTERNS = [("ENU", ["еңу", "еұу", "ену", "гумилев", "еуразия", "евраз", "enu"]),
                ("KATRU", ["катиу", "катзу", "кату", "сейфуллин", "агротех", "казату", "kazatu", "katu", "катру"])]


def uni_code(s):
    s = norm(s)
    for code, pats in UNI_PATTERNS:
        if any(p in s for p in pats): return code
    return "OTHER"


def faculty_of(s):
    s = str(s)
    parts = re.split(r"[,;]| - | – ", s, maxsplit=1)
    return parts[1].strip() if len(parts) > 1 else ""


def cohort_year(group):
    m = re.search(r"-(\d\d)-", str(group))
    return 2000 + int(m.group(1)) if m else np.nan


def read_form_v2(path, lang):
    df = pd.read_excel(path) if path.lower().endswith((".xlsx", ".xls")) else pd.read_csv(path, encoding="utf-8-sig")
    g = group_columns(df)
    need = ["A1", "A2", "A3", "B1", "B2", "B3", "C1", "D1", "E1", "E5"]
    miss = [k for k in need if k not in g]
    if miss: sys.exit(f"[{path}] не найдены колонки с кодами {miss}")
    cols = list(df.columns)
    ts_col = cols[0]
    consent_col = next((c for c in cols[1:4] if "согласие" in c.lower() or "келісім" in c.lower()), None)
    adult_col = next((c for c in cols[1:4] if "18" in c), None)
    if consent_col is None or adult_col is None: sys.exit(f"[{path}] не найдены колонки согласия / «18 лет» среди {cols[1:4]}")
    out = pd.DataFrame()
    out["row_no"] = np.arange(1, len(df) + 1)              # номер строки в выгрузке = идентификатор отправки
    out["submitted_at"] = pd.to_datetime(df[ts_col], errors="coerce", format="mixed", dayfirst=False)
    out["form_lang"] = lang
    out["consent"] = df[consent_col].map(lambda v: 1 if norm(v) else 0)
    out["adult"] = df[adult_col].map(lambda v: 1 if norm(v) in ("да", "иә") else 0)
    out["self_code"] = df[g["A1"][0]].astype(str).str.strip()
    out["university_raw"] = df[g["A2"][0]].astype(str).str.strip()
    out["university"] = out.university_raw.map(uni_code)
    out["faculty_raw"] = out.university_raw.map(faculty_of)
    out["group_raw"] = df[g["A3"][0]].astype(str).str.strip()
    out["group_norm"] = out.group_raw.str.upper().str.replace(r"\s+", "", regex=True)
    out["cohort_year"] = out.group_raw.map(cohort_year)
    out["gender"] = df[g["A4"][0]].map(lambda v: GENDER.get(norm(v), 0)) if "A4" in g else 0
    out["school_lang"] = df[g["A5"][0]].map(lambda v: SCHOOL_LANG.get(norm(v), "")) if "A5" in g else ""
    def inet(v):
        s = norm(v)
        for k, val in INTERNET.items():
            if k in s: return val
        return np.nan
    out["internet_hours"] = df[g["A6"][0]].map(inet) if "A6" in g else np.nan
    for i, c in enumerate(g["B1"][:12], 1): out[f"int_{i}"] = pd.to_numeric(df[c], errors="coerce")
    fm = FORMATS_RU if lang == "ru" else FORMATS_KZ
    b2 = df[g["B2"][0]].fillna("").astype(str)
    for i, f in enumerate(fm, 1): out[f"fmt_{i}"] = b2.map(lambda s: int(f.lower() in s.lower()))
    for i, c in enumerate(g["B3"][:6], 1): out[f"mot_{i}"] = pd.to_numeric(df[c], errors="coerce")
    out["b4_text"] = df[g["B4"][0]].fillna("").astype(str) if "B4" in g else ""
    for i, c in enumerate(g["C1"][:13], 1): out[f"cog_{i}"] = pd.to_numeric(df[c], errors="coerce")
    for q in ["D1", "D2", "D3", "D4", "D5", "D6"]:
        raw = df[g[q][0]].fillna("").astype(str) if q in g else ""
        out[q.lower()] = raw
        keys = [norm(k) for k in D_KEYS[q]]
        out[f"{q.lower()}_correct"] = raw.map(lambda v: int(norm(v) in keys))
    for q in QUESTIONS:
        out[f"{q.lower()}_text"] = df[g[q][0]].fillna("").astype(str).str.strip() if q in g else ""
    out["satisfaction"] = pd.to_numeric(df[g["F1"][0]], errors="coerce") if "F1" in g else np.nan
    out["f2_text"] = df[g["F2"][0]].fillna("").astype(str) if "F2" in g else ""
    return out


def cmd_ingest(a):
    frames = []
    if a.ru: frames.append(read_form_v2(a.ru, "ru"))
    if a.kz: frames.append(read_form_v2(a.kz, "kz"))
    if not frames: sys.exit("Укажите --ru и/или --kz")
    df = pd.concat(frames, ignore_index=True)
    os.makedirs(a.out, exist_ok=True)
    rep = ["# Отчёт ingest v2 — " + dt.datetime.now().strftime("%Y-%m-%d %H:%M"), "",
           f"Строк в выгрузках: {len(df)} (RU {int((df.form_lang=='ru').sum())}, KZ {int((df.form_lang=='kz').sum())})"]

    # --- идентификаторы и флаги
    df["student_id"] = df.form_lang.str.upper() + df.row_no.map(lambda n: f"{n:04d}")
    df["test_row"] = df.self_code.isin(TEST_CODES).astype(int)
    df["course"] = np.nan   # курс НЕ вычисляется: значение года в номере группы (ВТ-22-3 → 22) требует уточнения у вуза
    df["attention_fail"] = (df.cog_7 != 2).astype(int)
    df["ct_closed"] = df[[f"d{i}_correct" for i in range(1, 7)]].sum(axis=1)
    for q in QUESTIONS: df[f"{q.lower()}_nchars"] = df[f"{q.lower()}_text"].str.len()
    df["n_empty"] = sum((df[f"{q.lower()}_nchars"] == 0).astype(int) for q in QUESTIONS)
    df["n_short"] = sum((df[f"{q.lower()}_nchars"] < MIN_CHARS).astype(int) for q in QUESTIONS)
    # полный дубль: все пять E совпадают с более ранней строкой → повторная отправка
    key = df[[f"{q.lower()}_text" for q in QUESTIONS]].agg("||".join, axis=1)
    nonempty = df[[f"{q.lower()}_nchars" for q in QUESTIONS]].sum(axis=1) > 0
    df["full_duplicate"] = (key.duplicated(keep="first") & nonempty).astype(int)
    # код студента: дубли (важно только для отзыва согласия)
    df["self_code_dup"] = df.self_code.duplicated(keep=False).astype(int)
    df["form_version"] = FORM_VERSION; df["data_source"] = a.source
    df["session_id"] = df.university + "_" + df.group_norm + "_" + df.submitted_at.dt.strftime("%Y-%m-%d").fillna("nodate")

    # --- отбор
    df["excluded_reason"] = ""
    df.loc[df.test_row == 1, "excluded_reason"] = "test"
    df.loc[(df.excluded_reason == "") & (df.consent == 0), "excluded_reason"] = "no_consent"
    df.loc[(df.excluded_reason == "") & (df.adult == 0), "excluded_reason"] = "under_18"
    df.loc[(df.excluded_reason == "") & (df.full_duplicate == 1), "excluded_reason"] = "full_duplicate"
    keep = df[df.excluded_reason == ""].copy()

    rep += ["", "## Проверки",
            f"- тестовые строки (коды {sorted(TEST_CODES)}): {int(df.test_row.sum())}",
            f"- без согласия: {int((df.consent==0).sum())}; «18 лет» = нет: {int((df.adult==0).sum())}",
            f"- повторная отправка (все 5 ответов совпадают с более ранней строкой): {int(df.full_duplicate.sum())}",
            f"- после исключений: {len(keep)} студентов (RU {int((keep.form_lang=='ru').sum())}, KZ {int((keep.form_lang=='kz').sum())})",
            f"- attention_fail (cog_7 ≠ 2) среди оставшихся: {int(keep.attention_fail.sum())} — помечены, не удалены (пререгистрация)",
            f"- совпадающие самопридуманные коды: {int(keep.self_code_dup.sum())} строк — на анализ не влияет",
            f"- вуз: {dict(Counter(keep.university))}; вуз не распознан (OTHER): {sorted(keep.loc[keep.university=='OTHER','university_raw'].unique().tolist())[:10]}",
            f"- год в номере группы (cohort_year, смысл уточнить): {dict(sorted(Counter(keep.cohort_year.dropna().astype(int)).items()))}; не распознан: {int(keep.cohort_year.isna().sum())}",
            f"- язык обучения в школе: {dict(Counter(keep.school_lang))}",
            f"- пол (1 м, 2 ж, 0 не указан): {dict(Counter(keep.gender))}"]
    rep += ["", "## Закрытые задания D1–D6 (доля верных среди оставшихся)"]
    for i in range(1, 7): rep.append(f"- D{i}: {keep[f'd{i}_correct'].mean():.2f}")
    rep.append(f"- ct_closed среднее: {keep.ct_closed.mean():.2f}")

    # --- responses (длинный формат)
    rows = []
    for _, r in keep.iterrows():
        for k, q in enumerate(QUESTIONS, 1):
            t = r[f"{q.lower()}_text"]
            rows.append({"response_id": f"{r.student_id}_{q}", "student_id": r.student_id, "question": q, "level": k,
                         "text": t, "text_lang_detected": detect_lang(t), "n_chars": len(t),
                         "n_words": len(re.findall(r"\w+", t)), "form_lang": r.form_lang, "university": r.university,
                         "group_norm": r.group_norm, "school_lang": r.school_lang, "consent": r.consent, "data_source": a.source})
    resp = pd.DataFrame(rows)
    # ответ, дословно совпадающий с ответом другой строки (переписан у соседа) → флаг; в обучение идёт первый экземпляр
    seen = defaultdict(list)
    for i, t in zip(resp.index, resp.text):
        if len(t) >= 30: seen[t].append(i)
    resp["copied_text"] = 0; resp["copy_of"] = ""
    for t, idx in seen.items():
        if len(idx) > 1:
            for j in idx[1:]:
                resp.loc[j, "copied_text"] = 1; resp.loc[j, "copy_of"] = resp.loc[idx[0], "response_id"]
    resp["is_empty"] = (resp.n_chars == 0).astype(int)
    resp["short"] = ((resp.n_chars > 0) & (resp.n_chars < MIN_CHARS)).astype(int)
    # партии разметки: calib (n_calib на вопрос, оба языка поровну), затем b01.. — одинаковы для всех экспертов
    rng = np.random.default_rng(a.seed)
    resp["batch_id"] = ""
    elig = resp[(resp.is_empty == 0) & (resp.copied_text == 0)]
    calib = []
    for q in QUESTIONS:
        for lang in ("ru", "kz"):
            pool = elig[(elig.question == q) & (elig.form_lang == lang) & (elig.n_chars >= MIN_CHARS)].index.to_numpy()
            take = rng.choice(pool, size=min(a.n_calib // 2, len(pool)), replace=False) if len(pool) else []
            calib += list(take)
    resp.loc[calib, "batch_id"] = "calib"
    rest = np.array(elig.index.difference(calib).tolist()); rng.shuffle(rest)
    for i, idx in enumerate(rest): resp.loc[idx, "batch_id"] = f"b{i // a.batch_size + 1:02d}"
    resp.loc[resp.batch_id == "", "batch_id"] = "excluded"
    rep += ["", "## Открытые ответы (после исключения строк)",
            f"- всего: {len(resp)}; пустых: {int(resp.is_empty.sum())}; короче {MIN_CHARS} симв. (не пустых): {int(resp.short.sum())}; переписанных у другой строки: {int(resp.copied_text.sum())}",
            f"- язык текста (эвристика): {dict(Counter(resp.text_lang_detected))}; по форме: {dict(Counter(resp.form_lang))}",
            f"- несовпадение языка формы и текста (kz-форма→ru-текст или наоборот): {int(((resp.form_lang=='kz')&(resp.text_lang_detected=='ru')).sum())} + {int(((resp.form_lang=='ru')&(resp.text_lang_detected=='kz')).sum())}",
            "- длина (знаков, медиана) по вопросам и языкам: " + "; ".join(
                f"{q} ru {int(resp[(resp.question==q)&(resp.form_lang=='ru')].n_chars.median())} / kz {int(resp[(resp.question==q)&(resp.form_lang=='kz')].n_chars.median())}" for q in QUESTIONS),
            f"- партии разметки: {dict(sorted(Counter(resp.batch_id).items()))}",
            f"- в разметку (не excluded): {int((resp.batch_id!='excluded').sum())} ответов; из них ≥{MIN_CHARS} знаков: {int(((resp.batch_id!='excluded')&(resp.n_chars>=MIN_CHARS)).sum())}"]

    # --- sessions
    ses = keep.groupby("session_id").agg(university=("university", "first"), group=("group_norm", "first"),
                                          date=("submitted_at", lambda s: s.dt.date.min()), first_submit=("submitted_at", "min"),
                                          last_submit=("submitted_at", "max"), n_submitted=("student_id", "size"),
                                          n_attention_fail=("attention_fail", "sum")).reset_index()
    ses["form_version"] = FORM_VERSION; ses["data_source"] = a.source

    # --- survey_raw (все строки, включая исключённые, с причиной)
    cols = ["student_id", "row_no", "session_id", "form_lang", "form_version", "submitted_at", "consent", "adult", "self_code", "self_code_dup",
            "university", "university_raw", "faculty_raw", "group_raw", "group_norm", "cohort_year", "course", "gender", "school_lang", "internet_hours"] + \
           [f"int_{i}" for i in range(1, 13)] + [f"fmt_{i}" for i in range(1, 10)] + [f"mot_{i}" for i in range(1, 7)] + \
           [f"cog_{i}" for i in range(1, 14)] + [c for i in range(1, 7) for c in (f"d{i}", f"d{i}_correct")] + ["ct_closed", "attention_fail"] + \
           [f"{q.lower()}_text" for q in QUESTIONS] + [f"{q.lower()}_nchars" for q in QUESTIONS] + \
           ["n_empty", "n_short", "full_duplicate", "test_row", "excluded_reason", "satisfaction", "f2_text", "b4_text", "data_source"]
    df[cols].to_csv(os.path.join(a.out, "survey_raw.csv"), index=False, encoding="utf-8-sig")
    # students.csv — метаданные без текстов (для анализа и Datasheet)
    keep[["student_id", "form_lang", "university", "group_norm", "cohort_year", "course", "gender", "school_lang", "internet_hours",
          "attention_fail", "ct_closed", "n_empty", "n_short", "submitted_at"]].to_csv(os.path.join(a.out, "students.csv"), index=False, encoding="utf-8-sig")
    resp.to_csv(os.path.join(a.out, "responses.csv"), index=False, encoding="utf-8-sig")
    ses.to_csv(os.path.join(a.out, "sessions.csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame(columns=["response_id", "rater_id", "score", "flag", "comment", "rated_at", "batch_id"]).to_csv(
        os.path.join(a.out, "labels_expert.csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame(columns=["response_id", "score_r1", "score_r2", "score_r3", "score_final", "adjudicated", "split"]).to_csv(
        os.path.join(a.out, "labels_final.csv"), index=False, encoding="utf-8-sig")
    resp[["response_id", "question", "level", "text", "batch_id"]][resp.batch_id != "excluded"].to_csv(
        os.path.join(a.out, "for_annotation.csv"), index=False, encoding="utf-8-sig")
    rep += ["", "## Файлы", f"- {a.out}/survey_raw.csv — {len(df)} строк (все, с excluded_reason)",
            f"- {a.out}/students.csv — {len(keep)} студентов (метаданные без текстов)",
            f"- {a.out}/responses.csv — {len(resp)} ответов", f"- {a.out}/sessions.csv — {len(ses)} сеансов (вуз × группа × дата)",
            f"- {a.out}/for_annotation.csv — {int((resp.batch_id!='excluded').sum())} ответов для инструмента разметки",
            "- labels_expert.csv, labels_final.csv — пустые шаблоны (команда labels из forms_to_protocol.py)",
            "", "Правила (пререгистрация): attention_fail и short помечаются, не удаляются; copied_text — дословные повторы другой строки, "
            "в разметку и обучение идёт только первый экземпляр; полные дубли строк — повторная отправка, оставлена первая."]
    open(os.path.join(a.out, "ingest_report.md"), "w", encoding="utf-8").write("\n".join(rep))
    print("\n".join(rep))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = p.add_subparsers(dest="cmd", required=True)
    i = sp.add_parser("ingest")
    i.add_argument("--ru"); i.add_argument("--kz"); i.add_argument("--out", default="clean_v2")
    i.add_argument("--source", default="field_2026_students"); i.add_argument("--year", type=int, default=2026, help="учебный год начала (для курса)")
    i.add_argument("--n-calib", type=int, default=6, help="калибровочных ответов на вопрос (3 RU + 3 KZ)")
    i.add_argument("--batch-size", type=int, default=100); i.add_argument("--seed", type=int, default=42)
    i.set_defaults(fn=cmd_ingest)
    a = p.parse_args(); a.fn(a)
