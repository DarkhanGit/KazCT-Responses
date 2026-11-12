"""
supplement.py (07.11.2026) — дополнительные пре-регистрированные анализы и рисунки для статьи 3.
 S1  sensitivity: QWK ключевых условий (а) без студентов с провалом attention-check, (б) без ответов с флагом
     ai_suspect (любой эксперт), (в) по языку ФОРМЫ вместо детектированного языка (KZ-форма включает 28 RU-ответов — они
     и так в KZ-тесте: лог проверяет, сколько их) — на тех же бутстреп-индексах, что и основной отчёт (seed 11).
 S2  fairness secondary: модель H6 + пол, вуз, язык школы (референс: мужской, KATRU, kz) для mBERT joint и XLM-R joint.
 S3  рисунки: F2 — QWK по вопросам (mono-модели и LLM vs потолок), F3 — кривая обучения по числу KZ-примеров.
 S4  LLM: QWK KazLLM few-shot KZ без 78 ответов без цифры (верхняя граница).
Запуск: python supplement.py <predictions_dir> <clean_dir> <report_dir> <out_dir>
"""
import sys, re, glob, json
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

PRED, CLEAN, REP, OUT = (Path(x) for x in sys.argv[1:5]); OUT.mkdir(parents=True, exist_ok=True)
B, SEED, K, P = 2000, 11, 5, ["E1", "E2", "E3", "E4", "E5"]
W = (np.subtract.outer(np.arange(K), np.arange(K)) ** 2) / (K - 1) ** 2
lab = pd.read_csv(CLEAN / "labels_final.csv", encoding="utf-8-sig").set_index("response_id")
stu = pd.read_csv(CLEAN / "students.csv", encoding="utf-8-sig").set_index("student_id")
resp = pd.read_csv(CLEAN / "responses.csv", encoding="utf-8-sig").set_index("response_id")
exp = pd.read_csv(CLEAN / "labels_expert.csv", encoding="utf-8-sig")
assert {"R1", "R2"} <= set(exp.rater_id), "labels_expert.csv must contain both main raters (use the master clean folder)"
ai_ids = set(exp[exp.rater_id.isin(["R1", "R2"]) & exp.flag.fillna("").str.contains("ai_suspect")].response_id)  # either main rater


def qwk1(yt, yp):
    cm = np.zeros((K, K)); np.add.at(cm, (yt, yp), 1); E = np.outer(cm.sum(1), cm.sum(0)) / cm.sum()
    d = (W * E).sum(); return 1 - (W * cm).sum() / d if d > 0 else np.nan


def load(model, cond, lang, k=0):
    out = []
    for f in glob.glob(str(PRED / f"{model}__{cond}__{lang}__k{k}__s*__seed*.csv")):
        if f.endswith(".partial.csv"): continue
        d = pd.read_csv(f, encoding="utf-8-sig", usecols=["response_id", "question", "y_pred"]); out.append(d)
    return out


base = {}
for lang in ("kz", "ru"):
    b = load("xlmr", "mono", lang)[0][["response_id", "question"]].drop_duplicates().copy()
    b["y"] = b.response_id.map(lab.score_final).astype(int); b["student_id"] = b.response_id.map(resp.student_id)
    b["att_fail"] = b.student_id.map(stu.attention_fail).astype(int); b["ai"] = b.response_id.isin(ai_ids).astype(int)
    b["form_lang"] = b.response_id.map(resp.form_lang); b["det_lang"] = b.response_id.map(resp.text_lang_detected)
    base[lang] = b.sort_values(["question", "response_id"]).reset_index(drop=True)
print("KZ-тест: язык формы", base["kz"].form_lang.value_counts().to_dict(), "| детектированный", base["kz"].det_lang.value_counts().to_dict())
print("RU-тест: язык формы", base["ru"].form_lang.value_counts().to_dict())


def cond_qwk(lang, dfs, mask=None):
    """точечная оценка + CI (бутстреп по ответам внутри вопроса) для подмножества mask (bool Series по base[lang])."""
    b = base[lang] if mask is None else base[lang][mask.values].reset_index(drop=True)
    rng = np.random.default_rng(SEED); per = []
    for q in P:
        pos = np.where(b.question.values == q)[0]; n = len(pos)
        if n < 10: continue
        idx = np.vstack([np.arange(n), rng.integers(0, n, (B, n))])
        ids = b.response_id.values[pos]; yt = b.y.values[pos]; vals = []
        for d in dfs:
            yp = d.set_index("response_id").y_pred.reindex(ids).values.astype(int)
            codes = (yt * K + yp)[idx]; nb = idx.shape[0]
            cm = np.bincount((codes + 25 * np.arange(nb)[:, None]).ravel(), minlength=25 * nb).reshape(nb, K, K).astype(float)
            E = np.einsum("bi,bj->bij", cm.sum(2), cm.sum(1)) / cm.sum((1, 2))[:, None, None]
            den = (W * E).sum((1, 2)); vals.append(np.where(den > 0, 1 - (W * cm).sum((1, 2)) / den, np.nan))
        per.append(np.nanmean(vals, 0))
    tot = np.array(per).mean(0); lo, hi = np.nanpercentile(tot[1:], [2.5, 97.5])
    return round(tot[0], 3), round(lo, 3), round(hi, 3), len(b)


# ---------- S1 sensitivity ----------
KEY = [("mbert", "joint", 0), ("kazroberta", "mono", 0), ("mbert", "mono", 0), ("xlmr", "mono", 0), ("xlmr", "joint", 0), ("lr", "mono", 0), ("qwen_kz", "fewshot_llm", 25)]
rows = []
for lang in ("kz", "ru"):
    b = base[lang]
    subsets = {"all": None, "no_attention_fail": b.att_fail == 0, "no_ai_suspect": b.ai == 0,
               "detected_lang_only": b.det_lang == lang}
    for model, cond, k in KEY:
        dfs = load(model, cond, lang, k)
        if not dfs: continue
        for name, m in subsets.items():
            q, lo, hi, n = cond_qwk(lang, dfs, m)
            rows.append(dict(lang=lang, model=model, condition=cond, subset=name, n=n, qwk=q, ci_lo=lo, ci_hi=hi))
S1 = pd.DataFrame(rows); S1.to_csv(OUT / "S1_sensitivity.csv", index=False, encoding="utf-8-sig")
print(S1.pivot_table(index=["lang", "model", "condition"], columns="subset", values="qwk").round(3))

# ---------- S2 fairness secondary ----------
def fair2(model, cond, k=0):
    parts = []
    for lang in ("kz", "ru"):
        dfs = load(model, cond, lang, k); avg = pd.concat(dfs).groupby("response_id").y_pred.mean()
        b = base[lang].copy(); b["m"] = b.response_id.map(avg); b["lang_kz"] = int(lang == "kz"); parts.append(b)
    d = pd.concat(parts, ignore_index=True).dropna(subset=["m"])
    d["n_chars"] = d.response_id.map(resp.n_chars); g = d.student_id.map(stu.gender); u = d.student_id.map(stu.university); s = d.student_id.map(stu.school_lang)
    cols = {"human": d.y, "lang_kz": d.lang_kz, "lang_kz_x_human": d.lang_kz * d.y, "log_chars": np.log(d.n_chars.clip(lower=1)),
            "female": (g == 2).astype(int), "gender_not_stated": (g == 0).astype(int), "ENU": (u == "ENU").astype(int),
            "school_ru": (s == "ru").astype(int), "school_mixed": (s == "mixed").astype(int)}
    X = np.column_stack([np.ones(len(d))] + [v.values for v in cols.values()]); y = d.m.values
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    studs = d.student_id.unique(); grp = {st: np.where(d.student_id.values == st)[0] for st in studs}; r = np.random.default_rng(SEED); bs = []
    for _ in range(B):
        ii = np.concatenate([grp[st] for st in r.choice(studs, len(studs))]); bs.append(np.linalg.lstsq(X[ii], y[ii], rcond=None)[0])
    ci = np.percentile(np.array(bs), [2.5, 97.5], axis=0); names = ["intercept"] + list(cols)
    return [dict(model=f"{model} {cond}", term=n, coef=round(beta[i], 3), ci_lo=round(ci[0, i], 3), ci_hi=round(ci[1, i], 3), includes_0=bool(ci[0, i] <= 0 <= ci[1, i])) for i, n in enumerate(names)]

S2 = pd.DataFrame(fair2("mbert", "joint") + fair2("xlmr", "joint") + fair2("kazroberta", "mono") if False else fair2("mbert", "joint") + fair2("xlmr", "joint"))
S2.to_csv(OUT / "S2_fairness_secondary.csv", index=False, encoding="utf-8-sig"); print(S2.to_string(index=False))

# ---------- S4 KazLLM without no-digit ----------
f = glob.glob(str(PRED / "kazllm_supp__fewshot_llm__kz__k25__s0__seed11.csv"))[0]
d = pd.read_csv(f, encoding="utf-8-sig"); d = d[d.response_id.isin(base["kz"].response_id)]
nd = ~d.raw_output.fillna("").astype(str).str.contains(r"[0-4]")
q_all = np.mean([qwk1(g.y_true.values, g.y_pred.values) for _, g in d.groupby("question")])
q_wo = np.mean([qwk1(g.y_true.values, g.y_pred.values) for _, g in d[~nd].groupby("question")])
json.dump({"kazllm_fewshot_kz_all": round(q_all, 3), "n_no_digit": int(nd.sum()), "kazllm_fewshot_kz_without_no_digit": round(q_wo, 3)}, open(OUT / "S4_kazllm_nodigit.json", "w"), indent=1)
print("S4:", round(q_all, 3), int(nd.sum()), round(q_wo, 3))

# ---------- S5 training-set sizes, few-shot range over samples ----------
def qmean(d):
    return float(np.mean([qwk1(base_y[g.response_id.values[0][:2].lower()].reindex(g.response_id).values, g.y_pred.values.astype(int)) for _, g in d.groupby("question")]))
base_y = {lang: base[lang].set_index("response_id").y for lang in base}
S5 = {}
for lang in ("kz", "ru"):
    f0 = glob.glob(str(PRED / f"mbert__mono__{lang}__k0__s0__seed11.csv"))[0]
    d0 = pd.read_csv(f0, encoding="utf-8-sig"); d0 = d0[d0.response_id.isin(base[lang].response_id)]
    n_q = d0.groupby("question").size(); test = d0.groupby(["question", "fold"]).size()
    fold_train = (n_q.reindex(test.index.get_level_values(0)).values - test.values)
    stud_train = d0.student_id.nunique() * 0.8; val_share = np.ceil(0.1 * stud_train) / stud_train
    S5[lang] = dict(n_per_prompt_min=int(n_q.min()), n_per_prompt_max=int(n_q.max()), train_fold_mean=round(float(fold_train.mean()), 1), train_fold_min=int(fold_train.min()),
                    train_fold_max=int(fold_train.max()), val_mean=round(float(fold_train.mean() * val_share), 1), fit_mean=round(float(fold_train.mean() * (1 - val_share)), 1))
rng_rows = []
for model, cond in (("xlmr", "fewshot"), ("mbert", "fewshot"), ("kazroberta", "mono_k")):
    for lang in ("kz", "ru"):
        for k in (50, 100, 150):
            per_sample = []
            for smp in (0, 1, 2):
                fs = [f for f in glob.glob(str(PRED / f"{model}__{cond}__{lang}__k{k}__s{smp}__seed*.csv")) if not f.endswith(".partial.csv")]
                if not fs: continue
                vals = []
                for f in fs:
                    d = pd.read_csv(f, encoding="utf-8-sig", usecols=["response_id", "question", "y_pred"]); d = d[d.response_id.isin(base[lang].response_id)]
                    yt = base_y[lang].reindex(d.response_id).values; d = d.assign(y=yt)
                    vals.append(np.mean([qwk1(g.y.values.astype(int), g.y_pred.values.astype(int)) for _, g in d.groupby("question")]))
                per_sample.append(np.mean(vals))
            if per_sample: rng_rows.append(dict(model=model, condition=cond, lang=lang, k=k, n_samples=len(per_sample), mean=round(float(np.mean(per_sample)), 3), min=round(float(min(per_sample)), 3), max=round(float(max(per_sample)), 3)))
S5R = pd.DataFrame(rng_rows); S5R.to_csv(OUT / "S5_fewshot_range_over_samples.csv", index=False, encoding="utf-8-sig")
json.dump(S5, open(OUT / "S5_training_set_sizes.json", "w"), indent=1); print(json.dumps(S5, indent=1)); print(S5R.to_string(index=False))
ct = pd.crosstab(stu.form_lang, stu.school_lang); ct.to_csv(OUT / "S6_form_language_by_school_language.csv", encoding="utf-8-sig"); print(ct)
print("AI-suspect (either main rater) in test sets:", {l: int(base[l].ai.sum()) for l in base})

# ---------- S3 figures ----------
T = pd.read_csv(REP / "T3_all_conditions.csv", encoding="utf-8-sig")
plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans"})
# F2: per-prompt monolingual
fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
series = [("human", "R1_R2", 0, "Human ceiling (R1–R2)", "k", "-"), ("mbert", "joint", 0, "mBERT joint", "#1f77b4", "-"), ("kazroberta", "mono", 0, "Kaz-RoBERTa mono", "#2ca02c", "-"),
          ("mbert", "mono", 0, "mBERT mono", "#1f77b4", "--"), ("lr", "mono", 0, "LR n-grams", "#7f7f7f", "-"), ("xlmr", "mono", 0, "XLM-R mono", "#d62728", "--"),
          ("qwen_kz", "fewshot_llm", 25, "Qwen3.5 five-shot", "#9467bd", ":"), ("length", "mono", 0, "Length only", "#7f7f7f", ":")]
for ax, lang, title in zip(axes, ("kz", "ru"), ("Kazakh test folds", "Russian test folds")):
    for m, c, k, label, col, ls in series:
        r = T[(T.lang == lang) & (T.model == m) & (T.condition == c) & (T.k == k)]
        if len(r): ax.plot(P, r[P].values[0], ls, color=col, marker="o", ms=3, label=label, lw=1.6 if m in ("human", "mbert") else 1.2)
    ax.set_title(title); ax.set_ylim(0, 1); ax.grid(alpha=.3)
axes[0].set_ylabel("QWK"); h, l = axes[0].get_legend_handles_labels(); fig.legend(h, l, fontsize=7, loc="lower center", ncol=4, frameon=False)
fig.tight_layout(rect=(0, 0.1, 1, 1)); fig.savefig(OUT / "F2_per_prompt.png", dpi=300); fig.savefig(OUT / "F2_per_prompt.pdf"); plt.close(fig)
# F3: learning curve (v2, 12.11.2026: CI error bars, ceiling as a band, length-only zero-shot reference, legend outside)
fig, ax = plt.subplots(figsize=(6.2, 3.9))
xs = [0, 50, 100, 150, 200]; xl = ["0\n(zero-shot)", "50", "100", "150", "all (mono):\n≈150 fitted\n+ ≈17 validation"]
def sel(model, cond, k=None):
    r = T[(T.lang == "kz") & (T.model == model) & (T.condition == cond)]
    if k is not None: r = r[r.k == k]
    return r.iloc[0] if len(r) else None
def curve(model, zs, fs, mono):
    rows = [sel(model, zs)] + [sel(model, fs, k) for k in (50, 100, 150)] + [sel(model, mono)]
    v = np.array([r.qwk if r is not None else np.nan for r in rows]); lo = np.array([r.ci_lo if r is not None else np.nan for r in rows]); hi = np.array([r.ci_hi if r is not None else np.nan for r in rows])
    return v, v - lo, hi - v
for model, zs, fs, col, mk, ls, lbl, sl in (("xlmr", "zeroshot", "fewshot", "#d62728", "o", "-", "XLM-R (RU, then k KZ)", slice(0, 5)), ("mbert", "zeroshot", "fewshot", "#1f77b4", "^", "-", "mBERT (RU, then k KZ)", slice(0, 5)),
                                         ("kazroberta", "zeroshot", "mono_k", "#2ca02c", "s", "--", "Kaz-RoBERTa (k KZ only)", slice(1, 5))):
    v, el, eh = curve(model, zs, fs, "mono")
    ax.errorbar(np.array(xs)[sl], v[sl], yerr=[el[sl], eh[sl]], fmt=mk + ls, color=col, ms=5, lw=1.4, capsize=2.5, label=lbl)
r = sel("length", "zeroshot"); ax.errorbar([0], [r.qwk], yerr=[[r.qwk - r.ci_lo], [r.ci_hi - r.qwk]], fmt="D", color="#555555", ms=5, capsize=2.5, label="Length-only LR (trained on RU)")
for m, col, ls, lbl in (("mbert", "#1f77b4", ":", "mBERT joint (RU + all KZ)"), ("xlmr", "#d62728", "-.", "XLM-R joint (RU + all KZ)")):
    ax.axhline(sel(m, "joint").qwk, color=col, ls=ls, lw=1.1, label=lbl)
h = sel("human", "R1_R2"); ax.axhspan(h.ci_lo, h.ci_hi, color="0.8", alpha=.6, lw=0); ax.axhline(h.qwk, color="k", lw=1.3, label="Human ceiling (R1–R2) with 95% CI")
ax.set_xticks(xs); ax.set_xticklabels(xl); ax.set_xlabel("Kazakh training examples per prompt"); ax.set_ylabel("QWK (Kazakh test folds)"); ax.set_ylim(0, 1); ax.set_xlim(-12, 215); ax.grid(alpha=.3)
ax.legend(fontsize=7, loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
fig.tight_layout(); fig.savefig(OUT / "F3_learning_curve.png", dpi=600); fig.savefig(OUT / "F3_learning_curve.pdf"); plt.close(fig)
print("figures saved")
