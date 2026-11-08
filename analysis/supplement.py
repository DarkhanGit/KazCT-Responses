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
ai_ids = set(exp[exp.flag.fillna("").str.contains("ai_suspect")].response_id)


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
# F3: learning curve
fig, ax = plt.subplots(figsize=(5.2, 3.6))
xs = [0, 50, 100, 150, 200]; xl = ["0\n(zero-shot)", "50", "100", "150", "all\n(mono)"]
def curve(model, zs, fs, mono):
    v = [T[(T.lang == "kz") & (T.model == model) & (T.condition == zs)].qwk.values]
    for k in (50, 100, 150): v.append(T[(T.lang == "kz") & (T.model == model) & (T.condition == fs) & (T.k == k)].qwk.values)
    v.append(T[(T.lang == "kz") & (T.model == model) & (T.condition == mono)].qwk.values)
    return [x[0] if len(x) else np.nan for x in v]
ax.plot(xs, curve("xlmr", "zeroshot", "fewshot", "mono"), "o-", color="#d62728", label="XLM-R (RU → k KZ)")
ax.plot(xs, curve("mbert", "zeroshot", "fewshot", "mono"), "o-", color="#1f77b4", label="mBERT (RU → k KZ)")
ax.plot(xs[1:], curve("kazroberta", "zeroshot", "mono_k", "mono")[1:], "s--", color="#2ca02c", label="Kaz-RoBERTa (k KZ only)")
for m, col, lbl in (("mbert", "#1f77b4", "mBERT joint"), ("xlmr", "#d62728", "XLM-R joint")):
    ax.axhline(T[(T.lang == "kz") & (T.model == m) & (T.condition == "joint")].qwk.values[0], color=col, ls=":", lw=1, label=lbl)
ax.axhline(T[(T.lang == "kz") & (T.model == "human")].qwk.values[0], color="k", lw=1.4, label="Human ceiling")
ax.set_xticks(xs); ax.set_xticklabels(xl); ax.set_xlabel("Kazakh training examples per prompt"); ax.set_ylabel("QWK (Kazakh test folds)"); ax.set_ylim(0, 1); ax.grid(alpha=.3); ax.legend(fontsize=7, loc="lower right")
fig.tight_layout(); fig.savefig(OUT / "F3_learning_curve.png", dpi=300); fig.savefig(OUT / "F3_learning_curve.pdf"); plt.close(fig)
print("figures saved")
