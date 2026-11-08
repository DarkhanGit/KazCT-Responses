"""Results 5.1–5.2: корпус и надёжность человеческих оценок. Только описательная статистика по labels_final;
никаких моделей. Запуск: python results_5_1_5_2.py <папка clean> <папка вывода>"""
import sys, re, json
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
CL, OUT = Path(sys.argv[1]), Path(sys.argv[2]); OUT.mkdir(parents=True, exist_ok=True)
B, SEED = 2000, 11
def qwk(a, b, k=5):
    a = np.asarray(a, int); b = np.asarray(b, int); O = np.zeros((k, k))
    np.add.at(O, (a, b), 1)
    W = np.array([[(i - j) ** 2 for j in range(k)] for i in range(k)]) / (k - 1) ** 2
    E = np.outer(O.sum(1), O.sum(0)) / len(a); d = (W * E).sum()
    return 1 - (W * O).sum() / d if d > 0 else np.nan
def boot(f, a, b, groups=None):
    """бутстреп по студентам (группам), как в пререгистрации"""
    rng = np.random.default_rng(SEED); a = np.asarray(a); b = np.asarray(b)
    if groups is None: groups = np.arange(len(a))
    g = pd.Series(np.arange(len(a))).groupby(np.asarray(groups)).apply(list).tolist()
    vals = []
    for _ in range(B):
        idx = np.concatenate([g[i] for i in rng.integers(0, len(g), len(g))])
        vals.append(f(a[idx], b[idx]))
    return np.nanpercentile(vals, [2.5, 97.5])
def spearman(x, y):
    return pd.Series(x).rank().corr(pd.Series(y).rank())
def stats(a, b, groups):
    a = np.asarray(a, int); b = np.asarray(b, int); lo, hi = boot(qwk, a, b, groups)
    return {"n": len(a), "QWK": qwk(a, b), "CI_low": lo, "CI_high": hi, "exact": (a == b).mean(),
            "within1": (np.abs(a - b) <= 1).mean(), "mean_A": a.mean(), "mean_B": b.mean()}

resp = pd.read_csv(CL / "responses.csv", encoding="utf-8-sig")
lab = pd.read_csv(CL / "labels_final.csv", encoding="utf-8-sig")
ex = pd.read_csv(CL / "labels_expert.csv", encoding="utf-8-sig")
stu = pd.read_csv(CL / "students.csv", encoding="utf-8-sig")
d = resp.merge(lab, on="response_id", how="left")
d["no_letters"] = d.text.fillna("").map(lambda t: re.search(r"\w", str(t)) is None)
d["model_set"] = d.score_final.notna() & ~d.no_letters & (d.copied_text != 1)
d["lang"] = d.form_lang
M = d[d.model_set].copy(); M["y"] = M.score_final.astype(int)
rep = {}

# ---------------- 5.1 корпус
t0 = pd.DataFrame({
    "students": stu.groupby("form_lang").student_id.nunique(),
    "KATRU": stu[stu.university == "KATRU"].groupby("form_lang").size(),
    "ENU": stu[stu.university == "ENU"].groupby("form_lang").size(),
    "responses_all": d.groupby("lang").size(),
    "empty_or_no_letters": d[d.no_letters].groupby("lang").size(),
    "copied_text": d[d.copied_text == 1].groupby("lang").size(),
    "modelling_set": M.groupby("lang").size(),
    "short_lt150": M[M.n_chars < 150].groupby("lang").size(),
}).fillna(0).astype(int)
t0.loc["total"] = t0.sum(); t0.to_csv(OUT / "T0_corpus.csv", encoding="utf-8-sig")
L = M.groupby(["question", "lang"]).n_chars.describe(percentiles=[.25, .5, .75])[["count", "25%", "50%", "75%"]].round(0)
L.to_csv(OUT / "T0b_length_by_prompt_lang.csv", encoding="utf-8-sig")
# T1: распределение итоговых баллов по вопросу × языку (множество моделирования)
T1 = pd.crosstab([M.question, M.lang], M.y)
T1p = (T1.div(T1.sum(1), axis=0) * 100).round(1)
T1.assign(n=T1.sum(1), mean=M.groupby(["question", "lang"]).y.mean().round(2)).to_csv(OUT / "T1_scores_counts.csv", encoding="utf-8-sig")
T1p.to_csv(OUT / "T1_scores_percent.csv", encoding="utf-8-sig")
tot = pd.crosstab(M.lang, M.y); tot.to_csv(OUT / "T1_scores_by_lang.csv", encoding="utf-8-sig")
rep["scores_total"] = M.y.value_counts().sort_index().to_dict()
rep["scores_by_lang_mean"] = M.groupby("lang").y.mean().round(3).to_dict()
rep["len_score_spearman"] = {l: round(spearman(g.n_chars, g.y), 3) for l, g in M.groupby("lang")}
rep["len_score_spearman_all"] = round(spearman(M.n_chars, M.y), 3)
# флаги экспертов
fl = ex[ex.comment.astype(str) != "rule: no letters/digits"].copy(); fl["flag"] = fl.flag.fillna("")
fl = fl.merge(resp[["response_id", "form_lang"]], on="response_id")
flags = {}
for f_ in ["ai_suspect", "offtopic", "lang_switch", "pii", "empty"]:
    has = fl[fl.flag.str.contains(f_)]
    flags[f_] = {"R1": int((has.rater_id == "R1").sum()), "R2": int((has.rater_id == "R2").sum()), "R3": int((has.rater_id == "R3").sum()),
                 "any_R1R2_responses": int(has[has.rater_id.isin(["R1", "R2"])].response_id.nunique()),
                 "both_R1R2_responses": int((has[has.rater_id.isin(["R1", "R2"])].groupby("response_id").rater_id.nunique() == 2).sum())}
rep["flags"] = flags
ai_any = fl[fl.flag.str.contains("ai_suspect") & fl.rater_id.isin(["R1", "R2"])].response_id.unique()
rep["ai_suspect_by_lang"] = resp[resp.response_id.isin(ai_any)].form_lang.value_counts().to_dict()

# ---------------- 5.2 надёжность (только реально оценённые, без calib)
w = ex[ex.comment.astype(str) != "rule: no letters/digits"].pivot_table(index="response_id", columns="rater_id", values="score", aggfunc="first")
bt = ex.drop_duplicates("response_id").set_index("response_id").batch_id
w = w.join(bt).join(resp.set_index("response_id")[["form_lang", "question", "student_id"]])
rows = []
def add(name, sub, x, y):
    s = stats(sub[x], sub[y], sub.student_id); s["pair"] = f"{x}-{y}"; s["subset"] = name; rows.append(s)
both = w.dropna(subset=["R1", "R2"]); main = both[both.batch_id != "calib"]
add("all (excl. calibration)", main, "R1", "R2")
for l in ["kz", "ru"]: add(f"language={l}", main[main.form_lang == l], "R1", "R2")
for q in sorted(main.question.unique()): add(f"prompt={q}", main[main.question == q], "R1", "R2")
add("calibration set (30)", both[both.batch_id == "calib"], "R1", "R2")
r3 = w.dropna(subset=["R3"]); r3m = r3[r3.batch_id != "calib"]
for x in ["R1", "R2"]:
    add("R3 subset (200, excl. calibration)", r3m, x, "R3")
    for l in ["kz", "ru"]: add(f"R3 subset, language={l}", r3m[r3m.form_lang == l], x, "R3")
T2 = pd.DataFrame(rows)[["pair", "subset", "n", "QWK", "CI_low", "CI_high", "exact", "within1", "mean_A", "mean_B"]].round(3)
T2.to_csv(OUT / "T2_human_agreement.csv", index=False, encoding="utf-8-sig")
# среднее попарное QWK трёх экспертов на подвыборке R3
pw = [qwk(r3m[a], r3m[b]) for a, b in [("R1", "R2"), ("R1", "R3"), ("R2", "R3")]]
rep["R3_subset_pairwise"] = {"R1-R2": round(pw[0], 3), "R1-R3": round(pw[1], 3), "R2-R3": round(pw[2], 3), "mean": round(float(np.mean(pw)), 3)}
cm = pd.crosstab(main.R1.astype(int), main.R2.astype(int)); cm.to_csv(OUT / "T2b_confusion_R1xR2.csv", encoding="utf-8-sig")
rep["diff_R1_minus_R2"] = (main.R1 - main.R2).value_counts().sort_index().astype(int).to_dict()
rep["adjudicated"] = int(lab.adjudicated.sum())
adj = lab[lab.adjudicated == 1]; rep["adjudicated_final"] = adj.score_final.astype(int).value_counts().sort_index().to_dict()
rep["final_rule_higher_of_two"] = int(((lab.adjudicated == 0) & lab.score_r1.notna() & (lab.score_r1 != lab.score_r2)).sum())
# смещение по длине у экспертов: разница R1-R2 vs длина
main2 = main.join(resp.set_index("response_id").n_chars)
rep["len_vs_rater_diff_spearman"] = round(spearman(main2.n_chars, main2.R1 - main2.R2), 3)
# H1
h1 = {l: T2[(T2.pair == "R1-R2") & (T2.subset == f"language={l}")].iloc[0] for l in ["kz", "ru"]}
rep["H1"] = {l: {"QWK": float(v.QWK), "CI": [float(v.CI_low), float(v.CI_high)], "meets_0.60": bool(v.QWK >= 0.60)} for l, v in h1.items()}
rep["H1_supported"] = all(x["meets_0.60"] for x in rep["H1"].values())
json.dump(rep, open(OUT / "summary_5_1_5_2.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)

# ---------------- F1: распределение итоговых баллов по вопросу и языку
fig, axes = plt.subplots(1, 5, figsize=(13, 3.2), sharey=True)
cols = {"kz": "#2b6cb0", "ru": "#c05621"}
for ax, q in zip(axes, sorted(M.question.unique())):
    for j, l in enumerate(["kz", "ru"]):
        p = T1p.loc[(q, l)].reindex(range(5), fill_value=0)
        ax.bar(np.arange(5) + (j - 0.5) * 0.38, p.values, width=0.38, color=cols[l], label={"kz": "Kazakh", "ru": "Russian"}[l])
    ax.set_title(q); ax.set_xticks(range(5)); ax.set_xlabel("final score")
    ax.spines[["top", "right"]].set_visible(False)
axes[0].set_ylabel("% of responses"); axes[0].legend(frameon=False, fontsize=8)
fig.tight_layout(); fig.savefig(OUT / "F1_score_distribution.png", dpi=300); fig.savefig(OUT / "F1_score_distribution.pdf")
print(json.dumps(rep, ensure_ascii=False, indent=1, default=str)); print(t0.to_string()); print(T2.to_string()); print(L.to_string()); print(T1p.to_string())
