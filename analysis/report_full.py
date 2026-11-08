"""
report_full.py — полный отчёт по пре-регистрированному плану (Analysis Plan, OSF SDGYE + updates 1–3).

Метрика условия: QWK по каждому вопросу (предсказания всех 5 фолдов объединены), среднее по сидам (и выборкам k),
затем среднее по 5 вопросам. 95 % CI — бутстреп по тестовым ответам (2000, стратифицировано по вопросу, seed 11),
одни и те же индексы для всех условий → парные разности. Holm (α = 0.05) по семье пре-регистрированных сравнений.
Потолок: QWK R1–R2 на тех же ответах (тот же бутстреп). H6: OLS machine ~ human + lang + lang:human + log(chars),
бутстреп по студентам (2000). Только numpy/pandas.

Запуск: python report_full.py <predictions_dir> <labels_final.csv> <out_dir>
"""
import sys, re, glob, json
from pathlib import Path
import numpy as np, pandas as pd

PRED, LABELS, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
OUT.mkdir(parents=True, exist_ok=True)
B, SEED, K = 2000, 11, 5
PROMPTS = ["E1", "E2", "E3", "E4", "E5"]
W = (np.subtract.outer(np.arange(K), np.arange(K)) ** 2) / (K - 1) ** 2

lab = pd.read_csv(LABELS, encoding="utf-8-sig").set_index("response_id")


def qwk_boot(yt, yp, idx):
    """yt, yp: (n,) int; idx: (B+1, n) — строка 0 = исходная выборка. Возвращает (B+1,) QWK."""
    codes = (yt * K + yp)[idx]
    nb = idx.shape[0]
    cm = np.bincount((codes + 25 * np.arange(nb)[:, None]).ravel(), minlength=25 * nb).reshape(nb, K, K).astype(float)
    n = cm.sum((1, 2))
    E = np.einsum("bi,bj->bij", cm.sum(2), cm.sum(1)) / n[:, None, None]
    num = (W * cm).sum((1, 2)); den = (W * E).sum((1, 2))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, 1 - num / den, np.nan)


# ---------- загрузка предсказаний ----------
runs = {}
for f in sorted(glob.glob(str(PRED / "*.csv"))):
    name = Path(f).stem
    if name.endswith(".partial"): continue
    m = re.match(r"(.+?)__(.+?)__(kz|ru)__k(\d+)__s(\d+)__seed(\d+)$", name)
    if not m: continue
    model, cond, lang, k, s, seed = m.groups()
    d = pd.read_csv(f, encoding="utf-8-sig", usecols=lambda c: c in ("response_id", "question", "y_pred", "n_chars", "student_id"))
    runs.setdefault((model, cond, lang, int(k)), []).append(d.assign(seed=int(seed), sample=int(s)))

# множество моделирования по языку = ответы xlmr mono seed 11 (1038 / 688)
base = {}
for lang in ("kz", "ru"):
    b = [d for d in runs[("xlmr", "mono", lang, 0)] if d.seed.iloc[0] == 11][0]
    b = b[["response_id", "question", "student_id", "n_chars"]].copy()
    b["y"] = b.response_id.map(lab.score_final).astype(int)
    b["r1"] = b.response_id.map(lab.score_r1); b["r2"] = b.response_id.map(lab.score_r2)
    base[lang] = b.sort_values(["question", "response_id"]).reset_index(drop=True)

# бутстреп-индексы по вопросу (стратифицировано), общие для всех условий одного языка
rng = np.random.default_rng(SEED)
bidx = {}
for lang, b in base.items():
    for q in PROMPTS:
        pos = np.where(b.question.values == q)[0]
        n = len(pos)
        bidx[(lang, q)] = (pos, np.vstack([np.arange(n), rng.integers(0, n, (B, n))]))


def cond_dist(lang, dfs, col="y_pred"):
    """(B+1,) распределение метрики условия: среднее по прогонам и вопросам."""
    b = base[lang]; per_q = []
    for q in PROMPTS:
        pos, idx = bidx[(lang, q)]
        ids = b.response_id.values[pos]; yt = b.y.values[pos]
        vals = []
        for d in dfs:
            yp = d.set_index("response_id")[col].reindex(ids)
            if yp.isna().any():  # LLM zero-shot покрывает больше ответов; недостающих быть не должно
                raise SystemExit(f"нет предсказаний для {int(yp.isna().sum())} ответов ({lang} {q})")
            vals.append(qwk_boot(yt, yp.values.astype(int), idx))
        per_q.append(np.nanmean(vals, axis=0))
    per_q = np.array(per_q)                      # (5, B+1)
    return per_q.mean(0), per_q[:, 0]


rows, dists = [], {}
for (model, cond, lang, k), dfs in sorted(runs.items()):
    tot, pq = cond_dist(lang, dfs)
    dists[(model, cond, lang, k)] = tot
    lo, hi = np.nanpercentile(tot[1:], [2.5, 97.5])
    rows.append(dict(lang=lang, model=model, condition=cond, k=k, n_runs=len(dfs), qwk=round(tot[0], 3), ci_lo=round(lo, 3), ci_hi=round(hi, 3),
                     **{q: round(v, 3) for q, v in zip(PROMPTS, pq)}))

# потолок людей на тех же ответах
for lang, b in base.items():
    per_q = []
    for q in PROMPTS:
        pos, idx = bidx[(lang, q)]
        r1 = b.r1.values[pos]; r2 = b.r2.values[pos]
        ok = ~(np.isnan(r1) | np.isnan(r2))
        if not ok.all():  # редкие пропуски: заменить отсутствующую оценку итоговой
            r1 = np.where(np.isnan(r1), b.y.values[pos], r1); r2 = np.where(np.isnan(r2), b.y.values[pos], r2)
        per_q.append(qwk_boot(r1.astype(int), r2.astype(int), idx))
    per_q = np.array(per_q); tot = per_q.mean(0)
    dists[("human", "R1_R2", lang, 0)] = tot
    lo, hi = np.nanpercentile(tot[1:], [2.5, 97.5])
    rows.append(dict(lang=lang, model="human", condition="R1_R2", k=0, n_runs=1, qwk=round(tot[0], 3), ci_lo=round(lo, 3), ci_hi=round(hi, 3),
                     **{q: round(v, 3) for q, v in zip(PROMPTS, per_q[:, 0])}))

T = pd.DataFrame(rows)
ceil = T[T.model == "human"].set_index("lang")
T["human_level"] = [bool(r.ci_hi >= ceil.loc[r.lang, "ci_lo"]) if r.model != "human" else "" for r in T.itertuples()]
T.sort_values(["lang", "model", "condition", "k"]).to_csv(OUT / "T3_all_conditions.csv", index=False, encoding="utf-8-sig")


# ---------- пре-регистрированные сравнения (Holm) ----------
def holm(p):
    p = np.asarray(p); o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0
    for i, j in enumerate(o):
        run = max(run, min(1, (m - i) * p[j])); adj[j] = run
    return adj


def compare(model, lang="kz"):
    fam = [("mono", 0, c, k) for c, k in [("zeroshot", 0), ("fewshot", 50), ("fewshot", 100), ("fewshot", 150), ("joint", 0),
                                           ("translate_test", 0), ("translate_train", 0)]] + [("fewshot", 150, "translate_test", 0)]
    out = []
    for a, ka, c, kc in fam:
        A, C_ = dists.get((model, a, lang, ka)), dists.get((model, c, lang, kc))
        if A is None or C_ is None: continue
        diff = A - C_; bs = diff[1:]
        lo, hi = np.nanpercentile(bs, [2.5, 97.5])
        p = min(1.0, 2 * min(np.mean(bs <= 0), np.mean(bs >= 0)))   # бутстреп-p (двусторонний), только для Holm
        out.append(dict(model=model, lang=lang, comparison=f"{a}{ka if ka else ''} − {c}{kc if kc else ''}", diff=round(diff[0], 3),
                        ci_lo=round(lo, 3), ci_hi=round(hi, 3), p_boot=round(p, 4)))
    D = pd.DataFrame(out)
    if len(D): D["p_holm"] = holm(D.p_boot).round(4); D["sig_holm_0.05"] = D.p_holm < 0.05
    return D


C = pd.concat([compare("xlmr", "kz"), compare("mbert", "kz"), compare("xlmr", "ru"), compare("mbert", "ru")], ignore_index=True)
C.to_csv(OUT / "T4_preregistered_comparisons_holm.csv", index=False, encoding="utf-8-sig")

# ---------- гипотезы H2–H5 (XLM-R, KZ) ----------
g = lambda key: dists[key]
mono, zs, fs150, tt = g(("xlmr", "mono", "kz", 0)), g(("xlmr", "zeroshot", "kz", 0)), g(("xlmr", "fewshot", "kz", 150)), g(("xlmr", "translate_test", "kz", 0))
mono_pq = T[(T.model == "xlmr") & (T.condition == "mono") & (T.lang == "kz")][PROMPTS].iloc[0]
gap = mono - zs; rec = (fs150 - zs) / gap
H = {
    "H2_mono_kz_per_prompt": mono_pq.to_dict(), "H2_n_prompts_ge_0.50": int((mono_pq >= 0.50).sum()), "H2_n_prompts_ge_0.60": int((mono_pq >= 0.60).sum()),
    "H2_supported": bool((mono_pq >= 0.50).sum() >= 3),
    "H3_gap_mono_minus_zs": round(gap[0], 3), "H3_ci": [round(x, 3) for x in np.nanpercentile(gap[1:], [2.5, 97.5])], "H3_supported": bool(gap[0] >= 0.15),
    "H4_recovery_fs150": round(rec[0], 3), "H4_ci": [round(x, 3) for x in np.nanpercentile(rec[1:], [2.5, 97.5])], "H4_supported": bool(rec[0] >= 0.5),
    "H5_tt_minus_fs150": round((tt - fs150)[0], 3), "H5_ci": [round(x, 3) for x in np.nanpercentile((tt - fs150)[1:], [2.5, 97.5])],
    "H5_supported": bool(abs((tt - fs150)[0]) <= 0.05),
}

# ---------- H6: справедливость по языку ----------
def fairness(model, cond, k=0, label=None):
    parts = []
    for lang in ("kz", "ru"):
        dfs = runs.get((model, cond, lang, k))
        if not dfs: return None
        avg = pd.concat(dfs).groupby("response_id").y_pred.mean()         # среднее по сидам/выборкам
        b = base[lang].copy(); b["m"] = b.response_id.map(avg); b["lang_kz"] = int(lang == "kz"); parts.append(b)
    d = pd.concat(parts, ignore_index=True).dropna(subset=["m"])
    X = np.column_stack([np.ones(len(d)), d.y, d.lang_kz, d.lang_kz * d.y, np.log(d.n_chars.clip(lower=1))])
    beta = np.linalg.lstsq(X, d.m.values, rcond=None)[0]
    studs = d.student_id.unique(); grp = {s: np.where(d.student_id.values == s)[0] for s in studs}
    r2 = np.random.default_rng(SEED); bs = []
    for _ in range(B):
        ii = np.concatenate([grp[s] for s in r2.choice(studs, len(studs))])
        bs.append(np.linalg.lstsq(X[ii], d.m.values[ii], rcond=None)[0])
    bs = np.array(bs); ci = np.percentile(bs, [2.5, 97.5], axis=0)
    names = ["intercept", "human", "lang_kz", "lang_kz_x_human", "log_chars"]
    return {"model": label or f"{model} {cond}", "n": len(d), **{f"{n}": round(beta[i], 3) for i, n in enumerate(names)},
            **{f"{n}_ci": [round(ci[0, i], 3), round(ci[1, i], 3)] for i, n in enumerate(names)},
            "lang_ci_includes_0": bool(ci[0, 2] <= 0 <= ci[1, 2])}

F = [x for x in [fairness("mbert", "joint", label="mBERT joint (best model with both languages)"), fairness("xlmr", "joint"), fairness("xlmr", "mono"),
                 fairness("mbert", "mono"), fairness("lr", "mono"), fairness("qwen_kz", "fewshot_llm", 25, "Qwen few-shot")] if x]
pd.DataFrame(F).to_csv(OUT / "T5_fairness_H6.csv", index=False, encoding="utf-8-sig")
H["H6_best_model"] = F[0]
json.dump(H, open(OUT / "hypotheses_H2_H6.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(json.dumps(H, ensure_ascii=False, indent=1))
print(T[T.lang == "kz"].sort_values("qwk", ascending=False)[["model", "condition", "k", "qwk", "ci_lo", "ci_hi", "human_level"]].to_string(index=False))
print(C.to_string(index=False))
print(pd.DataFrame(F)[["model", "lang_kz", "lang_kz_ci", "lang_kz_x_human", "lang_kz_x_human_ci", "lang_ci_includes_0"]].to_string(index=False))
