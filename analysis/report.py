"""
report.py — таблицы и рисунки статьи из файлов предсказаний (analysis_out/predictions) и меток.

Таблицы (CSV + Markdown в analysis_out/tables/):
  T2 reliability      — QWK/exact/within-1 человек–человек: R1–R2 по корпусу, по языкам, по промптам; R1–R3, R2–R3 на подвыборке
  T3 distribution     — распределение итоговых баллов по промптам и языкам; длины ответов
  T4 monolingual      — все модели × mono по языкам: QWK [CI], MAE, exact, within-1, macro-F1, потолок, Δ до потолка, «at human level»
  T5 transfer         — XLM-R/mBERT: mono / zeroshot / fewshot-50/100/200 / joint / translate_test / translate_train на KZ (и RU — обратное)
  T6 per_prompt       — лучшая модель по промптам (QWK) + потолок по промптам
  T7 hypotheses       — H1–H6: проверка порогов из пре-регистрации
  + fairness.csv (из fairness.py), сравнения с поправкой Холма (comparisons.csv)
Рисунки (PNG+PDF в analysis_out/figures/):
  F2 score distribution; F3 QWK моделей vs потолок (mono, по языкам); F4 ЦЕНТРАЛЬНЫЙ: QWK на KZ vs число KZ-примеров по условиям;
  F5 fairness: machine − human по языкам при равном человеческом балле; F6 калибровка (ECE) если есть вероятности.
Запуск: python report.py --data <clean> --labels labels_final.csv --expert labels_expert.csv
"""
import argparse, itertools
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import common as cm

PAL = {"xlmr": "#2a78d6", "mbert": "#eb6834", "kazroberta": "#1baf7a", "lr": "#eda100", "length": "#e87ba4", "llm": "#4a3aa7", "xlmr_large": "#008300", "tiny": "#999999"}
COND_ORDER = ["mono", "zeroshot", "fewshot", "joint", "translate_test", "translate_train", "zeroshot_llm", "fewshot_llm"]
COND_LABEL = {"mono": "mono (target only)", "zeroshot": "zero-shot", "fewshot": "few-shot", "joint": "joint", "translate_test": "translate-test",
              "translate_train": "translate-train", "zeroshot_llm": "LLM zero-shot", "fewshot_llm": "LLM few-shot"}
MODEL_LABEL = {"xlmr": "XLM-R base", "mbert": "mBERT", "kazroberta": "Kaz-RoBERTa", "lr": "LR n-grams", "length": "length only", "llm": "LLM", "xlmr_large": "XLM-R large", "tiny": "tiny (smoke)"}
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#b5b4ae", "axes.labelcolor": "#52514e",
                     "xtick.color": "#52514e", "ytick.color": "#52514e", "grid.color": "#e6e5e0", "grid.linewidth": 0.6, "figure.dpi": 150, "savefig.dpi": 300})


def md_table(df: pd.DataFrame, floatfmt=".3f") -> str:
    cols = list(df.columns); lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(f"{v:{floatfmt}}" if isinstance(v, float) else str(v) for v in r.values) + " |")
    return "\n".join(lines)


def save_table(df, name, tdir, floatfmt=".3f"):
    df.to_csv(tdir / f"{name}.csv", index=False, encoding="utf-8-sig")
    (tdir / f"{name}.md").write_text(md_table(df, floatfmt), encoding="utf-8")


def agg_runs(P: pd.DataFrame, lab: pd.DataFrame, by=("model", "condition", "target_lang", "k")):
    """Среднее по seed/sample метрик + бутстреп CI по ответам (объединённым по seed) + потолок на тех же ответах."""
    rows = []
    for key, g in P.groupby(list(by)):
        per = [cm.metrics(s.y_true, s.y_pred) for _, s in g.groupby(["seed", "sample"])]
        m = {k: float(np.mean([p[k] for p in per])) for k in ("qwk", "mae", "exact", "within1", "macro_f1")}
        q, lo, hi = cm.bootstrap_ci(g.y_true.values, g.y_pred.values, seed=1)
        hc = cm.human_ceiling(lab, g.response_id.unique(), seed=1)
        rec = dict(zip(by, key)); rec.update({"n_resp": int(g.response_id.nunique()), "n_runs": len(per), **m, "qwk_lo": lo, "qwk_hi": hi,
                   "human_qwk": hc["qwk"], "human_lo": hc["lo"], "human_hi": hc["hi"], "delta_to_human": m["qwk"] - hc["qwk"],
                   "at_human_level": bool(hi >= hc["lo"]) if not np.isnan(hc["qwk"]) else None})
        if {"p0", "p1", "p2", "p3", "p4"} <= set(g.columns) and g[["p0", "p1", "p2", "p3", "p4"]].notna().all(axis=None):
            rec["ece"] = cm.ece(g.y_true.values, g[["p0", "p1", "p2", "p3", "p4"]].values)
        rows.append(rec)
    return pd.DataFrame(rows)


def reliability(lab: pd.DataFrame, df: pd.DataFrame, calib_ids=None):
    L = lab.merge(df[["response_id", "lang", "question"]], on="response_id", how="inner")
    if calib_ids is not None: L = L[~L.response_id.isin(calib_ids)]
    rows = []
    def add(name, sub, a, b):
        s = sub[[a, b]].dropna()
        if len(s) < 10: return
        q, lo, hi = cm.bootstrap_ci(s[a].astype(int), s[b].astype(int), seed=1)
        rows.append({"subset": name, "raters": f"{a[-2:].upper()}–{b[-2:].upper()}", "n": len(s), "qwk": q, "qwk_lo": lo, "qwk_hi": hi,
                     "exact": float((s[a] == s[b]).mean()), "within1": float((abs(s[a] - s[b]) <= 1).mean())})
    add("all", L, "score_r1", "score_r2")
    for lg, g in L.groupby("lang"): add(f"lang={lg}", g, "score_r1", "score_r2")
    for q, g in L.groupby("question"): add(f"prompt={q}", g, "score_r1", "score_r2")
    if "score_r3" in L:
        S = L[L.score_r3.notna()]
        for a, b in (("score_r1", "score_r3"), ("score_r2", "score_r3"), ("score_r1", "score_r2")): add("R3 subset", S, a, b)
        for lg, g in S.groupby("lang"):
            add(f"R3 subset lang={lg}", g, "score_r1", "score_r3"); add(f"R3 subset lang={lg}", g, "score_r2", "score_r3")
    return pd.DataFrame(rows)


def hypotheses(T4, T5, fair, rel):
    H = []
    r = rel[rel.subset.str.startswith("lang=")]
    H.append({"H": "H1", "statement": "human–human QWK ≥ 0.60 in both languages", "value": "; ".join(f"{a.subset}: {a.qwk:.3f}" for a in r.itertuples()),
              "supported": bool(len(r) == 2 and (r.qwk >= 0.60).all())})
    x = T5[(T5.model == "xlmr") & (T5.target_lang == "kz")]
    mono = x[x.condition == "mono"]; zs = x[x.condition == "zeroshot"]; fs200 = x[(x.condition == "fewshot") & (x.k == 200)]; tt = x[x.condition == "translate_test"]
    pp = T4[(T4.model == "xlmr") & (T4.condition == "mono") & (T4.target_lang == "kz")] if "question" in T4 else None
    H.append({"H": "H2", "statement": "XLM-R mono-KZ QWK ≥ 0.50 on ≥3 of 5 prompts (secondary ≥0.60)",
              "value": "; ".join(f"{a.question}: {a.qwk:.3f}" for a in pp.itertuples()) if pp is not None and len(pp) else "n/a",
              "supported": bool(pp is not None and len(pp) and (pp.qwk >= 0.50).sum() >= 3)})
    def val(d): return float(d.qwk.iloc[0]) if len(d) else np.nan
    H.append({"H": "H3", "statement": "zero-shot ≥ 0.15 QWK below mono-KZ", "value": f"mono {val(mono):.3f} − zeroshot {val(zs):.3f} = {val(mono)-val(zs):.3f}", "supported": bool(val(mono) - val(zs) >= 0.15)})
    gap = val(mono) - val(zs); rec = val(fs200) - val(zs)
    H.append({"H": "H4", "statement": "few-shot-200 recovers ≥ ½ of the zero-shot gap", "value": f"recovered {rec:.3f} of gap {gap:.3f}", "supported": bool(gap > 0 and rec >= 0.5 * gap)})
    H.append({"H": "H5", "statement": "translate-test within 0.05 QWK of few-shot-200", "value": f"|{val(tt):.3f} − {val(fs200):.3f}| = {abs(val(tt)-val(fs200)):.3f}", "supported": bool(abs(val(tt) - val(fs200)) <= 0.05)})
    if fair is not None and len(fair):
        best = fair.sort_values("n", ascending=False).iloc[0]
        H.append({"H": "H6", "statement": "language coefficient CI includes 0 after length control (best model)",
                  "value": f"{best.model}/{best.condition}: {best.lang_coef_len_ctrl:+.3f} [{best.ci_lo_len:+.3f}, {best.ci_hi_len:+.3f}]", "supported": bool(best.includes_zero_len)})
    return pd.DataFrame(H)


# ---------- рисунки ----------
def fig_distribution(df, fdir):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6), sharey=True)
    for ax, lg in zip(axes, cm.LANGS):
        sub = df[df.lang == lg]; ct = pd.crosstab(sub.question, sub.y, normalize="index").reindex(columns=list(cm.LEVELS), fill_value=0)
        bottom = np.zeros(len(ct)); greys = ["#cfd8e6", "#9ec5f4", "#5598e7", "#256abf", "#0d366b"]
        for j, lvl in enumerate(cm.LEVELS):
            ax.barh(ct.index, ct[lvl], left=bottom, color=greys[j], edgecolor="#fcfcfb", linewidth=1.5, label=f"score {lvl}"); bottom += ct[lvl].values
        ax.set_title(f"{lg.upper()} (n = {len(sub)})", loc="left"); ax.set_xlim(0, 1); ax.set_xlabel("share of responses"); ax.grid(axis="x"); ax.set_axisbelow(True)
    axes[0].legend(ncol=5, frameon=False, loc="upper center", bbox_to_anchor=(1.05, -0.28), fontsize=8)
    fig.tight_layout(); fig.savefig(fdir / "F2_score_distribution.png"); fig.savefig(fdir / "F2_score_distribution.pdf"); plt.close(fig)


def fig_mono(T, fdir):
    T = T[T.condition == "mono"].copy(); models = [m for m in PAL if m in set(T.model)]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), sharey=True)
    for ax, lg in zip(axes, cm.LANGS):
        sub = T[T.target_lang == lg].set_index("model").reindex(models).dropna(subset=["qwk"])
        y = np.arange(len(sub))
        ax.barh(y, sub.qwk, color=[PAL[m] for m in sub.index], height=0.55)
        ax.errorbar(sub.qwk, y, xerr=[sub.qwk - sub.qwk_lo, sub.qwk_hi - sub.qwk], fmt="none", ecolor="#0b0b0b", elinewidth=1, capsize=2)
        if sub.human_qwk.notna().any():
            h = sub.human_qwk.iloc[0]; ax.axvspan(sub.human_lo.iloc[0], sub.human_hi.iloc[0], color="#e6e5e0", zorder=0); ax.axvline(h, color="#52514e", lw=1, ls="--")
            ax.text(h, len(sub) - 0.4, f"human–human {h:.2f}", fontsize=7, color="#52514e", ha="center", va="bottom")
        for yi, v in zip(y, sub.qwk): ax.text(v + 0.01, yi, f"{v:.2f}", va="center", fontsize=7, color="#0b0b0b")
        ax.set_yticks(y); ax.set_yticklabels([MODEL_LABEL.get(m, m) for m in sub.index]); ax.set_xlim(0, 1); ax.set_xlabel("QWK (mean over folds × seeds, 95% bootstrap CI)")
        ax.set_title(f"test language: {lg.upper()}", loc="left"); ax.grid(axis="x"); ax.set_axisbelow(True)
    fig.tight_layout(); fig.savefig(fdir / "F3_monolingual_vs_ceiling.png"); fig.savefig(fdir / "F3_monolingual_vs_ceiling.pdf"); plt.close(fig)


def fig_transfer(T5, P, fdir, target="kz"):
    """Центральный рисунок: QWK на target vs число target-примеров (0 = zero-shot; 50/100/200 = few-shot; all = mono/joint)."""
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    pm = P[(P.target_lang == target) & (P.condition == "mono")]
    n_all = int(pm.groupby("question").response_id.nunique().mean() * (1 - 1 / cm.N_FOLDS)) if len(pm) else 0  # ≈ target-ответов на промпт в обучающем фолде
    xs_pos = {0: 0, 50: 1, 100: 2, 200: 3, "all": 4}
    for mi, model in enumerate([m for m in ("xlmr", "mbert", "kazroberta", "tiny") if m in set(T5.model)]):
        t = T5[(T5.model == model) & (T5.target_lang == target)]
        pts = []
        z = t[t.condition == "zeroshot"]; pts += [(0, z.qwk.iloc[0], z.qwk_lo.iloc[0], z.qwk_hi.iloc[0])] if len(z) else []
        for k in cm.FEWSHOT_K:
            f = t[(t.condition == "fewshot") & (t.k == k)]
            if len(f): pts.append((k, f.qwk.iloc[0], f.qwk_lo.iloc[0], f.qwk_hi.iloc[0]))
        if pts:
            xp = [xs_pos[p[0]] for p in pts]; ax.plot(xp, [p[1] for p in pts], "-o", color=PAL[model], lw=2, ms=5, label=f"{MODEL_LABEL[model]}: RU→{target.upper()} few-shot")
            ax.fill_between(xp, [p[2] for p in pts], [p[3] for p in pts], color=PAL[model], alpha=0.12, lw=0)
        for cond, mk in (("mono", "s"), ("joint", "D"), ("translate_test", "^"), ("translate_train", "v")):
            c = t[t.condition == cond]
            if len(c): ax.plot([xs_pos["all"] + (0.12 * ["mono", "joint", "translate_test", "translate_train"].index(cond) - 0.18)], [c.qwk.iloc[0]], mk, color=PAL[model], ms=6,
                              label=f"{MODEL_LABEL[model]}: {COND_LABEL[cond]}" if mi == 0 or cond == "mono" else None)
    hc = T5[(T5.target_lang == target) & T5.human_qwk.notna()]
    if len(hc): ax.axhspan(hc.human_lo.iloc[0], hc.human_hi.iloc[0], color="#e6e5e0", zorder=0); ax.axhline(hc.human_qwk.iloc[0], color="#52514e", lw=1, ls="--"); ax.text(4.3, hc.human_qwk.iloc[0], "human–human", fontsize=7, color="#52514e", va="bottom", ha="right")
    ax.set_xticks(list(xs_pos.values())); ax.set_xticklabels(["0\n(zero-shot)", "50", "100", "200", f"all (~{n_all})"]); ax.set_xlabel(f"number of {target.upper()} training responses per prompt")
    ax.set_ylabel("QWK on " + target.upper()); ax.set_ylim(0, 1); ax.grid(axis="y"); ax.set_axisbelow(True); ax.legend(fontsize=6.5, frameon=False, loc="lower right")
    fig.tight_layout(); fig.savefig(fdir / f"F4_transfer_{target}.png"); fig.savefig(fdir / f"F4_transfer_{target}.pdf"); plt.close(fig)


def fig_fairness(P, fdir, model="xlmr", condition="mono"):
    g = P[(P.model == model) & (P.condition == condition)]
    if g.empty or g.lang.nunique() < 2: return
    fig, ax = plt.subplots(figsize=(4.6, 2.8)); w = 0.38
    for i, lg in enumerate(cm.LANGS):
        gl = g[g.lang == lg].copy(); gl["err"] = gl.y_pred - gl.y_true
        s = gl.groupby("y_true").err.agg(m="mean", se=lambda e: e.std() / np.sqrt(len(e)))
        ax.bar(s.index + (i - 0.5) * w, s.m, width=w, color=["#2a78d6", "#eb6834"][i], yerr=1.96 * s.se, capsize=2, ecolor="#0b0b0b", error_kw={"elinewidth": 1}, label=lg.upper())
    ax.axhline(0, color="#52514e", lw=1); ax.set_xlabel("human score"); ax.set_ylabel("machine − human (mean, 95% CI)"); ax.set_xticks(list(cm.LEVELS))
    ax.set_title(f"{MODEL_LABEL.get(model, model)}, {COND_LABEL.get(condition, condition)}", loc="left"); ax.legend(frameon=False, fontsize=8); ax.grid(axis="y"); ax.set_axisbelow(True)
    fig.tight_layout(); fig.savefig(fdir / "F5_fairness_by_language.png"); fig.savefig(fdir / "F5_fairness_by_language.pdf"); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(cm.DEFAULT_DATA)); ap.add_argument("--labels", default="labels_final.csv"); ap.add_argument("--expert", default=None)
    ap.add_argument("--pred", default=str(cm.PRED_DIR)); ap.add_argument("--out", default=str(cm.OUT_DIR)); ap.add_argument("--calib_ids", default=None, help="csv с response_id калибровочных ответов (исключаются из надёжности)")
    a = ap.parse_args(); out = Path(a.out); tdir, fdir = out / "tables", out / "figures"; tdir.mkdir(parents=True, exist_ok=True); fdir.mkdir(parents=True, exist_ok=True)
    df = cm.load_data(a.data, a.labels); lab = pd.read_csv(Path(a.data) / a.labels, encoding="utf-8-sig")
    calib = set(pd.read_csv(a.calib_ids).response_id) if a.calib_ids else None
    synthetic = "SYNTHETIC" in a.labels.upper() or (("data_source" in df) and (df.data_source == "synthetic").any())
    banner = "⚠ SYNTHETIC DATA — pipeline test only, NOT results\n\n" if synthetic else ""
    # T2, T3
    rel = reliability(lab, df, calib); save_table(rel, "T2_reliability", tdir)
    dist = df.groupby(["lang", "question"]).agg(n=("y", "size"), mean_score=("y", "mean"), share_3_4=("y", lambda s: float((s >= 3).mean())),
                                                 median_chars=("n_chars", "median"), mean_chars=("n_chars", "mean")).reset_index(); save_table(dist, "T3_distribution", tdir, ".2f")
    fig_distribution(df, fdir)
    # предсказания
    P = cm.load_all_predictions(Path(a.pred)); P["sample"] = P.get("sample", 0)
    T = agg_runs(P, lab); save_table(T.sort_values(["target_lang", "condition", "model", "k"]), "T4_T5_all_conditions", tdir)
    T4 = T[T.condition == "mono"]; save_table(T4, "T4_monolingual", tdir)
    T5 = T[T.model.isin(["xlmr", "mbert", "kazroberta", "xlmr_large", "tiny", "lr", "llm"]) & T.condition.isin(COND_ORDER)]; save_table(T5, "T5_transfer", tdir)
    Tq = agg_runs(P, lab, by=("model", "condition", "target_lang", "k", "question")); save_table(Tq[Tq.condition == "mono"], "T6_per_prompt_mono", tdir)
    # сравнения с Холмом: mono vs каждое условие переноса (XLM-R, KZ) и fewshot-200 vs translate_test
    comps = []
    x = P[(P.model == "xlmr") & (P.target_lang == "kz")] if "xlmr" in set(P.model) else P[(P.model == P.model.iloc[0]) & (P.target_lang == "kz")]
    if len(x):
        base = x[(x.condition == "mono")].groupby("response_id").y_pred.agg(lambda s: int(round(s.mean())))
        for cond, k in [("zeroshot", 0), ("fewshot", 50), ("fewshot", 100), ("fewshot", 200), ("joint", 0), ("translate_test", 0), ("translate_train", 0)]:
            o = x[(x.condition == cond) & (x.k == k)].groupby("response_id").y_pred.agg(lambda s: int(round(s.mean())))
            ids = base.index.intersection(o.index)
            if len(ids) < 20: continue
            yt = x.drop_duplicates("response_id").set_index("response_id").loc[ids].y_true.values
            d, lo, hi, p = cm.paired_diff_ci(yt, base.loc[ids].values, o.loc[ids].values, seed=1)
            comps.append({"comparison": f"mono − {cond}{'-'+str(k) if k else ''}", "n": len(ids), "diff_qwk": d, "ci_lo": lo, "ci_hi": hi, "p_boot": p})
        if comps:
            C2 = pd.DataFrame(comps); C2["reject_holm"] = cm.holm(C2.p_boot.tolist()); save_table(C2, "comparisons_holm", tdir)
    fair = pd.read_csv(out / "fairness.csv", encoding="utf-8-sig") if (out / "fairness.csv").exists() else None
    H = hypotheses(Tq[Tq.condition == "mono"].rename(columns={}) if len(Tq) else T4, T5, fair, rel); save_table(H, "T7_hypotheses", tdir)
    fig_mono(T, fdir); fig_transfer(T5, P, fdir, "kz"); fig_transfer(T5, P, fdir, "ru"); fig_fairness(P, fdir, model=("xlmr" if "xlmr" in set(P.model) else P.model.iloc[0]))
    (out / "REPORT.md").write_text(banner + "# Tables\n\n" + "\n\n".join(f"## {p.stem}\n\n{p.read_text(encoding='utf-8')}" for p in sorted(tdir.glob("*.md"))) +
                                   "\n\n# Figures\n\n" + "\n".join(f"- {p.name}" for p in sorted(fdir.glob("*.png"))), encoding="utf-8")
    print(banner + f"tables -> {tdir}\nfigures -> {fdir}\nREPORT.md -> {out/'REPORT.md'}")
    print(H.to_string(index=False))


if __name__ == "__main__":
    main()
