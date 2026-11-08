"""
fairness.py — справедливость по языку ответа (Methods 4.5, гипотеза H6).

Для каждого файла предсказаний (модель × условие с target=kz и target=ru объединяются по модели/условию/seed):
  (a) метрики отдельно по языкам: QWK, MAE, средняя знаковая ошибка (machine − human);
  (b) линейная модель  y_pred ~ y_true + lang + y_true:lang            → коэффициент lang;
  (c) та же модель + log_len                                            → коэффициент lang при контроле длины;
      CI коэффициентов — bootstrap по ученикам (2000), а не по ответам (ответы одного ученика коррелированы);
  (d) ECE по языкам (если есть p0..p4);
  (e) то же для gender и grade как вторичные проверки (если колонки есть в survey_raw.csv).
Выход: analysis_out/fairness.csv и fairness_by_lang.csv
"""
import argparse
from pathlib import Path
import numpy as np, pandas as pd
import statsmodels.formula.api as smf
import common as cm

def lang_coef(d: pd.DataFrame, with_len: bool):
    f = "y_pred ~ y_true * C(lang, Treatment('ru'))" + (" + log_len" if with_len else "")
    m = smf.ols(f, data=d).fit()
    key = [k for k in m.params.index if k.startswith("C(lang") and ":" not in k][0]
    return float(m.params[key]), f, m

def boot_coef(d: pd.DataFrame, with_len: bool, n_boot=cm.N_BOOT, seed=0):
    rng = np.random.RandomState(seed); studs = d["student_id"].unique(); vals = []
    by = {s: g for s, g in d.groupby("student_id")}
    for _ in range(n_boot):
        pick = rng.choice(studs, len(studs), replace=True)
        bd = pd.concat([by[s] for s in pick], ignore_index=True)
        if bd["lang"].nunique() < 2: continue
        try: vals.append(lang_coef(bd, with_len)[0])
        except Exception: continue
    vals = np.array(vals)
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--pred", default=str(cm.PRED_DIR)); ap.add_argument("--out", default=str(cm.OUT_DIR))
    ap.add_argument("--survey", default=None, help="survey_raw.csv для gender/grade"); ap.add_argument("--n_boot", type=int, default=cm.N_BOOT)
    a = ap.parse_args(); P = cm.load_all_predictions(Path(a.pred)); P["log_len"] = np.log1p(P["n_chars"])
    if a.survey:
        s = pd.read_csv(a.survey, encoding="utf-8-sig")[["student_id", "gender", "grade"]]; P = P.merge(s, on="student_id", how="left")
    rows, bylang = [], []
    for (model, cond, k, sample, seed), g in P.groupby(["model", "condition", "k", "sample", "seed"]):
        # объединяем target=kz и target=ru одной конфигурации
        if g["lang"].nunique() < 2: continue
        for lang, gl in g.groupby("lang"):
            m = cm.metrics(gl.y_true, gl.y_pred); m.update({"model": model, "condition": cond, "k": k, "sample": sample, "seed": seed, "lang": lang,
                                                           "signed_err": float((gl.y_pred - gl.y_true).mean())})
            if "p0" in gl and gl[["p0","p1","p2","p3","p4"]].notna().all(axis=None):
                m["ece"] = cm.ece(gl.y_true.values, gl[["p0","p1","p2","p3","p4"]].values)
            bylang.append(m)
        c0, f0, _ = lang_coef(g, False); lo0, hi0 = boot_coef(g, False, a.n_boot, seed)
        c1, f1, _ = lang_coef(g, True); lo1, hi1 = boot_coef(g, True, a.n_boot, seed)
        rows.append({"model": model, "condition": cond, "k": k, "sample": sample, "seed": seed, "n": len(g),
                     "lang_coef_kz_vs_ru": c0, "ci_lo": lo0, "ci_hi": hi0, "includes_zero": lo0 <= 0 <= hi0,
                     "lang_coef_len_ctrl": c1, "ci_lo_len": lo1, "ci_hi_len": hi1, "includes_zero_len": lo1 <= 0 <= hi1, "formula": f1})
        print(f"{model:10s} {cond:15s} k={k:<3} seed={seed}: lang coef {c0:+.3f} [{lo0:+.3f},{hi0:+.3f}] | len-ctrl {c1:+.3f} [{lo1:+.3f},{hi1:+.3f}]")
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "fairness.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(bylang).to_csv(out / "fairness_by_lang.csv", index=False, encoding="utf-8-sig")
    print("->", out / "fairness.csv")

if __name__ == "__main__":
    main()
