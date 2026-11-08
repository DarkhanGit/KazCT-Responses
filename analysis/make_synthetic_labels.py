"""
make_synthetic_labels.py — СИНТЕТИЧЕСКИЕ метки для отладки конвейера. НЕ ДАННЫЕ СТАТЬИ.

Берёт responses.csv синтетического прогона (1 500 ответов) и имитирует двух экспертов:
латентное качество ответа ~ f(длина, число связок/маркеров рассуждения, шум); R1 и R2 — зашумлённые
копии с заданным согласием; R3 — на 200 ответах; адъюдикация при |R1−R2| ≥ 2.
Выход: labels_final_SYNTHETIC.csv, labels_expert_SYNTHETIC.csv в той же папке.
Запуск: python make_synthetic_labels.py --data ../test_clean
"""
import argparse, re
from pathlib import Path
import numpy as np, pandas as pd

MARKERS = {
    "ru": ["потому что", "например", "поэтому", "если", "но ", "однако", "с другой стороны", "во-первых", "во-вторых", "значит", "хотя", "проверить", "источник"],
    "kz": ["өйткені", "мысалы", "сондықтан", "егер", "бірақ", "алайда", "екінші жағынан", "біріншіден", "екіншіден", "демек", "тексеру", "дереккөз"],
}

def latent_quality(text: str, lang: str, rng) -> float:
    t = (text or "").lower(); n = len(t)
    if n < 20: return 0.0
    m = sum(t.count(w) for w in MARKERS.get(lang, MARKERS["ru"]))
    sents = max(1, len(re.findall(r"[.!?]+", t)))
    q = 0.9 * np.log1p(n) / np.log1p(800) * 4 + 0.35 * min(m, 6) + 0.15 * min(sents, 6) - 1.0 + rng.normal(0, 0.5)
    return float(np.clip(q, 0, 4))

def rater(q, rng, noise):
    return int(np.clip(round(q + rng.normal(0, noise)), 0, 4))

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", default="../test_clean"); ap.add_argument("--seed", type=int, default=2026)
    a = ap.parse_args(); d = Path(a.data); rng = np.random.RandomState(a.seed)
    r = pd.read_csv(d / "responses.csv", encoding="utf-8-sig")
    q = np.array([latent_quality(t, l, rng) for t, l in zip(r["text"], r["form_lang"])])
    r1 = np.array([rater(x, rng, 0.55) for x in q]); r2 = np.array([rater(x, rng, 0.60) for x in q])
    r3 = np.full(len(r), np.nan)
    sub = r.sample(n=min(200, len(r)), random_state=1).index
    r3[sub] = [rater(q[i], rng, 0.6) for i in sub]
    final = np.where(np.abs(r1 - r2) <= 1, np.ceil((r1 + r2) / 2), np.round(q)).astype(int)
    adjud = (np.abs(r1 - r2) >= 2).astype(int)
    lab = pd.DataFrame({"response_id": r["response_id"], "score_r1": r1, "score_r2": r2, "score_r3": r3,
                        "score_final": final, "adjudicated": adjud})
    # split 80/20 по ученикам
    students = r["student_id"].unique(); rng2 = np.random.RandomState(7); test_students = set(rng2.choice(students, int(0.2 * len(students)), replace=False))
    lab["split"] = np.where(r["student_id"].isin(test_students), "test", "train")
    lab["data_source"] = "SYNTHETIC"
    lab.to_csv(d / "labels_final_SYNTHETIC.csv", index=False, encoding="utf-8-sig")
    long = pd.concat([pd.DataFrame({"response_id": r["response_id"], "rater_id": rid, "score": sc}) for rid, sc in (("R1", r1), ("R2", r2), ("R3", r3))])
    long = long.dropna(); long["data_source"] = "SYNTHETIC"
    long.to_csv(d / "labels_expert_SYNTHETIC.csv", index=False, encoding="utf-8-sig")
    from sklearn.metrics import cohen_kappa_score
    print("SYNTHETIC labels:", len(lab), "| QWK R1-R2 =", round(cohen_kappa_score(r1, r2, weights="quadratic"), 3),
          "| распределение final:", lab.score_final.value_counts().sort_index().to_dict(), "| adjudicated:", int(adjud.sum()))

if __name__ == "__main__":
    main()
