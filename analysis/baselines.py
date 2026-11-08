"""
baselines.py — признаковые базлайны (Methods 4.2): логистическая регрессия на TF-IDF n-граммах
(слова 1–3 + символы 2–5, как у Horbach et al. 2024) и length-only (log числа символов).

Условия для LR: mono (обучение и тест на одном языке), zeroshot (обучение на другом языке), joint (оба языка).
translate_* для LR требуют файла переводов (см. train_encoder.py --mt) и здесь не считаются.
Модели обучаются ПО ПРОМПТАМ (per prompt), фолды — GroupKFold по ученику, 3 seed.

Запуск: python baselines.py --data ../test_clean --labels labels_final_SYNTHETIC.csv
Выход: analysis_out/predictions/lr__*.csv, length__*.csv
"""
import argparse
from pathlib import Path
import numpy as np, pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import common as C


def featurize(train_text, test_text):
    wv = TfidfVectorizer(analyzer="word", ngram_range=(1, 3), min_df=2, sublinear_tf=True, lowercase=True)
    cv = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True, lowercase=True)
    Xtr = hstack([wv.fit_transform(train_text), cv.fit_transform(train_text)]).tocsr()
    Xte = hstack([wv.transform(test_text), cv.transform(test_text)]).tocsr()
    return Xtr, Xte


def fit_predict_lr(Xtr, ytr, Xte, seed):
    clf = LogisticRegression(C=2.0, max_iter=2000, class_weight="balanced", random_state=seed)
    clf.fit(Xtr, ytr)
    probs = np.zeros((Xte.shape[0], 5))
    probs[:, clf.classes_] = clf.predict_proba(Xte)
    return probs.argmax(1), probs


def run(df: pd.DataFrame, model: str, condition: str, target: str, seed: int) -> list[dict]:
    rows = []
    folds = C.make_folds(df, seed)
    other = "ru" if target == "kz" else "kz"
    for q in C.PROMPTS:
        for f in range(C.N_FOLDS):
            test = df[(df.question == q) & (df.lang == target) & (folds == f)]
            train_all = df[(df.question == q) & (folds != f)]
            if condition == "mono":
                train = train_all[train_all.lang == target]
            elif condition == "zeroshot":
                train = train_all[train_all.lang == other]
            elif condition == "joint":
                train = train_all
            else:
                raise ValueError(condition)
            if len(test) == 0 or train["y"].nunique() < 2:
                continue
            if model == "lr":
                Xtr, Xte = featurize(train["text"].fillna(""), test["text"].fillna(""))
                y_pred, probs = fit_predict_lr(Xtr, train["y"].values, Xte, seed)
            elif model == "length":
                sc = StandardScaler().fit(train[["log_len"]])
                y_pred, probs = fit_predict_lr(sc.transform(train[["log_len"]]), train["y"].values, sc.transform(test[["log_len"]]), seed)
            else:
                raise ValueError(model)
            for i, (rid, r) in enumerate(test.iterrows()):
                rows.append({"response_id": r.response_id, "student_id": r.student_id, "lang": r.lang, "question": q,
                             "n_chars": r.n_chars, "fold": f, "y_true": int(r.y), "y_pred": int(y_pred[i]),
                             **{f"p{j}": float(probs[i, j]) for j in range(5)}})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(C.DEFAULT_DATA)); ap.add_argument("--labels", default="labels_final.csv")
    ap.add_argument("--models", nargs="*", default=["lr", "length"]); ap.add_argument("--conditions", nargs="*", default=["mono", "zeroshot", "joint"])
    ap.add_argument("--out", default=str(C.PRED_DIR))
    a = ap.parse_args()
    df = C.load_data(a.data, a.labels)
    for model in a.models:
        for cond in a.conditions:
            for target in C.LANGS:
                for seed in C.SEEDS:
                    spec = C.RunSpec(model=model, condition=cond, target_lang=target, seed=seed)
                    rows = run(df, model, cond, target, seed)
                    if rows:
                        p = C.save_predictions(spec, rows, Path(a.out))
                        m = C.metrics([r["y_true"] for r in rows], [r["y_pred"] for r in rows])
                        print(f"{spec.name():50s} n={m['n']:4d} QWK={m['qwk']:.3f} MAE={m['mae']:.2f}")


if __name__ == "__main__":
    main()
