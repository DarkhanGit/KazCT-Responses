"""
common.py — общие функции конвейера анализа статьи 3 (KazCT-Responses, AP25796141).

Правила, зашитые здесь (совпадают с Methods 4.4 и пре-регистрацией):
  * кросс-валидация: 5 фолдов GroupKFold по student_id, seeds = (11, 22, 33);
  * внутри обучающего фолда 10 % учеников — валидация (early stopping);
  * метрики: QWK (основная), MAE, exact, within-1, macro-F1; ECE для энкодеров;
  * доверительные интервалы: bootstrap по ответам, 2000 ресэмплов, 95 %;
  * сравнение условий: bootstrap CI парной разности QWK на тех же ответах; поправка Холма;
  * «на уровне человека» = CI модели перекрывает CI человек–человек на тех же ответах.

Формат файла предсказаний (predictions/*.csv), один файл = одна модель × условие × k × выборка × seed:
  response_id, student_id, lang, question, n_chars, fold, seed, y_true, y_pred, p0..p4 (вероятности, если есть),
  model, condition, k, sample
"""
from __future__ import annotations
import json, math, re
from pathlib import Path
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, mean_absolute_error, f1_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

SEEDS = (11, 22, 33)
N_FOLDS = 5
N_BOOT = 2000
ALPHA = 0.05
PROMPTS = ("E1", "E2", "E3", "E4", "E5")
LANGS = ("kz", "ru")
LEVELS = (0, 1, 2, 3, 4)
FEWSHOT_K = (50, 100, 200)
CONDITIONS = ("mono", "zeroshot", "fewshot", "joint", "translate_test", "translate_train")

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA = ROOT / "test_clean"          # синтетика по умолчанию; реальные данные — --data <папка clean>
PRED_DIR = ROOT / "analysis_out" / "predictions"
OUT_DIR = ROOT / "analysis_out"


# ---------- данные ----------
def load_data(data_dir: Path | str = DEFAULT_DATA, labels_file: str = "labels_final.csv") -> pd.DataFrame:
    """Объединяет responses.csv и labels_final.csv протокола в одну таблицу ответов с итоговым баллом."""
    data_dir = Path(data_dir)
    resp = pd.read_csv(data_dir / "responses.csv", encoding="utf-8-sig")
    lab = pd.read_csv(data_dir / labels_file, encoding="utf-8-sig")
    df = resp.merge(lab, on="response_id", how="inner")
    df = df.rename(columns={"form_lang": "lang"})
    df["lang"] = df["lang"].str.lower().map({"kz": "kz", "kk": "kz", "ru": "ru"}).fillna(df["lang"])
    df = df[df["score_final"].notna()].copy()
    df["y"] = df["score_final"].round().astype(int).clip(0, 4)
    if "n_chars" not in df:
        df["n_chars"] = df["text"].fillna("").str.len()
    df["log_len"] = np.log1p(df["n_chars"])
    if "data_source" in df and (df["data_source"] == "synthetic").any():
        print(f"[common] ВНИМАНИЕ: {int((df['data_source']=='synthetic').sum())} из {len(df)} ответов — СИНТЕТИКА. "
              "Результаты на них не являются результатами статьи.")
    return df


def make_folds(df: pd.DataFrame, seed: int, n_folds: int = N_FOLDS) -> pd.Series:
    """Номер тестового фолда для каждого ответа; группировка по student_id, перемешивание учеников по seed."""
    students = df["student_id"].unique()
    rng = np.random.RandomState(seed)
    perm = rng.permutation(students)
    fold_of_student = {s: i % n_folds for i, s in enumerate(perm)}
    return df["student_id"].map(fold_of_student).astype(int)


def split_val(train_idx: np.ndarray, groups: np.ndarray, seed: int, val_frac: float = 0.10):
    """Отщепляет 10 % учеников обучающего фолда под валидацию."""
    gss = GroupShuffleSplit(n_splits=1, test_size=val_frac, random_state=seed)
    tr, va = next(gss.split(train_idx, groups=groups[train_idx]))
    return train_idx[tr], train_idx[va]


# ---------- метрики ----------
def qwk(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    if len(set(y_true) | set(y_pred)) < 2:
        return float("nan")
    return cohen_kappa_score(y_true, y_pred, weights="quadratic", labels=list(LEVELS))


def metrics(y_true, y_pred) -> dict:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return {
        "qwk": qwk(y_true, y_pred),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "exact": float((y_true == y_pred).mean()),
        "within1": float((np.abs(y_true - y_pred) <= 1).mean()),
        "macro_f1": float(f1_score(y_true, y_pred, labels=list(LEVELS), average="macro", zero_division=0)),
        "n": int(len(y_true)),
    }


def ece(y_true, probs, n_bins: int = 10) -> float:
    """Expected calibration error по максимальной вероятности."""
    probs = np.asarray(probs); y_true = np.asarray(y_true)
    conf = probs.max(1); pred = probs.argmax(1); acc = (pred == y_true).astype(float)
    bins = np.linspace(0, 1, n_bins + 1); e = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return float(e)


# ---------- бутстреп ----------
def bootstrap_ci(y_true, y_pred, stat=qwk, n_boot: int = N_BOOT, seed: int = 0, alpha: float = ALPHA):
    """Точечная оценка и перцентильный CI статистики по ресэмплингу ответов."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    rng = np.random.RandomState(seed); n = len(y_true); vals = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        vals.append(stat(y_true[idx], y_pred[idx]))
    vals = np.array(vals, dtype=float); vals = vals[~np.isnan(vals)]
    return float(stat(y_true, y_pred)), float(np.percentile(vals, 100 * alpha / 2)), float(np.percentile(vals, 100 * (1 - alpha / 2)))


def paired_diff_ci(y_true, pred_a, pred_b, stat=qwk, n_boot: int = N_BOOT, seed: int = 0, alpha: float = ALPHA):
    """CI разности stat(a) − stat(b) на одних и тех же ответах (парный бутстреп). Возвращает (diff, lo, hi, p_two_sided)."""
    y_true, pred_a, pred_b = map(np.asarray, (y_true, pred_a, pred_b))
    rng = np.random.RandomState(seed); n = len(y_true); d = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        d.append(stat(y_true[idx], pred_a[idx]) - stat(y_true[idx], pred_b[idx]))
    d = np.array(d, dtype=float); d = d[~np.isnan(d)]
    point = stat(y_true, pred_a) - stat(y_true, pred_b)
    p = 2 * min((d <= 0).mean(), (d >= 0).mean())
    return float(point), float(np.percentile(d, 100 * alpha / 2)), float(np.percentile(d, 100 * (1 - alpha / 2))), float(min(1.0, p))


def holm(pvals: list[float], alpha: float = ALPHA) -> list[bool]:
    """Поправка Холма: True = отвергаем H0 на уровне alpha (семейный)."""
    m = len(pvals); order = np.argsort(pvals); reject = [False] * m
    for rank, i in enumerate(order):
        if pvals[i] <= alpha / (m - rank):
            reject[i] = True
        else:
            break
    return reject


# ---------- потолок человек–человек ----------
def human_ceiling(labels: pd.DataFrame, response_ids, r_a: str = "score_r1", r_b: str = "score_r2", seed: int = 0):
    """QWK между двумя экспертами на заданном множестве ответов, с бутстреп-CI."""
    sub = labels.set_index("response_id").loc[list(response_ids)]
    sub = sub[[r_a, r_b]].dropna()
    if len(sub) < 10:
        return {"qwk": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": int(len(sub))}
    q, lo, hi = bootstrap_ci(sub[r_a].astype(int), sub[r_b].astype(int), seed=seed)
    return {"qwk": q, "lo": lo, "hi": hi, "n": int(len(sub))}


# ---------- предсказания ----------
@dataclass
class RunSpec:
    model: str          # lr | length | xlmr | mbert | kazroberta | llm
    condition: str      # mono | zeroshot | fewshot | joint | translate_test | translate_train | zeroshot_llm | fewshot_llm
    target_lang: str    # язык тестовых ответов
    k: int = 0          # число целевых примеров для fewshot
    sample: int = 0     # номер выборки few-shot (0..2)
    seed: int = 11

    def name(self) -> str:
        return f"{self.model}__{self.condition}__{self.target_lang}__k{self.k}__s{self.sample}__seed{self.seed}"


def save_predictions(spec: RunSpec, rows: list[dict], pred_dir: Path = PRED_DIR) -> Path:
    pred_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    for k, v in (("model", spec.model), ("condition", spec.condition), ("target_lang", spec.target_lang),
                 ("k", spec.k), ("sample", spec.sample), ("seed", spec.seed)):
        df[k] = v
    p = pred_dir / (spec.name() + ".csv")
    df.to_csv(p, index=False, encoding="utf-8-sig")
    return p


def load_all_predictions(pred_dir: Path = PRED_DIR) -> pd.DataFrame:
    files = sorted(Path(pred_dir).glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"нет предсказаний в {pred_dir}")
    return pd.concat([pd.read_csv(f, encoding="utf-8-sig") for f in files], ignore_index=True)


def select_fewshot(df_train_target: pd.DataFrame, k: int, sample: int, seed_base: int = 1000) -> pd.DataFrame:
    """k целевых примеров, стратифицированных по баллу; sample = 0,1,2 → разные выборки. Если примеров меньше k — вернуть None."""
    if len(df_train_target) < k:
        return None
    rng = np.random.RandomState(seed_base + 7 * k + sample)
    parts = []
    props = df_train_target["y"].value_counts(normalize=True)
    for lvl, p in props.items():
        pool = df_train_target[df_train_target["y"] == lvl]
        n_take = max(1, int(round(p * k)))
        parts.append(pool.sample(n=min(n_take, len(pool)), random_state=rng))
    out = pd.concat(parts)
    if len(out) > k:
        out = out.sample(n=k, random_state=rng)
    elif len(out) < k:
        rest = df_train_target.drop(out.index)
        out = pd.concat([out, rest.sample(n=k - len(out), random_state=rng)])
    return out


def write_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")
