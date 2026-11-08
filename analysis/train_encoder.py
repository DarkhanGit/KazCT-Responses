"""
train_encoder.py — дообучение энкодеров и шесть условий переноса (Methods 4.2–4.3, пре-регистрация).

Модели (ключ → репозиторий HF):
  xlmr       FacebookAI/xlm-roberta-base
  mbert      google-bert/bert-base-multilingual-cased
  kazroberta kz-transformers/kaz-roberta-conversational   (только mono / fewshot на KZ)
  xlmr_large FacebookAI/xlm-roberta-large                 (только по правилу из пре-регистрации)
  tiny       локальная случайная модель для smoke-теста конвейера (без интернета)

Условия (target = язык теста; other = второй язык):
  mono            обучение на target-фолде
  zeroshot        обучение только на other
  fewshot         обучение на other, затем дообучение на k target-примерах (k=50/100/200, 3 выборки)
  joint           other + target вместе
  translate_test  обучение на other; тест — переводы target-ответов на other (нужен --mt translations.csv)
  translate_train обучение на переводах other-ответов на target; тест — оригинальные target-ответы (нужен --mt)

Гиперпараметры: lr 2e-5, batch 16, 20 эпох с warmup 10 % и линейным убыванием lr, выбор лучшей эпохи
по QWK на валидации (10 % учеников обучающего фолда) — OSF update 3 (04.11.2026); --legacy = как в регистрации (10 эпох, patience 2), max_len 256, fp16 на GPU, seeds 11/22/33.
Стадия 2 few-shot: фиксированные 3 эпохи на k целевых примерах без early stopping (k слишком мало для валидации).
Фолды: 5 × GroupKFold по ученику. Модель — отдельная на каждый промпт.

Файл переводов (--mt): CSV с колонками response_id, mt_lang (язык перевода: ru|kz), text_mt, mt_system.

Запуск (полный, на Legion 5):
  python train_encoder.py --data <clean> --labels labels_final.csv --model xlmr --conditions mono zeroshot fewshot joint --targets kz ru
  python train_encoder.py ... --conditions translate_test translate_train --mt <clean>/translations.csv
Smoke-тест без GPU и интернета:
  python train_encoder.py --data ../test_clean --labels labels_final_SYNTHETIC.csv --model tiny --conditions mono fewshot --targets kz --prompts E1 --seeds 11 --epochs 1 --k 50
"""
import argparse, math, os, time
os.environ.setdefault("HF_HUB_OFFLINE", "1"); os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
from pathlib import Path
import numpy as np, pandas as pd, torch
from torch.utils.data import Dataset, DataLoader
import common as C

HF = {"xlmr": "FacebookAI/xlm-roberta-base", "mbert": "google-bert/bert-base-multilingual-cased",
      "kazroberta": "kz-transformers/kaz-roberta-conversational", "xlmr_large": "FacebookAI/xlm-roberta-large"}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class TextDS(Dataset):
    def __init__(self, texts, labels, tok, max_len):
        self.enc = tok(list(texts), truncation=True, max_length=max_len, padding=False)
        self.labels = list(labels)
    def __len__(self): return len(self.labels)
    def __getitem__(self, i):
        d = {k: torch.tensor(v[i]) for k, v in self.enc.items()}
        d["labels"] = torch.tensor(self.labels[i]); return d


def collate(batch, pad_id):
    L = max(len(b["input_ids"]) for b in batch)
    out = {}
    for k in batch[0]:
        if k == "labels":
            out[k] = torch.stack([b[k] for b in batch])
        else:
            pad = pad_id if k == "input_ids" else 0
            out[k] = torch.stack([torch.cat([b[k], torch.full((L - len(b[k]),), pad, dtype=b[k].dtype)]) for b in batch])
    return out


def load_backbone(model_key: str, cache: dict):
    """Токенизатор + функция, создающая свежую модель классификации (веса — предобученные)."""
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
    if model_key == "tiny":
        return tiny_backbone(cache)
    repo = HF[model_key]
    tok = AutoTokenizer.from_pretrained(repo)
    rev = None
    try:
        from huggingface_hub import scan_cache_dir
        for r_ in scan_cache_dir().repos:
            if r_.repo_id == repo: rev = ",".join(x.commit_hash[:12] for x in r_.revisions)
    except Exception:
        pass
    def make():
        return AutoModelForSequenceClassification.from_pretrained(repo, num_labels=5)
    return tok, make, rev


def tiny_backbone(cache: dict):
    """Крошечная случайная XLM-R-подобная модель + BPE-токенизатор, обученный на текстах корпуса. Только для smoke-теста."""
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers
    from transformers import PreTrainedTokenizerFast, XLMRobertaConfig, XLMRobertaForSequenceClassification
    texts = cache["texts"]
    tk = Tokenizer(models.BPE(unk_token="<unk>")); tk.pre_tokenizer = pre_tokenizers.Whitespace()
    tk.train_from_iterator(texts, trainers.BpeTrainer(vocab_size=2000, special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"]))
    tok = PreTrainedTokenizerFast(tokenizer_object=tk, bos_token="<s>", eos_token="</s>", unk_token="<unk>", pad_token="<pad>", mask_token="<mask>")
    cfg = XLMRobertaConfig(vocab_size=2000, hidden_size=64, num_hidden_layers=2, num_attention_heads=2, intermediate_size=128,
                           max_position_embeddings=258, num_labels=5, pad_token_id=1)
    def make():
        torch.manual_seed(0); return XLMRobertaForSequenceClassification(cfg)
    return tok, make, "tiny-random"


@torch.no_grad()
def predict(model, ds, tok, bs=32):
    model.eval(); dl = DataLoader(ds, batch_size=bs, collate_fn=lambda b: collate(b, tok.pad_token_id))
    probs = []
    for b in dl:
        b = {k: v.to(DEVICE) for k, v in b.items() if k != "labels"}
        with torch.autocast(device_type="cuda", enabled=(DEVICE == "cuda")):
            logits = model(**b).logits
        probs.append(torch.softmax(logits.float(), -1).cpu().numpy())
    return np.concatenate(probs) if probs else np.zeros((0, 5))


def train(model, tok, train_df, val_df, seed, epochs, lr, batch, max_len, patience=None, log="", sched=True, warmup=0.1):
    """Стадия 1 (с валидацией): warmup 10 % + линейное убывание lr, все эпохи, берётся лучшая по val QWK эпоха
    (OSF update 3, 04.11.2026; диагностика diag_encoder.py). Стадия 2 few-shot (без валидации): sched=False, все эпохи."""
    torch.manual_seed(seed); np.random.seed(seed)
    tr = TextDS(train_df["text"].fillna(""), train_df["y"], tok, max_len)
    va = TextDS(val_df["text"].fillna(""), val_df["y"], tok, max_len) if len(val_df) else None
    g = torch.Generator(); g.manual_seed(seed)
    dl = DataLoader(tr, batch_size=batch, shuffle=True, generator=g, collate_fn=lambda b: collate(b, tok.pad_token_id))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total = max(1, epochs * len(dl)); warm = max(1, int(warmup * total))
    lam = (lambda s: min((s + 1) / warm, max(0.0, (total - s) / max(1, total - warm)))) if sched else (lambda s: 1.0)
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lam)
    scaler = torch.amp.GradScaler("cuda", enabled=(DEVICE == "cuda"))
    best, best_state, bad = -1.0, None, 0
    for ep in range(epochs):
        model.train(); t0 = time.time()
        for b in dl:
            b = {k: v.to(DEVICE) for k, v in b.items()}
            with torch.autocast(device_type="cuda", enabled=(DEVICE == "cuda")):
                loss = model(**b).loss
            opt.zero_grad(); scaler.scale(loss).backward(); scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); scaler.step(opt); scaler.update(); sch.step()
        if va is not None and len(val_df["y"].unique()) > 1:
            q = C.qwk(val_df["y"].values, predict(model, va, tok).argmax(1))
        else:
            q = float(ep)  # без валидации — просто идём до конца
        print(f"    {log} ep{ep+1} val_qwk={q:.3f} ({time.time()-t0:.0f}s)")
        if q > best:
            best, bad = q, 0; best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if patience is not None and bad >= patience: break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def run_condition(df, mt, model_key, tok, make, cond, target, k, sample, seed, a, rev):
    other = "ru" if target == "kz" else "kz"
    folds = C.make_folds(df, seed); rows = []
    for q in a.prompts:
        sub = df[df.question == q]
        for f in range(C.N_FOLDS):
            test = sub[(sub.lang == target) & (folds[sub.index] == f)]
            train_all = sub[folds[sub.index] != f]
            tr_t, tr_o = train_all[train_all.lang == target], train_all[train_all.lang == other]
            if len(test) == 0: continue
            # --- состав обучающих данных по условию ---
            stage2 = None; test_texts = test
            if cond == "mono":
                stage1 = tr_t
            elif cond == "zeroshot":
                stage1 = tr_o
            elif cond == "joint":
                stage1 = train_all
            elif cond == "fewshot":
                stage1 = tr_o; stage2 = C.select_fewshot(tr_t, k, sample)
                if stage2 is None:
                    print(f"    {q} fold{f}: k={k} > доступных {len(tr_t)} target-примеров — не оценивается"); continue
            elif cond == "translate_test":
                stage1 = tr_o
                m = mt[(mt.mt_lang == other)].set_index("response_id")["text_mt"]
                test_texts = test.copy(); test_texts["text"] = test_texts["response_id"].map(m)
                if test_texts["text"].isna().any():
                    raise SystemExit(f"нет переводов для {int(test_texts['text'].isna().sum())} тестовых ответов ({q}, fold {f})")
            elif cond == "translate_train":
                m = mt[(mt.mt_lang == target)].set_index("response_id")["text_mt"]
                stage1 = tr_o.copy(); stage1["text"] = stage1["response_id"].map(m)
                if stage1["text"].isna().any():
                    raise SystemExit("нет переводов обучающих ответов для translate_train")
                if k:  # опционально: + k аутентичных target-примеров
                    stage2 = C.select_fewshot(tr_t, k, sample)
            else:
                raise ValueError(cond)
            if stage1["y"].nunique() < 2: continue
            # валидация: 10 % учеников stage1 (для fewshot — из stage2, если хватает; иначе из stage1)
            groups = stage1["student_id"].values
            tr_idx, va_idx = C.split_val(np.arange(len(stage1)), groups, seed)
            model = make().to(DEVICE)
            model = train(model, tok, stage1.iloc[tr_idx], stage1.iloc[va_idx], seed, a.epochs, a.lr, a.batch, a.max_len, log=f"{q} f{f} s1", patience=a.patience, sched=a.sched)
            if stage2 is not None:
                model = train(model, tok, stage2, stage2.iloc[:0], seed, a.stage2_epochs, a.lr, min(a.batch, 8), a.max_len, log=f"{q} f{f} s2(k={k})", sched=False)  # стадия 2: фиксированное число эпох, без early stopping (k примеров слишком мало для валидации)
            if a.save_dir and cond == "mono" and f == 0 and seed == a.seeds[0] and a.model != "tiny":
                sd = Path(a.save_dir) / f"{a.model}__{target}__{q}"; sd.mkdir(parents=True, exist_ok=True)
                model.save_pretrained(sd); tok.save_pretrained(sd)   # чекпойнт для interpret.py
            probs = predict(model, TextDS(test_texts["text"].fillna(""), test["y"], tok, a.max_len), tok)
            for i, (_, r) in enumerate(test.iterrows()):
                rows.append({"response_id": r.response_id, "student_id": r.student_id, "lang": r.lang, "question": q,
                             "n_chars": r.n_chars, "fold": f, "y_true": int(r.y), "y_pred": int(probs[i].argmax()),
                             **{f"p{j}": float(probs[i, j]) for j in range(5)}, "revision": rev})
            del model; torch.cuda.empty_cache() if DEVICE == "cuda" else None
    return rows


def run_zs_fewshot(df, tok, make, target, seed, a, rev, ks, samples, want_zs):
    """zeroshot и fewshot за один проход: модель стадии 1 (обучение на втором языке) обучается ОДИН раз на
    (вопрос, фолд, seed); её предсказания = zeroshot; затем для каждого k и выборки — копия + стадия 2 на k
    целевых примерах. Процедура совпадает с пререгистрацией; базовая модель для few-shot и zero-shot — одна и та же,
    поэтому сравнения парные. Возвращает {имя спецификации: строки}."""
    other = "ru" if target == "kz" else "kz"
    folds = C.make_folds(df, seed); out = {}
    def put(spec, test, probs, f, q):
        rows = out.setdefault(spec.name(), (spec, []))[1]
        for i, (_, r) in enumerate(test.iterrows()):
            rows.append({"response_id": r.response_id, "student_id": r.student_id, "lang": r.lang, "question": q,
                         "n_chars": r.n_chars, "fold": f, "y_true": int(r.y), "y_pred": int(probs[i].argmax()),
                         **{f"p{j}": float(probs[i, j]) for j in range(5)}, "revision": rev})
    for q in a.prompts:
        sub = df[df.question == q]
        for f in range(C.N_FOLDS):
            test = sub[(sub.lang == target) & (folds[sub.index] == f)]
            train_all = sub[folds[sub.index] != f]
            tr_t, tr_o = train_all[train_all.lang == target], train_all[train_all.lang == other]
            if len(test) == 0 or tr_o["y"].nunique() < 2: continue
            tr_idx, va_idx = C.split_val(np.arange(len(tr_o)), tr_o["student_id"].values, seed)
            model = make().to(DEVICE)
            model = train(model, tok, tr_o.iloc[tr_idx], tr_o.iloc[va_idx], seed, a.epochs, a.lr, a.batch, a.max_len, log=f"{q} f{f} s1", patience=a.patience, sched=a.sched)
            base = {k_: v.detach().cpu().clone() for k_, v in model.state_dict().items()}
            ds_test = TextDS(test["text"].fillna(""), test["y"], tok, a.max_len)
            if want_zs:
                put(C.RunSpec(model=a.model, condition="zeroshot", target_lang=target, seed=seed), test, predict(model, ds_test, tok), f, q)
            for k in ks:
                for sample in range(samples):
                    st2 = C.select_fewshot(tr_t, k, sample)
                    if st2 is None:
                        if sample == 0: print(f"    {q} fold{f}: k={k} > доступных {len(tr_t)} target-примеров — не оценивается")
                        continue
                    model.load_state_dict(base)
                    model = train(model, tok, st2, st2.iloc[:0], seed, a.stage2_epochs, a.lr, min(a.batch, 8), a.max_len, log=f"{q} f{f} s2(k={k},s{sample})", sched=False)
                    put(C.RunSpec(model=a.model, condition="fewshot", target_lang=target, k=k, sample=sample, seed=seed), test, predict(model, ds_test, tok), f, q)
            del model, base; torch.cuda.empty_cache() if DEVICE == "cuda" else None
    return out


def run_mono_k(df, tok, make, target, seed, a, rev, k, sample):
    """Только для kazroberta (казахская модель, без русского входа): кривая обучения на k казахских примерах
    без стадии на втором языке. Та же схема фолдов и выборок k, что у fewshot."""
    folds = C.make_folds(df, seed); rows = []
    for q in a.prompts:
        sub = df[df.question == q]
        for f in range(C.N_FOLDS):
            test = sub[(sub.lang == target) & (folds[sub.index] == f)]
            tr_t = sub[(folds[sub.index] != f) & (sub.lang == target)]
            st = C.select_fewshot(tr_t, k, sample)
            if len(test) == 0 or st is None or st["y"].nunique() < 2: continue
            tr_idx, va_idx = C.split_val(np.arange(len(st)), st["student_id"].values, seed)
            model = make().to(DEVICE)
            model = train(model, tok, st.iloc[tr_idx], st.iloc[va_idx], seed, a.epochs, a.lr, a.batch, a.max_len, log=f"{q} f{f} mono_k(k={k},s{sample})", patience=a.patience, sched=a.sched)
            probs = predict(model, TextDS(test["text"].fillna(""), test["y"], tok, a.max_len), tok)
            for i, (_, r) in enumerate(test.iterrows()):
                rows.append({"response_id": r.response_id, "student_id": r.student_id, "lang": r.lang, "question": q,
                             "n_chars": r.n_chars, "fold": f, "y_true": int(r.y), "y_pred": int(probs[i].argmax()),
                             **{f"p{j}": float(probs[i, j]) for j in range(5)}, "revision": rev})
            del model; torch.cuda.empty_cache() if DEVICE == "cuda" else None
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(C.DEFAULT_DATA)); ap.add_argument("--labels", default="labels_final.csv")
    ap.add_argument("--model", default="xlmr", choices=list(HF) + ["tiny"])
    ap.add_argument("--conditions", nargs="*", default=["mono", "zeroshot", "fewshot", "joint"])
    ap.add_argument("--targets", nargs="*", default=["kz", "ru"]); ap.add_argument("--prompts", nargs="*", default=list(C.PROMPTS))
    ap.add_argument("--seeds", nargs="*", type=int, default=list(C.SEEDS)); ap.add_argument("--k", nargs="*", type=int, default=list(C.FEWSHOT_K))
    ap.add_argument("--samples", type=int, default=3); ap.add_argument("--mt", default=None)
    ap.add_argument("--epochs", type=int, default=20); ap.add_argument("--legacy", action="store_true", help="процедура до update 3: 10 эпох, постоянный lr, patience 2"); ap.add_argument("--stage2_epochs", type=int, default=3); ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch", type=int, default=16); ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--out", default=str(C.PRED_DIR)); ap.add_argument("--save_dir", default=None, help="куда сохранять mono-чекпойнты fold 0 для interpret.py")
    a = ap.parse_args()
    a.patience, a.sched = (2, False) if a.legacy else (None, True)
    if a.legacy and a.epochs == 20: a.epochs = 10
    df = C.load_data(a.data, a.labels)
    mt = pd.read_csv(a.mt, encoding="utf-8-sig") if a.mt else None
    if any(c.startswith("translate") for c in a.conditions) and mt is None:
        raise SystemExit("translate_* требуют --mt translations.csv")
    if a.model == "kazroberta":
        a.targets = ["kz"]
        a.conditions = [("mono_k" if c == "fewshot" else c) for c in a.conditions if c in ("mono", "fewshot", "mono_k")]
        print("[kazroberta] только KZ: mono и mono_k (k казахских примеров без русской стадии)")
    tok, make, rev = load_backbone(a.model, {"texts": df["text"].fillna("").tolist()})
    print(f"device={DEVICE} model={a.model} rev={rev}")
    def done(spec): return (Path(a.out) / (spec.name() + ".csv")).exists()
    def report(spec, rows):
        C.save_predictions(spec, rows, Path(a.out))
        m = C.metrics([r["y_true"] for r in rows], [r["y_pred"] for r in rows])
        print(f"   -> {spec.name()} n={m['n']} QWK={m['qwk']:.3f} MAE={m['mae']:.2f}", flush=True)
    # zeroshot + fewshot одним проходом (общая модель стадии 1)
    if any(c in a.conditions for c in ("zeroshot", "fewshot")):
        want_zs = "zeroshot" in a.conditions; ks = a.k if "fewshot" in a.conditions else []
        for target in a.targets:
            for seed in a.seeds:
                specs = ([C.RunSpec(model=a.model, condition="zeroshot", target_lang=target, seed=seed)] if want_zs else []) + \
                        [C.RunSpec(model=a.model, condition="fewshot", target_lang=target, k=k, sample=s_, seed=seed) for k in ks for s_ in range(a.samples)]
                if all(done(sp) for sp in specs):
                    print("skip (exists) zeroshot/fewshot", target, seed); continue
                print(f"== zeroshot+fewshot target={target} seed={seed}", flush=True)
                res = run_zs_fewshot(df, tok, make, target, seed, a, rev, ks, a.samples, want_zs)
                for name, (spec, rows) in res.items():
                    if rows and not done(spec): report(spec, rows)
        a.conditions = [c for c in a.conditions if c not in ("zeroshot", "fewshot")]
    for cond in a.conditions:
        if cond == "mono_k":
            for target in a.targets:
                for k in a.k:
                    for sample in range(a.samples):
                        for seed in a.seeds:
                            spec = C.RunSpec(model=a.model, condition="mono_k", target_lang=target, k=k, sample=sample, seed=seed)
                            if done(spec): print("skip (exists)", spec.name()); continue
                            print("==", spec.name(), flush=True)
                            rows = run_mono_k(df, tok, make, target, seed, a, rev, k, sample)
                            if rows: report(spec, rows)
            continue
        for target in a.targets:
            ks = a.k if cond == "fewshot" else [0]
            for k in ks:
                for sample in (range(a.samples) if cond == "fewshot" else [0]):
                    for seed in a.seeds:
                        spec = C.RunSpec(model=a.model, condition=cond, target_lang=target, k=k, sample=sample, seed=seed)
                        out_path = Path(a.out) / (spec.name() + ".csv")
                        if out_path.exists():
                            print("skip (exists)", spec.name()); continue
                        print("==", spec.name())
                        rows = run_condition(df, mt, a.model, tok, make, cond, target, k, sample, seed, a, rev)
                        if rows:
                            C.save_predictions(spec, rows, Path(a.out))
                            m = C.metrics([r["y_true"] for r in rows], [r["y_pred"] for r in rows])
                            print(f"   -> n={m['n']} QWK={m['qwk']:.3f} MAE={m['mae']:.2f}")


if __name__ == "__main__":
    main()
