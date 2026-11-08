"""
diag_encoder.py - диагностика обучения энкодера (04.11.2026), ТОЛЬКО на валидационных частях обучающих фолдов.
Тестовые фолды НЕ используются и НЕ предсказываются: выбор процедуры обучения не опирается на тестовые результаты.

Причина: в полном прогоне XLM-R после 1-й эпохи val_qwk = 0.000 (модель предсказывает один класс),
early stopping (patience 2) останавливает обучение на 3-4 эпохе - модель почти не обучена.

Сравниваются варианты процедуры на mono-KZ, E1-E5, фолды 0-4, seed 11 (25 обучений на вариант):
  A  как в регистрации/коде: 10 эпох, постоянный lr 2e-5, early stopping по val QWK, patience 2
  B  10 эпох, warmup 10 % + линейное убывание lr, patience 2
  C  10 эпох, warmup + линейное убывание, без досрочной остановки, берётся лучшая по val эпоха
  D  20 эпох, warmup + линейное убывание, лучшая по val эпоха
Метрики (по объединённым валидационным предсказаниям 25 обучений): QWK, доля самого частого предсказанного класса,
число эпох до лучшей; плюс train loss последней эпохи.

Запуск (C:\\meta_ozat\\analysis, окружение .venv):
  python diag_encoder.py --data C:\\meta_ozat\\data\\clean_2026-10-20 --model xlmr
Результат: C:\\meta_ozat\\out\\diag\\diag_<model>.csv и diag_<model>_summary.csv
"""
import argparse, os, time, json
os.environ.setdefault("HF_HUB_OFFLINE", "1")
from pathlib import Path
import numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader
import common as C
import train_encoder as T

VARIANTS = {
    "A_registered": dict(epochs=10, sched=False, patience=2),
    "B_warmup_p2":  dict(epochs=10, sched=True,  patience=2),
    "C_warmup_best10": dict(epochs=10, sched=True, patience=None),
    "D_warmup_best20": dict(epochs=20, sched=True, patience=None),
}


def train_v(model, tok, tr_df, va_df, seed, lr, batch, max_len, epochs, sched, patience):
    torch.manual_seed(seed); np.random.seed(seed)
    tr = T.TextDS(tr_df["text"].fillna(""), tr_df["y"], tok, max_len)
    va = T.TextDS(va_df["text"].fillna(""), va_df["y"], tok, max_len)
    g = torch.Generator(); g.manual_seed(seed)
    dl = DataLoader(tr, batch_size=batch, shuffle=True, generator=g, collate_fn=lambda b: T.collate(b, tok.pad_token_id))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total = epochs * len(dl); warm = max(1, int(0.1 * total))
    lam = (lambda s: min((s + 1) / warm, max(0.0, (total - s) / max(1, total - warm)))) if sched else (lambda s: 1.0)
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lam)
    scaler = torch.amp.GradScaler("cuda", enabled=(T.DEVICE == "cuda"))
    best, best_ep, best_pred, bad, hist = -2.0, 0, None, 0, []
    for ep in range(epochs):
        model.train(); losses = []
        for b in dl:
            b = {k: v.to(T.DEVICE) for k, v in b.items()}
            with torch.autocast(device_type="cuda", enabled=(T.DEVICE == "cuda")):
                loss = model(**b).loss
            opt.zero_grad(); scaler.scale(loss).backward(); scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); scaler.step(opt); scaler.update(); sch.step()
            losses.append(float(loss))
        pred = T.predict(model, va, tok).argmax(1)
        q = C.qwk(va_df["y"].values, pred); q = -1.0 if np.isnan(q) else q
        hist.append((ep + 1, round(float(np.mean(losses)), 3), round(q, 3), int(np.bincount(pred, minlength=5).max())))
        if q > best:
            best, best_ep, best_pred, bad = q, ep + 1, pred, 0
        else:
            bad += 1
            if patience is not None and bad >= patience: break
    return best_pred, best_ep, hist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--labels", default="labels_final.csv")
    ap.add_argument("--model", default="xlmr"); ap.add_argument("--target", default="kz")
    ap.add_argument("--seed", type=int, default=11); ap.add_argument("--variants", nargs="*", default=list(VARIANTS))
    ap.add_argument("--out", default=r"C:\meta_ozat\out\diag")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    df = C.load_data(a.data, a.labels)
    tok, make, rev = T.load_backbone(a.model, {"texts": df["text"].fillna("").tolist()})
    print(f"device={T.DEVICE} model={a.model} rev={rev}", flush=True)
    folds = C.make_folds(df, a.seed); rows = []
    for vname in a.variants:
        v = VARIANTS[vname]; t0 = time.time()
        for q in C.PROMPTS:
            sub = df[df.question == q]
            for f in range(C.N_FOLDS):
                tr_t = sub[(folds[sub.index] != f) & (sub.lang == a.target)]   # только обучающий фолд; тест не трогаем
                if len(tr_t) < 10: continue
                tr_idx, va_idx = C.split_val(np.arange(len(tr_t)), tr_t["student_id"].values, a.seed)
                trd, vad = tr_t.iloc[tr_idx], tr_t.iloc[va_idx]
                model = make().to(T.DEVICE)
                pred, bep, hist = train_v(model, tok, trd, vad, a.seed, 2e-5, 16, 256, v["epochs"], v["sched"], v["patience"])
                for (rid, yt), yp in zip(vad[["response_id", "y"]].values, pred):
                    rows.append({"variant": vname, "question": q, "fold": f, "response_id": rid, "y_true": int(yt), "y_pred": int(yp),
                                 "best_epoch": bep, "epochs_run": len(hist), "last_train_loss": hist[-1][1]})
                print(f"  {vname} {q} f{f}: best_ep={bep}/{len(hist)} hist={hist}", flush=True)
                del model; torch.cuda.empty_cache() if T.DEVICE == "cuda" else None
        print(f"== {vname} done in {time.time()-t0:.0f}s", flush=True)
        pd.DataFrame(rows).to_csv(out / f"diag_{a.model}.csv", index=False, encoding="utf-8-sig")
    R = pd.DataFrame(rows); S = []
    for vname, g in R.groupby("variant", sort=False):
        per = g.groupby(["question", "fold"]).first()
        S.append({"variant": vname, "n_val": len(g), "val_qwk_pooled": round(C.qwk(g.y_true, g.y_pred), 3),
                  "val_qwk_mean_prompt": round(np.nanmean([C.qwk(x.y_true, x.y_pred) for _, x in g.groupby("question")]), 3),
                  "share_majority_pred": round(g.y_pred.value_counts(normalize=True).max(), 3),
                  "pred_dist": json.dumps(g.y_pred.value_counts(normalize=True).sort_index().round(2).to_dict()),
                  "median_best_epoch": float(per.best_epoch.median()), "median_epochs_run": float(per.epochs_run.median()),
                  "median_last_train_loss": round(float(per.last_train_loss.median()), 3)})
    S = pd.DataFrame(S); S.to_csv(out / f"diag_{a.model}_summary.csv", index=False, encoding="utf-8-sig")
    print(S.to_string(index=False))


if __name__ == "__main__":
    main()
