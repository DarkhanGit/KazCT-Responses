"""
interpret.py v2 (07.11.2026) — интерпретируемость (Methods 4.6): Integrated Gradients для лучшего mono-энкодера
в каждом языке на 100 ответах (50/язык, 10/вопрос, стратифицировано по баллу) из ТЕСТОВОГО фолда 0 (seed 11) —
т. е. на ответах, которых модель (чекпойнт fold 0, seed 11) не видела при обучении.

Изменения против v1: IG реализован напрямую в torch (без captum); атрибуции суммируются от субтокенов к словам
(offset_mapping); масса атрибуций считается по классам слов: маркеры рассуждения, слова темы вопроса, прочие;
отдельные модели для KZ и RU; IoU с опорными фрагментами экспертов — только если передан --spans.

Запуск (C:\\meta_ozat\\analysis, .venv):
  python interpret.py --data C:\\meta_ozat\\data\\clean_2026-10-20 --checkpoint_dir C:\\meta_ozat\\out\\checkpoints --model_kz kazroberta --model_ru mbert --out C:\\meta_ozat\\out\\interpret
Выход: attributions_words.csv (слово, атрибуция), interpret_summary.csv (по ответам), interpret_top_words.csv, interpret_overview.csv
"""
import argparse, re
from pathlib import Path
import numpy as np, pandas as pd
import common as C
from prompts_llm import PROMPTS

MARKERS = {"ru": ["потому что", "так как", "поэтому", "например", "если", "но", "однако", "с другой стороны", "во-первых",
                  "во-вторых", "значит", "хотя", "зато", "с одной стороны", "наоборот", "следовательно", "в итоге", "кроме того"],
           "kz": ["өйткені", "себебі", "сондықтан", "мысалы", "егер", "бірақ", "алайда", "екінші жағынан", "біріншіден",
                  "екіншіден", "демек", "дегенмен", "керісінше", "сол себепті", "қорыта айтқанда", "яғни"]}
STOP = {"ru": {"или", "что", "как", "это", "твой", "твоя", "после", "когда", "какие", "почему", "сделал", "сделала"},
        "kz": {"немесе", "қандай", "неге", "қалай", "кейін", "сенің", "не", "үшін"}}
WORD = re.compile(r"\w+(?:-\w+)*", re.U)


def topic_stems(q, lang):
    words = [w.lower() for w in WORD.findall(PROMPTS[q][lang]) if len(w) >= 4 and w.lower() not in STOP[lang]]
    return {w[:5] for w in words}


def sample_100(df, seed=11):
    folds = C.make_folds(df, seed); d = df[folds == 0]
    rng = np.random.RandomState(seed); parts = []
    for lang in C.LANGS:
        for q in sorted(df.question.unique()):
            pool = d[(d.lang == lang) & (d.question == q)]
            take = []
            for lvl in C.LEVELS:
                pl = pool[pool.y == lvl]
                if len(pl): take.append(pl.sample(n=min(2, len(pl)), random_state=rng))
            sel = pd.concat(take) if take else pool.iloc[:0]
            if len(sel) < 10:   # редкие уровни (например, нет «4» в E2) — добираем случайно
                rest = pool.drop(sel.index)
                sel = pd.concat([sel, rest.sample(n=min(10 - len(sel), len(rest)), random_state=rng)])
            parts.append(sel.head(10))
    return pd.concat(parts)


def ig_words(model, tok, text, target, device, n_steps=32):
    import torch
    enc = tok(text, return_tensors="pt", truncation=True, max_length=256, return_offsets_mapping=True)
    offs = enc.pop("offset_mapping")[0].tolist(); enc = {k: v.to(device) for k, v in enc.items()}
    emb = model.get_input_embeddings()
    with torch.no_grad():
        x = emb(enc["input_ids"])                                             # (1, L, H)
        base = emb(torch.full_like(enc["input_ids"], tok.pad_token_id))
    alphas = (torch.arange(n_steps, device=device, dtype=x.dtype) + 0.5) / n_steps   # средние точки (Riemann)
    path = (base + alphas[:, None, None] * (x - base)).detach().requires_grad_(True)  # (n, L, H)
    mask = enc["attention_mask"].expand(n_steps, -1)
    extra = {k: v.expand(n_steps, -1) for k, v in enc.items() if k == "token_type_ids"}
    logits = model(inputs_embeds=path, attention_mask=mask, **extra).logits
    grads = torch.autograd.grad(logits[:, target].sum(), path)[0]
    attr = ((x - base) * grads.mean(0, keepdim=True)).sum(-1)[0].detach().float().cpu().numpy()   # (L,)
    with torch.no_grad():
        pred = int(model(**enc).logits.argmax(-1))
    # субтокены → слова по смещениям
    covered = max((e for s, e in offs), default=0)                          # текст после обрезки до 256 токенов не входит
    words = [(m.start(), m.end(), m.group(0)) for m in WORD.finditer(text[:covered])]
    wa = np.zeros(len(words))
    for (s, e), a in zip(offs, attr):
        if e <= s: continue                                                    # спецтокены
        for i, (ws, we, _) in enumerate(words):
            if s < we and e > ws: wa[i] += a; break
    tot = np.abs(attr).sum() + 1e-9
    return [w for _, _, w in words], wa / tot, pred


def word_classes(words, lang, q):
    low = [w.lower() for w in words]; cls = ["other"] * len(words)
    stems = topic_stems(q, lang)
    for i, w in enumerate(low):
        if len(w) >= 4 and w[:5] in stems: cls[i] = "topic"
    for m in MARKERS[lang]:
        parts = m.split(); n = len(parts)
        for i in range(len(low) - n + 1):
            if low[i:i + n] == parts:
                for j in range(i, i + n): cls[j] = "marker"
    return cls


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(C.DEFAULT_DATA)); ap.add_argument("--labels", default="labels_final.csv")
    ap.add_argument("--checkpoint_dir", required=True); ap.add_argument("--model_kz", default="kazroberta"); ap.add_argument("--model_ru", default="mbert")
    ap.add_argument("--spans", default=None); ap.add_argument("--out", default=r"C:\meta_ozat\out\interpret"); ap.add_argument("--n_steps", type=int, default=32)
    a = ap.parse_args()
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    device = "cuda" if torch.cuda.is_available() else "cpu"
    df = C.load_data(a.data, a.labels); S = sample_100(df); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    print(f"device={device}; ответов: {len(S)} ({S.lang.value_counts().to_dict()}); модели KZ={a.model_kz}, RU={a.model_ru}", flush=True)
    cache, rows, summ = {}, [], []
    for i, (_, r) in enumerate(S.iterrows()):
        key = f"{a.model_kz if r.lang == 'kz' else a.model_ru}__{r.lang}__{r.question}"
        if key not in cache:
            cache.clear(); torch.cuda.empty_cache() if device == "cuda" else None
            p = Path(a.checkpoint_dir) / key
            cache[key] = (AutoTokenizer.from_pretrained(p), AutoModelForSequenceClassification.from_pretrained(p).to(device).eval())
        tok, model = cache[key]
        words, wa, pred = ig_words(model, tok, str(r.text), int(r.y), device, a.n_steps)
        cls = word_classes(words, r.lang, r.question)
        for w, v, c in zip(words, wa, cls):
            rows.append({"response_id": r.response_id, "lang": r.lang, "question": r.question, "y": int(r.y), "word": w, "attr": float(v), "class": c})
        absw = np.abs(wa); tot = absw.sum() + 1e-9
        rec = {"response_id": r.response_id, "lang": r.lang, "question": r.question, "y": int(r.y), "pred": pred, "n_words": len(words), "model": key,
               **{f"mass_{c}": float(absw[[x == c for x in cls]].sum() / tot) for c in ("marker", "topic", "other")},
               **{f"share_words_{c}": float(np.mean([x == c for x in cls])) if cls else np.nan for c in ("marker", "topic")}}
        summ.append(rec)
        if (i + 1) % 10 == 0: print(f"  {i+1}/{len(S)}", flush=True)
    W = pd.DataFrame(rows); W.to_csv(out / "attributions_words.csv", index=False, encoding="utf-8-sig")
    S2 = pd.DataFrame(summ)
    if a.spans:  # IoU: топ-20 % слов по |атрибуции| vs слова фрагментов экспертов
        sp = pd.read_csv(a.spans, encoding="utf-8-sig"); norm = lambda s: re.sub(r"[^\w]", "", s.lower()); iou = []
        for rid, g in W.groupby("response_id"):
            k = max(1, int(round(0.2 * len(g)))); top = {norm(w) for w in g.loc[g.attr.abs().nlargest(k).index, "word"]} - {""}
            vals = [len(top & ({norm(w) for w in s.split()} - {""})) / max(1, len(top | ({norm(w) for w in s.split()} - {""}))) for s in sp[sp.response_id == rid].span_text]
            iou.append((rid, np.mean(vals) if vals else np.nan))
        S2 = S2.merge(pd.DataFrame(iou, columns=["response_id", "iou"]), on="response_id", how="left")
    S2.to_csv(out / "interpret_summary.csv", index=False, encoding="utf-8-sig")
    # обзор: масса на классах (сравнить с долей слов этого класса = «ожидание при равномерной атрибуции»)
    ov = S2.groupby("lang")[[c for c in S2.columns if c.startswith(("mass_", "share_words_", "iou"))]].mean().round(3)
    ov["n"] = S2.groupby("lang").size(); ov["acc_pred_eq_y"] = S2.assign(ok=S2.pred == S2.y).groupby("lang").ok.mean().round(3)
    ov.to_csv(out / "interpret_overview.csv", encoding="utf-8-sig")
    # самые «весомые» слова (средняя |атрибуция|, слово встречается в ≥ 3 ответах)
    W["w"] = W.word.str.lower(); W["abs"] = W.attr.abs()
    top = (W.groupby(["lang", "w"]).agg(mean_abs=("abs", "mean"), mean_signed=("attr", "mean"), n_resp=("response_id", "nunique"), cls=("class", lambda s: s.mode().iat[0]))
           .query("n_resp >= 3").sort_values(["lang", "mean_abs"], ascending=[True, False]).groupby("lang").head(30).round(4))
    top.to_csv(out / "interpret_top_words.csv", encoding="utf-8-sig")
    print(ov.to_string()); print(top.groupby("lang").head(12).to_string())


if __name__ == "__main__":
    main()
