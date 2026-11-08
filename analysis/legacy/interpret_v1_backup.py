"""
interpret.py — интерпретируемость (Methods 4.6): Integrated Gradients (captum) для дообученного энкодера
на 100 стратифицированных ответах (50/язык, 10/промпт) и сравнение с опорными фрагментами экспертов.

Вход:
  --checkpoint_dir  папка с чекпойнтами <model>__<lang>__<q> (train_encoder.py --save_dir сохраняет mono-модели fold 0)
  --spans       spans.csv: response_id, rater_id, span_text   (фрагменты, которые эксперт отметил как решающие; собираются при калибровке)
Выход: analysis_out/attributions.csv (токен, атрибуция), analysis_out/interpret_summary.csv (IoU по ответам/языкам,
       доля массы атрибуций на маркерах рассуждения vs тематических словах vs длине).
Smoke: python interpret.py --dry  (случайные атрибуции на синтетике — только проверка формата)
"""
import argparse, re
from pathlib import Path
import numpy as np, pandas as pd
import common as C

REASONING_MARKERS = {"ru": ["потому что","например","поэтому","если","но","однако","с другой стороны","во-первых","во-вторых","значит","хотя","зато","с одной стороны","наоборот"],
                     "kz": ["өйткені","мысалы","сондықтан","егер","бірақ","алайда","екінші жағынан","біріншіден","екіншіден","демек","дегенмен","керісінше"]}

def sample_100(df: pd.DataFrame, seed=11) -> pd.DataFrame:
    rng = np.random.RandomState(seed); parts = []
    for lang in C.LANGS:
        for q in C.PROMPTS:
            pool = df[(df.lang == lang) & (df.question == q)]
            # стратификация по баллу: по 2 на уровень
            for lvl in C.LEVELS:
                pl = pool[pool.y == lvl]
                if len(pl): parts.append(pl.sample(n=min(2, len(pl)), random_state=rng))
    return pd.concat(parts).drop_duplicates("response_id")

def token_iou(attr_tokens: list[str], attr_vals: np.ndarray, span_text: str, top_frac=0.2) -> float:
    """IoU между top-20 % токенов по |атрибуции| и токенами опорного фрагмента (по нормализованным словам)."""
    norm = lambda s: re.sub(r"[^\w]", "", s.lower().replace("▁", ""))
    k = max(1, int(len(attr_tokens) * top_frac)); top = set(norm(attr_tokens[i]) for i in np.argsort(-np.abs(attr_vals))[:k]) - {""}
    span = set(norm(w) for w in span_text.split()) - {""}
    return len(top & span) / max(1, len(top | span))

def attributions_ig(model, tok, text: str, target: int, device):
    import torch
    from captum.attr import LayerIntegratedGradients
    enc = tok(text, return_tensors="pt", truncation=True, max_length=256).to(device)
    emb_layer = model.get_input_embeddings()
    def fwd(ids, mask): return model(input_ids=ids, attention_mask=mask).logits
    lig = LayerIntegratedGradients(fwd, emb_layer)
    base = torch.full_like(enc["input_ids"], tok.pad_token_id)
    attr = lig.attribute(enc["input_ids"], baselines=base, additional_forward_args=(enc["attention_mask"],), target=target, n_steps=32)
    a = attr.sum(-1).squeeze(0).detach().cpu().numpy(); a = a / (np.abs(a).sum() + 1e-9)
    return tok.convert_ids_to_tokens(enc["input_ids"][0]), a

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(C.DEFAULT_DATA)); ap.add_argument("--labels", default="labels_final.csv")
    ap.add_argument("--checkpoint_dir", default=None, help="папка с чекпойнтами <model>__<lang>__<q> из train_encoder.py --save_dir"); ap.add_argument("--model", default="xlmr"); ap.add_argument("--spans", default=None); ap.add_argument("--dry", action="store_true")
    ap.add_argument("--out", default=str(C.OUT_DIR))
    a = ap.parse_args(); df = C.load_data(a.data, a.labels); S = sample_100(df); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    spans = pd.read_csv(a.spans, encoding="utf-8-sig") if a.spans else None
    if not a.dry:
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        device = "cuda" if torch.cuda.is_available() else "cpu"
        cache = {}
        def get(lang, q):
            key = f"{a.model}__{lang}__{q}"
            if key not in cache:
                p = Path(a.checkpoint_dir) / key
                cache[key] = (AutoTokenizer.from_pretrained(p), AutoModelForSequenceClassification.from_pretrained(p).to(device).eval())
            return cache[key]
    rows, summ = [], []
    for _, r in S.iterrows():
        if a.dry:
            toks = str(r.text).split(); vals = np.random.RandomState(len(toks)).randn(len(toks)); vals /= np.abs(vals).sum()
        else:
            tok, model = get(r.lang, r.question); toks, vals = attributions_ig(model, tok, str(r.text), int(r.y), device)
        for t, v in zip(toks, vals): rows.append({"response_id": r.response_id, "lang": r.lang, "question": r.question, "token": t, "attr": float(v)})
        mk = REASONING_MARKERS[r.lang]; mass_markers = float(sum(abs(v) for t, v in zip(toks, vals) if any(m == t.lower().replace("▁","") for m in mk)))
        rec = {"response_id": r.response_id, "lang": r.lang, "question": r.question, "y": int(r.y), "n_tokens": len(toks), "mass_on_reasoning_markers": mass_markers}
        if spans is not None:
            sp = spans[spans.response_id == r.response_id]
            rec["iou_mean"] = float(np.mean([token_iou(toks, vals, s) for s in sp.span_text])) if len(sp) else np.nan
        summ.append(rec)
    pd.DataFrame(rows).to_csv(out / "attributions.csv", index=False, encoding="utf-8-sig")
    S2 = pd.DataFrame(summ); S2.to_csv(out / "interpret_summary.csv", index=False, encoding="utf-8-sig")
    print(S2.groupby("lang")[["mass_on_reasoning_markers"] + (["iou_mean"] if "iou_mean" in S2 else [])].mean().round(3))

if __name__ == "__main__":
    main()
