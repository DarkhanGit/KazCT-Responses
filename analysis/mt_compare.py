"""
mt_compare.py — выбор системы машинного перевода для условий translate-test / translate-train (пре-регистрация, Analysis Plan).

Шаг 0 (носитель): python mt_compare.py make-template --data <clean> --labels labels_final.csv --n 50
    → human_translation_template.csv: 50 казахских ответов (стратифицированно по промптам и баллам; если меток нет — случайно).
      Носитель заполняет колонку text_ru_human (перевод на русский, по смыслу, без «улучшения» текста). ~1,5 часа.
Шаг 1: python mt_compare.py run --human human_translation_template.csv --systems nllb google yandex
    → переводы каждой системой → chrF (основная), BLEU, TER (sacrebleu) против человеческого перевода → mt_compare_results.csv + отчёт.
    nllb   — facebook/nllb-200-distilled-600M, локально (GPU/CPU), бесплатно, воспроизводимо
    google — Google Cloud Translation API v2; нужен ключ в переменной GOOGLE_API_KEY (или файл google_key.txt)
    yandex — Yandex Cloud Translate v2; нужны YANDEX_API_KEY и YANDEX_FOLDER_ID
    Любую систему можно пропустить — не указывать в --systems.
Шаг 2 (после выбора): python mt_compare.py translate-all --data <clean> --system <winner> --out <clean>/translations.csv
    → переводит ВСЕ ответы в обе стороны (kz→ru и ru→kz) в формат, который читает train_encoder.py --mt
    (response_id, mt_lang, text_mt, mt_system, mt_date). Переводы публикуются с корпусом.
"""
import argparse, os, sys, time, json
os.environ.setdefault("HF_HUB_OFFLINE", "1")   # модели только из кэша; тексты никуда не уходят
from pathlib import Path
from datetime import date
import numpy as np, pandas as pd

NLLB = "facebook/nllb-200-distilled-600M"
NLLB_LANG = {"kz": "kaz_Cyrl", "ru": "rus_Cyrl"}


def make_template(a):
    try:
        sys.path.insert(0, str(Path(__file__).parent)); import common as C
        df = C.load_data(a.data, a.labels)
    except Exception:
        r = pd.read_csv(Path(a.data) / "responses.csv", encoding="utf-8-sig").rename(columns={"form_lang": "lang"}); r["y"] = -1; df = r
    kz = df[(df.lang == "kz") & (df.text.fillna("").str.len() >= 60)]
    rng = np.random.RandomState(11); parts = []
    per_q = a.n // 5
    for q in ("E1", "E2", "E3", "E4", "E5"):
        pool = kz[kz.question == q]
        if "y" in pool and (pool.y >= 0).any():
            take = []
            for lvl in sorted(pool.y.unique()):
                take.append(pool[pool.y == lvl].sample(n=min(2, (pool.y == lvl).sum()), random_state=rng))
            sel = pd.concat(take)
            if len(sel) < per_q: sel = pd.concat([sel, pool.drop(sel.index).sample(n=min(per_q - len(sel), len(pool) - len(sel)), random_state=rng)])
            parts.append(sel.head(per_q))
        else:
            parts.append(pool.sample(n=min(per_q, len(pool)), random_state=rng))
    T = pd.concat(parts)[["response_id", "question", "text"]].rename(columns={"text": "text_kz"})
    T["text_ru_human"] = ""
    T.to_csv("human_translation_template.csv", index=False, encoding="utf-8-sig")
    print(f"human_translation_template.csv: {len(T)} ответов. Заполните колонку text_ru_human (перевод по смыслу, без правок содержания).")


class NLLBTranslator:
    def __init__(self):
        import torch
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        self.tok = AutoTokenizer.from_pretrained(NLLB); self.model = AutoModelForSeq2SeqLM.from_pretrained(NLLB)
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"; self.model.to(self.dev).eval(); self.torch = torch
    def __call__(self, texts, src, tgt, bs=8):
        out = []
        self.tok.src_lang = NLLB_LANG[src]
        for i in range(0, len(texts), bs):
            enc = self.tok(texts[i:i+bs], return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.dev)
            with self.torch.no_grad():
                gen = self.model.generate(**enc, forced_bos_token_id=self.tok.convert_tokens_to_ids(NLLB_LANG[tgt]), max_new_tokens=512, num_beams=4, no_repeat_ngram_size=4)  # защита от зацикливания (пилот 04.11: «интеллектуальная ×N» в KZ0108_E3)
            out += self.tok.batch_decode(gen, skip_special_tokens=True)
        return out


class GoogleTranslator:
    def __init__(self):
        import requests; self.r = requests
        self.key = os.environ.get("GOOGLE_API_KEY") or (Path("google_key.txt").read_text().strip() if Path("google_key.txt").exists() else None)
        if not self.key: raise SystemExit("Google: нет GOOGLE_API_KEY / google_key.txt")
    def __call__(self, texts, src, tgt, bs=50):
        out = []
        for i in range(0, len(texts), bs):
            r = self.r.post("https://translation.googleapis.com/language/translate/v2", params={"key": self.key},
                            json={"q": texts[i:i+bs], "source": "kk" if src == "kz" else "ru", "target": "kk" if tgt == "kz" else "ru", "format": "text"}, timeout=60)
            r.raise_for_status(); out += [t["translatedText"] for t in r.json()["data"]["translations"]]; time.sleep(0.2)
        return out


class YandexTranslator:
    def __init__(self):
        import requests; self.r = requests
        self.key, self.folder = os.environ.get("YANDEX_API_KEY"), os.environ.get("YANDEX_FOLDER_ID")
        if not (self.key and self.folder): raise SystemExit("Yandex: нет YANDEX_API_KEY / YANDEX_FOLDER_ID")
    def __call__(self, texts, src, tgt, bs=50):
        out = []
        for i in range(0, len(texts), bs):
            r = self.r.post("https://translate.api.cloud.yandex.net/translate/v2/translate", headers={"Authorization": f"Api-Key {self.key}"},
                            json={"folderId": self.folder, "texts": texts[i:i+bs], "sourceLanguageCode": "kk" if src == "kz" else "ru", "targetLanguageCode": "kk" if tgt == "kz" else "ru"}, timeout=60)
            r.raise_for_status(); out += [t["text"] for t in r.json()["translations"]]; time.sleep(0.2)
        return out


import re as _re
_SENT = _re.compile(r"(?<=[.!?…])\s+")


def split_sentences(text):
    """Простое разбиение на предложения по . ! ? … + пробел/перенос; переносы строк — тоже граница."""
    parts = []
    for line in str(text).splitlines():
        parts += [s.strip() for s in _SENT.split(line) if s.strip()]
    return parts or [""]


class NLLBSentTranslator(NLLBTranslator):
    """NLLB по предложениям: каждый ответ режется на предложения, переводится, склеивается пробелом."""
    def __call__(self, texts, src, tgt, bs=16):
        segs, owner = [], []
        for i, t in enumerate(texts):
            for s in split_sentences(t): segs.append(s); owner.append(i)
        hyp = NLLBTranslator.__call__(self, segs, src, tgt, bs=bs)
        out = [[] for _ in texts]
        for i, h in zip(owner, hyp): out[i].append(h)
        return [" ".join(x).strip() for x in out]


# google / yandex оставлены в коде только для полноты: НЕ использовать на реальных ответах —
# облачный перевод противоречит согласию участников (DEVIATIONS 03.11.2026).
SYSTEMS = {"nllb": NLLBTranslator, "nllb_sent": NLLBSentTranslator, "google": GoogleTranslator, "yandex": YandexTranslator}


def run(a):
    import sacrebleu
    H = pd.read_csv(a.human, encoding="utf-8-sig"); H = H[H.text_ru_human.fillna("").str.strip() != ""]
    if len(H) < 20: raise SystemExit(f"заполнено только {len(H)} переводов — нужно ≥ 20 (лучше все 50)")
    print(f"{len(H)} ответов с человеческим переводом")
    res, cols = [], {}
    for name in a.systems:
        try:
            tr = SYSTEMS[name](); t0 = time.time(); hyp = tr(H.text_kz.tolist(), "kz", "ru"); dt = time.time() - t0
        except Exception as e:
            print(f"[{name}] пропуск: {e}"); continue
        refs = [H.text_ru_human.tolist()]
        chrf = sacrebleu.corpus_chrf(hyp, refs).score; bleu = sacrebleu.corpus_bleu(hyp, refs).score; ter = sacrebleu.corpus_ter(hyp, refs).score
        res.append({"system": name, "n": len(H), "chrF": round(chrf, 2), "BLEU": round(bleu, 2), "TER": round(ter, 2), "seconds": round(dt, 1), "date": str(date.today())})
        cols[f"mt_{name}"] = hyp; print(f"{name:7s} chrF={chrf:.1f} BLEU={bleu:.1f} TER={ter:.1f} ({dt:.0f}s)")
    for k, v in cols.items(): H[k] = v
    H.to_csv("mt_compare_outputs.csv", index=False, encoding="utf-8-sig")
    R = pd.DataFrame(res).sort_values("chrF", ascending=False); R.to_csv("mt_compare_results.csv", index=False, encoding="utf-8-sig")
    if len(R):
        print(f"\nЛучшая по chrF: {R.iloc[0].system}. Зафиксировать в пре-регистрации: система, версия/дата, chrF={R.iloc[0].chrF}.")


def translate_all(a):
    r = pd.read_csv(Path(a.data) / "responses.csv", encoding="utf-8-sig").rename(columns={"form_lang": "lang"})
    tr = SYSTEMS[a.system](); rows = []
    for src, tgt in (("kz", "ru"), ("ru", "kz")):
        sub = r[r.lang == src]; texts = sub.text.fillna("").tolist()
        hyp = tr(texts, src, tgt)
        rows += [{"response_id": rid, "mt_lang": tgt, "text_mt": h, "mt_system": a.system + ("/" + NLLB if a.system.startswith("nllb") else ""), "mt_date": str(date.today())} for rid, h in zip(sub.response_id, hyp)]
        print(f"{src}->{tgt}: {len(sub)}")
    pd.DataFrame(rows).to_csv(a.out, index=False, encoding="utf-8-sig"); print("->", a.out)


def main():
    ap = argparse.ArgumentParser(); sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("make-template"); p.add_argument("--data", required=True); p.add_argument("--labels", default="labels_final.csv"); p.add_argument("--n", type=int, default=50)
    p = sp.add_parser("run"); p.add_argument("--human", default="human_translation_template.csv"); p.add_argument("--systems", nargs="*", default=["nllb", "nllb_sent"])
    p = sp.add_parser("translate-all"); p.add_argument("--data", required=True); p.add_argument("--system", required=True, choices=list(SYSTEMS)); p.add_argument("--out", required=True)
    a = ap.parse_args(); {"make-template": make_template, "run": run, "translate-all": translate_all}[a.cmd](a)

if __name__ == "__main__":
    main()
