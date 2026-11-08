"""
score_llm.py — оценка ответов открытой LLM (Methods 4.2): zero-shot и few-shot (5 размеченных примеров на уровень
из обучающего фолда) с рубрикой v1.1 на языке ответа; greedy, reasoning отключён, ответ — одна цифра 0–4.
Та же кросс-валидация по ученику (нужна только для few-shot: примеры берутся из обучающего фолда; zero-shot
вычисляется один раз, но записывается в тот же формат с fold по seed 11 для сопоставимости).

Запуск на Legion 5 (4-бит):  python score_llm.py --data <clean> --labels labels_final.csv --repo issai/LLama-3.1-KazLLM-1.0-8B --model llm
Smoke без модели (случайный «оценщик», проверка формата файлов): python score_llm.py --data ../test_clean --labels labels_final_SYNTHETIC.csv --dry
Промпт и рубрика — те же, что в gpu/smoke_test_llm.py (единый источник: prompts_llm.py).
"""
import argparse, re, time
from pathlib import Path
import numpy as np, pandas as pd
import common as C
from prompts_llm import RUBRIC, PROMPTS, build_messages, DIGIT

def fewshot_block(examples: pd.DataFrame, lang: str) -> str:
    head = "Примеры оценённых ответов:" if lang == "ru" else "Бағаланған жауаптар мысалдары:"
    lines = [head]
    for _, e in examples.iterrows():
        lines.append(("Ответ: «{t}» → Балл: {y}" if lang == "ru" else "Жауап: «{t}» → Балл: {y}").format(t=str(e.text)[:400], y=int(e.y)))
    return "\n".join(lines)

def pick_examples(train_df: pd.DataFrame, n_per_level: int, seed: int) -> pd.DataFrame:
    rng = np.random.RandomState(seed); parts = []
    for lvl in C.LEVELS:
        pool = train_df[train_df.y == lvl]
        if len(pool): parts.append(pool.sample(n=min(n_per_level, len(pool)), random_state=rng))
    return pd.concat(parts).sample(frac=1, random_state=rng) if parts else train_df.iloc[:0]

class Scorer:
    def __init__(self, repo, quant=True, dry=False):
        self.dry = dry; self.repo = repo; self.rev = None
        if dry: return
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        kw = {"device_map": "auto", "dtype": torch.float16}
        if quant:
            kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
        self.tok = AutoTokenizer.from_pretrained(repo)
        try:
            self.model = AutoModelForCausalLM.from_pretrained(repo, **kw)
        except Exception:
            from transformers import AutoModelForImageTextToText
            self.model = AutoModelForImageTextToText.from_pretrained(repo, **kw)
        self.model.eval()
        try:
            from huggingface_hub import model_info; self.rev = model_info(repo).sha
        except Exception: pass
        self.torch = torch

    def score(self, messages) -> tuple[int | None, str]:
        if self.dry:
            g = str(np.random.randint(0, 5)) if np.random.rand() > 0.03 else "Балл: 3"
        else:
            kw = dict(add_generation_prompt=True, tokenize=False)
            try: prompt = self.tok.apply_chat_template(messages, enable_thinking=False, **kw)
            except TypeError: prompt = self.tok.apply_chat_template(messages, **kw)
            enc = self.tok(prompt, return_tensors="pt").to(self.model.device)
            with self.torch.no_grad():
                out = self.model.generate(**enc, max_new_tokens=8, do_sample=False, temperature=None, top_p=None, pad_token_id=self.tok.eos_token_id)
            g = self.tok.decode(out[0][enc["input_ids"].shape[-1]:], skip_special_tokens=True)
        m = DIGIT.match(g)
        if m: return int(m.group(1)), g
        m2 = re.search(r"[0-4]", g)          # нестрогий разбор — помечается invalid_format=1
        return (int(m2.group(0)) if m2 else None), g

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(C.DEFAULT_DATA)); ap.add_argument("--labels", default="labels_final.csv")
    ap.add_argument("--repo", default="issai/LLama-3.1-KazLLM-1.0-8B"); ap.add_argument("--model", default="llm")
    ap.add_argument("--modes", nargs="*", default=["zeroshot_llm", "fewshot_llm"]); ap.add_argument("--n_per_level", type=int, default=5)
    ap.add_argument("--targets", nargs="*", default=["kz", "ru"]); ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--dry", action="store_true"); ap.add_argument("--no-quant", action="store_true"); ap.add_argument("--out", default=str(C.PRED_DIR))
    a = ap.parse_args()
    df = C.load_data(a.data, a.labels); sc = Scorer(a.repo, quant=not a.no_quant, dry=a.dry)
    folds = C.make_folds(df, a.seed)
    for mode in a.modes:
        for target in a.targets:
            spec = C.RunSpec(model=a.model, condition=mode, target_lang=target, k=(a.n_per_level * 5 if mode == "fewshot_llm" else 0), seed=a.seed)
            rows = []; t0 = time.time(); invalid = 0
            for q in C.PROMPTS:
                for f in range(C.N_FOLDS):
                    test = df[(df.question == q) & (df.lang == target) & (folds == f)]
                    ex = pick_examples(df[(df.question == q) & (df.lang == target) & (folds != f)], a.n_per_level, a.seed + f) if mode == "fewshot_llm" else None
                    for _, r in test.iterrows():
                        msgs = build_messages(target, q, r.text)
                        if ex is not None and len(ex):
                            msgs[1]["content"] = fewshot_block(ex, target) + "\n\n" + msgs[1]["content"]
                        y, raw = sc.score(msgs)
                        bad = int(y is None or not DIGIT.match(raw)); invalid += bad
                        rows.append({"response_id": r.response_id, "student_id": r.student_id, "lang": r.lang, "question": q, "n_chars": r.n_chars,
                                     "fold": f, "y_true": int(r.y), "y_pred": int(y if y is not None else 0), "invalid_format": bad,
                                     "raw_output": raw.replace("\n", "\\n")[:60], "revision": sc.rev, "repo": a.repo})
            C.save_predictions(spec, rows, Path(a.out))
            m = C.metrics([r["y_true"] for r in rows], [r["y_pred"] for r in rows])
            print(f"{spec.name():50s} n={m['n']} QWK={m['qwk']:.3f} invalid={invalid} ({time.time()-t0:.0f}s)")

if __name__ == "__main__":
    main()
