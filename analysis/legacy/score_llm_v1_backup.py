"""
score_llm.py — оценка ответов открытой LLM (Methods 4.2): zero-shot и few-shot (5 размеченных примеров на уровень
из обучающего фолда) с рубрикой v1.1 на языке ответа; greedy, reasoning отключён, ответ — одна цифра 0–4.
Та же кросс-валидация по ученику (нужна только для few-shot: примеры берутся из обучающего фолда; zero-shot
вычисляется один раз, но записывается в тот же формат с fold по seed 11 для сопоставимости).

Запуск на Legion 5 (4-бит):  python score_llm.py --data <clean> --labels labels_final.csv --repo issai/LLama-3.1-KazLLM-1.0-8B --model llm
Smoke без модели (случайный «оценщик», проверка формата файлов): python score_llm.py --data ../test_clean --labels labels_final_SYNTHETIC.csv --dry
Промпт и рубрика — те же, что в gpu/smoke_test_llm.py (единый источник: prompts_llm.py).
"""
import argparse, os, re, time
from pathlib import Path
os.environ.setdefault("HF_HUB_OFFLINE", "1"); os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
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
        self.tok = AutoTokenizer.from_pretrained(repo)
        def bnb(off):
            return BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16,
                                      bnb_4bit_use_double_quant=True, llm_int8_enable_fp32_cpu_offload=off)
        attempts = [("GPU only", {"device_map": {"": 0}, "dtype": torch.float16, "quantization_config": bnb(False)}),
                    ("GPU + CPU offload", {"device_map": "auto", "max_memory": {0: "6.5GiB", "cpu": "24GiB"}, "dtype": torch.float16, "quantization_config": bnb(True)})] if quant else [("fp16", {"device_map": "auto", "dtype": torch.float16})]
        last = None
        for name, kw in attempts:
            try:
                self.model = AutoModelForCausalLM.from_pretrained(repo, **kw); print(f"[score_llm] loaded via {name}"); break
            except Exception as e:
                last = e; print(f"[score_llm] load failed ({name}): {str(e)[:200]}"); import gc; gc.collect(); torch.cuda.empty_cache()
        else:
            raise last
        self.model.eval()
        try:
            from huggingface_hub import scan_cache_dir
            for rep in scan_cache_dir().repos:
                if rep.repo_id == repo: self.rev = ",".join(r.commit_hash[:12] for r in rep.revisions)
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
    ap.add_argument("--repo", default="issai/Qwen3.5-9B-Kazakh"); ap.add_argument("--model", default="qwen_kz")  # KazLLM supplementary: --repo issai/LLama-3.1-KazLLM-1.0-8B --model kazllm_supp
    ap.add_argument("--modes", nargs="*", default=["zeroshot_llm", "fewshot_llm"]); ap.add_argument("--n_per_level", type=int, default=5)
    ap.add_argument("--targets", nargs="*", default=["kz", "ru"]); ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--dry", action="store_true"); ap.add_argument("--no-quant", action="store_true"); ap.add_argument("--out", default=str(C.PRED_DIR))
    a = ap.parse_args()
    if a.labels.lower() == "none":
        # без итоговых меток: только zero-shot; y = -1, берутся все ответы, допущенные к разметке (batch_id != excluded)
        df = pd.read_csv(Path(a.data) / "responses.csv", encoding="utf-8-sig").rename(columns={"form_lang": "lang"})
        if "batch_id" in df: df = df[df.batch_id != "excluded"]
        df = df[df.text.fillna("").str.len() > 0].copy(); df["y"] = -1
        if "n_chars" not in df: df["n_chars"] = df.text.str.len()
        a.modes = [m for m in a.modes if m != "fewshot_llm"]
        print(f"[score_llm] без меток: {len(df)} ответов, режимы {a.modes}")
    else:
        df = C.load_data(a.data, a.labels)
    sc = Scorer(a.repo, quant=not a.no_quant, dry=a.dry)
    folds = C.make_folds(df, a.seed)
    out_dir = Path(a.out); out_dir.mkdir(parents=True, exist_ok=True)
    for mode in a.modes:
        for target in a.targets:
            spec = C.RunSpec(model=a.model, condition=mode, target_lang=target, k=(a.n_per_level * 5 if mode == "fewshot_llm" else 0), seed=a.seed)
            rows = []; t0 = time.time(); invalid = 0
            part = out_dir / (spec.name() + ".partial.csv")          # возобновление после прерывания
            done = set()
            if part.exists():
                prev = pd.read_csv(part, encoding="utf-8-sig"); rows = prev.to_dict("records"); done = set(prev.response_id); invalid = int(prev.invalid_format.sum())
                print(f"[score_llm] {spec.name()}: продолжаю, уже есть {len(done)}")
            n_since_save = 0
            for q in C.PROMPTS:
                for f in range(C.N_FOLDS):
                    test = df[(df.question == q) & (df.lang == target) & (folds == f)]
                    ex = pick_examples(df[(df.question == q) & (df.lang == target) & (folds != f)], a.n_per_level, a.seed + f) if mode == "fewshot_llm" else None
                    for _, r in test.iterrows():
                        if r.response_id in done: continue
                        msgs = build_messages(target, q, r.text)
                        if ex is not None and len(ex):
                            msgs[1]["content"] = fewshot_block(ex, target) + "\n\n" + msgs[1]["content"]
                        y, raw = sc.score(msgs)
                        bad = int(y is None or not DIGIT.match(raw)); invalid += bad
                        rows.append({"response_id": r.response_id, "student_id": r.student_id, "lang": r.lang, "question": q, "n_chars": r.n_chars,
                                     "fold": f, "y_true": int(r.y), "y_pred": int(y if y is not None else 0), "invalid_format": bad,
                                     "raw_output": raw.replace("\n", "\\n")[:60], "revision": sc.rev, "repo": a.repo})
                        n_since_save += 1
                        if n_since_save >= 25:
                            pd.DataFrame(rows).to_csv(part, index=False, encoding="utf-8-sig"); n_since_save = 0
                            print(f"  {spec.name()}: {len(rows)} готово, {(time.time()-t0)/60:.0f} мин", flush=True)
            C.save_predictions(spec, rows, out_dir)
            if part.exists(): part.unlink()
            if len(rows) and all(r["y_true"] >= 0 for r in rows):
                m = C.metrics([r["y_true"] for r in rows], [r["y_pred"] for r in rows])
                print(f"{spec.name():50s} n={m['n']} QWK={m['qwk']:.3f} invalid={invalid} ({time.time()-t0:.0f}s)")
            else:
                print(f"{spec.name():50s} n={len(rows)} (без меток — QWK будет посчитан после разметки) invalid={invalid} ({time.time()-t0:.0f}s)")

if __name__ == "__main__":
    main()
