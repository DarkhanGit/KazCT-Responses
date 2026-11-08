#!/usr/bin/env python3
"""
smoke_test_llm.py — smoke-тест кандидатов LLM для статьи 3 (AP25796141, KazCT-Responses).

Что проверяет для каждой модели (на 60 СИНТЕТИЧЕСКИХ ответах, 30 RU + 30 KZ, smoke_responses.csv):
  1. load_ok        — загружается ли в 4-битном квантовании на GPU 8 ГБ (bitsandbytes NF4)
  2. vram_gb        — пиковая занятая видеопамять
  3. format_valid   — доля ответов, где модель вернула ровно одну цифру 0–4 (критерий ≥ 0.95)
  4. sec_per_resp   — секунд на один ответ (zero-shot, greedy, ≤ 8 новых токенов)
  5. kz_valid / ru_valid — доля валидных отдельно по языкам
Качество оценок НЕ проверяется — данные синтетические, меток нет. Это тест пригодности, не точности.

Запуск (после setup_env.ps1 и huggingface-cli login):
    python smoke_test_llm.py                      # все три кандидата
    python smoke_test_llm.py --models kazllm      # один
    python smoke_test_llm.py --no-quant           # без 4-бит (только если VRAM ≥ 20 ГБ)
Результат: smoke_results.csv (сводка) и smoke_outputs.csv (все сырые ответы) — прислать Claude.
Если модель не загружается, в csv будет строка с load_ok=False и текстом ошибки; это тоже результат.
"""
import argparse, csv, gc, json, os, re, sys, time, traceback
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("HF_HUB_OFFLINE", "1")          # модели уже в кэше: не ходить в интернет
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")   # консоль Windows и кириллица
except Exception:
    pass

import pandas as pd
import torch

MODELS = {
    "kazllm":   "issai/LLama-3.1-KazLLM-1.0-8B",          # gated, CC BY-NC 4.0
    "qwen":     "issai/Qwen3.5-9B-Kazakh",                # Apache-2.0; reasoning-модель, think отключаем
    "sherkala": "inception42/Llama-3.1-Sherkala-8B-Chat",  # gated, CC BY-NC-SA 4.0
}

from prompts_llm import PROMPTS, RUBRIC, build_messages, DIGIT  # единый источник промптов

def load_model(repo, quant):
    """Три попытки: (1) CausalLM целиком на GPU; (2) CausalLM с выгрузкой лишнего на CPU (fp32 offload, нужно 32 ГБ RAM);
    (3) мультимодальный класс (Qwen3.5) с выгрузкой блока изображений на CPU. Каждая неудача печатается."""
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    tok = AutoTokenizer.from_pretrained(repo)
    def bnb(offload):
        return BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16,
                                  bnb_4bit_use_double_quant=True, llm_int8_enable_fp32_cpu_offload=offload)
    attempts = []
    if quant:
        attempts.append(("CausalLM, GPU only", AutoModelForCausalLM, {"device_map": {"": 0}, "dtype": torch.float16, "quantization_config": bnb(False)}))
        attempts.append(("CausalLM, GPU + CPU offload", AutoModelForCausalLM, {"device_map": "auto", "max_memory": {0: "6.5GiB", "cpu": "24GiB"}, "dtype": torch.float16, "quantization_config": bnb(True)}))
        try:
            from transformers import AutoModelForImageTextToText
            attempts.append(("ImageTextToText, GPU + CPU offload", AutoModelForImageTextToText, {"device_map": "auto", "max_memory": {0: "6.5GiB", "cpu": "24GiB"}, "dtype": torch.float16, "quantization_config": bnb(True)}))
        except Exception:
            pass
    else:
        attempts.append(("CausalLM fp16", AutoModelForCausalLM, {"device_map": "auto", "dtype": torch.float16}))
    last = None
    for name, cls, kw in attempts:
        try:
            print(f"  load attempt: {name}")
            model = cls.from_pretrained(repo, **kw); model.eval(); print(f"  loaded via: {name}")
            return tok, model
        except Exception as e:
            last = e; print(f"  failed ({name}): {type(e).__name__}: {str(e)[:300]}")
            gc.collect(); torch.cuda.empty_cache()
    raise last

LLAMA3_TEMPLATE = (
    "{{ bos_token }}{% for message in messages %}"
    "<|start_header_id|>{{ message['role'] }}<|end_header_id|>\n\n{{ message['content'] | trim }}<|eot_id|>"
    "{% endfor %}{% if add_generation_prompt %}<|start_header_id|>assistant<|end_header_id|>\n\n{% endif %}"
)

def apply_template(tok, messages):
    """Шаблон чата из токенизатора; если его нет (Sherkala-8B-Chat — Llama-3.1 без chat_template) — стандартный шаблон Llama-3."""
    kwargs = dict(add_generation_prompt=True, tokenize=False)
    if not getattr(tok, "chat_template", None):
        tok.chat_template = LLAMA3_TEMPLATE
        print("  note: tokenizer has no chat_template, using the standard Llama-3 template")
    try:
        return tok.apply_chat_template(messages, enable_thinking=False, **kwargs)  # Qwen3.x: отключить <think>
    except TypeError:
        return tok.apply_chat_template(messages, **kwargs)

@torch.no_grad()
def score_one(tok, model, messages, max_new=8):
    prompt = apply_template(tok, messages)
    enc = tok(prompt, return_tensors="pt").to(model.device)
    eos = [tok.eos_token_id] + [i for i in [tok.convert_tokens_to_ids("<|eot_id|>")] if isinstance(i, int) and i >= 0 and i != tok.unk_token_id]
    out = model.generate(**enc, max_new_tokens=max_new, do_sample=False, temperature=None, top_p=None,
                         pad_token_id=tok.eos_token_id, eos_token_id=eos)
    gen = tok.decode(out[0][enc["input_ids"].shape[-1]:], skip_special_tokens=True)
    return gen

def run_model(key, repo, data, quant, outputs):
    row = {"model_key": key, "repo": repo, "quant4bit": quant, "load_ok": False, "error": "", "vram_gb": None,
           "n": len(data), "format_valid": None, "ru_valid": None, "kz_valid": None, "sec_per_resp": None,
           "transformers": None, "revision": None}
    import transformers; row["transformers"] = transformers.__version__
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    try:
        t0 = time.time(); tok, model = load_model(repo, quant); load_s = time.time() - t0
        row["load_ok"] = True
        try:
            from huggingface_hub import scan_cache_dir
            for rep in scan_cache_dir().repos:
                if rep.repo_id == repo: row["revision"] = ",".join(r.commit_hash[:12] for r in rep.revisions)
        except Exception: pass
        valid = {"ru": [], "kz": []}; times = []
        for r in data.itertuples():
            t1 = time.time()
            try:
                gen = score_one(tok, model, build_messages(r.lang, r.question, r.text))
                err = ""
            except Exception as e:
                gen, err = "", f"{type(e).__name__}: {e}"[:200]
            dt = time.time() - t1; times.append(dt)
            # допускаем ответ вида "3" или "3." или "Балл: 3" — но строгий критерий = одна цифра
            strict = bool(DIGIT.match(gen))
            loose = re.search(r"[0-4]", gen) is not None
            valid[r.lang].append(strict)
            outputs.append({"model_key": key, "response_id": r.response_id, "lang": r.lang, "question": r.question,
                            "raw_output": gen.replace("\n", "\\n"), "strict_valid": strict, "loose_valid": loose,
                            "seconds": round(dt, 2), "error": err})
        row["ru_valid"] = round(sum(valid["ru"]) / max(1, len(valid["ru"])), 3)
        row["kz_valid"] = round(sum(valid["kz"]) / max(1, len(valid["kz"])), 3)
        row["format_valid"] = round((sum(valid["ru"]) + sum(valid["kz"])) / len(data), 3)
        row["sec_per_resp"] = round(sum(times) / len(times), 2)
        row["vram_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
        row["load_seconds"] = round(load_s, 1)
        del model, tok
    except Exception as e:
        row["error"] = f"{type(e).__name__}: {e}"[:400]
        traceback.print_exc()
    gc.collect(); torch.cuda.empty_cache()
    return row

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=list(MODELS), choices=list(MODELS))
    ap.add_argument("--data", default="smoke_responses.csv")
    ap.add_argument("--no-quant", action="store_true")
    a = ap.parse_args()
    if not torch.cuda.is_available():
        sys.exit("CUDA недоступна — сначала setup_env.ps1 / check_gpu.py")
    data = pd.read_csv(a.data, encoding="utf-8-sig")
    assert {"response_id", "question", "lang", "text"} <= set(data.columns)
    print(f"GPU: {torch.cuda.get_device_name(0)} | {len(data)} ответов | модели: {a.models}")
    results, outputs = [], []
    for key in a.models:
        print(f"\n=== {key}: {MODELS[key]} ===")
        results.append(run_model(key, MODELS[key], data, not a.no_quant, outputs))
        pd.DataFrame(results).to_csv("smoke_results.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(outputs).to_csv("smoke_outputs.csv", index=False, encoding="utf-8-sig")
        print(json.dumps(results[-1], ensure_ascii=False, indent=1))
    print("\nСводка (критерий: load_ok и format_valid >= 0.95):")
    for r in results:
        verdict = "PASS" if r["load_ok"] and (r["format_valid"] or 0) >= 0.95 else "FAIL"
        print(f"  {r['model_key']:9s} {verdict}  load={r['load_ok']} valid={r['format_valid']} "
              f"ru={r['ru_valid']} kz={r['kz_valid']} vram={r['vram_gb']}GB sec/resp={r['sec_per_resp']} {r['error'][:80]}")
    print("Пришлите smoke_results.csv и smoke_outputs.csv.")

if __name__ == "__main__":
    main()
