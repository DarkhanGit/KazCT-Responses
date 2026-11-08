# download_models.ps1 (v2, 31.10.2026) - download ALL models for article 3 into the Hugging Face cache.
# Run on a fast network from C:\meta_ozat\gpu:  Set-ExecutionPolicy -Scope Process Bypass; .\download_models.ps1
# Needs: setup_env.ps1 done, 'hf auth login' done. Total ~50 GB. Safe to rerun: finished files are skipped.

$ErrorActionPreference = "Continue"
& .\.venv\Scripts\Activate.ps1

$repos = @(
  "FacebookAI/xlm-roberta-base",
  "google-bert/bert-base-multilingual-cased",
  "kz-transformers/kaz-roberta-conversational",
  "facebook/nllb-200-distilled-600M",
  "issai/LLama-3.1-KazLLM-1.0-8B",
  "issai/Qwen3.5-9B-Kazakh",
  "inception42/Llama-3.1-Sherkala-8B-Chat"
)

foreach ($r in $repos) {
  Write-Host ""
  Write-Host "===== $r ====="
  # one --exclude per pattern (the CLI treats a list after --exclude as file names)
  hf download $r --exclude "original/*" --exclude "*.pth" --exclude "*.gguf" --exclude "*.h5" --exclude "*.msgpack" --exclude "*.ckpt"
  if ($LASTEXITCODE -ne 0) { Write-Host "!!! FAILED: $r  (rerun this script later; finished parts are kept)" }
}

Write-Host ""
Write-Host "===== cache summary ====="
hf cache ls
Write-Host "Expected: 7 repos, roughly 45-55 GB in total. If a model shows ~0 B, rerun."
Write-Host "Done. Next (any network, GPU): python smoke_test_llm.py"
