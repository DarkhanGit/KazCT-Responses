# run_encoders.ps1 - baselines + all encoder conditions for article 3 (no machine translation yet).
# Run from C:\meta_ozat\analysis:
#   ..\gpu\.venv\Scripts\Activate.ps1 ; .\run_encoders.ps1
# Safe to stop (Ctrl+C) and rerun at any time: finished runs (csv in the output folder) are skipped.
# ASCII only (Windows PowerShell 5.1).

$DATA = "C:\meta_ozat\data\clean_2026-10-20"
$OUT  = "C:\meta_ozat\out\predictions"
$CKPT = "C:\meta_ozat\out\checkpoints"
$LOG  = "C:\meta_ozat\out\run_encoders_log.txt"
New-Item -ItemType Directory -Force -Path $OUT, $CKPT | Out-Null
$env:PYTHONIOENCODING = "utf-8"
$env:HF_HUB_OFFLINE = "1"
$env:PYTHONUNBUFFERED = "1"
$ErrorActionPreference = "Continue"

function Step($title, $argsList) {
  Write-Host ""
  Write-Host "==================== $title  ($(Get-Date -Format 'yyyy-MM-dd HH:mm'))"
  "==================== $title  $(Get-Date -Format 'yyyy-MM-dd HH:mm')" | Out-File -Append -Encoding utf8 $LOG
  & python @argsList 2>&1 | ForEach-Object { "$_" } | Tee-Object -Append -FilePath $LOG | Where-Object { $_ -notmatch "val_qwk" }
}

Step "1/4 baselines (LR n-grams, length-only)" @("baselines.py","--data",$DATA,"--labels","labels_final.csv","--out",$OUT)
Step "2/4 XLM-R base: mono, joint, zeroshot+fewshot" @("train_encoder.py","--data",$DATA,"--labels","labels_final.csv","--model","xlmr","--conditions","mono","joint","zeroshot","fewshot","--out",$OUT,"--save_dir",$CKPT)
Step "3/4 mBERT: mono, joint, zeroshot+fewshot" @("train_encoder.py","--data",$DATA,"--labels","labels_final.csv","--model","mbert","--conditions","mono","joint","zeroshot","fewshot","--out",$OUT,"--save_dir",$CKPT)
Step "4/4 Kaz-RoBERTa: mono, mono_k (Kazakh only)" @("train_encoder.py","--data",$DATA,"--labels","labels_final.csv","--model","kazroberta","--conditions","mono","fewshot","--out",$OUT,"--save_dir",$CKPT)

Write-Host ""
Write-Host "ALL DONE $(Get-Date -Format 'yyyy-MM-dd HH:mm'). Copy folder C:\meta_ozat\out to the project folder analysis_out and tell Claude."
