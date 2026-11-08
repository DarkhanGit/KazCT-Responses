# run_translate.ps1 - translate-test and translate-train for XLM-R and mBERT (article 3, OSF update 2: NLLB sentence-level).
# Run from C:\meta_ozat\analysis after run_encoders.ps1 (one GPU job at a time):
#   ..\gpu\.venv\Scripts\Activate.ps1 ; .\run_translate.ps1
# Safe to stop and rerun: finished runs are skipped. ASCII only.

$DATA = "C:\meta_ozat\data\clean_2026-10-20"
$MT   = "C:\meta_ozat\data\clean_2026-10-20\translations.csv"
$OUT  = "C:\meta_ozat\out\predictions"
$LOG  = "C:\meta_ozat\out\run_translate_log.txt"
New-Item -ItemType Directory -Force -Path $OUT | Out-Null
$env:PYTHONIOENCODING = "utf-8"
$env:HF_HUB_OFFLINE = "1"
$env:PYTHONUNBUFFERED = "1"
$ErrorActionPreference = "Continue"
if (-not (Test-Path $MT)) { Write-Host "translations.csv not found: $MT"; exit 1 }

function Step($title, $argsList) {
  Write-Host ""
  Write-Host "==================== $title  ($(Get-Date -Format 'yyyy-MM-dd HH:mm'))"
  "==================== $title  $(Get-Date -Format 'yyyy-MM-dd HH:mm')" | Out-File -Append -Encoding utf8 $LOG
  & python @argsList 2>&1 | ForEach-Object { "$_" } | Tee-Object -Append -FilePath $LOG | Where-Object { $_ -notmatch "val_qwk" }
}

Step "1/2 XLM-R base: translate_test, translate_train" @("train_encoder.py","--data",$DATA,"--labels","labels_final.csv","--model","xlmr","--conditions","translate_test","translate_train","--mt",$MT,"--out",$OUT)
Step "2/2 mBERT: translate_test, translate_train" @("train_encoder.py","--data",$DATA,"--labels","labels_final.csv","--model","mbert","--conditions","translate_test","translate_train","--mt",$MT,"--out",$OUT)

Write-Host ""
Write-Host "ALL DONE $(Get-Date -Format 'yyyy-MM-dd HH:mm'). Copy folder C:\meta_ozat\out to the project folder analysis_out and tell Claude."
