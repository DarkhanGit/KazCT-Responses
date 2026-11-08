# setup_env.ps1 - environment for RTX 5060 Laptop (Blackwell, CUDA 12.8), Windows 11.
# Run from C:\meta_ozat\gpu in PowerShell:  Set-ExecutionPolicy -Scope Process Bypass; .\setup_env.ps1
# ASCII only (no Cyrillic) so that Windows PowerShell 5.1 parses it regardless of code page.

$ErrorActionPreference = "Stop"
Write-Host "== 1. Python =="
$py = Get-Command python -ErrorAction SilentlyContinue
if (-not $py) { Write-Host "Python not found. Install Python 3.11 or 3.12 from python.org (tick 'Add to PATH'), then rerun."; exit 1 }
python --version

Write-Host "== 2. NVIDIA driver =="
$smi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
if ($smi) { nvidia-smi } else { Write-Host "nvidia-smi not found - install/update the NVIDIA driver (GeForce Experience), then rerun." }

Write-Host "== 3. Virtual environment =="
if (-not (Test-Path ".venv")) { python -m venv .venv }
& .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip wheel

Write-Host "== 4. PyTorch (CUDA 12.8 build) =="
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

Write-Host "== 5. Libraries =="
pip install -U transformers "accelerate>=1.0" "bitsandbytes>=0.45" sentencepiece protobuf safetensors scikit-learn pandas numpy scipy tqdm huggingface_hub datasets evaluate sacrebleu openpyxl

Write-Host "== 6. GPU check from PyTorch =="
python check_gpu.py
if ($LASTEXITCODE -ne 0) { Write-Host "GPU check FAILED - send the whole output to Claude (save as setup_log.txt)."; exit 1 }

Write-Host "== 7. Hugging Face =="
Write-Host "KazLLM and Sherkala are gated: create an HF account, click 'Agree and access' on each model page,"
Write-Host "then run once:  huggingface-cli login   (token: huggingface.co -> Settings -> Access Tokens -> Read)"
Write-Host "Done. Next:  .\.venv\Scripts\Activate.ps1 ; python smoke_test_llm.py"
