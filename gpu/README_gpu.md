# Папка gpu/ — запуск на Lenovo Legion 5 (RTX 5060 Laptop, 8 ГБ)

Порядок:
1. Скопировать папку `gpu` на ноутбук (например, в `C:\meta_ozat\gpu`).
2. PowerShell в этой папке: `Set-ExecutionPolicy -Scope Process Bypass; .\setup_env.ps1` — 10–20 минут. В конце должно быть `matmul on GPU: OK` и версия bitsandbytes. Если нет — прислать весь вывод.
3. Аккаунт Hugging Face: на страницах https://huggingface.co/issai/LLama-3.1-KazLLM-1.0-8B и https://huggingface.co/inception42/Llama-3.1-Sherkala-8B-Chat нажать «Agree and access» (одобрение автоматическое). Затем в PowerShell: `.\.venv\Scripts\Activate.ps1; huggingface-cli login` и вставить токен (Settings → Access Tokens → Read).
4. `python smoke_test_llm.py` — скачает три модели (~5 ГБ каждая в 4 бит; суммарно ~15–20 ГБ, скачивается полный fp16, ~16 ГБ на модель, поэтому место на диске: ≥ 60 ГБ свободно) и прогонит 60 синтетических ответов. 30–90 минут в зависимости от интернета.
5. Прислать `smoke_results.csv` и `smoke_outputs.csv`.

Критерий выбора модели (зафиксирован в Methods 4.8 и пре-регистрации): load_ok = True, format_valid ≥ 0.95, при равенстве — предпочтение руководителя (KazLLM), затем скорость.

Файлы: `setup_env.ps1`, `check_gpu.py`, `smoke_test_llm.py`, `smoke_responses.csv` (60 СИНТЕТИЧЕСКИХ ответов, 30 RU + 30 KZ, по 6 на промпт; не полевые данные).
Типичные проблемы: «CUDA not available» — установлена сборка не cu128 или старый драйвер; «401 Unauthorized / gated» — не принята лицензия или нет login; «OutOfMemory» — закрыть браузер и другие программы, использующие GPU, повторить с одной моделью `--models kazllm`.
