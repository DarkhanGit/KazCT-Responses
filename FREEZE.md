# Заморозка кода для пре-регистрации

1. Убедиться, что `analysis/` проходит на синтетике: `cd analysis && bash run_all.sh ../synthetic labels_final_SYNTHETIC.csv` (или по частям: baselines → report).
2. `pip freeze > requirements-lock.txt`.
3. `git add -A && git commit -m "Pre-registration freeze: analysis pipeline v1.0"`
4. `git tag -a prereg-v1.0 -m "Analysis pipeline as pre-registered on OSF"` и `git push --tags`.
5. Вставить в OSF-регистрацию: URL репозитория, тег `prereg-v1.0` и хеш коммита (`git rev-parse prereg-v1.0`).
6. После регистрации изменения в analysis/ допускаются только с записью в `preregistration/DEVIATIONS.md` (что, почему, дата).
