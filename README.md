# Workflow-CI (contoh struktur repo)

Folder ini contoh isi repo untuk Kriteria 3.
Di repo GitHub PUBLIC Anda, letakkan file-file berikut di root repo (bukan hanya di zip):
- MLProject
- modelling.py / src training
- requirements.txt
- .github/workflows/ci.yml

Workflow contoh melakukan:
1) install dependency
2) run training
3) upload artifacts (contoh: model file / report) sebagai artifact workflow
4) (opsional advanced) build & push docker image (butuh secrets)
