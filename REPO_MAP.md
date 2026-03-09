# Repo Map

Canonical vision and research scope live in [tasks/project_vision.md](/data/mehmet/projects/mixture-of-LoRA/tasks/project_vision.md).

The current research core is:

- `cascade/`: cascade distillation experiments, phase scripts, logging, calibration, analysis
- `cascade/results/experiment_archive_summary.json`: compact archive of deleted raw `exp_*` experiment folders
- `src/models/`: student and teacher wrappers used by cascade
- `src/training/`: data formatting and LoRA training used by cascade
- `src/datasets/`: Spider and BIRD loading plus schema augmentation
- `src/evaluation/sql_*`: SQL extraction and execution-based evaluation
- `bird_data/`, `spider_data/`: benchmark data and databases

The main execution path today is:

- `python -m cascade.run_experiment`
- `python -m cascade.phase1.exp_...`

The cleanup goal is to preserve the cascade-distillation research path and remove code and docs that describe the repo as a general framework or app.

The old generic framework layer has been removed. If new code is added, it should align with `project_vision.md` and the `cascade/` experiment path instead of reintroducing a parallel framework abstraction.
