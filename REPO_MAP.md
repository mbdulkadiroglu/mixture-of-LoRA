# Repo Map

Canonical vision and research scope live in [tasks/project_vision.md](/data/mehmet/projects/mixture-of-LoRA/tasks/project_vision.md).

The current research core is:

- `cascade/`: cascade distillation experiments, phase scripts, logging, calibration, analysis
- `src/models/`: student and teacher wrappers used by cascade
- `src/training/`: data formatting and LoRA training used by cascade
- `src/datasets/`: Spider and BIRD loading plus schema augmentation
- `src/evaluation/sql_*`: SQL extraction and execution-based evaluation
- `bird_data/`, `spider_data/`: benchmark data and databases

The main execution path today is:

- `python -m cascade.run_experiment`
- `python -m cascade.phase1.exp_...`

Likely legacy code, kept for now until deletion is verified:

- `src/framework.py`
- `src/router/router.py`
- `src/adapters/manager.py`
- `src/evaluation/evaluator.py`
- `configs/config.yaml`
- `cascade/scripts/*.py` files that import `AdaptiveSLMFramework`

The cleanup goal is to preserve the cascade-distillation research path and remove code and docs that describe the repo as a general framework or app.
