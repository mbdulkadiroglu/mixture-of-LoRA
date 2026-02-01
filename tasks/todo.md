# BIRD Dataset Training Plan

## Status: Blocked on train.zip

**Issue**: `train.zip` (3.6GB) is corrupted/incomplete - missing end-of-central-directory signature. Needs re-download.

## Available Data
- ✅ `dev.zip` extracted: 1534 samples, 11 databases
- ❌ `train.zip`: Corrupted, cannot extract

## Tasks

### Phase 1: Infrastructure (Can proceed now)
- [x] Add BIRD dataset loader to `src/datasets/loader.py`
- [x] Add BIRD SQL execution evaluator (extend `sql_executor.py`)
- [x] Test infrastructure with dev set - **WORKING**
- [ ] Update `configs/config.yaml` with BIRD paths (optional)

### Phase 2: Training (Blocked on train.zip)
- [ ] Re-download train.zip
- [ ] Extract train data and databases
- [ ] Train LoRA adapter on BIRD train set
- [ ] Evaluate on dev set with execution accuracy

## BIRD Dataset Structure
```
bird_data/
├── dev_20240627/
│   ├── dev.json          # 1534 samples
│   ├── dev.sql           # SQL queries
│   ├── dev_tables.json   # Schema info
│   └── dev_databases/    # 11 SQLite databases
└── train/ (need re-download)
    ├── train.json
    └── train_databases/
```

## Sample Format
```json
{
  "question_id": 0,
  "db_id": "california_schools",
  "question": "What is the highest eligible free rate...",
  "evidence": "Eligible free rate for K-12 = ...",
  "SQL": "SELECT ... FROM frpm WHERE ...",
  "difficulty": "simple"
}
```
