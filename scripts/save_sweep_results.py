"""Save all sweep experiment results to a JSON file. Run anytime to update."""
import json
import sqlite3
from datetime import datetime
from pathlib import Path

import yaml


def save_results(output_path: str = "results/cascade/sweep_results.json"):
    all_results = {}
    for d in sorted(Path("results/cascade").glob("exp_sweep_*")):
        dbs = list(d.glob("*.db"))
        if not dbs:
            continue
        name = d.name.replace("exp_sweep_", "")
        conn = sqlite3.connect(dbs[0])
        details = conn.execute("""
            SELECT round, eval_accuracy, training_loss, new_examples_count, replay_examples_count
            FROM adapter_versions ORDER BY round
        """).fetchall()
        conn.close()

        config_path = d / "config.yaml"
        config = {}
        if config_path.exists():
            with open(config_path) as f:
                config = yaml.safe_load(f)

        rounds = {
            f"R{r}": {
                "eval_accuracy": acc,
                "training_loss": loss,
                "new_examples": new_ex,
                "replay_examples": replay_ex,
            }
            for r, acc, loss, new_ex, replay_ex in details
            if acc is not None
        }

        all_results[name] = {
            "status": "complete" if len(rounds) == 4 else "in_progress",
            "rounds": rounds,
            "config": {
                "training_lr": config.get("training_lr"),
                "training_epochs": config.get("training_epochs"),
                "training_grad_accum": config.get("training_grad_accum"),
                "training_batch_size": config.get("training_batch_size"),
                "replay_ratio": config.get("replay_ratio"),
                "training_source": config.get("training_source"),
                "train_start_round": config.get("train_start_round"),
            },
        }

    output = {
        "generated_at": datetime.now().isoformat(),
        "base_model_accuracy": 0.392,
        "baseline_old_run": {"R0": 0.389, "R1": 0.320, "R2": 0.286, "R3": 0.297},
        "teacher_cache_accuracy": 0.628,
        "experiments": all_results,
    }

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    complete = sum(1 for v in all_results.values() if v["status"] == "complete")
    print(f"Saved {len(all_results)} experiments ({complete} complete) to {out_path}")


if __name__ == "__main__":
    save_results()
