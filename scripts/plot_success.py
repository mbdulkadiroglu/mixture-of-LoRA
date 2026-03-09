"""
Plot: Adaptive Teacher-Student System Success Visualization

Three accuracy lines:
  1. Student-only (full test set, no routing) -- measures learning
  2. System (routed: student + teacher together) -- measures user experience
  3. Teacher-only (upper bound) -- the ceiling

Plus teacher usage % on the right axis, and a cost plot.

Usage:
    python scripts/plot_success.py                    # mock data demo
    python scripts/plot_success.py --data results.json # real data
"""

import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from pathlib import Path

EXAMPLES_PER_ROUND = 100


def load_mock_data():
    """Simulated trajectory for visualization prototyping."""
    rounds = np.arange(0, 11)

    rng = np.random.default_rng(42)

    # Student-only accuracy: evaluated on full test set, no teacher help
    # Measures how much the student has actually learned
    base_accuracy = 72.3
    teacher_accuracy = 91.0
    student_accuracy = base_accuracy + (teacher_accuracy - base_accuracy) * (
        1 - np.exp(-0.25 * rounds)
    )
    student_accuracy += rng.normal(0, 0.5, len(rounds))
    student_accuracy[0] = base_accuracy

    # Teacher usage drops as router gains confidence in student
    teacher_usage = 100 * np.exp(-0.3 * rounds)
    teacher_usage = np.clip(teacher_usage + rng.normal(0, 1.5, len(rounds)), 5, 100)
    teacher_usage[0] = 100.0

    # System accuracy: what the user actually experiences
    # = (teacher_usage% * teacher_acc) + ((1 - teacher_usage%) * student_acc)
    # Always >= student-only because teacher catches student failures
    system_accuracy = (
        (teacher_usage / 100) * teacher_accuracy
        + (1 - teacher_usage / 100) * np.array(student_accuracy)
    )

    # Blended cost per query (teacher=$0.02/query, student=~free)
    teacher_cost, student_cost = 0.02, 0.001
    cost_per_query = (
        (teacher_usage / 100) * teacher_cost
        + (1 - teacher_usage / 100) * student_cost
    )

    return {
        "rounds": rounds.tolist(),
        "examples_per_round": EXAMPLES_PER_ROUND,
        "base_accuracy": float(base_accuracy),
        "teacher_accuracy": float(teacher_accuracy),
        "student_accuracy": student_accuracy.tolist(),
        "system_accuracy": system_accuracy.tolist(),
        "teacher_usage_pct": teacher_usage.tolist(),
        "cost_per_query": cost_per_query.tolist(),
    }


def plot_main(data, output_path: Path):
    """Dual-axis plot: three accuracy lines (left) vs teacher usage (right)."""
    rounds = data["rounds"]
    student_acc = data["student_accuracy"]
    system_acc = data["system_accuracy"]
    teacher_usage = data["teacher_usage_pct"]
    base_acc = data["base_accuracy"]
    teacher_acc = data["teacher_accuracy"]
    epr = data.get("examples_per_round", EXAMPLES_PER_ROUND)

    fig, ax1 = plt.subplots(figsize=(11, 6.5))

    # -- Colors --
    c_student = "#2E7D32"
    c_system = "#1565C0"
    c_base = "#9E9E9E"
    c_teacher = "#7B1FA2"
    c_usage = "#E65100"

    # -- Left axis: Accuracy lines --
    ax1.plot(rounds, student_acc, color=c_student, marker="o", linewidth=2.5,
             markersize=7, label="Student-only (measures learning)", zorder=5)
    ax1.plot(rounds, system_acc, color=c_system, marker="^", linewidth=2.5,
             markersize=7, label="System (routed, user experience)", zorder=5)
    ax1.axhline(y=base_acc, color=c_base, linestyle="--", linewidth=1.5,
                label=f"Base model ({base_acc:.1f}%)")
    ax1.axhline(y=teacher_acc, color=c_teacher, linestyle="--", linewidth=1.5,
                label=f"Teacher-only ({teacher_acc:.1f}%)")

    # Shade student improvement over base
    ax1.fill_between(rounds, base_acc, student_acc, alpha=0.10, color=c_student)
    # Shade system advantage over student (the teacher's contribution)
    ax1.fill_between(rounds, student_acc, system_acc, alpha=0.10, color=c_system)

    ax1.set_xlabel(
        f"Training Round ({epr} teacher-distilled examples each)",
        fontsize=12, fontweight="bold",
    )
    ax1.set_ylabel("Execution Accuracy (%)", fontsize=12, fontweight="bold")
    ax1.tick_params(axis="y")
    ax1.set_ylim(base_acc - 5, teacher_acc + 4)
    ax1.yaxis.set_major_formatter(mtick.FormatStrFormatter("%.1f%%"))

    # -- Right axis: Teacher usage --
    ax2 = ax1.twinx()

    ax2.plot(rounds, teacher_usage, color=c_usage, marker="s", linewidth=2.5,
             markersize=7, linestyle="-.", label="Teacher usage %", zorder=4)
    ax2.fill_between(rounds, 0, teacher_usage, alpha=0.06, color=c_usage)

    ax2.set_ylabel("Queries Routed to Teacher (%)", fontsize=12, fontweight="bold",
                    color=c_usage)
    ax2.tick_params(axis="y", labelcolor=c_usage)
    ax2.set_ylim(0, 115)
    ax2.yaxis.set_major_formatter(mtick.FormatStrFormatter("%.0f%%"))

    # -- Combined legend --
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center left",
               fontsize=9.5, framealpha=0.95,
               bbox_to_anchor=(0.01, 0.45))

    # -- Annotations --
    final_student = student_acc[-1]
    final_system = system_acc[-1]
    final_usage = teacher_usage[-1]
    student_gain = final_student - base_acc

    # Annotate student gain
    ax1.annotate(
        f"Student: +{student_gain:.1f}%\nover base",
        xy=(rounds[-1], final_student),
        xytext=(rounds[-1] - 2.8, final_student - 5),
        fontsize=9.5, fontweight="bold", color=c_student,
        arrowprops=dict(arrowstyle="->", color=c_student, lw=1.5),
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                  edgecolor=c_student, alpha=0.9),
    )

    # Annotate system + teacher usage
    ax1.annotate(
        f"System: {final_system:.1f}%\n({final_usage:.0f}% teacher)",
        xy=(rounds[-1], final_system),
        xytext=(rounds[-1] - 2.8, final_system + 3.5),
        fontsize=9.5, fontweight="bold", color=c_system,
        arrowprops=dict(arrowstyle="->", color=c_system, lw=1.5),
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                  edgecolor=c_system, alpha=0.9),
    )

    # -- Subtitle explaining the two shaded regions --
    ax1.text(
        0.5, -0.13,
        "Green region = student learned from distillation  |  "
        "Blue region = remaining teacher contribution via routing",
        transform=ax1.transAxes, fontsize=9, ha="center", color="#555555",
    )

    plt.title("Adaptive Teacher-Student: Learning Progress & Teacher Dependency",
              fontsize=13, fontweight="bold", pad=15)
    ax1.set_xticks(rounds)
    ax1.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.15)

    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close(fig)


def plot_cost(data, output_path: Path):
    """Cost per query over training rounds."""
    rounds = data["rounds"]
    cost = data["cost_per_query"]
    epr = data.get("examples_per_round", EXAMPLES_PER_ROUND)

    fig, ax = plt.subplots(figsize=(10, 4))

    ax.plot(rounds, [c * 1000 for c in cost], color="#1565C0", marker="D",
            linewidth=2.5, markersize=7)
    ax.fill_between(rounds, 0, [c * 1000 for c in cost], alpha=0.1, color="#1565C0")

    ax.set_xlabel(
        f"Training Round ({epr} teacher-distilled examples each)",
        fontsize=12, fontweight="bold",
    )
    ax.set_ylabel("Blended Cost per Query (x$0.001)", fontsize=12, fontweight="bold")
    ax.set_xticks(rounds)
    ax.grid(axis="y", alpha=0.3)

    savings = (1 - cost[-1] / cost[0]) * 100
    ax.annotate(
        f"{savings:.0f}% cost reduction",
        xy=(rounds[-1], cost[-1] * 1000),
        xytext=(rounds[-1] - 3, cost[0] * 1000 * 0.7),
        fontsize=11, fontweight="bold", color="#1565C0",
        arrowprops=dict(arrowstyle="->", color="#1565C0", lw=1.5),
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                  edgecolor="#1565C0", alpha=0.9),
    )

    plt.title("Blended Inference Cost per Query Over Training",
              fontsize=13, fontweight="bold", pad=15)
    fig.tight_layout()

    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot success metrics")
    parser.add_argument("--data", type=str,
                        help="Path to results JSON (omit for mock data)")
    parser.add_argument("--output-dir", type=str, default="results/plots")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.data:
        with open(args.data) as f:
            data = json.load(f)
    else:
        data = load_mock_data()

    # Save data for reference
    with open(out_dir / "plot_data.json", "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved: {out_dir / 'plot_data.json'}")

    plot_main(data, out_dir / "accuracy_vs_teacher_usage.png")
    plot_cost(data, out_dir / "cost_per_query.png")


if __name__ == "__main__":
    main()
