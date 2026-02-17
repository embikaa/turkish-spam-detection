"""
Model comparison script for Turkish spam detection.

Loads multi-model training results, selects the top 5 models by F1 score,
and generates comparison charts and tables.

Outputs:
    results/comparison_table.txt   — formatted top-5 comparison table
    results/comparison_chart.png   — bar chart of top-5 metrics
    results/weak_label_dist.png    — weak supervision label distribution
    results/confusion_matrices.png — confusion matrices for top 5
    results/all_models_ranking.png — full ranking of all models
"""

import json
import os
import sys
import numpy as np

sys.path.append(os.path.dirname(__file__))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

from config.settings import Config
from src.utils import ensure_dir

# Style
plt.rcParams.update({
    "figure.facecolor": "#0f172a",
    "axes.facecolor": "#1e293b",
    "axes.edgecolor": "#334155",
    "axes.labelcolor": "#e2e8f0",
    "text.color": "#e2e8f0",
    "xtick.color": "#94a3b8",
    "ytick.color": "#94a3b8",
    "grid.color": "#334155",
    "font.family": "sans-serif",
    "font.size": 11,
})

COLORS = ["#6366f1", "#22c55e", "#f59e0b", "#a855f7", "#06b6d4"]
METRIC_COLORS = {
    "Accuracy": "#6366f1",
    "F1 Score": "#22c55e",
    "Precision": "#f59e0b",
    "Recall": "#a855f7",
    "ROC AUC": "#06b6d4",
}


def load_results() -> dict:
    """Load multi-model results JSON."""
    path = os.path.join(Config.MODELS_DIR, "multi_model_results.json")
    if not os.path.exists(path):
        print(f"ERROR: Results file not found at {path}")
        print("Run 'python train_all_models.py' first.")
        sys.exit(1)

    with open(path) as f:
        return json.load(f)


def get_top_n(models: dict, n: int = 5) -> list:
    """Return top N model names sorted by F1 score (descending)."""
    valid = {k: v for k, v in models.items() if "error" not in v}
    ranked = sorted(valid.items(), key=lambda x: x[1]["f1_score"], reverse=True)
    return ranked[:n]


def print_comparison_table(top_models: list, out_dir: str):
    """Print and save a formatted comparison table."""
    header = (f"{'Rank':<6} {'Model':<25} {'Accuracy':>10} {'F1 Score':>10} "
              f"{'Precision':>10} {'Recall':>10} {'ROC AUC':>10} {'Time':>8}")
    sep = "-" * len(header)

    lines = [
        "=" * len(header),
        "TOP 5 MODEL COMPARISON — Turkish Spam Detection",
        "Features: TF-IDF (500) + BERTurk (768) = 1268 dimensions",
        "=" * len(header),
        "",
        header,
        sep,
    ]

    for rank, (name, m) in enumerate(top_models, 1):
        auc = f"{m.get('roc_auc', 0):.4f}" if "roc_auc" in m else "N/A"
        lines.append(
            f"{rank:<6} {name:<25} {m['accuracy']:>10.4f} {m['f1_score']:>10.4f} "
            f"{m['precision']:>10.4f} {m['recall']:>10.4f} "
            f"{auc:>10} {m['training_time']:>7.1f}s"
        )

    lines.append(sep)

    # Best per metric
    lines.append("")
    lines.append("BEST MODEL PER METRIC:")
    for metric_name, metric_key in [("Accuracy", "accuracy"), ("F1 Score", "f1_score"),
                                      ("Precision", "precision"), ("Recall", "recall"),
                                      ("ROC AUC", "roc_auc")]:
        best = max(top_models, key=lambda x: x[1].get(metric_key, 0))
        val = best[1].get(metric_key, 0)
        lines.append(f"  {metric_name:<12} -> {best[0]} ({val:.4f})")

    lines.append("=" * len(header))

    table_text = "\n".join(lines)
    print(table_text)

    # Save to file
    path = os.path.join(out_dir, "comparison_table.txt")
    with open(path, "w") as f:
        f.write(table_text)
    print(f"\nTable saved to {path}")


def plot_comparison_chart(top_models: list, out_dir: str):
    """Generate grouped bar chart comparing top 5 models across metrics."""
    names = [n for n, _ in top_models]
    metrics = {
        "Accuracy": [m["accuracy"] * 100 for _, m in top_models],
        "F1 Score": [m["f1_score"] * 100 for _, m in top_models],
        "Precision": [m["precision"] * 100 for _, m in top_models],
        "Recall": [m["recall"] * 100 for _, m in top_models],
        "ROC AUC": [m.get("roc_auc", 0) * 100 for _, m in top_models],
    }

    fig, ax = plt.subplots(figsize=(14, 7))
    x = np.arange(len(names))
    n_metrics = len(metrics)
    width = 0.15

    for i, (label, values) in enumerate(metrics.items()):
        offset = (i - n_metrics / 2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=label,
                      color=METRIC_COLORS[label], alpha=0.85, edgecolor="none")
        # Value labels on bars
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    f"{val:.1f}", ha="center", va="bottom", fontsize=7,
                    color="#e2e8f0")

    ax.set_xlabel("Model", fontsize=13, fontweight="bold")
    ax.set_ylabel("Score (%)", fontsize=13, fontweight="bold")
    ax.set_title("Top 5 Model Comparison — All Metrics", fontsize=16, fontweight="bold",
                 pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=10)
    ax.set_ylim(0, 105)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.legend(loc="lower right", framealpha=0.8, facecolor="#1e293b", edgecolor="#334155")
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()
    path = os.path.join(out_dir, "comparison_chart.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Comparison chart saved to {path}")


def plot_weak_label_distribution(stats: dict, out_dir: str):
    """Generate weak supervision label distribution chart (count + percentage)."""
    genuine = stats["genuine_count"]
    spam = stats["spam_count"]
    total = stats["total_samples"]
    gen_pct = stats["genuine_pct"]
    spam_pct = stats["spam_pct"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Bar chart — counts
    bars = ax1.bar(["Genuine", "Spam"], [genuine, spam],
                   color=["#22c55e", "#ef4444"], alpha=0.85,
                   edgecolor="none", width=0.5)
    ax1.set_title("Weak Labels — Sample Count", fontsize=14, fontweight="bold", pad=12)
    ax1.set_ylabel("Number of Samples", fontsize=12)
    for bar, count in zip(bars, [genuine, spam]):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + total * 0.01,
                 f"{count:,}", ha="center", va="bottom", fontsize=14, fontweight="bold")
    ax1.set_ylim(0, max(genuine, spam) * 1.15)
    ax1.grid(axis="y", alpha=0.3)
    ax1.set_axisbelow(True)

    # Pie chart — percentages
    wedges, texts, autotexts = ax2.pie(
        [genuine, spam],
        labels=["Genuine", "Spam"],
        autopct=lambda p: f"{p:.1f}%\n({int(p * total / 100):,})",
        colors=["#22c55e", "#ef4444"],
        startangle=90,
        textprops={"fontsize": 12, "color": "#e2e8f0"},
        wedgeprops={"edgecolor": "#0f172a", "linewidth": 2},
    )
    for autotext in autotexts:
        autotext.set_fontweight("bold")
    ax2.set_title("Weak Labels — Distribution", fontsize=14, fontweight="bold", pad=12)

    fig.suptitle(f"Weak Supervision Labeling Results (n={total:,})",
                 fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = os.path.join(out_dir, "weak_label_distribution.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Weak label distribution saved to {path}")


def plot_confusion_matrices(top_models: list, out_dir: str):
    """Plot confusion matrices for all top-5 models side by side."""
    n = len(top_models)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]

    labels_cm = ["Genuine", "Spam"]

    for ax, (name, m) in zip(axes, top_models):
        cm = np.array(m["confusion_matrix"])
        im = ax.imshow(cm, interpolation="nearest", cmap="YlOrRd", alpha=0.85)

        # Text annotations
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]}",
                        ha="center", va="center", fontsize=14, fontweight="bold",
                        color="white" if cm[i, j] > cm.max() / 2 else "#1e293b")

        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(labels_cm, fontsize=9)
        ax.set_yticklabels(labels_cm, fontsize=9)
        ax.set_xlabel("Predicted", fontsize=10)
        ax.set_ylabel("Actual", fontsize=10)
        ax.set_title(name, fontsize=11, fontweight="bold", pad=8)

    fig.suptitle("Confusion Matrices — Top 5 Models",
                 fontsize=15, fontweight="bold", y=1.05)
    plt.tight_layout()
    path = os.path.join(out_dir, "confusion_matrices.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Confusion matrices saved to {path}")


def plot_all_models_ranking(all_models: dict, out_dir: str):
    """Horizontal bar chart ranking ALL models by F1 score."""
    valid = {k: v for k, v in all_models.items() if "error" not in v}
    ranked = sorted(valid.items(), key=lambda x: x[1]["f1_score"])

    names = [n for n, _ in ranked]
    f1_scores = [m["f1_score"] * 100 for _, m in ranked]

    fig, ax = plt.subplots(figsize=(10, max(6, len(names) * 0.7)))

    colors_bar = [COLORS[i % len(COLORS)] for i in range(len(names))]
    bars = ax.barh(names, f1_scores, color=colors_bar, alpha=0.85, edgecolor="none", height=0.6)

    for bar, score in zip(bars, f1_scores):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{score:.2f}%", va="center", fontsize=10, fontweight="bold")

    ax.set_xlabel("F1 Score (%)", fontsize=13, fontweight="bold")
    ax.set_title("All Models Ranked by F1 Score", fontsize=16, fontweight="bold", pad=15)
    ax.set_xlim(0, max(f1_scores) + 8)
    ax.grid(axis="x", alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()
    path = os.path.join(out_dir, "all_models_ranking.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"All models ranking saved to {path}")


def main():
    print("=" * 70)
    print("MODEL COMPARISON & VISUALIZATION")
    print("=" * 70)

    # Load results
    data = load_results()
    models = data["models"]
    out_dir = ensure_dir(Config.RESULTS_DIR)

    print(f"\nLoaded results for {len(models)} models")
    print(f"Training timestamp: {data.get('timestamp', 'N/A')}")

    # Get top 5
    top5 = get_top_n(models, n=5)
    print(f"Top 5 models selected by F1 Score:\n")

    # 1. Comparison table
    print_comparison_table(top5, out_dir)

    # 2. Comparison chart
    print("\nGenerating comparison charts...")
    plot_comparison_chart(top5, out_dir)

    # 3. Weak label distribution
    if "weak_labels" in data:
        plot_weak_label_distribution(data["weak_labels"], out_dir)
    else:
        print("WARNING: Weak label stats not found in results. Re-run training.")

    # 4. Confusion matrices
    plot_confusion_matrices(top5, out_dir)

    # 5. All models ranking
    plot_all_models_ranking(models, out_dir)

    print("\n" + "=" * 70)
    print("All outputs saved to: results/")
    print("  - comparison_table.txt")
    print("  - comparison_chart.png")
    print("  - weak_label_distribution.png")
    print("  - confusion_matrices.png")
    print("  - all_models_ranking.png")
    print("=" * 70)


if __name__ == "__main__":
    main()
