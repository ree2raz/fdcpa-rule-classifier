"""Visualization: per-rule F1, confusion matrices, cost comparison."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

RESULTS_DIR = Path(__file__).resolve().parent.parent.parent / "results"

sns.set_theme(style="whitegrid", font_scale=1.1)
PALETTE = {"o3-mini": "#10a37f", "Qwen Base": "#FFA15A", "Qwen QLoRA": "#00CC96"}


def load_eval_results(path: Path | None = None) -> dict:
    if path is None:
        path = RESULTS_DIR / "eval_results.json"
    with open(path) as f:
        return json.load(f)


def plot_per_rule_f1(results: dict, output_path: Path | None = None):
    if output_path is None:
        output_path = RESULTS_DIR / "per_rule_f1.png"

    model_names = list(results.keys())
    per_rule_data = {}
    for model_name in model_names:
        metrics = results[model_name].get("metrics", {})
        per_rule = metrics.get("per_rule", {})
        for rule_id, rule_metrics in per_rule.items():
            per_rule_data.setdefault(rule_id, {})[model_name] = rule_metrics.get("f1_macro", 0)

    rule_ids = sorted(per_rule_data.keys())
    x = np.arange(len(rule_ids))
    width = 0.25

    fig, ax = plt.subplots(figsize=(14, 6))
    for i, model_name in enumerate(model_names):
        values = [per_rule_data.get(rid, {}).get(model_name, 0) for rid in rule_ids]
        color = PALETTE.get(model_name, None)
        bars = ax.bar(x + i * width, values, width, label=model_name, color=color, edgecolor="white")
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=7)

    ax.set_xlabel("FDCPA Rule")
    ax.set_ylabel("F1 Score (macro)")
    ax.set_title("Per-Rule F1 Score: o3-mini vs Base Qwen vs QLoRA Fine-tuned")
    ax.set_xticks(x + width)
    ax.set_xticklabels(rule_ids, rotation=45, ha="right")
    ax.set_ylim(0, 1.1)
    ax.legend(loc="upper right")
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_confusion_matrices(results: dict, output_path: Path | None = None):
    if output_path is None:
        output_path = RESULTS_DIR / "confusion_matrices.png"

    model_names = list(results.keys())
    fig, axes = plt.subplots(1, len(model_names), figsize=(5 * len(model_names), 4))
    if len(model_names) == 1:
        axes = [axes]

    labels = ["pass", "fail"]
    for ax, model_name in zip(axes, model_names):
        cm = results[model_name].get("metrics", {}).get("confusion_matrix", [[0, 0], [0, 0]])
        cm = np.array(cm)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(model_name)

    plt.suptitle("Confusion Matrices", fontsize=14)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_cost_comparison(results: dict, output_path: Path | None = None):
    if output_path is None:
        output_path = RESULTS_DIR / "cost_comparison.png"

    model_names = list(results.keys())
    latencies = []
    for mn in model_names:
        overall = results[mn].get("metrics", {}).get("overall", {})
        latencies.append(overall.get("avg_latency_ms", 0))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    colors = [PALETTE.get(mn, "#999") for mn in model_names]

    bars1 = ax1.bar(model_names, latencies, color=colors, edgecolor="white")
    for bar, val in zip(bars1, latencies):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(latencies) * 0.02,
                 f"{val:.0f}ms", ha="center", va="bottom", fontsize=10)
    ax1.set_ylabel("Avg Latency (ms)")
    ax1.set_title("Inference Latency per Transcript")

    costs = []
    for mn in model_names:
        raw = results[mn].get("raw_predictions", [])
        total_input = sum(r.get("input_tokens", 0) for r in raw)
        total_output = sum(r.get("output_tokens", 0) for r in raw)
        input_cost = total_input * 3 / 1_000_000
        output_cost = total_output * 15 / 1_000_000
        total = input_cost + output_cost
        per_example = total / max(1, len(raw))
        costs.append(per_example)

    bars2 = ax2.bar(model_names, costs, color=colors, edgecolor="white")
    for bar, val in zip(bars2, costs):
        label = f"${val:.4f}" if val > 0 else "$0.00 (local)"
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(costs) * 0.02 + 0.0001,
                 label, ha="center", va="bottom", fontsize=10)
    ax2.set_ylabel("Cost per Transcript ($)")
    ax2.set_title("API Cost per Transcript")

    plt.suptitle("Cost & Latency Comparison", fontsize=14)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def generate_all_plots(results_path: Path | None = None):
    results = load_eval_results(results_path)
    plot_per_rule_f1(results)
    plot_confusion_matrices(results)
    plot_cost_comparison(results)


if __name__ == "__main__":
    generate_all_plots()
