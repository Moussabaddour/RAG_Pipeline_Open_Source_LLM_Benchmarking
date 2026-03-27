import os
import matplotlib
matplotlib.use("Agg") 
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from llm_loader import load_llm
from rag_pipeline import (
    load_documents,
    split_documents,
    create_vectorstore,
    build_rag_chain
)
from evaluation import evaluate_rag
from dataset import DATASET
from evaluation_utils import build_results_table

# -------------------------
# Plotting config
# -------------------------

PAPERS       = ["EigenGAN", "MFGAN", "CMGAN", "Ontology Depth"]
METRICS      = ["answer_quality", "faithfulness", "context_utilization"]
METRIC_NAMES = ["Answer Quality", "Faithfulness", "Context Utilization"]
COLORS       = ["#3266ad", "#1D9E75", "#D85A30", "#7F77DD", "#BA7517"]


def save_fig(fig, name):
    os.makedirs("outputs", exist_ok=True)
    path = f"outputs/{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"   Saved → {path}")


# ------------------------------------------------------------------
# Plot 1 — Overall summary: grouped bars (metrics) + latency line
# ------------------------------------------------------------------
def plot_summary(summary):
    print("\n Plotting overall summary...")
    model_list    = list(summary["model"])
    n_models      = len(model_list)
    x             = np.arange(n_models)
    width         = 0.22
    metric_colors = ["#3d7ebf", "#a8c97f", "#e8945a"]

    fig, ax = plt.subplots(figsize=(11, 5))

    for i, (metric, label, color) in enumerate(zip(METRICS, METRIC_NAMES, metric_colors)):
        vals = list(summary[metric])
        ax.bar(x + i * width, vals, width, label=label, color=color, zorder=3)

    ax.set_xticks(x + width)
    ax.set_xticklabels(model_list, fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("Model comparison — overall metrics", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, loc="upper left")
    ax.yaxis.grid(True, alpha=0.35, zorder=0)
    ax.set_axisbelow(True)

    # Latency on secondary y-axis
    ax2 = ax.twinx()
    ax2.plot(x + width, summary["latency"], marker="o", color="#c0392b",
             linewidth=2, markersize=7, label="Latency (s)", zorder=4)
    ax2.set_ylabel("Avg latency (s)", fontsize=11, color="#c0392b")
    ax2.tick_params(axis="y", colors="#c0392b")
    ax2.legend(loc="upper right", fontsize=10)

    fig.tight_layout()
    save_fig(fig, "01_overall_summary")


# ------------------------------------------------------------------
# Plot 2 — Heatmap: models × papers (answer quality)
# ------------------------------------------------------------------
def plot_heatmap(df):
    print("\n Plotting per-paper heatmap...")
    model_list = list(df["model"].unique())
    data = np.zeros((len(model_list), len(PAPERS)))

    for i, model in enumerate(model_list):
        for j, paper in enumerate(PAPERS):
            subset = df[(df["model"] == model) & (df["paper"] == paper)]
            data[i, j] = subset["answer_quality"].mean() if len(subset) else 0

    fig, ax = plt.subplots(figsize=(9, len(model_list) * 0.9 + 1.5))
    im = ax.imshow(data, cmap="Blues", vmin=0, vmax=1, aspect="auto")

    ax.set_xticks(range(len(PAPERS)))
    ax.set_xticklabels(PAPERS, fontsize=11)
    ax.set_yticks(range(len(model_list)))
    ax.set_yticklabels(model_list, fontsize=11)
    ax.set_title("Answer quality per paper × model", fontsize=13, fontweight="bold")

    for i in range(len(model_list)):
        for j in range(len(PAPERS)):
            val = data[i, j]
            ax.text(j, i, f"{val:.0%}", ha="center", va="center",
                    fontsize=11, color="white" if val > 0.55 else "black")

    fig.colorbar(im, ax=ax, label="Answer quality", fraction=0.03, pad=0.04)
    fig.tight_layout()
    save_fig(fig, "02_heatmap_answer_quality")


# ------------------------------------------------------------------
# Plot 3 — Per-paper grouped bars for each metric
# ------------------------------------------------------------------
def plot_per_paper_metrics(df):
    print("\n Plotting per-paper metric breakdown...")
    model_list = list(df["model"].unique())
    n_models   = len(model_list)
    n_papers   = len(PAPERS)
    width      = 0.8 / n_models

    for metric, label in zip(METRICS, METRIC_NAMES):
        fig, axes = plt.subplots(1, n_papers, figsize=(5 * n_papers, 4.5), sharey=True)
        if n_papers == 1:
            axes = [axes]

        for ax, paper in zip(axes, PAPERS):
            for k, (model, color) in enumerate(zip(model_list, COLORS)):
                subset = df[(df["model"] == model) & (df["paper"] == paper)]
                val    = subset[metric].mean() if len(subset) else 0
                ax.bar([k * width], val, width, color=color, zorder=3)

            ax.set_title(paper, fontsize=11, fontweight="bold")
            ax.set_xticks([])
            ax.yaxis.grid(True, alpha=0.3, zorder=0)
            ax.set_axisbelow(True)
            ax.set_ylim(0, 1.05)

        axes[0].set_ylabel(label, fontsize=11)
        fig.suptitle(f"{label} — breakdown by paper", fontsize=13, fontweight="bold")

        patches = [mpatches.Patch(color=COLORS[i], label=m) for i, m in enumerate(model_list)]
        fig.legend(handles=patches, loc="lower center", ncol=n_models,
                   fontsize=10, bbox_to_anchor=(0.5, -0.06))
        fig.tight_layout()
        safe_metric = metric.replace(" ", "_")
        save_fig(fig, f"03_per_paper_{safe_metric}")


# ------------------------------------------------------------------
# Plot 4 — Latency bar chart
# ------------------------------------------------------------------
def plot_latency(summary):
    print("\n Plotting latency comparison...")
    model_list = list(summary["model"])
    latencies  = list(summary["latency"])

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(model_list, latencies,
                  color=COLORS[:len(model_list)], zorder=3, width=0.5)

    for bar, val in zip(bars, latencies):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.3,
                f"{val:.1f}s", ha="center", va="bottom",
                fontsize=11, fontweight="bold")

    ax.set_ylabel("Avg latency per question (s)", fontsize=11)
    ax.set_title("Inference latency by model", fontsize=13, fontweight="bold")
    ax.yaxis.grid(True, alpha=0.35, zorder=0)
    ax.set_axisbelow(True)
    fig.tight_layout()
    save_fig(fig, "04_latency_comparison")


# ------------------------------------------------------------------
# Plot 5 — Radar chart: one polygon per model across 3 metrics
# ------------------------------------------------------------------
def plot_radar(summary):
    print("\n Plotting radar chart...")
    categories = METRIC_NAMES
    n_cats     = len(categories)
    angles     = np.linspace(0, 2 * np.pi, n_cats, endpoint=False).tolist()
    angles    += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))

    for i, row in summary.iterrows():
        vals  = [row["answer_quality"], row["faithfulness"], row["context_utilization"]]
        vals += vals[:1]
        color = COLORS[i % len(COLORS)]
        ax.plot(angles, vals, color=color, linewidth=2, label=row["model"])
        ax.fill(angles, vals, color=color, alpha=0.12)

    ax.set_thetagrids(np.degrees(angles[:-1]), categories, fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_title("Model profiles — radar chart", fontsize=13,
                 fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=10)
    fig.tight_layout()
    save_fig(fig, "05_radar_chart")


# -------------------------
# Main
# -------------------------

def main():

    print(" Loading documents...")
    docs = load_documents("data/")

    print(" Splitting documents...")
    chunks = split_documents(docs)

    print(" Creating vector store...")
    vectorstore = create_vectorstore(chunks)

    models      = ["mistral", "tinyllama", "gemma", "qwen", "phi3"]
    all_results = []

    for model_name in models:
        print(f"\n Running model: {model_name}")
        llm     = load_llm(model_name)
        rag     = build_rag_chain(llm, vectorstore)
        results = evaluate_rag(rag, DATASET, model_name)
        all_results.extend(results)

    df, summary = build_results_table(all_results)

    print("\n Detailed Results:")
    print(df.to_string(index=False))

    print("\n Summary:")
    print(summary.to_string(index=False))

    os.makedirs("outputs", exist_ok=True)
    df.to_csv("outputs/results.csv", index=False)
    summary.to_csv("outputs/summary.csv", index=False)
    print("\n CSVs saved to outputs/")

    print("\n Generating plots...")
    plot_summary(summary)
    plot_heatmap(df)
    plot_per_paper_metrics(df)
    plot_latency(summary)
    plot_radar(summary)
    print("\n All plots saved to outputs/")


if __name__ == "__main__":
    main()