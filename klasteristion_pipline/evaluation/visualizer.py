"""
Visualisation suite for the clustering pipeline.

All ``plot_*`` functions return a matplotlib Figure; ``save_all`` persists them.
Interactive Plotly variants are generated for 2-D / 3-D scatter and BERTopic
built-in visualisations.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from config import PLOTS_DIR

log = logging.getLogger(__name__)
sns.set_theme(style="whitegrid", palette="deep", font_scale=1.1)

PALETTE = sns.color_palette("husl", 20)


class PipelineVisualizer:
    """Stateless helper that generates and saves pipeline visualisations."""

    def __init__(self, output_dir: Path = PLOTS_DIR):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)


    def plot_umap_2d(
        self,
        umap_2d: np.ndarray,
        labels: list[int] | np.ndarray,
        title: str = "UMAP 2-D",
        texts: list[str] | None = None,
    ) -> plt.Figure:
        labels = np.asarray(labels)
        fig, ax = plt.subplots(figsize=(12, 8))

        unique_labels = sorted(set(labels))
        for lab in unique_labels:
            mask = labels == lab
            color = "lightgrey" if lab == -1 else PALETTE[lab % len(PALETTE)]
            alpha = 0.3 if lab == -1 else 0.7
            label_str = "Noise" if lab == -1 else f"Topic {lab}"
            ax.scatter(
                umap_2d[mask, 0], umap_2d[mask, 1],
                c=[color], alpha=alpha, s=20, label=label_str, edgecolors="none",
            )

        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlabel("UMAP-1")
        ax.set_ylabel("UMAP-2")
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8, ncol=1)
        fig.tight_layout()
        return fig


    def plot_umap_3d_interactive(
        self,
        umap_3d: np.ndarray,
        labels: list[int] | np.ndarray,
        texts: list[str] | None = None,
        title: str = "UMAP 3-D",
    ) -> Any:
        import plotly.express as px

        labels = np.asarray(labels)
        df = pd.DataFrame({
            "x": umap_3d[:, 0],
            "y": umap_3d[:, 1],
            "z": umap_3d[:, 2],
            "topic": [str(l) for l in labels],
            "text": [t[:120] + "…" if len(t) > 120 else t for t in (texts or [""] * len(labels))],
        })
        fig = px.scatter_3d(
            df, x="x", y="y", z="z",
            color="topic", hover_data=["text"],
            title=title, opacity=0.7,
        )
        fig.update_traces(marker_size=3)
        fig.update_layout(width=1000, height=700)
        return fig


    def plot_cluster_sizes(
        self,
        topic_info: pd.DataFrame,
        title: str = "Cluster sizes",
    ) -> plt.Figure:
        df = topic_info[topic_info["Topic"] != -1].copy()
        df = df.sort_values("Count", ascending=True)

        fig, ax = plt.subplots(figsize=(10, max(4, len(df) * 0.5)))
        colors = [PALETTE[i % len(PALETTE)] for i in range(len(df))]
        ax.barh(df["Name"].astype(str), df["Count"], color=colors)
        ax.set_xlabel("Number of reviews")
        ax.set_title(title, fontsize=14, fontweight="bold")

        for i, (_, row) in enumerate(df.iterrows()):
            ax.text(row["Count"] + 0.5, i, str(row["Count"]), va="center", fontsize=9)

        fig.tight_layout()
        return fig


    def plot_silhouette_per_cluster(
        self,
        embeddings: np.ndarray,
        labels: list[int] | np.ndarray,
        title: str = "Silhouette per cluster",
    ) -> plt.Figure:
        from sklearn.metrics import silhouette_samples

        labels = np.asarray(labels)
        mask = labels != -1
        if mask.sum() < 2:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "Not enough data", ha="center", va="center")
            return fig

        emb_clean = embeddings[mask]
        lab_clean = labels[mask]
        sil_vals = silhouette_samples(emb_clean, lab_clean, metric="cosine")

        unique = sorted(set(lab_clean))
        fig, ax = plt.subplots(figsize=(10, max(4, len(unique) * 0.7)))
        y_lower = 0

        for i, tid in enumerate(unique):
            cluster_sil = np.sort(sil_vals[lab_clean == tid])
            size = len(cluster_sil)
            y_upper = y_lower + size
            color = PALETTE[i % len(PALETTE)]
            ax.fill_betweenx(
                np.arange(y_lower, y_upper), 0, cluster_sil,
                facecolor=color, edgecolor=color, alpha=0.7,
            )
            ax.text(-0.05, y_lower + 0.5 * size, f"T{tid}", fontsize=9, va="center")
            y_lower = y_upper + 2

        avg = np.mean(sil_vals)
        ax.axvline(x=avg, color="red", linestyle="--", label=f"Mean = {avg:.3f}")
        ax.set_xlabel("Silhouette coefficient")
        ax.set_ylabel("Cluster (sorted samples)")
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.legend()
        fig.tight_layout()
        return fig


    def plot_top_words(
        self,
        topic_model: Any,
        n_topics: int = 10,
        n_words: int = 8,
        title: str = "Top words per topic",
    ) -> plt.Figure:
        topics_dict = topic_model.get_topics()
        real_topics = {k: v for k, v in topics_dict.items() if k != -1}
        topic_ids = sorted(real_topics.keys())[:n_topics]
        n = len(topic_ids)

        fig, axes = plt.subplots(1, n, figsize=(n * 3, 4), sharey=False)
        if n == 1:
            axes = [axes]

        for ax, tid in zip(axes, topic_ids):
            words_scores = real_topics[tid][:n_words]
            words = [w for w, _ in words_scores]
            scores = [s for _, s in words_scores]
            color = PALETTE[tid % len(PALETTE)]
            ax.barh(words[::-1], scores[::-1], color=color)
            ax.set_title(f"Topic {tid}", fontsize=10, fontweight="bold")
            ax.tick_params(axis="y", labelsize=8)

        fig.suptitle(title, fontsize=14, fontweight="bold")
        fig.tight_layout()
        return fig


    def plot_word_clouds(
        self,
        topic_model: Any,
        n_topics: int = 8,
        title: str = "Word Clouds",
    ) -> plt.Figure:
        from wordcloud import WordCloud

        topics_dict = topic_model.get_topics()
        real_topics = {k: v for k, v in topics_dict.items() if k != -1}
        topic_ids = sorted(real_topics.keys())[:n_topics]
        n = len(topic_ids)

        cols = min(4, n)
        rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
        axes_flat = np.array(axes).flatten() if n > 1 else [axes]

        for i, tid in enumerate(topic_ids):
            freq = {w: abs(s) for w, s in real_topics[tid]}
            wc = WordCloud(
                width=400, height=300,
                background_color="white",
                max_words=30,
                colormap="viridis",
            ).generate_from_frequencies(freq)
            axes_flat[i].imshow(wc, interpolation="bilinear")
            axes_flat[i].set_title(f"Topic {tid}", fontsize=10)
            axes_flat[i].axis("off")

        for j in range(i + 1, len(axes_flat)):
            axes_flat[j].axis("off")

        fig.suptitle(title, fontsize=14, fontweight="bold")
        fig.tight_layout()
        return fig


    def plot_rating_per_cluster(
        self,
        labels: list[int] | np.ndarray,
        ratings: list[int] | np.ndarray,
        title: str = "Rating distribution per cluster",
    ) -> plt.Figure:
        labels = np.asarray(labels)
        ratings = np.asarray(ratings)
        mask = labels != -1

        df = pd.DataFrame({"topic": labels[mask], "rating": ratings[mask]})
        topic_ids = sorted(df["topic"].unique())
        n = len(topic_ids)

        fig, axes = plt.subplots(1, min(n, 8), figsize=(min(n, 8) * 2.5, 3), sharey=True)
        if n == 1:
            axes = [axes]

        for ax, tid in zip(axes, topic_ids[:8]):
            sub = df[df["topic"] == tid]
            ax.hist(sub["rating"], bins=np.arange(0.5, 6.5, 1),
                    color=PALETTE[tid % len(PALETTE)], edgecolor="white")
            ax.set_title(f"T{tid} (n={len(sub)})", fontsize=9)
            ax.set_xticks(range(1, 6))
            ax.set_xlabel("★")

        axes[0].set_ylabel("Count")
        fig.suptitle(title, fontsize=14, fontweight="bold")
        fig.tight_layout()
        return fig


    def plot_experiment_heatmap(
        self,
        results_df: pd.DataFrame,
        x_col: str,
        y_col: str,
        value_col: str,
        title: str = "Grid Search",
    ) -> plt.Figure:
        pivot = results_df.pivot_table(index=y_col, columns=x_col, values=value_col, aggfunc="mean")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlGnBu", ax=ax)
        ax.set_title(title, fontsize=14, fontweight="bold")
        fig.tight_layout()
        return fig


    def plot_metrics_comparison(
        self,
        metrics_list: list[dict],
        label_key: str = "label",
        title: str = "Metrics Comparison",
    ) -> plt.Figure:
        metric_keys = ["silhouette_score", "calinski_harabasz_index", "noise_pct"]
        available = [k for k in metric_keys if any(m.get(k) is not None for m in metrics_list)]

        fig, axes = plt.subplots(1, len(available), figsize=(len(available) * 4, 4))
        if len(available) == 1:
            axes = [axes]

        labels = [m.get(label_key, str(i)) for i, m in enumerate(metrics_list)]

        for ax, key in zip(axes, available):
            vals = [m.get(key, 0) or 0 for m in metrics_list]
            colors = [PALETTE[i % len(PALETTE)] for i in range(len(vals))]
            ax.bar(range(len(vals)), vals, color=colors)
            ax.set_xticks(range(len(vals)))
            ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
            nice = key.replace("_", " ").title()
            ax.set_title(nice, fontsize=11, fontweight="bold")

        fig.suptitle(title, fontsize=14, fontweight="bold")
        fig.tight_layout()
        return fig


    def save_bertopic_plots(
        self,
        topic_model: Any,
        texts: list[str],
        embeddings: np.ndarray,
        prefix: str = "",
    ) -> None:
        """Generate and save all BERTopic interactive Plotly visualisations."""
        try:
            fig = topic_model.visualize_topics()
            fig.write_html(str(self.output_dir / f"{prefix}intertopic_distance.html"))
        except Exception as e:
            log.warning("visualize_topics failed: %s", e)

        try:
            fig = topic_model.visualize_barchart(top_n_topics=12, n_words=8)
            fig.write_html(str(self.output_dir / f"{prefix}barchart.html"))
        except Exception as e:
            log.warning("visualize_barchart failed: %s", e)

        try:
            fig = topic_model.visualize_hierarchy()
            fig.write_html(str(self.output_dir / f"{prefix}hierarchy.html"))
        except Exception as e:
            log.warning("visualize_hierarchy failed: %s", e)

        try:
            fig = topic_model.visualize_heatmap()
            fig.write_html(str(self.output_dir / f"{prefix}heatmap.html"))
        except Exception as e:
            log.warning("visualize_heatmap failed: %s", e)


    def save_figures(self, figures: dict[str, plt.Figure], prefix: str = "") -> None:
        """Save all figures to output_dir as PNG."""
        for name, fig in figures.items():
            path = self.output_dir / f"{prefix}{name}.png"
            fig.savefig(path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            log.info("Saved plot: %s", path.name)

    def save_plotly(self, figures: dict[str, Any], prefix: str = "") -> None:
        """Save Plotly figures as HTML."""
        for name, fig in figures.items():
            path = self.output_dir / f"{prefix}{name}.html"
            fig.write_html(str(path))
            log.info("Saved interactive plot: %s", path.name)
