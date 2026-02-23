"""
IDRIS Visualisation Module
===========================
Chart functions used by the Quarto report (index.qmd).
All figures return matplotlib Figure objects, PDF-safe.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.ticker as mticker

# ── Brand palette ─────────────────────────────────────────────
MINT    = "#33FFA2"
MAGENTA = "#FF33FF"
GREY    = "#737373"
INK     = "#1A1A1A"
LIGHT   = "#F7F7F5"

BAND_COLORS = {
    "Dark Green": "#1a6b3a",
    "Green":      "#33AA66",
    "Amber":      "#E8A020",
    "Red":        "#CC3333",
}

DIM_LABELS = {
    "climate":    "Climate & Environment",
    "water":      "Water & Resources",
    "gender":     "Gender Equity",
    "social":     "Social Mobility",
    "territory":  "Territory & Wealth",
    "governance": "Governance",
    "pollution":  "Pollution & Health",
    "innovation": "Innovation",
}


def _style(ax):
    ax.set_facecolor(LIGHT)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#DDDDDD")
    ax.tick_params(colors=GREY, labelsize=9)
    ax.yaxis.label.set_color(GREY)
    ax.xaxis.label.set_color(GREY)


# ─────────────────────────────────────────────────────────────
# 1. IDRIS SCORE DISTRIBUTION
# ─────────────────────────────────────────────────────────────

def plot_score_distribution(df: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 4.5))
    fig.patch.set_facecolor(LIGHT)

    counts, bins, patches = ax.hist(
        df["idris_score"], bins=40, edgecolor="white", linewidth=0.4
    )
    for patch, left in zip(patches, bins):
        if left >= 75:   patch.set_facecolor(BAND_COLORS["Dark Green"])
        elif left >= 58: patch.set_facecolor(BAND_COLORS["Green"])
        elif left >= 40: patch.set_facecolor(BAND_COLORS["Amber"])
        else:            patch.set_facecolor(BAND_COLORS["Red"])

    for x, label, color in [
        (75, "Dark Green", BAND_COLORS["Dark Green"]),
        (58, "Green",      BAND_COLORS["Green"]),
        (40, "Amber",      BAND_COLORS["Amber"]),
    ]:
        ax.axvline(x, color=color, linewidth=1, linestyle="--", alpha=0.7)
        ax.text(x + 0.5, counts.max() * 0.9, label,
                color=color, fontsize=8, va="top")

    _style(ax)
    ax.set_xlabel("IDRIS Score", fontsize=10)
    ax.set_ylabel("Number of Projects", fontsize=10)
    ax.set_title("Distribution of IDRIS Composite Scores (n=2,000)", fontsize=12,
                 color=INK, pad=12, fontweight="semibold")
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────
# 2. RADAR CHART — single project
# ─────────────────────────────────────────────────────────────

def plot_radar(dims: dict, project_name: str = "Project") -> plt.Figure:
    labels = list(DIM_LABELS.values())
    values = [dims.get(k, 0) for k in DIM_LABELS.keys()]
    values += values[:1]   # close the polygon

    N = len(labels)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"polar": True})
    fig.patch.set_facecolor(LIGHT)
    ax.set_facecolor(LIGHT)

    ax.plot(angles, values, color=MINT, linewidth=2, linestyle="solid")
    ax.fill(angles, values, color=MINT, alpha=0.18)

    # Reference ring at 50
    ref = [50] * (N + 1)
    ax.plot(angles, ref, color=GREY, linewidth=0.8, linestyle="--", alpha=0.5)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, size=9, color=INK)
    ax.set_ylim(0, 100)
    ax.set_yticks([25, 50, 75, 100])
    ax.set_yticklabels(["25", "50", "75", "100"], size=7, color=GREY)
    ax.yaxis.set_tick_params(labelcolor=GREY)
    ax.grid(color="#CCCCCC", linewidth=0.5)
    ax.spines["polar"].set_color("#CCCCCC")

    ax.set_title(f"Impact Profile — {project_name}", size=12,
                 color=INK, pad=20, fontweight="semibold")
    return fig


# ─────────────────────────────────────────────────────────────
# 3. DIMENSION HEATMAP by sector
# ─────────────────────────────────────────────────────────────

def plot_sector_heatmap(df: pd.DataFrame) -> plt.Figure:
    dim_cols = [f"dim_{k}" for k in DIM_LABELS.keys()]
    pivot = df.groupby("sector")[dim_cols].mean().round(1)
    pivot.columns = list(DIM_LABELS.values())
    pivot = pivot.sort_values("Climate & Environment")

    fig, ax = plt.subplots(figsize=(14, max(6, len(pivot) * 0.45)))
    fig.patch.set_facecolor(LIGHT)
    ax.set_facecolor(LIGHT)

    cmap = LinearSegmentedColormap.from_list(
        "idris", ["#CC3333", "#E8A020", "#33AA66", "#1a6b3a"]
    )
    im = ax.imshow(pivot.values, aspect="auto", cmap=cmap, vmin=20, vmax=95)

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=30, ha="right", fontsize=9, color=INK)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=9, color=INK)

    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.values[i, j]
            ax.text(j, i, f"{val:.0f}", ha="center", va="center",
                    fontsize=8, color="white" if val < 60 else INK)

    plt.colorbar(im, ax=ax, fraction=0.02, pad=0.01, label="Score (0–100)")
    ax.set_title("Average Impact Scores by Sector", fontsize=13,
                 color=INK, pad=14, fontweight="semibold")
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────
# 4. SFDR CLASSIFICATION BREAKDOWN
# ─────────────────────────────────────────────────────────────

def plot_sfdr_breakdown(df: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor(LIGHT)

    # Pie: SFDR distribution
    ax = axes[0]
    ax.set_facecolor(LIGHT)
    counts = df["sfdr_article"].value_counts()
    colors = {"Article 9": BAND_COLORS["Dark Green"],
              "Article 8": BAND_COLORS["Green"],
              "Article 6": GREY}
    ax.pie(counts.values,
           labels=counts.index,
           colors=[colors.get(k, GREY) for k in counts.index],
           autopct="%1.0f%%", startangle=90,
           textprops={"fontsize": 10, "color": INK},
           wedgeprops={"linewidth": 1.5, "edgecolor": "white"})
    ax.set_title("SFDR Article Classification", fontsize=11,
                 color=INK, fontweight="semibold")

    # Bar: mean IDRIS by SFDR
    ax2 = axes[1]
    ax2.set_facecolor(LIGHT)
    means = df.groupby("sfdr_article")["idris_score"].mean().reindex(
        ["Article 6", "Article 8", "Article 9"]
    )
    bars = ax2.bar(means.index, means.values,
                   color=[colors.get(k, GREY) for k in means.index],
                   edgecolor="white", linewidth=1.2, width=0.5)
    for bar, val in zip(bars, means.values):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.8, f"{val:.1f}",
                 ha="center", fontsize=10, color=INK, fontweight="semibold")
    _style(ax2)
    ax2.set_ylabel("Mean IDRIS Score", fontsize=10)
    ax2.set_ylim(0, 100)
    ax2.set_title("Mean IDRIS Score by SFDR Article", fontsize=11,
                  color=INK, fontweight="semibold")

    fig.tight_layout(pad=2)
    return fig


# ─────────────────────────────────────────────────────────────
# 5. CORRELATION HEATMAP
# ─────────────────────────────────────────────────────────────

def plot_correlation(df: pd.DataFrame) -> plt.Figure:
    dim_cols = [f"dim_{k}" for k in DIM_LABELS.keys()] + ["idris_score"]
    corr = df[dim_cols].corr()
    labels = list(DIM_LABELS.values()) + ["IDRIS Score"]

    fig, ax = plt.subplots(figsize=(10, 9))
    fig.patch.set_facecolor(LIGHT)
    cmap = LinearSegmentedColormap.from_list("rb", [MAGENTA, "white", MINT])
    im = ax.imshow(corr.values, cmap=cmap, vmin=-1, vmax=1)

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=8.5, color=INK)
    ax.set_yticklabels(labels, fontsize=8.5, color=INK)

    for i in range(len(labels)):
        for j in range(len(labels)):
            val = corr.values[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=7.5, color=INK if abs(val) < 0.5 else "white")

    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    ax.set_title("Correlation Matrix — Impact Dimensions & IDRIS Score",
                 fontsize=12, color=INK, pad=14, fontweight="semibold")
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────
# 6. SCORE BY INVESTMENT SIZE
# ─────────────────────────────────────────────────────────────

def plot_score_vs_size(df: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor(LIGHT)
    ax.set_facecolor(LIGHT)

    band_map = {"Dark Green": BAND_COLORS["Dark Green"],
                "Green": BAND_COLORS["Green"],
                "Amber": BAND_COLORS["Amber"],
                "Red": BAND_COLORS["Red"]}

    for band, grp in df.groupby("idris_band"):
        ax.scatter(grp["investment_eur"], grp["idris_score"],
                   c=band_map.get(band, GREY), alpha=0.35, s=12,
                   label=band, rasterized=True)

    ax.set_xscale("log")
    ax.set_xlabel("Investment Size (EUR, log scale)", fontsize=10)
    ax.set_ylabel("IDRIS Score", fontsize=10)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"€{x:,.0f}"
    ))
    ax.legend(title="IDRIS Band", fontsize=9, title_fontsize=9,
              framealpha=0.8, edgecolor="#DDDDDD")
    _style(ax)
    ax.set_title("IDRIS Score vs. Investment Size", fontsize=12,
                 color=INK, pad=12, fontweight="semibold")
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────
# 7. TAXONOMY ALIGNMENT BY SECTOR
# ─────────────────────────────────────────────────────────────

def plot_taxonomy_alignment(df: pd.DataFrame) -> plt.Figure:
    pct = (
        df[df["taxonomy_eligible"]]
        .groupby("sector")["taxonomy_aligned"]
        .mean()
        .sort_values()
        * 100
    ).round(1)

    fig, ax = plt.subplots(figsize=(10, max(5, len(pct) * 0.45)))
    fig.patch.set_facecolor(LIGHT)
    ax.set_facecolor(LIGHT)

    colors = [BAND_COLORS["Dark Green"] if v >= 60
              else BAND_COLORS["Green"] if v >= 40
              else BAND_COLORS["Amber"] for v in pct.values]

    bars = ax.barh(pct.index, pct.values, color=colors,
                   edgecolor="white", linewidth=0.8, height=0.65)
    for bar, val in zip(bars, pct.values):
        ax.text(val + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{val:.0f}%", va="center", fontsize=9, color=INK)

    ax.axvline(40, color=GREY, linewidth=1, linestyle="--", alpha=0.6)
    ax.text(40.5, -0.5, "40% threshold", fontsize=8, color=GREY)
    _style(ax)
    ax.set_xlabel("% Taxonomy Aligned (of eligible)", fontsize=10)
    ax.set_xlim(0, 105)
    ax.set_title("EU Taxonomy Alignment Rate by Sector",
                 fontsize=12, color=INK, pad=12, fontweight="semibold")
    fig.tight_layout()
    return fig
