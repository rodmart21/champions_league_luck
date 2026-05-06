"""
viz.py
------
All plotting helpers for the UCL luck analysis.
Each function returns a matplotlib Figure so it can be shown inline
in the notebook and also saved to figures/.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import seaborn as sns

FIGURES_DIR = Path(__file__).parent.parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

# UCL-inspired palette
UCL_DARK = "#0d1b2e"
UCL_BLUE = "#1a4b8c"
UCL_GOLD = "#c8a951"
UCL_WHITE = "#f5f5f5"

LUCKY_COLOR = "#2ecc71"    # green
UNLUCKY_COLOR = "#e74c3c"  # red
NEUTRAL_COLOR = "#95a5a6"  # grey

plt.rcParams.update({
    "figure.facecolor": UCL_DARK,
    "axes.facecolor": UCL_DARK,
    "axes.edgecolor": UCL_WHITE,
    "axes.labelcolor": UCL_WHITE,
    "xtick.color": UCL_WHITE,
    "ytick.color": UCL_WHITE,
    "text.color": UCL_WHITE,
    "grid.color": "#2c3e50",
    "grid.linestyle": "--",
    "grid.alpha": 0.5,
    "font.family": "sans-serif",
})


def _save(fig: plt.Figure, name: str, dpi: int = 150):
    path = FIGURES_DIR / f"{name}.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor=UCL_DARK)
    print(f"Saved → {path}")


# ---------------------------------------------------------------------------
# Figure 1: Luck Score Leaderboard
# ---------------------------------------------------------------------------

def plot_leaderboard(summary: pd.DataFrame, top_n: int = 20) -> plt.Figure:
    """
    Horizontal bar chart of mean luck score per team, ± 95% bootstrap CI.
    Color: green for positive luck, red for negative.
    """
    df = summary.head(top_n).sort_values("mean_luck")
    colors = [LUCKY_COLOR if v >= 0 else UNLUCKY_COLOR for v in df["mean_luck"]]

    fig, ax = plt.subplots(figsize=(10, max(6, len(df) * 0.45)))
    bars = ax.barh(df["team"], df["mean_luck"], color=colors, alpha=0.85, height=0.6)

    # Error bars (CI)
    if "ci_lower" in df.columns and "ci_upper" in df.columns:
        xerr_lo = df["mean_luck"] - df["ci_lower"]
        xerr_hi = df["ci_upper"] - df["mean_luck"]
        ax.errorbar(
            df["mean_luck"], df["team"],
            xerr=[xerr_lo, xerr_hi],
            fmt="none", color=UCL_WHITE, linewidth=1.2, capsize=3,
        )

    # Season count annotation
    for i, (_, row) in enumerate(df.iterrows()):
        ax.text(
            ax.get_xlim()[1] * 0.02 if row["mean_luck"] >= 0 else ax.get_xlim()[0] * 0.02,
            i,
            f"n={int(row['n_seasons'])}",
            va="center", ha="left", fontsize=7, color=UCL_WHITE, alpha=0.7,
        )

    ax.axvline(0, color=UCL_WHITE, linewidth=0.8, linestyle="-")
    ax.set_xlabel("Mean Composite Luck Score")
    ax.set_title("UCL Luck Leaderboard (2017-18 → 2023-24)\n±95% Bootstrap CI", pad=12)
    ax.grid(axis="x")
    fig.tight_layout()
    _save(fig, "fig1_leaderboard")
    return fig


# ---------------------------------------------------------------------------
# Figure 2: Component Breakdown (Stacked Bar)
# ---------------------------------------------------------------------------

def plot_component_breakdown(summary: pd.DataFrame, top_n: int = 15) -> plt.Figure:
    """
    Stacked bar chart showing AttackLuck, DefenseLuck, GKLuck for top teams.
    """
    df = summary.head(top_n).sort_values("mean_luck", ascending=False)

    x = np.arange(len(df))
    width = 0.55

    fig, ax = plt.subplots(figsize=(13, 5))

    ax.bar(x, df["mean_attack_luck"], width, label="Attack Luck", color="#3498db", alpha=0.85)
    ax.bar(x, df["mean_defense_luck"], width, bottom=df["mean_attack_luck"],
           label="Defense Luck", color=LUCKY_COLOR, alpha=0.85)
    ax.bar(
        x, df["mean_gk_luck"], width,
        bottom=df["mean_attack_luck"] + df["mean_defense_luck"],
        label="GK Luck (PSxG+/-)", color=UCL_GOLD, alpha=0.85,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(df["team"], rotation=35, ha="right", fontsize=8)
    ax.axhline(0, color=UCL_WHITE, linewidth=0.8)
    ax.set_ylabel("Mean Luck Component")
    ax.set_title("Luck Component Breakdown — Top 15 Teams", pad=12)
    ax.legend(facecolor=UCL_DARK, edgecolor=UCL_WHITE, fontsize=8)
    ax.grid(axis="y")
    fig.tight_layout()
    _save(fig, "fig2_component_breakdown")
    return fig


# ---------------------------------------------------------------------------
# Figure 3: Season × Team Heatmap
# ---------------------------------------------------------------------------

def plot_season_heatmap(master: pd.DataFrame, min_seasons: int = 3) -> plt.Figure:
    """
    Heatmap: teams (rows) × seasons (columns), colored by luck_score.
    Only shows teams with at least `min_seasons` seasons.
    """
    counts = master.groupby("team")["luck_score"].count()
    valid_teams = counts[counts >= min_seasons].index
    df = master[master["team"].isin(valid_teams)]

    pivot = df.pivot_table(index="team", columns="season", values="luck_score")
    # sort rows by mean luck
    pivot = pivot.loc[pivot.mean(axis=1).sort_values(ascending=False).index]

    vmax = max(abs(pivot.values[~np.isnan(pivot.values)])) if pivot.size > 0 else 1
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "luck", [UNLUCKY_COLOR, "#2c3e50", LUCKY_COLOR]
    )

    fig, ax = plt.subplots(figsize=(max(10, len(pivot.columns) * 1.2), max(7, len(pivot) * 0.4)))
    sns.heatmap(
        pivot,
        ax=ax,
        cmap=cmap,
        center=0,
        vmin=-vmax,
        vmax=vmax,
        linewidths=0.3,
        linecolor="#1a1a2e",
        annot=True,
        fmt=".2f",
        annot_kws={"size": 6},
        cbar_kws={"label": "Luck Score"},
    )
    ax.set_title("UCL Luck Score by Team & Season", pad=12)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", rotation=30, labelsize=8)
    ax.tick_params(axis="y", labelsize=7)
    fig.tight_layout()
    _save(fig, "fig3_season_heatmap")
    return fig


# ---------------------------------------------------------------------------
# Figure 4: Luck vs. UCL Round Reached (Scatter)
# ---------------------------------------------------------------------------

def plot_luck_vs_progress(master: pd.DataFrame) -> plt.Figure:
    """
    Scatter: luck_score (x) vs. UCL round reached (y).
    Requires a 'round_reached' column (1=Group, 2=R16, 3=QF, 4=SF, 5=F, 6=W).
    Adds a linear regression line.
    """
    if "round_reached" not in master.columns:
        print("Column 'round_reached' not found — skipping Figure 4.")
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "round_reached column required", ha="center", transform=ax.transAxes)
        return fig

    df = master.dropna(subset=["luck_score", "round_reached"])

    round_labels = {1: "Group", 2: "R16", 3: "QF", 4: "SF", 5: "Final", 6: "Winner"}
    jitter = np.random.default_rng(42).uniform(-0.1, 0.1, size=len(df))

    fig, ax = plt.subplots(figsize=(9, 6))
    sc = ax.scatter(
        df["luck_score"], df["round_reached"] + jitter,
        c=df["luck_score"], cmap="RdYlGn",
        alpha=0.7, s=55, edgecolors="none",
    )
    plt.colorbar(sc, ax=ax, label="Luck Score")

    # Regression line
    from scipy.stats import linregress
    slope, intercept, r, p, _ = linregress(df["luck_score"], df["round_reached"])
    x_line = np.linspace(df["luck_score"].min(), df["luck_score"].max(), 200)
    ax.plot(x_line, slope * x_line + intercept, color=UCL_GOLD, linewidth=1.8,
            label=f"OLS fit  r={r:.2f}  p={p:.3f}")

    ax.set_yticks(list(round_labels.keys()))
    ax.set_yticklabels(list(round_labels.values()))
    ax.set_xlabel("Composite Luck Score")
    ax.set_ylabel("UCL Round Reached")
    ax.set_title("Does Luck Correlate with UCL Advancement?", pad=12)
    ax.legend(facecolor=UCL_DARK, edgecolor=UCL_WHITE, fontsize=8)
    ax.grid(axis="both")
    fig.tight_layout()
    _save(fig, "fig4_luck_vs_progress")
    return fig


# ---------------------------------------------------------------------------
# Figure 5: xG Overperformance Time Series
# ---------------------------------------------------------------------------

def plot_xg_time_series(master: pd.DataFrame, teams: list[str] = None) -> plt.Figure:
    """
    Line chart of goals/xG ratio by season for a selection of teams.
    Auto-selects 3 luckiest + 3 unluckiest if `teams` is not provided.
    """
    if "goals_scored" not in master.columns or "xg" not in master.columns:
        print("Columns 'goals_scored'/'xg' not found — skipping Figure 5.")
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "goals_scored / xg columns required", ha="center", transform=ax.transAxes)
        return fig

    if teams is None:
        by_team = master.groupby("team")["luck_score"].mean().dropna()
        teams = list(by_team.nlargest(3).index) + list(by_team.nsmallest(3).index)

    palette = sns.color_palette("tab10", n_colors=len(teams))
    fig, ax = plt.subplots(figsize=(11, 5))

    for team, color in zip(teams, palette):
        df_t = master[master["team"] == team].sort_values("season")
        ratio = df_t["goals_scored"] / df_t["xg"].replace(0, np.nan)
        ax.plot(df_t["season"], ratio, marker="o", label=team, color=color, linewidth=1.8)

    ax.axhline(1.0, color=UCL_WHITE, linewidth=0.8, linestyle="--", label="xG baseline (ratio=1)")
    ax.set_xlabel("Season")
    ax.set_ylabel("Goals Scored / xG")
    ax.set_title("Attacking Overperformance vs. xG Over Time", pad=12)
    ax.legend(facecolor=UCL_DARK, edgecolor=UCL_WHITE, fontsize=8)
    ax.grid()
    plt.xticks(rotation=30)
    fig.tight_layout()
    _save(fig, "fig5_xg_time_series")
    return fig


# ---------------------------------------------------------------------------
# Figure 6: PSxG+/- Distribution (Violin)
# ---------------------------------------------------------------------------

def plot_psxg_distribution(master: pd.DataFrame, highlight_teams: list[str] = None) -> plt.Figure:
    """
    Violin plot of gk_luck (PSxG+/- per 90) across all team-seasons.
    Individual highlighted teams shown as colored dots on top.
    """
    if "gk_luck" not in master.columns:
        print("Column 'gk_luck' not found — skipping Figure 6.")
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "gk_luck column required", ha="center", transform=ax.transAxes)
        return fig

    df = master.dropna(subset=["gk_luck"])

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.violinplot(df["gk_luck"], positions=[0], showmedians=True, widths=0.6)
    ax.axhline(0, color=UCL_WHITE, linewidth=0.8, linestyle="--")

    # Jitter all points lightly
    rng = np.random.default_rng(42)
    jitter = rng.uniform(-0.06, 0.06, size=len(df))
    ax.scatter(jitter, df["gk_luck"], alpha=0.25, s=18, color=NEUTRAL_COLOR, zorder=3)

    # Highlight specific teams
    if highlight_teams:
        palette = sns.color_palette("tab10", n_colors=len(highlight_teams))
        for team, color in zip(highlight_teams, palette):
            t_df = df[df["team"] == team]
            t_jitter = rng.uniform(-0.06, 0.06, size=len(t_df))
            ax.scatter(t_jitter, t_df["gk_luck"], color=color, s=60,
                       zorder=5, label=team, edgecolors=UCL_WHITE, linewidths=0.5)

    ax.set_xticks([])
    ax.set_ylabel("GK Luck: PSxG+/- per 90")
    ax.set_title("Goalkeeper Luck Distribution — All UCL Team-Seasons", pad=12)
    if highlight_teams:
        ax.legend(facecolor=UCL_DARK, edgecolor=UCL_WHITE, fontsize=8)
    ax.grid(axis="y")
    fig.tight_layout()
    _save(fig, "fig6_psxg_distribution")
    return fig
