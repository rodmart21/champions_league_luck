"""
luck_plots.py
-------------
Visualisations for the UCL luck analysis.
Run standalone or import individual functions in a notebook.

Requires: master_with_luck.csv and luck_summary.csv in data/combined/
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from pathlib import Path

# ---------------------------------------------------------------------------
# Theme
# ---------------------------------------------------------------------------
DARK   = "#0d1b2e"
BLUE   = "#1a4b8c"
GOLD   = "#c8a951"
WHITE  = "#f5f5f5"
GREEN  = "#2ecc71"
RED    = "#e74c3c"
GREY   = "#7f8c8d"

plt.rcParams.update({
    "figure.facecolor": DARK,
    "axes.facecolor":   DARK,
    "axes.edgecolor":   WHITE,
    "axes.labelcolor":  WHITE,
    "xtick.color":      WHITE,
    "ytick.color":      WHITE,
    "text.color":       WHITE,
    "grid.color":       "#1e3050",
    "grid.linestyle":   "--",
    "grid.alpha":       0.5,
    "font.family":      "sans-serif",
    "font.size":        10,
})

FIGURES_DIR = Path(__file__).parent.parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

def _save(fig, name):
    path = FIGURES_DIR / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=DARK)
    print(f"Saved → {path}")


# ---------------------------------------------------------------------------
# Fig 1: Luck leaderboard — top 15 luckiest + unluckiest (± CI)
# ---------------------------------------------------------------------------
def plot_leaderboard(summary: pd.DataFrame, n: int = 15):
    top = summary.nlargest(n, "mean_luck")
    bot = summary.nsmallest(n, "mean_luck").sort_values("mean_luck")
    df = pd.concat([bot, top]).reset_index(drop=True)

    colors = [GREEN if v >= 0 else RED for v in df["mean_luck"]]
    xerr_lo = (df["mean_luck"] - df["ci_lower"]).clip(lower=0)
    xerr_hi = (df["ci_upper"] - df["mean_luck"]).clip(lower=0)

    fig, ax = plt.subplots(figsize=(10, len(df) * 0.42 + 1))
    ax.barh(df["team"], df["mean_luck"], color=colors, alpha=0.85, height=0.65)
    ax.errorbar(df["mean_luck"], df["team"],
                xerr=[xerr_lo, xerr_hi],
                fmt="none", color=WHITE, linewidth=1.1, capsize=3, alpha=0.7)
    ax.axvline(0, color=WHITE, linewidth=0.8)
    ax.set_xlabel("Mean Composite Luck Score")
    ax.set_title(f"UCL Luck Leaderboard — Top & Bottom {n} Teams\n±95 % Bootstrap CI", pad=10)
    ax.grid(axis="x")
    fig.tight_layout()
    _save(fig, "fig1_leaderboard")
    return fig


# ---------------------------------------------------------------------------
# Fig 2: Stacked bar — luck component breakdown per team (top 20)
# ---------------------------------------------------------------------------
def plot_component_breakdown(summary: pd.DataFrame, n: int = 20):
    df = summary.nlargest(n, "mean_luck").sort_values("mean_luck", ascending=False)
    x = np.arange(len(df))
    w = 0.55

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(x, df["mean_attack_luck"],  w, label="Attack luck",   color="#3498db", alpha=0.85)
    ax.bar(x, df["mean_defense_luck"], w, bottom=df["mean_attack_luck"],
           label="Defense luck", color=GREEN, alpha=0.85)
    ax.bar(x, df["mean_finish_luck"],  w,
           bottom=df["mean_attack_luck"] + df["mean_defense_luck"],
           label="Finishing luck", color=GOLD, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(df["team"], rotation=38, ha="right", fontsize=8)
    ax.axhline(0, color=WHITE, linewidth=0.8)
    ax.set_ylabel("Mean luck component")
    ax.set_title(f"Where Does Luck Come From? — Top {n} Teams", pad=10)
    ax.legend(facecolor=DARK, edgecolor=WHITE, fontsize=8)
    ax.grid(axis="y")
    fig.tight_layout()
    _save(fig, "fig2_component_breakdown")
    return fig


# ---------------------------------------------------------------------------
# Fig 3: Season heatmap — teams × seasons (min 3 seasons)
# ---------------------------------------------------------------------------
def plot_season_heatmap(master: pd.DataFrame, min_seasons: int = 3):
    counts = master.groupby("team")["luck_score"].count()
    teams = counts[counts >= min_seasons].index
    pivot = (
        master[master["team"].isin(teams)]
        .pivot_table(index="team", columns="season", values="luck_score")
    )
    pivot = pivot.loc[pivot.mean(axis=1).sort_values(ascending=False).index]

    vals = pivot.values[~np.isnan(pivot.values)]
    vmax = np.abs(vals).max() if len(vals) else 1
    cmap = mcolors.LinearSegmentedColormap.from_list("luck", [RED, "#1e2d40", GREEN])

    fig, ax = plt.subplots(figsize=(max(10, len(pivot.columns) * 1.1), max(6, len(pivot) * 0.38)))
    sns.heatmap(pivot, ax=ax, cmap=cmap, center=0, vmin=-vmax, vmax=vmax,
                linewidths=0.3, linecolor="#0a1520",
                annot=True, fmt=".2f", annot_kws={"size": 7},
                cbar_kws={"label": "Luck Score"})
    ax.set_title("UCL Luck Score by Team & Season", pad=10)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", rotation=35, labelsize=8)
    ax.tick_params(axis="y", labelsize=8)
    fig.tight_layout()
    _save(fig, "fig3_season_heatmap")
    return fig


# ---------------------------------------------------------------------------
# Fig 4: Luck over time — line chart for selected teams
# ---------------------------------------------------------------------------
def plot_luck_over_time(master: pd.DataFrame, summary: pd.DataFrame, n_each: int = 4):
    luckiest   = summary.nlargest(n_each, "mean_luck")["team"].tolist()
    unluckiest = summary.nsmallest(n_each, "mean_luck")["team"].tolist()
    teams = luckiest + unluckiest

    palette_luck   = sns.color_palette("Greens_r", n_colors=n_each)
    palette_unluck = sns.color_palette("Reds_r",   n_colors=n_each)
    colors = palette_luck + palette_unluck

    fig, ax = plt.subplots(figsize=(12, 5))
    for team, color in zip(teams, colors):
        df_t = master[master["team"] == team].sort_values("season")
        ax.plot(df_t["season"], df_t["luck_score"],
                marker="o", label=team, color=color, linewidth=1.8, markersize=5)

    ax.axhline(0, color=WHITE, linewidth=0.8, linestyle="--", alpha=0.6)
    ax.set_xlabel("Season")
    ax.set_ylabel("Luck Score")
    ax.set_title(f"Luck Score Over Time — {n_each} Luckiest (green) vs {n_each} Unluckiest (red)", pad=10)
    ax.legend(facecolor=DARK, edgecolor=WHITE, fontsize=8, ncol=2)
    ax.grid(axis="y")
    plt.xticks(rotation=30)
    fig.tight_layout()
    _save(fig, "fig4_luck_over_time")
    return fig


# ---------------------------------------------------------------------------
# Fig 5: Autocorrelation bar — is luck persistent?
# ---------------------------------------------------------------------------
def plot_autocorrelation(master: pd.DataFrame, min_seasons: int = 4):
    records = []
    for team, grp in master.groupby("team"):
        grp = grp.sort_values("season")
        luck = grp["luck_score"].dropna()
        if len(luck) >= min_seasons:
            records.append({"team": team, "autocorr": float(luck.autocorr(lag=1)), "n": len(luck)})
    ac = pd.DataFrame(records).sort_values("autocorr", ascending=False)

    colors = [GREEN if v >= 0 else RED for v in ac["autocorr"]]
    fig, ax = plt.subplots(figsize=(10, max(5, len(ac) * 0.38)))
    ax.barh(ac["team"], ac["autocorr"], color=colors, alpha=0.85, height=0.65)
    ax.axvline(0, color=WHITE, linewidth=0.8)
    ax.set_xlabel("Lag-1 Autocorrelation of Luck Score")
    ax.set_title(
        f"Year-to-Year Luck Persistence (min {min_seasons} seasons)\n"
        "Positive = luck repeats  |  Negative = luck reverses", pad=10
    )
    ax.grid(axis="x")

    mean_ac = ac["autocorr"].mean()
    ax.axvline(mean_ac, color=GOLD, linewidth=1.2, linestyle="--",
               label=f"Mean = {mean_ac:.2f}")
    ax.legend(facecolor=DARK, edgecolor=WHITE, fontsize=8)
    fig.tight_layout()
    _save(fig, "fig5_autocorrelation")
    return fig


# ---------------------------------------------------------------------------
# Fig 6: Attack luck vs Defense luck scatter — quadrant view
# ---------------------------------------------------------------------------
def plot_attack_vs_defense(summary: pd.DataFrame, min_seasons: int = 3):
    df = summary[summary["n_seasons"] >= min_seasons].copy()
    bubble = (df["n_seasons"] - df["n_seasons"].min() + 1) * 40

    fig, ax = plt.subplots(figsize=(10, 8))
    sc = ax.scatter(
        df["mean_attack_luck"], df["mean_defense_luck"],
        s=bubble, c=df["mean_luck"], cmap="RdYlGn",
        alpha=0.85, edgecolors=WHITE, linewidths=0.4,
    )
    plt.colorbar(sc, ax=ax, label="Composite Luck Score")

    for _, row in df.iterrows():
        ax.annotate(row["team"], (row["mean_attack_luck"], row["mean_defense_luck"]),
                    fontsize=7, color=WHITE, alpha=0.85,
                    xytext=(4, 4), textcoords="offset points")

    ax.axhline(0, color=WHITE, linewidth=0.7, linestyle="--", alpha=0.5)
    ax.axvline(0, color=WHITE, linewidth=0.7, linestyle="--", alpha=0.5)

    ax.text( 0.02,  0.98, "Lucky attack\nLucky defense",   transform=ax.transAxes, fontsize=8, color=GREEN,  va="top")
    ax.text( 0.75,  0.98, "Unlucky attack\nLucky defense", transform=ax.transAxes, fontsize=8, color=GOLD,   va="top")
    ax.text( 0.02,  0.05, "Lucky attack\nUnlucky defense", transform=ax.transAxes, fontsize=8, color=GOLD,   va="top")
    ax.text( 0.75,  0.05, "Unlucky attack\nUnlucky defense", transform=ax.transAxes, fontsize=8, color=RED, va="top")

    ax.set_xlabel("Mean Attack Luck  (goals scored vs xG)")
    ax.set_ylabel("Mean Defense Luck  (goals conceded vs xGA)")
    ax.set_title("Attack Luck vs Defense Luck\n(bubble size = seasons in dataset)", pad=10)
    ax.grid()
    fig.tight_layout()
    _save(fig, "fig6_attack_vs_defense")
    return fig


# ---------------------------------------------------------------------------
# Fig 7: Distribution of luck scores — all team-seasons
# ---------------------------------------------------------------------------
def plot_luck_distribution(master: pd.DataFrame):
    luck = master["luck_score"].dropna()

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(luck, bins=30, color=BLUE, edgecolor=DARK, alpha=0.85, density=True)

    from scipy.stats import norm
    mu, sigma = luck.mean(), luck.std()
    x = np.linspace(luck.min(), luck.max(), 300)
    ax.plot(x, norm.pdf(x, mu, sigma), color=GOLD, linewidth=2, label=f"Normal fit  μ={mu:.2f}  σ={sigma:.2f}")

    ax.axvline(0,  color=WHITE, linewidth=0.9, linestyle="--", alpha=0.6, label="Zero luck")
    ax.axvline(mu, color=GREEN, linewidth=1.2, linestyle="-",  label=f"Mean = {mu:.2f}")

    ax.set_xlabel("Luck Score")
    ax.set_ylabel("Density")
    ax.set_title("Distribution of Luck Scores — All UCL Team-Seasons", pad=10)
    ax.legend(facecolor=DARK, edgecolor=WHITE, fontsize=8)
    ax.grid(axis="y")
    fig.tight_layout()
    _save(fig, "fig7_luck_distribution")
    return fig


# ---------------------------------------------------------------------------
# Fig 8: Luck over time — top 10 European clubs
# ---------------------------------------------------------------------------
def plot_top10_luck_over_time(master: pd.DataFrame):
    TEAMS = [
        "Paris Saint-Germain",
        "Bayern Munich",
        "Real Madrid",
        "Barcelona",
        "Manchester City",
        "Liverpool",
        "Atlético Madrid",
    ]
    TEAM_COLORS = {
        "Paris Saint-Germain": "#003f7f",   # deep blue
        "Bayern Munich":       "#dc052d",   # Bayern red
        "Real Madrid":         "#f5d200",   # gold
        "Barcelona":           "#a50044",   # Barça crimson
        "Manchester City":     "#6cabdd",   # sky blue
        "Liverpool":           "#c8102e",   # Liverpool red
        "Atlético Madrid":     "#e8321a",   # Atleti orange-red
    }

    all_seasons = sorted(master["season"].unique())

    fig, ax = plt.subplots(figsize=(13, 6))

    for team in TEAMS:
        df_t = master[master["team"] == team].sort_values("season")
        color = TEAM_COLORS[team]
        lw = 2.2
        ax.plot(df_t["season"], df_t["luck_score"],
                marker="o", label=team, color=color,
                linewidth=lw, markersize=5.5, zorder=3)

    ax.axhline(0, color=WHITE, linewidth=0.8, linestyle="--", alpha=0.5)
    ax.fill_between(all_seasons,  0.15,  0.5, alpha=0.04, color=GREEN)
    ax.fill_between(all_seasons, -0.5,  -0.15, alpha=0.04, color=RED)

    ax.set_xlabel("Season")
    ax.set_ylabel("Luck Score")
    ax.set_title("Luck Over Time — Top 10 European Clubs", pad=12)
    ax.legend(facecolor=DARK, edgecolor="#2a3f5f", fontsize=8.5,
              ncol=2, loc="upper right", framealpha=0.85)
    ax.grid(axis="y")
    plt.xticks(rotation=30)
    fig.tight_layout()
    _save(fig, "fig8_top10_luck_over_time")
    return fig


# ---------------------------------------------------------------------------
# Main — generate all figures
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    master  = pd.read_csv("data/combined/master_with_luck.csv")
    summary = pd.read_csv("data/combined/luck_summary.csv")

    plot_leaderboard(summary)
    plot_component_breakdown(summary)
    plot_season_heatmap(master)
    plot_luck_over_time(master, summary)
    plot_autocorrelation(master)
    plot_attack_vs_defense(summary)
    plot_luck_distribution(master)
    plot_top10_luck_over_time(master)

    print("\nAll figures saved to figures/")
