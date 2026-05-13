"""
extended_analysis.py
--------------------
Four analyses using previously unused columns:

  1. Luck vs UCL Advancement  (mp as outcome)
  2. Penalty-clean luck       (pk / pk_att correction)
  3. Shot volume vs quality   (sot_pct + g_per_sh decomposition)
  4. League-of-origin effect  (country aggregation)
  5. Squad depth vs luck      (squad_size correlation)

Reads:  data/combined/master_with_luck.csv
        data/combined/luck_summary.csv
Saves:  data/combined/extended_summary.csv
        figures/ext_*.png
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from pathlib import Path
from scipy import stats

# ---------------------------------------------------------------------------
# Theme (same as luck_plots.py)
# ---------------------------------------------------------------------------
DARK  = "#0d1b2e"
BLUE  = "#1a4b8c"
GOLD  = "#c8a951"
WHITE = "#f5f5f5"
GREEN = "#2ecc71"
RED   = "#e74c3c"
GREY  = "#7f8c8d"

plt.rcParams.update({
    "figure.facecolor": DARK, "axes.facecolor": DARK,
    "axes.edgecolor": WHITE,  "axes.labelcolor": WHITE,
    "xtick.color": WHITE,     "ytick.color": WHITE,
    "text.color": WHITE,      "grid.color": "#1e3050",
    "grid.linestyle": "--",   "grid.alpha": 0.5,
    "font.family": "sans-serif", "font.size": 10,
})

FIGURES_DIR = Path("figures")
FIGURES_DIR.mkdir(exist_ok=True)

def _save(fig, name):
    path = FIGURES_DIR / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=DARK)
    print(f"  Saved → {path}")

TOP15_EUROPE = [
    "Paris Saint-Germain", "Bayern Munich", "Real Madrid", "Barcelona",
    "Manchester City", "Liverpool", "Atlético Madrid", "Juventus",
    "Inter", "Dortmund", "Chelsea", "Porto", "Benfica", "RB Leipzig", "Napoli",
]

COUNTRY_LABELS = {
    "eng": "England", "es": "Spain", "de": "Germany", "it": "Italy",
    "fr": "France",   "pt": "Portugal", "nl": "Netherlands", "be": "Belgium",
    "ru": "Russia",   "ua": "Ukraine",  "tr": "Turkey", "gr": "Greece",
    "at": "Austria",  "ch": "Switzerland", "dk": "Denmark", "sc": "Scotland",
    "hr": "Croatia",  "rs": "Serbia",
}

SEP = "=" * 60


# ===========================================================================
# ANALYSIS 1 — Luck vs UCL Advancement
# ===========================================================================

def analysis_luck_vs_advancement(df: pd.DataFrame):
    print(f"\n{SEP}")
    print("ANALYSIS 1 — Does Luck Help You Advance in UCL?")
    print(SEP)

    # mp: 6=group exit, 8=R16, 10=QF, 13=SF, 15=Final, 17/18=Winner
    round_map = {6: "Group", 8: "R16", 10: "QF", 13: "SF", 15: "Final", 17: "Winner", 18: "Winner"}

    def mp_to_round(mp):
        mp = int(mp)
        for threshold, label in sorted(round_map.items()):
            if mp <= threshold:
                return label
        return "Winner"

    df = df.copy()
    df["round"] = df["mp"].apply(mp_to_round)
    df["round_num"] = df["mp"]

    # Pearson correlation
    r, p = stats.pearsonr(df["luck_score"], df["round_num"])
    print(f"\n  Correlation luck_score vs matches played: r={r:.3f}  p={p:.4f}")
    print(f"  {'→ Statistically significant' if p < 0.05 else '→ Not statistically significant'}")

    # Mean luck by round
    round_order = ["Group", "R16", "QF", "SF", "Final", "Winner"]
    by_round = (
        df.groupby("round")[["luck_score", "attack_luck", "defense_luck", "finish_luck"]]
        .mean()
        .reindex([r for r in round_order if r in df["round"].unique()])
    )
    print("\n  Mean luck score by UCL round reached:")
    print(by_round.round(3).to_string())

    # Component correlations
    print("\n  Correlations per luck component vs mp:")
    for col in ["attack_luck", "defense_luck", "finish_luck"]:
        r2, p2 = stats.pearsonr(df[col], df["round_num"])
        print(f"    {col:<20} r={r2:+.3f}  p={p2:.4f}")

    # --- Plot 1a: scatter luck vs mp ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    rng = np.random.default_rng(42)
    jitter = rng.uniform(-0.25, 0.25, size=len(df))
    sc = ax.scatter(df["luck_score"], df["round_num"] + jitter,
                    c=df["luck_score"], cmap="RdYlGn",
                    alpha=0.65, s=40, edgecolors="none")
    plt.colorbar(sc, ax=ax, label="Luck Score")
    x_line = np.linspace(df["luck_score"].min(), df["luck_score"].max(), 200)
    slope, intercept, *_ = stats.linregress(df["luck_score"], df["round_num"])
    ax.plot(x_line, slope * x_line + intercept, color=GOLD, linewidth=2,
            label=f"OLS  r={r:.2f}  p={p:.3f}")
    ax.set_yticks(sorted(df["round_num"].unique()))
    ax.set_xlabel("Luck Score")
    ax.set_ylabel("Matches Played (proxy for round reached)")
    ax.set_title("Luck Score vs UCL Advancement", pad=8)
    ax.legend(facecolor=DARK, edgecolor=WHITE, fontsize=8)
    ax.grid()

    # --- Plot 1b: boxplot luck by round ---
    ax2 = axes[1]
    present_rounds = [r for r in round_order if r in df["round"].values]
    data_by_round = [df.loc[df["round"] == r, "luck_score"].values for r in present_rounds]
    bp = ax2.boxplot(data_by_round, patch_artist=True, labels=present_rounds,
                     medianprops={"color": GOLD, "linewidth": 2})
    cmap = plt.cm.RdYlGn
    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(cmap(i / len(present_rounds)))
        patch.set_alpha(0.75)
    ax2.axhline(0, color=WHITE, linewidth=0.8, linestyle="--", alpha=0.6)
    ax2.set_xlabel("UCL Round Reached")
    ax2.set_ylabel("Luck Score")
    ax2.set_title("Luck Distribution by Round Reached", pad=8)
    ax2.grid(axis="y")

    fig.tight_layout()
    _save(fig, "ext1_luck_vs_advancement")


# ===========================================================================
# ANALYSIS 2 — Penalty-Clean Luck
# ===========================================================================

def analysis_penalty_clean(df: pd.DataFrame, summary: pd.DataFrame):
    print(f"\n{SEP}")
    print("ANALYSIS 2 — Penalty-Clean Attack Luck")
    print(SEP)

    df = df.copy()

    # non-penalty goals (gls already per-season total, goals_scored is per-match)
    # rebuild per-match np_goals using total gls and pk
    df["np_goals_total"] = df["gls"] - df["pk"]
    df["xg_total"]       = df["xg"] * df["mp"]
    df["attack_luck_np"] = (df["np_goals_total"] - df["xg_total"]) / df["xg_total"].replace(0, np.nan)

    df["pk_rate"]        = df["pk_att"] / df["mp"]
    df["pk_conversion"]  = np.where(df["pk_att"] > 0, df["pk"] / df["pk_att"], np.nan)

    # compare original vs np attack luck per team
    comparison = (
        df.groupby("team")[["attack_luck", "attack_luck_np"]]
        .mean()
        .assign(delta=lambda x: x["attack_luck"] - x["attack_luck_np"])
        .sort_values("delta", ascending=False)
    )
    print("\n  Teams whose attack luck drops most after removing penalties:")
    print(comparison.head(8).round(3).to_string())
    print("\n  Teams whose attack luck changes least (penalties irrelevant):")
    print(comparison.tail(5).round(3).to_string())

    # penalty rate by country
    pk_by_country = (
        df.groupby("country")[["pk_att", "mp"]]
        .sum()
        .assign(pk_per_match=lambda x: x["pk_att"] / x["mp"])
        .sort_values("pk_per_match", ascending=False)
    )
    pk_by_country.index = pk_by_country.index.map(lambda c: COUNTRY_LABELS.get(c, c))
    print("\n  Penalty rate by country (pk per match):")
    print(pk_by_country[["pk_per_match"]].round(3).head(10).to_string())

    # --- Plot 2: scatter original vs np attack luck (top 15 Europe only) ---
    top15_df = comparison[comparison.index.isin(TOP15_EUROPE)].reset_index()
    colors = [GREEN if d > 0 else RED for d in top15_df["delta"]]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.scatter(top15_df["attack_luck"], top15_df["attack_luck_np"],
               c=colors, alpha=0.9, s=80, edgecolors=WHITE, linewidths=0.5)
    lims = [min(top15_df["attack_luck"].min(), top15_df["attack_luck_np"].min()) - 0.05,
            max(top15_df["attack_luck"].max(), top15_df["attack_luck_np"].max()) + 0.05]
    ax.plot(lims, lims, color=WHITE, linewidth=0.8, linestyle="--", alpha=0.5, label="No change line")
    for _, row in top15_df.iterrows():
        ax.annotate(row["team"], (row["attack_luck"], row["attack_luck_np"]),
                    fontsize=7.5, color=WHITE, xytext=(5, 3), textcoords="offset points")
    ax.set_xlabel("Original Attack Luck")
    ax.set_ylabel("Penalty-Clean Attack Luck")
    ax.set_title("How Much Do Penalties Inflate Attack Luck?\n(Top 15 European clubs)", pad=8)
    ax.legend(facecolor=DARK, edgecolor=WHITE, fontsize=8)
    ax.grid()

    ax2 = axes[1]
    top_pk = pk_by_country.head(12).reset_index()
    bar_colors = [GREEN if v > pk_by_country["pk_per_match"].median() else GREY
                  for v in top_pk["pk_per_match"]]
    ax2.barh(top_pk["country"], top_pk["pk_per_match"], color=bar_colors, alpha=0.85)
    ax2.axvline(pk_by_country["pk_per_match"].mean(), color=GOLD, linewidth=1.2,
                linestyle="--", label="Average")
    ax2.set_xlabel("Penalty Kicks per Match")
    ax2.set_title("Penalty Rate by Country", pad=8)
    ax2.legend(facecolor=DARK, edgecolor=WHITE, fontsize=8)
    ax2.grid(axis="x")

    fig.tight_layout()
    _save(fig, "ext2_penalty_clean")

    return df


# ===========================================================================
# ANALYSIS 3 — Shot Volume vs Quality
# ===========================================================================

def analysis_shot_decomposition(df: pd.DataFrame):
    print(f"\n{SEP}")
    print("ANALYSIS 3 — Shot Volume vs Finishing Quality")
    print(SEP)

    df = df.copy()
    league_sot_pct = df["sot_pct"].mean()
    league_g_per_sh = df["g_per_sh"].mean()

    # volume luck: did they get a higher % of shots on target than average?
    df["volume_luck"]  = (df["sot_pct"]  - league_sot_pct)  / league_sot_pct
    # quality luck: did they score more per total shot than average?
    df["quality_luck"] = (df["g_per_sh"] - league_g_per_sh) / league_g_per_sh

    r_vol,  p_vol  = stats.pearsonr(df["volume_luck"],  df["luck_score"])
    r_qual, p_qual = stats.pearsonr(df["quality_luck"], df["luck_score"])
    r_fin,  p_fin  = stats.pearsonr(df["finish_luck"],  df["luck_score"])

    print(f"\n  League avg shot-on-target rate : {league_sot_pct:.1f}%")
    print(f"  League avg goals per shot      : {league_g_per_sh:.3f}")
    print(f"\n  Correlation with composite luck_score:")
    print(f"    volume_luck  (sot_pct)   r={r_vol:+.3f}  p={p_vol:.4f}")
    print(f"    quality_luck (g_per_sh)  r={r_qual:+.3f}  p={p_qual:.4f}")
    print(f"    finish_luck  (g_per_sot) r={r_fin:+.3f}  p={p_fin:.4f}")

    team_decomp = (
        df.groupby("team")[["volume_luck", "quality_luck", "finish_luck", "luck_score"]]
        .mean()
        .sort_values("luck_score", ascending=False)
    )
    print("\n  Top 10 teams — volume vs quality breakdown:")
    print(team_decomp.head(10).round(3).to_string())

    # --- Plot 3a: scatter volume vs quality luck (top 15 Europe only) ---
    top15_decomp = team_decomp[team_decomp.index.isin(TOP15_EUROPE)].reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    sc = ax.scatter(top15_decomp["volume_luck"], top15_decomp["quality_luck"],
                    c=top15_decomp["luck_score"], cmap="RdYlGn",
                    s=90, alpha=0.9, edgecolors=WHITE, linewidths=0.5)
    plt.colorbar(sc, ax=ax, label="Luck Score")
    ax.axhline(0, color=WHITE, linewidth=0.7, linestyle="--", alpha=0.5)
    ax.axvline(0, color=WHITE, linewidth=0.7, linestyle="--", alpha=0.5)
    for _, row in top15_decomp.iterrows():
        ax.annotate(row["team"], (row["volume_luck"], row["quality_luck"]),
                    fontsize=7.5, color=WHITE, xytext=(5, 3), textcoords="offset points")
    ax.set_xlabel("Volume Luck  (shots on target % vs average)")
    ax.set_ylabel("Quality Luck  (goals per shot vs average)")
    ax.set_title("Shot Volume Luck vs Finishing Quality Luck\n(Top 15 European clubs)", pad=8)
    ax.text(0.02, 0.98, "High volume\nHigh quality", transform=ax.transAxes,
            fontsize=7, color=GREEN, va="top")
    ax.text(0.70, 0.02, "Low volume\nLow quality", transform=ax.transAxes,
            fontsize=7, color=RED, va="bottom")
    ax.grid()

    # --- Plot 3b: stacked bar volume vs quality for top 15 Europe ---
    ax2 = axes[1]
    top15_sorted = top15_decomp.sort_values("luck_score", ascending=False)
    x = np.arange(len(top15_sorted))
    ax2.bar(x, top15_sorted["volume_luck"],  0.5, label="Volume luck",  color=BLUE,  alpha=0.85)
    ax2.bar(x, top15_sorted["quality_luck"], 0.5, bottom=top15_sorted["volume_luck"],
            label="Quality luck", color=GOLD, alpha=0.85)
    ax2.set_xticks(x)
    ax2.set_xticklabels(top15_sorted["team"], rotation=38, ha="right", fontsize=8)
    ax2.axhline(0, color=WHITE, linewidth=0.8)
    ax2.set_ylabel("Luck component")
    ax2.set_title("Volume vs Quality Luck — Top 15 European Clubs", pad=8)
    ax2.legend(facecolor=DARK, edgecolor=WHITE, fontsize=8)
    ax2.grid(axis="y")

    fig.tight_layout()
    _save(fig, "ext3_shot_decomposition")

    return df


# ===========================================================================
# ANALYSIS 4 — League-of-Origin Effect
# ===========================================================================

def analysis_league_effect(df: pd.DataFrame):
    print(f"\n{SEP}")
    print("ANALYSIS 4 — League-of-Origin Effect")
    print(SEP)

    df = df.copy()
    df["country_label"] = df["country"].map(lambda c: COUNTRY_LABELS.get(c, c))

    min_teams = 5
    counts = df.groupby("country_label")["team"].count()
    valid_countries = counts[counts >= min_teams].index

    by_country = (
        df[df["country_label"].isin(valid_countries)]
        .groupby("country_label")[["luck_score", "attack_luck", "defense_luck", "finish_luck", "mp"]]
        .agg(["mean", "std", "count"])
    )
    by_country.columns = ["_".join(c) for c in by_country.columns]
    by_country = by_country.sort_values("luck_score_mean", ascending=False)

    print(f"\n  Countries with ≥ {min_teams} team-seasons:")
    print(by_country[["luck_score_mean", "luck_score_std", "luck_score_count",
                       "attack_luck_mean", "defense_luck_mean", "mp_mean"]].round(3).to_string())

    # ANOVA: is there a significant difference in luck across countries?
    groups = [df.loc[df["country_label"] == c, "luck_score"].values
              for c in valid_countries]
    f_stat, p_anova = stats.f_oneway(*groups)
    print(f"\n  One-way ANOVA across countries: F={f_stat:.3f}  p={p_anova:.4f}")
    print(f"  {'→ Significant difference between leagues' if p_anova < 0.05 else '→ No significant difference between leagues'}")

    # --- Plot 4: country comparison ---
    country_df = by_country.reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    colors = [GREEN if v >= 0 else RED for v in country_df["luck_score_mean"]]
    ax.barh(country_df["country_label"], country_df["luck_score_mean"],
            color=colors, alpha=0.85, height=0.6)
    ax.errorbar(country_df["luck_score_mean"], country_df["country_label"],
                xerr=country_df["luck_score_std"] / np.sqrt(country_df["luck_score_count"]),
                fmt="none", color=WHITE, linewidth=1.1, capsize=3, alpha=0.7)
    ax.axvline(0, color=WHITE, linewidth=0.8)
    ax.set_xlabel("Mean Luck Score")
    ax.set_title("UCL Luck by League of Origin\n(±1 SE)", pad=8)
    ax.grid(axis="x")

    ax2 = axes[1]
    x = np.arange(len(country_df)) * 1.4   # wider spacing between country groups
    w = 0.32
    bars_atk = ax2.bar(x - w, country_df["attack_luck_mean"],  w, label="Attack",
                       color="#3b82f6", alpha=0.9, edgecolor=DARK, linewidth=0.6)
    bars_def = ax2.bar(x,     country_df["defense_luck_mean"], w, label="Defense",
                       color="#22c55e", alpha=0.9, edgecolor=DARK, linewidth=0.6)
    bars_fin = ax2.bar(x + w, country_df["finish_luck_mean"],  w, label="Finishing",
                       color="#f59e0b", alpha=0.9, edgecolor=DARK, linewidth=0.6)
    # value labels on each bar
    for bars in (bars_atk, bars_def, bars_fin):
        for bar in bars:
            h = bar.get_height()
            if abs(h) > 0.01:
                ax2.text(bar.get_x() + bar.get_width() / 2,
                         h + (0.005 if h >= 0 else -0.018),
                         f"{h:+.2f}", ha="center", va="bottom" if h >= 0 else "top",
                         fontsize=5.5, color=WHITE)
    ax2.set_xticks(x)
    ax2.set_xticklabels(country_df["country_label"], rotation=35, ha="right", fontsize=8.5)
    ax2.axhline(0, color=WHITE, linewidth=0.8)
    ax2.set_ylabel("Mean luck component")
    ax2.set_title("Luck Components by League", pad=8)
    ax2.legend(facecolor=DARK, edgecolor=WHITE, fontsize=8.5)
    ax2.grid(axis="y", alpha=0.4)

    fig.tight_layout()
    _save(fig, "ext4_league_effect")


# ===========================================================================
# ANALYSIS 5 — Squad Depth vs Luck
# ===========================================================================

def analysis_squad_depth(df: pd.DataFrame):
    print(f"\n{SEP}")
    print("ANALYSIS 5 — Squad Depth vs Luck")
    print(SEP)

    df = df.copy()

    r, p = stats.pearsonr(df["squad_size"], df["luck_score"])
    print(f"\n  Correlation squad_size vs luck_score: r={r:.3f}  p={p:.4f}")
    print(f"  {'→ Significant' if p < 0.05 else '→ Not significant'}")

    for col in ["attack_luck", "defense_luck", "finish_luck"]:
        r2, p2 = stats.pearsonr(df["squad_size"], df[col])
        print(f"  squad_size vs {col:<20} r={r2:+.3f}  p={p2:.4f}")

    # also check squad size vs mp (do bigger squads go further?)
    r3, p3 = stats.pearsonr(df["squad_size"], df["mp"])
    print(f"  squad_size vs mp (advancement)     r={r3:+.3f}  p={p3:.4f}")

    # bins: small (<21), medium (21-24), large (>24)
    df["depth_tier"] = pd.cut(df["squad_size"], bins=[0, 21, 24, 99],
                               labels=["Small (<21)", "Medium (21-24)", "Large (>24)"])
    by_tier = df.groupby("depth_tier")[["luck_score", "attack_luck", "defense_luck", "mp"]].mean()
    print("\n  Mean stats by squad depth tier:")
    print(by_tier.round(3).to_string())

    # --- Plot 5 ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.scatter(df["squad_size"], df["luck_score"],
               c=df["luck_score"], cmap="RdYlGn",
               alpha=0.65, s=45, edgecolors="none")
    slope, intercept, *_ = stats.linregress(df["squad_size"], df["luck_score"])
    x_line = np.linspace(df["squad_size"].min(), df["squad_size"].max(), 200)
    ax.plot(x_line, slope * x_line + intercept, color=GOLD, linewidth=2,
            label=f"OLS  r={r:.2f}  p={p:.3f}")
    ax.axhline(0, color=WHITE, linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_xlabel("Squad Size (players used)")
    ax.set_ylabel("Luck Score")
    ax.set_title("Squad Depth vs Luck Score", pad=8)
    ax.legend(facecolor=DARK, edgecolor=WHITE, fontsize=8)
    ax.grid()

    ax2 = axes[1]
    tier_df = by_tier.reset_index()
    x = np.arange(len(tier_df))
    w = 0.28
    ax2.bar(x - w, tier_df["attack_luck"],  w, label="Attack",  color="#3498db", alpha=0.85)
    ax2.bar(x,     tier_df["defense_luck"], w, label="Defense", color=GREEN,     alpha=0.85)
    ax2.bar(x + w, tier_df["luck_score"],   w, label="Composite", color=GOLD,   alpha=0.85)
    ax2.set_xticks(x)
    ax2.set_xticklabels(tier_df["depth_tier"], fontsize=9)
    ax2.axhline(0, color=WHITE, linewidth=0.8)
    ax2.set_ylabel("Mean luck")
    ax2.set_title("Luck by Squad Depth Tier", pad=8)
    ax2.legend(facecolor=DARK, edgecolor=WHITE, fontsize=8)
    ax2.grid(axis="y")

    fig.tight_layout()
    _save(fig, "ext5_squad_depth")


# ===========================================================================
# Save extended summary
# ===========================================================================

def save_extended_summary(df: pd.DataFrame, summary: pd.DataFrame):
    league_sot_pct = df["sot_pct"].mean()
    league_g_per_sh = df["g_per_sh"].mean()

    df = df.copy()
    df["np_goals_total"] = df["gls"] - df["pk"]
    df["xg_total"]       = df["xg"] * df["mp"]
    df["attack_luck_np"] = (df["np_goals_total"] - df["xg_total"]) / df["xg_total"].replace(0, np.nan)
    df["volume_luck"]    = (df["sot_pct"]  - league_sot_pct)  / league_sot_pct
    df["quality_luck"]   = (df["g_per_sh"] - league_g_per_sh) / league_g_per_sh
    df["pk_rate"]        = df["pk_att"] / df["mp"]

    team_ext = (
        df.groupby("team")[["attack_luck_np", "volume_luck", "quality_luck", "pk_rate", "squad_size", "mp"]]
        .mean()
        .round(4)
        .reset_index()
    )

    extended = summary.merge(team_ext, on="team", how="left")

    # add autocorrelation
    records = []
    for team, grp in df.groupby("team"):
        luck = grp.sort_values("season")["luck_score"].dropna()
        ac = float(luck.autocorr(lag=1)) if len(luck) >= 4 else np.nan
        records.append({"team": team, "autocorr_lag1": round(ac, 4) if not np.isnan(ac) else np.nan})
    extended = extended.merge(pd.DataFrame(records), on="team", how="left")

    out = Path("data/combined/extended_summary.csv")
    extended.to_csv(out, index=False)
    print(f"\n  Saved → {out}  ({extended.shape[0]} teams × {extended.shape[1]} columns)")
    return extended


# ===========================================================================
# Main
# ===========================================================================

if __name__ == "__main__":
    master  = pd.read_csv("data/combined/master_with_luck.csv")
    summary = pd.read_csv("data/combined/luck_summary.csv")

    analysis_luck_vs_advancement(master)
    df_with_pk    = analysis_penalty_clean(master, summary)
    df_with_shots = analysis_shot_decomposition(master)
    analysis_league_effect(master)
    analysis_squad_depth(master)

    extended = save_extended_summary(master, summary)

    print(f"\n{SEP}")
    print("ALL ANALYSES COMPLETE")
    print(SEP)
    print("\nNew columns in extended_summary.csv:")
    new_cols = ["attack_luck_np", "volume_luck", "quality_luck", "pk_rate", "squad_size", "mp", "autocorr_lag1"]
    for c in new_cols:
        print(f"  {c}")
