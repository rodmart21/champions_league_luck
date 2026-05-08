"""
luck_analysis.py
----------------
First-draft analysis to quantify luck in the UEFA Champions League.

Luck dimensions (all computed per-match to normalise for varying MP):
  1. AttackLuck   = (goals_scored - xg) / xg
                    Positive → team scored more than xG model expected
  2. DefenseLuck  = (xga - goals_conceded) / xga
                    Positive → team conceded fewer than xG model expected
  3. FinishLuck   = (g_per_sot - league_avg_g_per_sot) / league_avg_g_per_sot
                    Positive → team converted shots on target at above-average rate
                    (proxy for GK luck + clinical finishing, since PSxG is unavailable)
  4. LuckScore    = mean(AttackLuck, DefenseLuck, FinishLuck)

Statistical layer:
  - Bootstrap 95% CI on mean LuckScore per team (10k resamples)
  - One-sample t-test per team (H0: mean luck == 0) + BH FDR correction
  - Lag-1 autocorrelation per team (is luck repeatable year-to-year?)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from statsmodels.stats.multitest import multipletests

DATA_PATH = Path("data/combined/master.csv")
MIN_SEASONS = 2  # minimum seasons for a team to appear in the summary


# ---------------------------------------------------------------------------
# 1. Load & clean
# ---------------------------------------------------------------------------

def load_clean() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["xg", "shots_on_target"])
    return df


# ---------------------------------------------------------------------------
# 2. Luck metrics
# ---------------------------------------------------------------------------

def add_luck_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["attack_luck"] = (df["goals_scored"] - df["xg"]) / df["xg"].replace(0, np.nan)
    df["defense_luck"] = (df["xga"] - df["goals_conceded"]) / df["xga"].replace(0, np.nan)

    league_avg_g_per_sot = df["g_per_sot"].mean()
    df["finish_luck"] = (
        (df["g_per_sot"] - league_avg_g_per_sot) / league_avg_g_per_sot
    )

    df["luck_score"] = df[["attack_luck", "defense_luck", "finish_luck"]].mean(axis=1)

    return df, league_avg_g_per_sot


# ---------------------------------------------------------------------------
# 3. Bootstrap CI
# ---------------------------------------------------------------------------

def bootstrap_ci(
    values: np.ndarray,
    n_boot: int = 10_000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) < 2:
        return np.nan, np.nan
    boot_means = np.array([
        rng.choice(values, size=len(values), replace=True).mean()
        for _ in range(n_boot)
    ])
    alpha = 1 - ci
    return (
        float(np.percentile(boot_means, 100 * alpha / 2)),
        float(np.percentile(boot_means, 100 * (1 - alpha / 2))),
    )


# ---------------------------------------------------------------------------
# 4. Team-level summary
# ---------------------------------------------------------------------------

def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    records = []
    for team, grp in df.groupby("team"):
        if len(grp) < MIN_SEASONS:
            continue
        luck = grp["luck_score"].dropna()
        lo, hi = bootstrap_ci(luck.values)
        t_stat, p_val = (
            stats.ttest_1samp(luck, popmean=0)
            if len(luck) >= 3
            else (np.nan, np.nan)
        )
        records.append({
            "team": team,
            "n_seasons": len(grp),
            "mean_luck": luck.mean(),
            "std_luck": luck.std(),
            "mean_attack_luck": grp["attack_luck"].mean(),
            "mean_defense_luck": grp["defense_luck"].mean(),
            "mean_finish_luck": grp["finish_luck"].mean(),
            "ci_lower": lo,
            "ci_upper": hi,
            "t_stat": t_stat,
            "p_raw": p_val,
            "best_season": grp.loc[grp["luck_score"].idxmax(), "season"],
            "worst_season": grp.loc[grp["luck_score"].idxmin(), "season"],
        })

    summary = pd.DataFrame(records).sort_values("mean_luck", ascending=False).reset_index(drop=True)

    # BH FDR correction on teams with enough data
    valid = summary["p_raw"].notna()
    p_adj = np.full(len(summary), np.nan)
    if valid.sum() > 0:
        _, p_corrected, _, _ = multipletests(summary.loc[valid, "p_raw"].values, method="fdr_bh")
        p_adj[valid] = p_corrected
    summary["p_adj"] = p_adj
    summary["significant"] = summary["p_adj"] < 0.05

    return summary


# ---------------------------------------------------------------------------
# 5. Autocorrelation (is luck persistent year-to-year?)
# ---------------------------------------------------------------------------

def compute_autocorrelation(df: pd.DataFrame) -> pd.DataFrame:
    records = []
    for team, grp in df.groupby("team"):
        grp = grp.sort_values("season")
        luck = grp["luck_score"].dropna()
        ac = float(luck.autocorr(lag=1)) if len(luck) >= 4 else np.nan
        records.append({"team": team, "autocorr_lag1": ac, "n_seasons": len(luck)})
    return (
        pd.DataFrame(records)
        .dropna(subset=["autocorr_lag1"])
        .sort_values("autocorr_lag1", ascending=False)
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# 6. Key findings helpers
# ---------------------------------------------------------------------------

def print_findings(summary: pd.DataFrame, autocorr: pd.DataFrame, df: pd.DataFrame, league_avg: float):
    sep = "-" * 60

    print(sep)
    print("DATASET")
    print(f"  Teams: {df['team'].nunique()}  |  Team-seasons: {len(df)}  |  Seasons: {sorted(df['season'].unique())}")
    print(f"  League avg goals per shot on target: {league_avg:.3f}")

    print(sep)
    print("TOP 10 LUCKIEST TEAMS (mean composite luck score)")
    print(summary[["team", "n_seasons", "mean_luck", "ci_lower", "ci_upper", "significant"]].head(10).to_string(index=False))

    print(sep)
    print("TOP 10 UNLUCKIEST TEAMS")
    print(summary[["team", "n_seasons", "mean_luck", "ci_lower", "ci_upper", "significant"]].tail(10).sort_values("mean_luck").to_string(index=False))

    print(sep)
    print("ATTACK LUCK — best finishers vs xG")
    atk = summary.nlargest(5, "mean_attack_luck")[["team", "mean_attack_luck", "n_seasons"]]
    print(atk.to_string(index=False))

    print(sep)
    print("DEFENSE LUCK — conceded least vs xGA")
    dfs = summary.nlargest(5, "mean_defense_luck")[["team", "mean_defense_luck", "n_seasons"]]
    print(dfs.to_string(index=False))

    print(sep)
    print("FINISHING LUCK — best shot conversion vs league average")
    fin = summary.nlargest(5, "mean_finish_luck")[["team", "mean_finish_luck", "n_seasons"]]
    print(fin.to_string(index=False))

    print(sep)
    print("STATISTICALLY SIGNIFICANT LUCK (BH-corrected p < 0.05)")
    sig = summary[summary["significant"] == True][["team", "mean_luck", "t_stat", "p_adj", "n_seasons"]]
    print(sig.to_string(index=False) if len(sig) else "  None found (expected — small sample per team)")

    print(sep)
    print("YEAR-TO-YEAR LUCK PERSISTENCE (lag-1 autocorrelation)")
    print(f"  Mean autocorr across all teams: {autocorr['autocorr_lag1'].mean():.3f}")
    print(f"  (Near 0 = luck is noise; consistently > 0 = structural advantage)")
    print(autocorr.head(5).to_string(index=False))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    df = load_clean()
    df, league_avg_g_per_sot = add_luck_columns(df)
    summary = build_summary(df)
    autocorr = compute_autocorrelation(df)

    print_findings(summary, autocorr, df, league_avg_g_per_sot)

    # Save outputs
    summary.to_csv("data/combined/luck_summary.csv", index=False)
    df.to_csv("data/combined/master_with_luck.csv", index=False)
    print("\nSaved → data/combined/luck_summary.csv")
    print("Saved → data/combined/master_with_luck.csv")
