"""
luck_metrics.py
---------------
Computes the three luck components and the composite luck score
from the cleaned master DataFrame.

Luck components:
  AttackLuck  = (Goals Scored - xG) / xG
                Positive = team scored more than model expected (lucky/clinical)

  DefenseLuck = (xGA - Goals Conceded) / xGA
                Positive = team conceded fewer goals than model expected (lucky/solid GK)

  GKLuck      = PSxG+/- per 90 minutes
                Positive = goalkeeper saved more than post-shot model expected

  LuckScore   = mean(AttackLuck, DefenseLuck, GKLuck)  [equal weights by default]

Statistical tests:
  - Bootstrap 95% CI per team (10k iterations, reproducible seed)
  - One-sample t-test per team (H0: mean luck == 0), with BH FDR correction
  - Lag-1 autocorrelation per team (does this season's luck predict next?)
  - ICC across all teams (is luck repeatable year-over-year?)
"""

import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------------
# Component calculations
# ---------------------------------------------------------------------------

def compute_attack_luck(df: pd.DataFrame) -> pd.Series:
    """
    (Goals Scored - xG) / xG
    Requires columns: goals_scored, xg
    Returns NaN where xG == 0 (avoid division by zero).
    """
    return (df["goals_scored"] - df["xg"]) / df["xg"].replace(0, np.nan)


def compute_defense_luck(df: pd.DataFrame) -> pd.Series:
    """
    (xGA - Goals Conceded) / xGA
    Requires columns: goals_conceded, xga
    """
    return (df["xga"] - df["goals_conceded"]) / df["xga"].replace(0, np.nan)


def compute_gk_luck(df: pd.DataFrame) -> pd.Series:
    """
    PSxG+/- per 90 minutes.
    Requires columns: psxg_pm (PSxG+/-), minutes_played.
    Returns raw PSxG+/- per 90; higher = GK outperformed model.
    """
    mins = df["minutes_played"].replace(0, np.nan)
    return df["psxg_pm"] / (mins / 90)


def compute_luck_score(
    df: pd.DataFrame,
    w_attack: float = 1 / 3,
    w_defense: float = 1 / 3,
    w_gk: float = 1 / 3,
) -> pd.Series:
    """
    Weighted composite luck score.
    Weights default to equal (1/3 each).
    """
    return (
        w_attack * df["attack_luck"]
        + w_defense * df["defense_luck"]
        + w_gk * df["gk_luck"]
    )


def add_luck_columns(df: pd.DataFrame, **weight_kwargs) -> pd.DataFrame:
    """
    In-place adds attack_luck, defense_luck, gk_luck, luck_score columns.
    Returns the modified DataFrame.
    """
    df = df.copy()
    df["attack_luck"] = compute_attack_luck(df)
    df["defense_luck"] = compute_defense_luck(df)
    df["gk_luck"] = compute_gk_luck(df)
    df["luck_score"] = compute_luck_score(df, **weight_kwargs)
    return df


# ---------------------------------------------------------------------------
# Team-level aggregation
# ---------------------------------------------------------------------------

def aggregate_by_team(df: pd.DataFrame, min_seasons: int = 3) -> pd.DataFrame:
    """
    Summarise per-(team, season) rows into one row per team.
    Only includes teams with at least `min_seasons` seasons of data.

    Returns a DataFrame with columns:
      team, n_seasons, mean_luck, std_luck,
      mean_attack_luck, mean_defense_luck, mean_gk_luck,
      best_season, worst_season
    """
    grouped = df.groupby("team")

    records = []
    for team, group in grouped:
        n = len(group)
        if n < min_seasons:
            continue
        luck = group["luck_score"].dropna()
        records.append({
            "team": team,
            "n_seasons": n,
            "mean_luck": luck.mean(),
            "std_luck": luck.std(),
            "mean_attack_luck": group["attack_luck"].mean(),
            "mean_defense_luck": group["defense_luck"].mean(),
            "mean_gk_luck": group["gk_luck"].mean(),
            "best_season": group.loc[group["luck_score"].idxmax(), "season"] if luck.any() else None,
            "worst_season": group.loc[group["luck_score"].idxmin(), "season"] if luck.any() else None,
        })

    summary = pd.DataFrame(records).sort_values("mean_luck", ascending=False).reset_index(drop=True)
    return summary


# ---------------------------------------------------------------------------
# Bootstrap confidence intervals
# ---------------------------------------------------------------------------

def bootstrap_ci(
    values: np.ndarray,
    n_boot: int = 10_000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    """
    Returns (lower, upper) bootstrap confidence interval for the mean.
    Uses a fixed seed for full reproducibility.
    """
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) < 2:
        return (np.nan, np.nan)
    boot_means = np.array([
        rng.choice(values, size=len(values), replace=True).mean()
        for _ in range(n_boot)
    ])
    alpha = 1 - ci
    return (
        float(np.percentile(boot_means, 100 * alpha / 2)),
        float(np.percentile(boot_means, 100 * (1 - alpha / 2))),
    )


def add_bootstrap_cis(summary: pd.DataFrame, master: pd.DataFrame, **bootstrap_kwargs) -> pd.DataFrame:
    """
    Adds ci_lower and ci_upper columns to the team summary DataFrame.
    Looks up per-season luck scores from `master`.
    """
    cis = []
    for team in summary["team"]:
        scores = master.loc[master["team"] == team, "luck_score"].dropna().values
        lo, hi = bootstrap_ci(scores, **bootstrap_kwargs)
        cis.append({"team": team, "ci_lower": lo, "ci_upper": hi})
    ci_df = pd.DataFrame(cis)
    return summary.merge(ci_df, on="team", how="left")


# ---------------------------------------------------------------------------
# Statistical tests
# ---------------------------------------------------------------------------

def run_ttests(summary: pd.DataFrame, master: pd.DataFrame) -> pd.DataFrame:
    """
    One-sample t-test per team (H0: mean luck == 0).
    Applies Benjamini-Hochberg FDR correction.
    Returns summary with added columns: t_stat, p_raw, p_adj, significant.
    """
    t_stats, p_vals = [], []

    for team in summary["team"]:
        scores = master.loc[master["team"] == team, "luck_score"].dropna().values
        if len(scores) < 3:
            t_stats.append(np.nan)
            p_vals.append(np.nan)
            continue
        t, p = stats.ttest_1samp(scores, popmean=0)
        t_stats.append(float(t))
        p_vals.append(float(p))

    summary = summary.copy()
    summary["t_stat"] = t_stats
    summary["p_raw"] = p_vals

    # BH correction
    valid_mask = ~np.isnan(summary["p_raw"])
    p_adj = np.full(len(summary), np.nan)
    if valid_mask.sum() > 0:
        from statsmodels.stats.multitest import multipletests
        _, p_corrected, _, _ = multipletests(
            summary.loc[valid_mask, "p_raw"].values, method="fdr_bh"
        )
        p_adj[valid_mask] = p_corrected

    summary["p_adj"] = p_adj
    summary["significant"] = summary["p_adj"] < 0.05
    return summary


def compute_autocorrelation(master: pd.DataFrame) -> pd.DataFrame:
    """
    Lag-1 autocorrelation of luck_score per team.
    Near-zero → luck is year-specific noise.
    Consistently positive → something structural (style, stadium, manager).
    Returns a DataFrame with columns: team, autocorr_lag1, n_seasons.
    """
    records = []
    for team, group in master.groupby("team"):
        group = group.sort_values("season")
        scores = group["luck_score"].dropna()
        if len(scores) >= 4:
            ac = float(scores.autocorr(lag=1))
        else:
            ac = np.nan
        records.append({"team": team, "autocorr_lag1": ac, "n_seasons": len(scores)})
    return pd.DataFrame(records).sort_values("autocorr_lag1", ascending=False)


def compute_icc(master: pd.DataFrame) -> float:
    """
    Intra-class correlation (ICC) of luck_score across all team-seasons.
    Uses a one-way random-effects model.
    ICC > 0.3 suggests meaningful repeatability across seasons.

    Requires pingouin; falls back to a manual calculation if unavailable.
    """
    try:
        import pingouin as pg
        data = master[["team", "luck_score"]].dropna()
        result = pg.intraclass_corr(data=data, targets="team", raters="luck_score")
        return float(result.loc[result["Type"] == "ICC1", "ICC"].values[0])
    except ImportError:
        # Manual one-way ANOVA-based ICC
        data = master[["team", "luck_score"]].dropna()
        groups = [g["luck_score"].values for _, g in data.groupby("team") if len(g) >= 2]
        grand_mean = data["luck_score"].mean()
        n_groups = len(groups)
        total_n = sum(len(g) for g in groups)

        ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups)
        ss_within = sum(((g - g.mean()) ** 2).sum() for g in groups)

        df_between = n_groups - 1
        df_within = total_n - n_groups

        ms_between = ss_between / df_between if df_between > 0 else 0
        ms_within = ss_within / df_within if df_within > 0 else 0

        n_avg = (total_n - sum(len(g) ** 2 for g in groups) / total_n) / df_between
        icc = (ms_between - ms_within) / (ms_between + (n_avg - 1) * ms_within)
        return float(icc)
