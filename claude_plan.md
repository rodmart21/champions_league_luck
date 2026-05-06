# UCL Luck Analysis Project

## Context
Build a data analysis project to quantify "luck" in the UEFA Champions League — measuring which teams consistently overperform their xG, benefit from opponent GK mistakes, or concede fewer goals than expected. Analysis spans 7 seasons (2017-18 through 2023-24) using free public data from FBref.com.

---

## Data Source
**FBref.com** via the `soccerdata` Python library (free, open source, auto-caches scraped HTML so FBref is only hit once).

> **Known issue**: `soccerdata` doesn't include the Champions League by default. Must create `~/.soccerdata/config/league_dict.json` with:
> ```json
> {"INT-UCL": {"FBref": "Champions League", "season_start": "Jul", "season_end": "Jun"}}
> ```
> Fallback if this proves fragile: `pandas.read_html()` directly on `https://fbref.com/en/comps/8/{season}-Champions-League-Stats`

---

## Project Structure
```
just_testing/
├── data/
│   ├── raw/                        # soccerdata cache (auto-populated)
│   └── processed/ucl_master.parquet  # cleaned, merged master table
├── figures/                         # exported PNGs
├── src/
│   ├── data_collection.py           # FBref scraping functions
│   ├── luck_metrics.py              # metric calculations
│   └── viz.py                       # plotting helpers
└── testing.ipynb                    # main analysis notebook
```

---

## Data Collection (4 FBref tables)

| Table | `stat_type` | Key columns |
|---|---|---|
| Team shooting stats | `"shooting"` | `Gls`, `xG`, `G-xG` |
| Keeper basic stats | `"keeper"` | `GA`, `SoTA`, `Saves`, `Save%` |
| Keeper advanced stats | `"keeper_adv"` | `PSxG`, `PSxG+/-` (post-shot xG — the cleanest GK luck metric) |
| Match schedule | `"schedule"` | `GF`, `GA`, `xG`, `xGA` per match |

---

## Luck Metrics

Three components, then one composite:

```
AttackLuck   = (Goals Scored - xG) / xG          # finishing over/under-performance
DefenseLuck  = (xGA - Goals Conceded) / xGA       # defensive over/under-performance
GKLuck       = PSxG+/- per 90                     # goalkeeper saves above/below post-shot expected
LuckScore    = (AttackLuck + DefenseLuck + GKLuck) / 3
```

Master table: one row per `(team, season)`, aggregated to team-level means/std across seasons.

---

## Statistical Testing

1. **Bootstrap 95% CI** per team (10k iterations, seed=42) — handles small samples (4-7 seasons per team)
2. **One-sample t-test** (mean luck ≠ 0) with Benjamini-Hochberg FDR correction
3. **Lag-1 autocorrelation** — does last year's luck predict next year's? (near-zero = noise, positive = structural)
4. **ICC** across teams — measures overall season-to-season repeatability
5. **OLS regression**: `LuckScore ~ xG_volume + possession + games_played + deep_run_dummy`

---

## Visualizations (6 figures)

1. **Luck Leaderboard** — horizontal bar chart, teams ranked by mean luck score ± 95% CI
2. **Component Breakdown** — stacked bar showing AttackLuck vs DefenseLuck vs GKLuck for top 15 teams
3. **Season Heatmap** — teams × seasons matrix, diverging color scale (white = zero luck)
4. **Luck vs. UCL Progress** — scatter: luck score vs. round reached, with regression line
5. **xG Overperformance Time Series** — 6 most interesting teams, goals/xG ratio by season
6. **PSxG+/- Distribution** — violin plot of all team-seasons, individual teams highlighted

---

## Notebook Sections

0. Setup & environment (`pip install`, constants, league config)
1. Data collection (4 FBref scrapes — first run ~20 min, cached thereafter)
2. Cleaning & merging (standardize team names, filter 3+ seasons, export `.parquet`)
3. Luck metric calculation (component scores → composite)
4. Statistical testing (bootstrap, t-tests, autocorrelation)
5. Rankings & summary table (headline findings)
6. Visualizations (all 6 figures inline)
7. Robustness checks (vary component weights, re-rank teams)
8. Conclusions & caveats (sample size, survivorship bias, PSxG = skill + luck)

---

## Known Limitations to Address in Notebook
- **Sample size**: 4-7 seasons per team → low statistical power, wide CIs. Be explicit.
- **Survivorship bias**: only qualified teams included; bad teams don't appear in UCL data.
- **PSxG conflates luck and skill**: persistent GK outperformance may be Alisson/Courtois quality, not luck.
- **xG model quality**: StatsBomb's UCL model may be less calibrated than domestic league models.

---

## Tech Stack
```
soccerdata, pandas, numpy, scipy, scikit-learn, matplotlib, seaborn, plotly, pingouin, tqdm
```

---

## Verification
1. Run Section 1 — confirm all 4 DataFrames scrape without errors and row counts are plausible
2. Run Section 2 — verify `ucl_master.parquet` has reasonable shape (~200-250 team-season rows)
3. Spot-check known facts: Real Madrid 2021-22 should show high luck score (legendary GK performances by Courtois)
4. Confirm bootstrap is deterministic: re-run Section 4 twice and verify identical CI values
5. Robustness check: verify top 5 ranking doesn't flip completely under alternative weights
