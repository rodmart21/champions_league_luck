# champions_league_luck

Analysis of luck in UEFA Champions League results using expected goals (xG) and goalkeeper performance models.

## Master dataset columns (`data/combined/master.csv`)

One row per team per season. Produced by `src/clean_and_merge.py` by joining the xG stats source (`combined_expected.json`) with the shooting detail source (`combined_all.json`).

### Key columns

| Column | Source | Description |
|---|---|---|
| `team` | both | Standardised team name |
| `season` | both | Season in `YY-YY` format (e.g. `23-24`) |

### From the xG / results source (`combined_expected.json`)

| Column | Description |
|---|---|
| `mp` | Matches played in the competition that season |
| `xg` | Expected goals scored per match (xG) |
| `xga` | Expected goals conceded per match (xGA) |
| `xgd` | Expected goal difference per match (`xg - xga`) |
| `goals_scored` | Actual goals scored per match |
| `goals_conceded` | Actual goals conceded per match |
| `xg_vs_actual` | Difference between actual goals scored and xG (`goals_scored - xg`) |

### From the shooting detail source (`combined_all.json`)

| Column | Description |
|---|---|
| `country` | Two-letter country code of the team |
| `squad_size` | Number of players used during the season |
| `90s_played` | Total 90-minute equivalents played |
| `gls` | Total goals scored |
| `shots` | Total shots attempted |
| `shots_on_target` | Total shots on target |
| `sot_pct` | Shot-on-target percentage (`shots_on_target / shots × 100`) |
| `sh_per_90` | Shots per 90 minutes |
| `sot_per_90` | Shots on target per 90 minutes |
| `g_per_sh` | Goals per shot (shooting accuracy) |
| `g_per_sot` | Goals per shot on target (conversion rate) |
| `pk` | Penalty kicks scored |
| `pk_att` | Penalty kicks attempted |