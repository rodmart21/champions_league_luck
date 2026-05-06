# Football Shooting Statistics Dataset

Season-by-season squad shooting data scraped from FBref, covering the **2016/17 through 2025/26** seasons.

Each file (e.g. `16_17.json`) contains one record per squad for that season. The combined file at `data/combined/combined.json` merges all seasons into a single dataset with a `season` field added.

---

## Column Definitions

| Column | Full Name | Description |
|---|---|---|
| `Squad` | Squad | Club/team name (e.g. `"Arsenal"`) |
| `Country` | Country | Three-letter country code of the league the team plays in (e.g. `"eng"` for England) |
| `Pl` | Players Used | Number of distinct players who appeared in at least one match |
| `90s` | 90s Played | Total minutes played by all players divided by 90 — represents the cumulative "full matches" of playing time |
| `Gls` | Goals | Total goals scored (excluding own goals) |
| `Sh` | Shots | Total shots taken (excluding penalty kicks) |
| `SoT` | Shots on Target | Shots that were on frame (on goal), excluding blocked shots that never reached the keeper |
| `SoT_Pct` | Shots on Target % | Percentage of shots that were on target: `SoT / Sh × 100` |
| `Sh_per_90` | Shots per 90 | Average number of shots taken per 90 minutes of play |
| `SoT_per_90` | Shots on Target per 90 | Average number of shots on target per 90 minutes of play |
| `G_per_Sh` | Goals per Shot | Shooting efficiency: goals scored per shot taken |
| `G_per_SoT` | Goals per Shot on Target | Conversion rate: goals scored per shot on target (equivalent to save percentage from the keeper's perspective) |
| `PK` | Penalty Kicks Made | Penalty kicks successfully converted |
| `PKatt` | Penalty Kicks Attempted | Total penalty kicks taken |

---

## Notes

- All shooting stats **exclude own goals**.
- `Gls` includes goals from penalty kicks (`PK`). To get open-play + free-kick goals only, subtract `PK` from `Gls`.
- `SoT_Pct`, `G_per_Sh`, and `G_per_SoT` are rate stats and are more meaningful when `Sh` is large enough to be statistically stable.
