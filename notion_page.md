# 🍀 How Much of the Champions League is Just Luck?

> A data analysis of 9 UEFA Champions League seasons (2017–2026) across 83 teams to measure which clubs consistently over or underperform what the stats predict.

---

## The Question

Every football fan has seen it: a team dominates a Champions League game, hits the post three times, and loses 1–0 to a counter-attack goal. Or a goalkeeper makes five impossible saves to keep his team alive. Is that just noise — or do some teams *consistently* benefit from these moments more than others?

This project tries to answer that using data. Specifically: **can we measure luck, and if so, who has the most of it in the UCL?**

---

## What Is "Luck" in Football?

We define luck as the gap between what the stats *predicted* would happen and what *actually* happened.

Modern football uses a metric called **Expected Goals (xG)** — a number that estimates how likely a shot is to result in a goal based on its position, angle, and type. If a team scores 2 goals from chances that collectively had an xG of 1.0, they outperformed their model. Over one game that's normal variance. Over multiple seasons, it starts to tell a story.

We measure luck across three dimensions:

| Dimension | What it measures | Formula |
|---|---|---|
| **Attack Luck** | Did you score more goals than your xG predicted? | `(Goals Scored − xG) / xG` |
| **Defense Luck** | Did you concede fewer goals than your opponents' xG predicted? | `(xGA − Goals Conceded) / xGA` |
| **Finishing Luck** | Did you convert shots on target above the UCL average rate? | `(Your conversion rate − 28.5% average) / 28.5%` |

These three are averaged into a single **Luck Score** per team per season. Positive = lucky. Negative = unlucky.

---

## The Data

- **Source:** FBref.com — the most comprehensive free football stats database
- **Coverage:** 9 UCL seasons from 2017-18 to 2025-26
- **Size:** 83 unique teams, 269 team-season records after cleaning
- **Key fields used:** xG, xGA, goals scored, goals conceded, shots on target, goals per shot on target

---

## Results

### 🏆 The Luckiest Teams

| Team | Seasons | Luck Score | Where it comes from |
|---|---|---|---|
| **Arsenal** | 3 | +0.34 | Strong across all three dimensions |
| **Paris Saint-Germain** | 9 | +0.26 | Primarily attack luck — they finish way above xG |
| **Bayern Munich** | 9 | +0.20 | Consistent attack overperformance across 9 seasons |
| **Sporting CP** | 5 | +0.18 | Exceptional finishing — goals far above xG |
| **Ajax** | 5 | +0.15 | Balanced across all three dimensions |

> ⚠️ Arsenal only have 3 seasons in the dataset. A high score with few seasons should be treated with more caution than PSG or Bayern who have 9 seasons of evidence.

---

### 💔 The Unluckiest Teams

| Team | Seasons | Luck Score | Where it hurts most |
|---|---|---|---|
| **Monaco** | 3 | −0.44 | Heavily penalised on defense — opponents scored far above xGA |
| **Slavia Prague** | 2 | −0.41 | Terrible finishing luck — scored well below their shot quality |
| **Marseille** | 3 | −0.35 | Consistently unlucky in both attack and defense |
| **Feyenoord** | 2 | −0.32 | Extreme defense bad luck — opponents converted almost everything |
| **Leverkusen** | 3 | −0.30 | Consistently star-crossed despite being a strong domestic side |

---

### 📊 Where Does Each Team's Luck Come From?

Not all luck is the same. Breaking it down:

- **PSG's luck is almost entirely in attack** — they score significantly more goals than their xG predicts. This could be Mbappé / Neymar brilliance that xG models undervalue, or genuine luck.
- **Arsenal's luck is balanced** — they outscore xG *and* concede less than xGA *and* convert shots above average. All three dimensions positive.
- **Monaco's bad luck is almost entirely defensive** — opponents scored 64% more than their xGA against Monaco. That's historically bad.
- **Slavia Prague's bad luck is finishing** — their shot conversion rate was 61% below the UCL average.

---

### 📅 Is Luck Stable Over Time?

One of the most important questions: does this year's luck predict next year's?

**The short answer: mostly no.**

The average lag-1 autocorrelation across all teams is **−0.16**. This means that if a team is lucky this season, they are *slightly more likely* to be unlucky next season (regression to the mean). That's exactly what you'd expect from genuine randomness.

However, a handful of teams show persistent positive autocorrelation:

| Team | Year-to-Year Luck Persistence |
|---|---|
| Benfica | +0.47 |
| Barcelona | +0.40 |
| RB Leipzig | +0.29 |
| Manchester City | +0.29 |

For these teams, their "luck" tends to repeat. This suggests their overperformance might not be pure luck — it could reflect **structural advantages** that xG models don't fully capture: elite goalkeeper quality, pressing systems that force low-quality shots, or superior tactical discipline.

---

### 🔬 Is Any of This Statistically Significant?

Honestly — not yet. With only 2 to 9 seasons per team, we don't have enough data to say with high confidence that any team's luck is *truly* different from zero. The statistical tests (t-tests with false discovery rate correction) return no significant results.

This is expected. You'd need roughly 15–20 seasons per team to achieve reliable statistical significance. The analysis gives us strong signals and interesting patterns, but we should be careful not to over-interpret them.

---

## How It Was Built

### Step 1 — Data collection
Raw JSON files per season were collected from FBref for two types of stats: xG/goals data and shooting detail data. Each file covers one UCL season.

### Step 2 — Cleaning & standardisation
The two datasets used different team names (e.g. "FC Bayern München" vs "Bayern Munich"). A mapping of 100+ team names was built to standardise them, and both datasets were merged into a single master table with one row per team per season.

### Step 3 — Luck metric calculation
The three luck components were computed per team-season row, then averaged into a composite Luck Score. The "finishing luck" baseline (28.5%) is the average goals-per-shot-on-target across all 269 team-seasons in the dataset.

### Step 4 — Statistical analysis
- **Bootstrap confidence intervals** (10,000 resamples) to estimate uncertainty around each team's mean luck
- **T-tests** per team to test whether luck is significantly different from zero
- **Benjamini-Hochberg correction** to avoid false positives from running many tests simultaneously
- **Lag-1 autocorrelation** to test year-to-year persistence

### Step 5 — Visualisation
Seven figures were produced covering the leaderboard, component breakdown, season heatmap, luck over time, autocorrelation, attack vs defense scatter, and the overall distribution of luck scores.

---

## Key Takeaways

1. **Luck exists and is measurable** — the gap between xG and actual goals is real and varies significantly across teams
2. **Most luck is noise** — the negative average autocorrelation confirms it largely doesn't repeat year-to-year
3. **A few elite teams show persistent patterns** — Benfica, Barcelona, Man City show year-to-year consistency that hints at structural factors beyond randomness
4. **PSG and Bayern stand out** — with 9 seasons of data each, their positive luck scores are the most credible findings in the dataset
5. **Small clubs get crushed** — Monaco, Slavia Prague and Feyenoord show how UCL can be brutal for teams that reach the group stage but face opponents whose quality is simply better than their xGA can absorb

---

## What's Next

Several columns in the dataset haven't been used yet and could unlock deeper analysis:

- **Matches played as an outcome variable** — directly test whether luckier teams advance further
- **Penalty correction** — strip out penalty goals to get a cleaner picture of open-play luck
- **Country-level patterns** — do Premier League teams systematically overperform xG in UCL compared to, say, Ligue 1 teams?
- **Luck vs squad value** — does a higher transfer budget make you "luckier" (i.e. your quality exceeds what xG captures)?

---

*Data source: FBref.com | Analysis: Python (pandas, scipy, matplotlib, seaborn) | Seasons: 2017-18 → 2025-26*
