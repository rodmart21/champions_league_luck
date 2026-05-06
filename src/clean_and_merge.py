"""
clean_and_merge.py
------------------
Standardizes team names across the two data sources and merges them
into a single master DataFrame with one row per (team, season).

Sources:
  - combined_expected.json  → xG / xGA / xGD / GF / GA stats
  - combined_all.json       → shooting detail (Sh, SoT, PSxG, etc.)
"""

import json
import pandas as pd
from pathlib import Path

# ---------------------------------------------------------------------------
# Team name mapping: expected source → canonical name
# ---------------------------------------------------------------------------
EXPECTED_TO_CANONICAL = {
    "AFC Ajax": "Ajax",
    "AC Milan": "Milan",
    "AEK Athens FC": "AEK Athens",
    "Arsenal FC": "Arsenal",
    "AS Monaco FC": "Monaco",
    "AS Roma": "Roma",
    "Atalanta Bergamasca Calcio": "Atalanta",
    "Aston Villa FC": "Aston Villa",
    "Bayer 04 Leverkusen": "Leverkusen",
    "BVB 09 Borussia Dortmund": "Dortmund",
    "Borussia VfL Mönchengladbach": "Gladbach",
    "BSC Young Boys": "Young Boys",
    "Celtic FC": "Celtic",
    "Chelsea FC": "Chelsea",
    "Club Atlético de Madrid": "Atlético Madrid",
    "Club Brugge KV": "Club Brugge",
    "FC Barcelona": "Barcelona",
    "FC Basel 1893": "Basel",
    "FC Bayern München": "Bayern Munich",
    "FC Dynamo Kyiv": "Dynamo Kyiv",
    "FC Internazionale Milano": "Inter",
    "FC Kairat Almaty": "FC Kairat",
    "FC København": "FC Copenhagen",
    "FC Midtjylland": "Midtjylland",
    "FC Porto": "Porto",
    "FC Salzburg": "RB Salzburg",
    "FC Schalke 04": "Schalke 04",
    "FC Shakhtar Donetsk": "Shakhtar Donetsk",
    "FC Viktoria Plzeň": "Viktoria Plzeň",
    "Fenerbahçe": "Fenerbahçe",
    "Ferencvárosi TC": "Ferencváros",
    "Feyenoord Rotterdam": "Feyenoord",
    "FK Bodo - Glimt": "Bodø/Glimt",
    "FK Krasnodar": "Krasnodar",
    "FK Lokomotiv Moskva": "Loko Moscow",
    "FK Sheriff Tiraspol": "FC Sheriff Tiraspol",
    "FK Spartak Moskva": "Spartak Moscow",
    "FK Zenit St. Petersburg": "Zenit",
    "GNK Dinamo Zagreb": "Dinamo Zagreb",
    "İstanbul Başakşehir FK": "Başakşehir",
    "Juventus FC": "Juventus",
    "KAA Gent": "Gent",
    "KKS Lech Poznań": "Lech Poznań",
    "KP Legia Warszawa": "Legia Warsaw",
    "KRC Genk": "Genk",
    "LASK Linz": "LASK",
    "Lille OSC Métropole": "Lille",
    "Liverpool FC": "Liverpool",
    "Maccabi Haifa FC": "Maccabi Haifa",
    "Malmö FF": "Malmö",
    "Manchester City FC": "Manchester City",
    "Manchester United FC": "Manchester Utd",
    "NK Celje": "Celje",
    "OGC Nice Côte d'Azur": "Nice",
    "Olympiakos CFP": "Olympiacos",
    "Olympique Lyonnais": "Lyon",
    "Olympique de Marseille": "Marseille",
    "PAOK Thessaloniki FC": "PAOK",
    "Paris Saint-Germain FC": "Paris Saint-Germain",
    "PFC Ludogorets 1945 Razgrad": "Ludogorets",
    "PFK CSKA Moskva": "CSKA Moscow",
    "PSV Eindhoven": "PSV",
    "Qarabağ Ağdam FK": "Qarabağ",
    "Rasen Ballsport Leipzig": "RB Leipzig",
    "Rangers FC": "Rangers",
    "Real Madrid CF": "Real Madrid",
    "Real Sociedad de Fútbol": "Real Sociedad",
    "Red Star Belgrade": "Red Star",
    "Royal Antwerp FC": "Antwerp",
    "Royal Standard de Liège": "Standard Liège",
    "Royal Union Saint-Gilloise": "Union SG",
    "RSC Anderlecht": "Anderlecht",
    "SC Dnipro-1": "Dnipro-1",
    "SC Fotbal Club FCSB SA": "FCSB",
    "SCS CFR 1907 Cluj": "CFR Cluj",
    "SK Rapid Wien": "Rapid Wien",
    "SK Slavia Praha": "Slavia Prague",
    "SK Sturm Graz": "Sturm Graz",
    "ŠK Slovan Bratislava": "Slovan Bratislava",
    "SL Benfica": "Benfica",
    "Sporting Braga": "Braga",
    "Sporting Clube de Portugal": "Sporting CP",
    "SS Lazio": "Lazio",
    "SSC Napoli": "Napoli",
    "Sevilla FC": "Sevilla",
    "Servette FC": "Servette",
    "Stade Brestois 29": "Brest",
    "Stade Rennais FC": "Rennes",
    "TSG 1899 Hoffenheim": "Hoffenheim",
    "Tottenham Hotspur FC": "Tottenham Hotspur",
    "1. FC Union Berlin": "Union Berlin",
    "AC Sparta Praha": "Sparta Prague",
    "Valencia CF": "Valencia",
    "VfB Stuttgart 1893": "Stuttgart",
    "VfL Wolfsburg": "Wolfsburg",
    "Villarreal CF": "Villarreal",
    "Newcastle United FC": "Newcastle United",
    "Molde FK": "Molde",
    "Rosenborg BK": "Rosenborg",
    "Maccabi Tel Aviv FC": "Maccabi Tel Aviv",
}

# ---------------------------------------------------------------------------
# Season normalisation: any format → "YY-YY"
# ---------------------------------------------------------------------------
def _norm_season(s: str) -> str:
    s = str(s).strip()
    if "/" in s:                      # "2017/18" → "17-18"
        parts = s.split("/")
        return parts[0][2:] + "-" + parts[1]
    return s


# ---------------------------------------------------------------------------
# Main cleaning + merge function
# ---------------------------------------------------------------------------
def load_and_merge(data_dir: str | Path = "data/combined") -> pd.DataFrame:
    data_dir = Path(data_dir)

    # --- load ---
    with open(data_dir / "combined_expected.json") as f:
        expected = pd.DataFrame(json.load(f))
    with open(data_dir / "combined_all.json") as f:
        raw = pd.DataFrame(json.load(f))

    # --- season ---
    expected["season"] = expected["season"].apply(_norm_season)
    raw["season"] = raw["season"].apply(_norm_season)

    # --- standardise team names ---
    expected["team"] = (
        expected["Team"]
        .str.strip()
        .map(lambda t: EXPECTED_TO_CANONICAL.get(t, t))
    )
    raw["team"] = raw["Squad"].str.strip()

    # --- drop duplicate/redundant columns before merge ---
    expected = expected.drop(columns=["Team", "Rank"], errors="ignore")
    raw = raw.drop(columns=["Squad", "12.0", "10.0"], errors="ignore")

    # rename expected columns to lowercase to avoid clashes
    expected = expected.rename(columns={
        "MP": "mp",
        "xG": "xg",
        "xGA": "xga",
        "xGD": "xgd",
        "GF": "goals_scored",
        "GA": "goals_conceded",
    })

    # rename raw columns to lowercase
    raw = raw.rename(columns={
        "Country": "country",
        "Pl": "squad_size",
        "90s": "90s_played",
        "Gls": "gls",
        "Sh": "shots",
        "SoT": "shots_on_target",
        "SoT_Pct": "sot_pct",
        "Sh_per_90": "sh_per_90",
        "SoT_per_90": "sot_per_90",
        "G_per_Sh": "g_per_sh",
        "G_per_SoT": "g_per_sot",
        "PK": "pk",
        "PKatt": "pk_att",
    })

    # --- merge on (team, season) ---
    master = pd.merge(expected, raw, on=["team", "season"], how="outer")

    # put key columns first
    front = ["team", "season"]
    rest = [c for c in master.columns if c not in front]
    master = master[front + rest].sort_values(["team", "season"]).reset_index(drop=True)

    return master


if __name__ == "__main__":
    master = load_and_merge()
    print(f"Master shape: {master.shape}")
    print(master.columns.tolist())
    print(master.head(5).to_string())

    out = Path("data/combined/master.csv")
    master.to_csv(out, index=False)
    print(f"\nSaved → {out}")
