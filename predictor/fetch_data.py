"""Download NBA results and turn them into per-quarter line scores.

stats.nba.com is often blocked from cloud machines, so this uses two mirrors:
- 2020-21 to 2025-26: cdn.nba.com play-by-play archived at github.com/shufinskiy/nba_data;
  each team's points per quarter come from the running score.
- The current season: ESPN's schedule with quarter line scores, published daily by
  sportsdataverse (github.com/sportsdataverse/sportsdataverse-data releases).
  The same file lists the games still to be played.

Usage:  python fetch_data.py [first_season] [last_season]
        (season = starting year, e.g. 2026 means 2026-27; default 2020-2026)
Output: data/line_scores.csv  (one row per finished regular-season game)
        data/schedule.csv     (upcoming games of the current season)
"""
import ast
import sys
import tarfile
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).parent
RAW = ROOT / "data" / "raw"
URL = "https://raw.githubusercontent.com/shufinskiy/nba_data/main/datasets/cdnnba_{y}.tar.xz"
MIRROR_LAST_SEASON = 2025  # last season the cdn.nba.com mirror covers
ESPN_URL = ("https://github.com/sportsdataverse/sportsdataverse-data/releases/download/"
            "espn_nba_schedules/nba_schedule_{y}.parquet")  # y = year the season ends
ESPN_TO_NBA = {"GS": "GSW", "NO": "NOP", "NY": "NYK", "SA": "SAS", "UTAH": "UTA", "WSH": "WAS"}
COLS = ["gameId", "period", "timeActual", "scoreHome", "scoreAway", "teamTricode", "actionType"]


def download(season: int) -> Path:
    RAW.mkdir(parents=True, exist_ok=True)
    path = RAW / f"cdnnba_{season}.tar.xz"
    if not path.exists():
        print(f"downloading {season}-{(season + 1) % 100:02d} ...")
        r = requests.get(URL.format(y=season), timeout=300)
        r.raise_for_status()
        path.write_bytes(r.content)
    return path


def read_pbp(path: Path) -> pd.DataFrame:
    with tarfile.open(path, "r:xz") as tar:
        member = next(m for m in tar.getmembers() if m.name.endswith(".csv"))
        return pd.read_csv(tar.extractfile(member), usecols=COLS, low_memory=False)


def line_scores(pbp: pd.DataFrame, season: int) -> pd.DataFrame:
    pbp = pbp.dropna(subset=["gameId"]).copy()
    pbp["scoreHome"] = pd.to_numeric(pbp["scoreHome"], errors="coerce")
    pbp["scoreAway"] = pd.to_numeric(pbp["scoreAway"], errors="coerce")
    pbp = pbp.dropna(subset=["scoreHome", "scoreAway"])

    # Which tricode is home: the team credited on plays where the home score rises.
    g = pbp.groupby("gameId", sort=False)
    pbp["dh"] = g["scoreHome"].diff().fillna(pbp["scoreHome"])
    pbp["da"] = g["scoreAway"].diff().fillna(pbp["scoreAway"])
    scored = pbp[pbp["teamTricode"].notna()]
    home = scored[scored["dh"] > 0].groupby("gameId")["teamTricode"].agg(lambda s: s.mode().iat[0])
    away = scored[scored["da"] > 0].groupby("gameId")["teamTricode"].agg(lambda s: s.mode().iat[0])

    # Cumulative score at the end of each period -> points per period.
    end = pbp.groupby(["gameId", "period"])[["scoreHome", "scoreAway"]].max().reset_index()
    end = end.sort_values(["gameId", "period"])
    end[["ph", "pa"]] = end.groupby("gameId")[["scoreHome", "scoreAway"]].diff().fillna(end[["scoreHome", "scoreAway"]])

    rows = []
    for gid, gdf in end.groupby("gameId"):
        if gid not in home.index or gid not in away.index or gdf["period"].max() < 4:
            continue
        rec = {"game_id": int(gid), "home": home[gid], "away": away[gid]}
        for q in (1, 2, 3, 4):
            r = gdf[gdf["period"] == q]
            rec[f"home_q{q}"] = int(r["ph"].iat[0])
            rec[f"away_q{q}"] = int(r["pa"].iat[0])
        ot = gdf[gdf["period"] > 4]
        rec["home_ot"], rec["away_ot"] = int(ot["ph"].sum()), int(ot["pa"].sum())
        rec["home_final"], rec["away_final"] = int(gdf["scoreHome"].max()), int(gdf["scoreAway"].max())
        rows.append(rec)
    out = pd.DataFrame(rows)

    dates = pd.to_datetime(pbp.groupby("gameId")["timeActual"].min(), utc=True, format="ISO8601", errors="coerce")
    # Shift UTC tip-off back 6h so late West Coast games keep their US calendar date.
    out["date"] = out["game_id"].map((dates - pd.Timedelta(hours=6)).dt.date.astype(str))
    out["season"] = season
    # Regular season ids start with 2 (002...), play-in 5, playoffs 4; keep regular season.
    out = out[out["game_id"].astype(str).str.zfill(10).str[2] == "2"]
    out = out[out["home"] != out["away"]]
    return out


def espn_schedule(season: int) -> pd.DataFrame:
    """ESPN schedule for a season, downloaded fresh each run since it updates daily."""
    RAW.mkdir(parents=True, exist_ok=True)
    path = RAW / f"espn_schedule_{season}.parquet"
    r = requests.get(ESPN_URL.format(y=season + 1), timeout=300)
    r.raise_for_status()
    path.write_bytes(r.content)
    s = pd.read_parquet(path)
    s = s[s["season_type"] == 2].copy()  # regular season
    for side in ("home", "away"):
        s[side] = s[f"{side}_abbreviation"].replace(ESPN_TO_NBA)
    return s


def _periods(cell) -> dict:
    """ESPN line scores arrive as text like "[{'value': 35.0, 'period': 1.0} ...]"."""
    if not isinstance(cell, str) or not cell.strip("[] "):
        return {}
    items = ast.literal_eval(cell.replace("}\n", "},").replace("} {", "}, {"))
    return {int(i["period"]): int(i["value"]) for i in items}


def espn_line_scores(s: pd.DataFrame, season: int) -> pd.DataFrame:
    rows = []
    for _, g in s[s["status_type_completed"]].iterrows():
        h, a = _periods(g["home_linescores"]), _periods(g["away_linescores"])
        if not all(q in h and q in a for q in (1, 2, 3, 4)):
            continue
        rec = {"game_id": int(g["game_id"]), "home": g["home"], "away": g["away"],
               "date": str(g["game_date"])[:10], "season": season}
        for q in (1, 2, 3, 4):
            rec[f"home_q{q}"], rec[f"away_q{q}"] = h[q], a[q]
        rec["home_ot"] = sum(v for k, v in h.items() if k > 4)
        rec["away_ot"] = sum(v for k, v in a.items() if k > 4)
        rec["home_final"], rec["away_final"] = int(g["home_score"]), int(g["away_score"])
        # Skip games whose quarters don't add up to the final score.
        if sum(h.values()) == rec["home_final"] and sum(a.values()) == rec["away_final"]:
            rows.append(rec)
    return pd.DataFrame(rows)


def upcoming(s: pd.DataFrame) -> pd.DataFrame:
    u = s[~s["status_type_completed"] & s["home"].ne("TBD") & s["away"].ne("TBD")]
    u = u.assign(date=u["game_date"].astype(str).str[:10], tipoff_utc=u["game_date_time"].astype(str))
    return u[["date", "tipoff_utc", "home", "away"]].sort_values(["tipoff_utc", "home"])


def main():
    first = int(sys.argv[1]) if len(sys.argv) > 1 else 2020
    last = int(sys.argv[2]) if len(sys.argv) > 2 else 2026
    frames = []
    for season in range(first, last + 1):
        if season <= MIRROR_LAST_SEASON:
            ls = line_scores(read_pbp(download(season)), season)
        else:
            sched = espn_schedule(season)
            ls = espn_line_scores(sched, season)
            if season == last:
                up = upcoming(sched)
                up.to_csv(ROOT / "data" / "schedule.csv", index=False)
                print(f"wrote {len(up)} upcoming games to data/schedule.csv")
        print(f"{season}-{(season + 1) % 100:02d}: {len(ls)} games")
        frames.append(ls)
    df = pd.concat(frames).sort_values(["date", "game_id"]).reset_index(drop=True)
    cols = ["season", "date", "game_id", "home", "away"] + \
        [f"{s}_q{q}" for q in (1, 2, 3, 4) for s in ("home", "away")] + \
        ["home_ot", "away_ot", "home_final", "away_final"]
    df[cols].to_csv(ROOT / "data" / "line_scores.csv", index=False)
    print(f"wrote {len(df)} games to data/line_scores.csv")


if __name__ == "__main__":
    main()
