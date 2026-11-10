"""Turn per-game line scores into pre-game team-form features.

Each team's form is an exponentially weighted average of its recent games
(points scored and allowed in each quarter, overall margin, wins), computed
only from games played before the one being predicted.
"""
import numpy as np
import pandas as pd

QUARTERS = (1, 2, 3, 4)
SPANS = (8, 30)  # short-term form and longer-term strength, in games
STATS = [f"pts_q{q}" for q in QUARTERS] + [f"opp_q{q}" for q in QUARTERS] + ["margin", "win"]
MIN_PRIOR_GAMES = 10
BLEND_K = 5  # games into a season at which this season's average counts as much as long-term form


def team_games(ls: pd.DataFrame) -> pd.DataFrame:
    """One row per team per game."""
    sides = []
    for side, other in (("home", "away"), ("away", "home")):
        t = pd.DataFrame({
            "game_id": ls["game_id"], "date": pd.to_datetime(ls["date"]), "season": ls["season"],
            "team": ls[side], "is_home": int(side == "home"),
        })
        for q in QUARTERS:
            t[f"pts_q{q}"] = ls[f"{side}_q{q}"]
            t[f"opp_q{q}"] = ls[f"{other}_q{q}"]
        t["margin"] = ls[f"{side}_final"] - ls[f"{other}_final"]
        t["win"] = (t["margin"] > 0).astype(int)
        sides.append(t)
    return pd.concat(sides).sort_values(["team", "date", "game_id"]).reset_index(drop=True)


def _state_after_each_game(tg: pd.DataFrame) -> pd.DataFrame:
    """Team form including the game on that row."""
    out = tg[["game_id", "date", "season", "team"]].copy()
    g = tg.groupby("team")
    for span in SPANS:
        ew = g[STATS].transform(lambda s: s.ewm(span=span, min_periods=1).mean())
        for c in STATS:
            out[f"{c}_e{span}"] = ew[c]
    # This-season average blended with long-term form: early in a season the long-term
    # numbers dominate, and after a few weeks the current season takes over.
    gs = tg.groupby(["team", "season"])
    w = (gs.cumcount() + 1) / (gs.cumcount() + 1 + BLEND_K)
    for c in STATS:
        season_avg = gs[c].transform(lambda s: s.expanding().mean())
        out[f"{c}_bl"] = w * season_avg + (1 - w) * out[f"{c}_e{max(SPANS)}"]
    out["n_games"] = g.cumcount() + 1
    return out


def form_columns():
    return [f"{c}_e{s}" for s in SPANS for c in STATS] + [f"{c}_bl" for c in STATS]


def pregame_features(ls: pd.DataFrame) -> pd.DataFrame:
    """Feature row per game, built only from earlier games."""
    tg = team_games(ls)
    after = _state_after_each_game(tg)
    cols = form_columns() + ["n_games"]
    before = after.groupby("team")[cols].shift(1)
    before["team"], before["game_id"] = after["team"], after["game_id"]
    before["rest"] = tg.groupby("team")["date"].diff().dt.days.clip(upper=4).fillna(4)

    games = ls[["season", "date", "game_id", "home", "away"]].copy()
    for side in ("home", "away"):
        f = before.rename(columns={c: f"{side}_{c}" for c in cols + ["rest"]})
        games = games.merge(f.rename(columns={"team": side}), on=[side, "game_id"], how="left")
    games = games[(games["home_n_games"] >= MIN_PRIOR_GAMES) & (games["away_n_games"] >= MIN_PRIOR_GAMES)]
    return add_matchup_columns(games.drop(columns=["home_n_games", "away_n_games"]))


def add_matchup_columns(games: pd.DataFrame) -> pd.DataFrame:
    """Expected quarter points for each side from offense vs. opposing defense."""
    games = games.copy()
    for s in SPANS:
        for q in QUARTERS:
            games[f"exp_home_q{q}_e{s}"] = (games[f"home_pts_q{q}_e{s}"] + games[f"away_opp_q{q}_e{s}"]) / 2
            games[f"exp_away_q{q}_e{s}"] = (games[f"away_pts_q{q}_e{s}"] + games[f"home_opp_q{q}_e{s}"]) / 2
    for side in ("home", "away"):
        games[f"{side}_b2b"] = (games[f"{side}_rest"] <= 1).astype(int)
    return games


def latest_team_form(ls: pd.DataFrame) -> pd.DataFrame:
    """Each team's form after its most recent game (used for new predictions)."""
    after = _state_after_each_game(team_games(ls))
    return after.groupby("team").tail(1).set_index("team")


def matchup_row(form: pd.DataFrame, home: str, away: str, home_rest=2, away_rest=2) -> pd.DataFrame:
    row = {"home": home, "away": away, "home_rest": home_rest, "away_rest": away_rest}
    for side, team in (("home", home), ("away", away)):
        for c in form_columns():
            row[f"{side}_{c}"] = form.at[team, c]
    return add_matchup_columns(pd.DataFrame([row]))


def feature_columns(games: pd.DataFrame):
    skip = {"season", "date", "game_id", "home", "away"}
    return [c for c in games.columns if c not in skip and not c.startswith("y_")]


def targets(ls: pd.DataFrame) -> pd.DataFrame:
    t = ls[["game_id"]].copy()
    for q in QUARTERS:
        h, a = ls[f"home_q{q}"], ls[f"away_q{q}"]
        t[f"y_spread_q{q}"] = h - a
        t[f"y_total_q{q}"] = h + a
        t[f"y_win_q{q}"] = np.sign(h - a)  # 1 home, -1 away, 0 tied quarter
    return t
