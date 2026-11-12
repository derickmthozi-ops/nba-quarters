"""Train per-quarter models and measure them on a held-out season.

For each quarter (Q1-Q4) three models:
  spread  - home points minus away points   (ridge regression)
  total   - home points plus away points    (ridge regression)
  winner  - probability the home team wins the quarter (logistic regression;
            tied quarters are left out of training)

Evaluation trains on every season before the last complete one and tests on
that full season. If the current season has started, it is also scored on its
games so far (trained on all earlier seasons). The saved models are then refit
on every game, including the current season.

Usage:  python train.py
Output: models/models.joblib, models/metrics.json
"""
import json
from pathlib import Path

import joblib
import numpy as np
from scipy.stats import norm
import pandas as pd
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from features import QUARTERS, feature_columns, latest_team_form, pregame_features, targets

ROOT = Path(__file__).parent
STRONG_GAME = 0.60     # full-game pick confidence that counts as a strong pick
STRONG_QUARTER = 0.65  # same for a single quarter


def fit_models(df: pd.DataFrame, X: list) -> dict:
    models = {}
    for q in QUARTERS:
        for kind in ("spread", "total"):
            y = df[f"y_{kind}_q{q}"]
            m = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-1, 4, 20))).fit(df[X], y)
            resid_sd = float(np.std(y - m.predict(df[X])))
            models[f"{kind}_q{q}"] = {"model": m, "resid_sd": resid_sd}
        y = df[f"y_win_q{q}"]
        keep = y != 0
        m = make_pipeline(StandardScaler(), LogisticRegression(C=0.01, max_iter=2000))
        models[f"win_q{q}"] = {"model": m.fit(df.loc[keep, X], (y[keep] == 1).astype(int))}
    game_pred = sum(models[f"spread_q{q}"]["model"].predict(df[X]) for q in QUARTERS)
    game_y = sum(df[f"y_spread_q{q}"] for q in QUARTERS)
    models["game_spread_resid_sd"] = float(np.std(game_y - game_pred))
    return models


def evaluate(models: dict, train: pd.DataFrame, test: pd.DataFrame, X: list) -> dict:
    out = {}
    pooled = {"right": 0, "strong": 0, "strong_right": 0, "decided": 0}
    for q in QUARTERS:
        r = {}
        for kind in ("spread", "total"):
            y = test[f"y_{kind}_q{q}"]
            pred = models[f"{kind}_q{q}"]["model"].predict(test[X])
            r[f"{kind}_mae"] = float(np.abs(y - pred).mean())
            r[f"{kind}_mae_baseline"] = float(np.abs(y - train[f"y_{kind}_q{q}"].mean()).mean())
        y = test[f"y_win_q{q}"]
        decided = y != 0
        p = models[f"win_q{q}"]["model"].predict_proba(test[X])[:, 1]
        pick = np.where(p >= 0.5, 1, -1)
        r["winner_accuracy"] = float((pick[decided] == y[decided]).mean())
        r["winner_accuracy_home_baseline"] = float((y[decided] == 1).mean())
        # Accuracy on the games where the model is most confident (top third by |p - 0.5|).
        conf = np.abs(p - 0.5)
        top = decided & (conf >= np.quantile(conf, 2 / 3))
        r["winner_accuracy_confident_third"] = float((pick[top] == y[top]).mean())
        r["tied_quarter_rate"] = float((~decided).mean())
        strong = decided & (np.maximum(p, 1 - p) >= STRONG_QUARTER)
        r["strong_accuracy"] = float((pick[strong] == y[strong]).mean()) if strong.any() else None
        r["strong_share"] = float(strong.sum() / decided.sum())
        pooled["right"] += int((pick[decided] == y[decided]).sum()); pooled["decided"] += int(decided.sum())
        pooled["strong"] += int(strong.sum()); pooled["strong_right"] += int((pick[strong] == y[strong]).sum())
        out[f"Q{q}"] = r

    # Whole game from summing the quarter predictions (regulation only).
    sp = sum(models[f"spread_q{q}"]["model"].predict(test[X]) for q in QUARTERS)
    tot = sum(models[f"total_q{q}"]["model"].predict(test[X]) for q in QUARTERS)
    y_sp = sum(test[f"y_spread_q{q}"] for q in QUARTERS)
    y_tot = sum(test[f"y_total_q{q}"] for q in QUARTERS)
    out["Game (regulation)"] = {
        "spread_mae": float(np.abs(y_sp - sp).mean()),
        "spread_mae_baseline": float(np.abs(y_sp - sum(train[f"y_spread_q{q}"].mean() for q in QUARTERS)).mean()),
        "total_mae": float(np.abs(y_tot - tot).mean()),
        "total_mae_baseline": float(np.abs(y_tot - sum(train[f"y_total_q{q}"].mean() for q in QUARTERS)).mean()),
        "winner_accuracy": float((np.sign(sp) == np.sign(y_sp))[y_sp != 0].mean()),
        "winner_accuracy_home_baseline": float((y_sp[y_sp != 0] > 0).mean()),
    }
    # Strong full-game picks: win probability from the summed spread, as the app shows it.
    p_home = norm.cdf(sp / models["game_spread_resid_sd"])
    decided = (y_sp != 0).values
    strong = decided & (np.maximum(p_home, 1 - p_home) >= STRONG_GAME)
    out["Game (regulation)"]["strong_accuracy"] = float((np.sign(sp) == np.sign(y_sp))[strong].mean())
    out["Game (regulation)"]["strong_share"] = float(strong.sum() / decided.sum())
    # All quarters together.
    out["All quarters"] = {"winner_accuracy": pooled["right"] / pooled["decided"],
                           "strong_accuracy": pooled["strong_right"] / max(1, pooled["strong"]),
                           "strong_share": pooled["strong"] / pooled["decided"]}
    return out


def main():
    ls = pd.read_csv(ROOT / "data" / "line_scores.csv")
    games = pregame_features(ls).merge(targets(ls), on="game_id")
    X = feature_columns(games)
    per_season = ls.groupby("season").size()
    test_season = int(per_season[per_season >= 1000].index.max())  # last complete season
    train, test = games[games["season"] < test_season], games[games["season"] == test_season]

    metrics = {
        "train_seasons": sorted(int(s) for s in train["season"].unique()),
        "test_season": test_season,
        "train_games": len(train), "test_games": len(test),
        "results": evaluate(fit_models(train, X), train, test, X),
    }
    current = int(games["season"].max())
    if current > test_season:
        tr, te = games[games["season"] < current], games[games["season"] == current]
        metrics["current_season"] = {
            "season": current, "games": len(te), "through": str(ls["date"].max()),
            "results": evaluate(fit_models(tr, X), tr, te, X),
        }

    metrics["strong_thresholds"] = {"game": STRONG_GAME, "quarter": STRONG_QUARTER}
    print(json.dumps(metrics, indent=2))

    final = fit_models(games, X)
    (ROOT / "models").mkdir(exist_ok=True)
    joblib.dump({
        "models": final, "features": X, "team_form": latest_team_form(ls),
        "data_through": str(ls["date"].max()),
    }, ROOT / "models" / "models.joblib")
    (ROOT / "models" / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print("saved models/models.joblib and models/metrics.json")


if __name__ == "__main__":
    main()
