"""Streamlit app: pick two teams, see per-quarter predictions.

Run:  streamlit run app.py
"""
import json
from math import erf, sqrt
from pathlib import Path

import pandas as pd
import streamlit as st

from features import QUARTERS, matchup_row

ROOT = Path(__file__).parent


@st.cache_resource
def load():
    import joblib
    bundle = joblib.load(ROOT / "models" / "models.joblib")
    metrics = json.loads((ROOT / "models" / "metrics.json").read_text())
    return bundle, metrics


def normal_cdf(x):
    return 0.5 * (1 + erf(x / sqrt(2)))


def predict(bundle, home, away, home_rest, away_rest):
    row = matchup_row(bundle["team_form"], home, away, home_rest, away_rest)[bundle["features"]]
    m = bundle["models"]
    rows = []
    for q in QUARTERS:
        spread = float(m[f"spread_q{q}"]["model"].predict(row)[0])
        total = float(m[f"total_q{q}"]["model"].predict(row)[0])
        p_home = float(m[f"win_q{q}"]["model"].predict_proba(row)[0, 1])
        rows.append({"Period": f"Q{q}", "spread": spread, "total": total, "p_home": p_home,
                     "spread_sd": m[f"spread_q{q}"]["resid_sd"], "total_sd": m[f"total_q{q}"]["resid_sd"]})
    spread = sum(r["spread"] for r in rows)
    total = sum(r["total"] for r in rows)
    rows.append({"Period": "Game", "spread": spread, "total": total,
                 "p_home": normal_cdf(spread / m["game_spread_resid_sd"]),
                 "spread_sd": m["game_spread_resid_sd"], "total_sd": None})
    return rows


def table(rows, home, away):
    out = []
    for r in rows:
        winner, p = (home, r["p_home"]) if r["p_home"] >= 0.5 else (away, 1 - r["p_home"])
        h, a = (r["total"] + r["spread"]) / 2, (r["total"] - r["spread"]) / 2
        out.append({
            "Period": r["Period"],
            "Winner": f"{winner} ({p:.0%})",
            f"Spread ({home})": f"{-r['spread']:+.1f}",
            "Total": f"{r['total']:.1f}",
            "Projected score": f"{home} {h:.0f} - {a:.0f} {away}",
        })
    return pd.DataFrame(out).set_index("Period")


def main():
    st.set_page_config(page_title="NBA Quarter Predictor", page_icon="🏀")
    st.title("NBA quarter-by-quarter predictor")
    bundle, metrics = load()
    teams = sorted(bundle["team_form"].index)

    c1, c2 = st.columns(2)
    away = c1.selectbox("Away team", teams, index=teams.index("BOS") if "BOS" in teams else 0)
    home = c2.selectbox("Home team", teams, index=teams.index("LAL") if "LAL" in teams else 1)
    r1, r2 = st.columns(2)
    away_rest = r1.slider("Away days of rest", 1, 4, 2, help="1 = second night of a back-to-back")
    home_rest = r2.slider("Home days of rest", 1, 4, 2)

    if home == away:
        st.warning("Pick two different teams.")
        return

    rows = predict(bundle, home, away, home_rest, away_rest)
    st.subheader(f"{away} at {home}")
    st.dataframe(table(rows, home, away), use_container_width=True)
    st.caption(
        f"Spread is from the home team's side, betting-line style ({home} -3.5 means {home} "
        "by 3.5). Game row sums the four quarters and ignores overtime. "
        f"Team form uses games through {bundle['data_through']}."
    )

    chart = pd.DataFrame({
        home: [(r["total"] + r["spread"]) / 2 for r in rows[:4]],
        away: [(r["total"] - r["spread"]) / 2 for r in rows[:4]],
    }, index=[r["Period"] for r in rows[:4]])
    st.bar_chart(chart, stack=False)

    with st.expander("How accurate is this?"):
        res = metrics["results"]
        acc = pd.DataFrame({
            p: {
                "Winner accuracy": f"{v['winner_accuracy']:.1%}",
                "Always-home baseline": f"{v['winner_accuracy_home_baseline']:.1%}",
                "Spread error (avg pts)": f"{v['spread_mae']:.1f} (baseline {v['spread_mae_baseline']:.1f})",
                "Total error (avg pts)": f"{v['total_mae']:.1f} (baseline {v['total_mae_baseline']:.1f})",
            } for p, v in res.items()
        }).T
        st.dataframe(acc, use_container_width=True)
        st.caption(
            f"Trained on {metrics['train_games']} games from earlier seasons and tested on all "
            f"{metrics['test_games']} games of the {metrics['test_season']}-"
            f"{(metrics['test_season'] + 1) % 100:02d} regular season, which the model never saw. "
            "Tied quarters are left out of winner accuracy."
        )


if __name__ == "__main__":
    main()
