# NBA quarter-by-quarter predictor

Predicts, for each quarter (and the whole game), the winner, the point spread and the total points.

## Run it

```bash
pip install -r requirements.txt pyarrow
python fetch_data.py      # build data/line_scores.csv (2020-21 to today) and data/schedule.csv
python train.py           # train models, print accuracy, save models/
streamlit run app.py      # open http://localhost:8501, pick away and home teams
```

`data/line_scores.csv` and `models/` are already built, so `streamlit run app.py` works straight away.

## Refresh with the latest games

```bash
python fetch_data.py && python train.py
python export_web.py web/model.json web/reference_predictions.json
python web/build_web.py web   # rebuilds ../index.html (the home screen app)
```

The current season's results update daily, so re-running this picks up every game played so far
and the next two weeks of the schedule.

## How it works

- **Data** (`fetch_data.py`):
  - 2020-21 to 2025-26: NBA play-by-play from cdn.nba.com, mirrored at github.com/shufinskiy/nba_data
    (stats.nba.com blocks most cloud machines). Quarter scores come from the running score at the end
    of each period.
  - The current season: ESPN's schedule with quarter line scores, published daily by sportsdataverse
    (github.com/sportsdataverse/sportsdataverse-data). It matched the NBA data quarter for quarter on
    99.9% of 2025-26 games. The same file gives the upcoming schedule.
- **Features** (`features.py`): each team's form as exponentially weighted averages over the last
  ~8 and ~30 games (points scored and allowed in each quarter, margin, win rate), a this-season average
  that takes over from long-term form after the first few games of a season, plus days of rest.
  Only games before the one being predicted are used.
- **Models** (`train.py`): per quarter, ridge regression for spread and total, logistic regression
  for the quarter winner. The game row sums the quarter predictions.

## Accuracy

**Last full season**: trained on 2020-21 to 2024-25, tested on all 1,230 games of 2025-26.

| Period | Winner accuracy | Always-home baseline | Spread error (pts) | Total error (pts) |
|---|---|---|---|---|
| Q1 | 58.8% | 52.1% | 6.7 (baseline 6.9) | 6.5 (baseline 6.8) |
| Q2 | 56.4% | 50.2% | 6.8 (baseline 7.0) | 6.7 (baseline 6.7) |
| Q3 | 57.3% | 51.9% | 6.8 (baseline 7.0) | 6.8 (baseline 7.0) |
| Q4 | 55.4% | 52.7% | 6.5 (baseline 6.6) | 7.1 (baseline 7.3) |
| Game | 68.2% | 55.2% | 11.5 (baseline 13.0) | 14.6 (baseline 15.8) |

**This season so far** (154 games of 2026-27 through November 9, 2026), compared with the earlier
version that kept team form frozen at the end of 2025-26:

| | Earlier version (frozen form) | Now (form updated every game) |
|---|---|---|
| Game winner | 62.3% | 68.5% |
| Quarter winner (average) | 54.0% | 57.3% |
| Game spread error | 12.5 pts | 11.9 pts |
| Game total error | 16.5 pts | 14.9 pts |

154 games is a small sample, so these numbers will move a few points as the season goes on.
Scoring is down this season (about 227 points a game vs 231 in 2025-26), and the model still
over-predicts totals by about 1 point per quarter; that shrinks as more of this season's games come in.
See `models/metrics.json` for everything.

## Limits

- Team form only, no player injuries, lineups or trades.
- The app's numbers only change when the refresh above is run and the app is republished.

## No-install version

The app at the top of this repo runs the same models in the browser (live at https://derickmthozi-ops.github.io/nba-quarters/).
`export_web.py` exports the trained weights, team form and upcoming games to JSON that gets embedded in that page.
