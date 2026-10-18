# NBA quarter-by-quarter predictor

Predicts, for each quarter (and the whole game), the winner, the point spread and the total points.

## Run it

```bash
pip install -r requirements.txt
python fetch_data.py      # download play-by-play, build data/line_scores.csv (2020-21 to 2025-26)
python train.py           # train models, print held-out accuracy, save models/
streamlit run app.py      # open http://localhost:8501, pick away and home teams
```

`data/line_scores.csv` and `models/` are already built, so `streamlit run app.py` works straight away.

## How it works

- **Data**: NBA play-by-play from cdn.nba.com, mirrored at github.com/shufinskiy/nba_data
  (stats.nba.com blocks most cloud machines). Quarter scores come from the running score at the
  end of each period; every game's quarters add up to the final score.
- **Features** (`features.py`): each team's recent form as exponentially weighted averages over the
  last ~8 and ~30 games: points scored and allowed in each quarter, margin, win rate, plus days of rest.
  Only games before the one being predicted are used.
- **Models** (`train.py`): per quarter, ridge regression for spread and total, logistic regression
  for the quarter winner. The game row sums the quarter predictions.

## Accuracy

Trained on 2020-21 to 2024-25, tested on all 1,230 games of the 2025-26 regular season.
See `models/metrics.json` for the full numbers.

| Period | Winner accuracy | Always-home baseline | Spread error (pts) | Total error (pts) |
|---|---|---|---|---|
| Q1 | 57.8% | 52.1% | 6.7 (baseline 6.9) | 6.6 (baseline 6.8) |
| Q2 | 57.3% | 50.2% | 6.8 (baseline 7.0) | 6.6 (baseline 6.7) |
| Q3 | 57.6% | 51.9% | 6.8 (baseline 7.0) | 6.8 (baseline 7.0) |
| Q4 | 56.0% | 52.7% | 6.5 (baseline 6.6) | 7.1 (baseline 7.3) |
| Game | 67.8% | 55.2% | 11.5 (baseline 13.0) | 14.6 (baseline 15.8) |

Single quarters are very noisy; on the third of games where the model is most confident,
quarter winner accuracy rises to about 63-66% for Q1-Q3.

## Limits

- Team form only, no player injuries, lineups or trades. Off-season roster changes are not reflected
  until new games are played.
- No live schedule feed (cdn.nba.com and stats.nba.com are blocked from this environment), so the app
  takes two teams rather than listing upcoming games. Re-run `fetch_data.py` and `train.py` once the
  mirror has 2026-27 data to refresh team form.

## No-install version

The app at the top of this repo runs the same models in the browser (live at https://derickmthozi-ops.github.io/nba-quarters/).
`export_web.py` exports the trained weights and team form to JSON that gets embedded in that page.
