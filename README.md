# FPL Predictor

A live, AI-powered Fantasy Premier League assistant. It predicts how many points each player will score in the next gameweek, builds the best squad you can afford, plans ahead over upcoming fixtures, and explains every recommendation in plain English.

Live demo: add your Render link here after deploy · Built by Husain Ali
What it does
Predicts next-gameweek points for every player with a machine-learning model trained on three seasons of real match data.
Builds your optimal 15-player squad and starting XI under the £100m budget, and picks a captain.
Plans ahead over the next several gameweeks so you can spot who has a good run of fixtures coming.
Surfaces the best value picks (points per £m) and low-owned differentials.
Explains each pick in a sentence or two, written by Claude, so you know why it is recommended.

## What it does

-Predicts next-gameweek points for every player with a machine-learning model trained on three seasons of real match data.
-Builds your optimal 15-player squad and starting XI under the £100m budget, and picks a captain.
-Plans ahead over the next several gameweeks so you can spot who has a good run of fixtures coming.
-Surfaces the best value picks (points per £m) and low-owned differentials.
-Explains each pick in a sentence or two, written by Claude, so you know why it is recommended.


## Why it is interesting

-Beats a "recent form average" baseline, and handles the messy real world: players who changed clubs over the summer, newly promoted teams, and signings who have never played in the league all get sensible predictions instead of breaking the model.
-Predicts each match from only what you would actually know beforehand (form so far plus the upcoming opponent and venue), not hindsight.
-Ships as a real API, so the predictions can power other tools, and it measures itself with usage analytics and a live A/B test on how recommendations are explained.


## Built with

Python · XGBoost · FastAPI · PuLP (squad optimization) · the Claude API · a zero-build HTML frontend. Deployed on Render.

## Run locally

```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp ../.env.example .env          # add your ANTHROPIC_API_KEY (optional)
export $(grep -v '^#' .env | xargs)
uvicorn main:app --reload
```

Open http://localhost:8000 for the app, or http://localhost:8000/docs for the API.
The first request trains the model (downloads a few season CSVs), which takes a minute; after that
it's cached.

## Deploy to Render (free)

1. Push this folder to a GitHub repo.
2. In Render: **New → Blueprint**,  select the repo. It reads `render.yaml` automatically.
3. Add `ANTHROPIC_API_KEY` as a secret env var on the web service (or omit it for the fallback).
4. After the first deploy, copy your `https://<app>.onrender.com` URL into the `WARM_URL` env var of
   the `fpl-warm` cron so the free instance stays awake and the model stays fresh.




