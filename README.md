# FPL Predictor

**A live, AI-powered Fantasy Premier League assistant.** It predicts how many points each player
will score in the next gameweek, builds the best squad you can afford, plans ahead over upcoming
fixtures, and explains every recommendation in plain English.

### ▶ Live demo: **https://fpl-predictor-app.onrender.com**
Built by [Husain Ali](https://github.com/husainal1). (Free hosting sleeps when idle, so the first
load after a quiet spell can take up to a minute to wake, then it's fast.)

![Best XI lineup](<img width="1989" height="1381" alt="0E680363-0482-4A02-91E7-92C142012512_1_201_a" src="https://github.com/user-attachments/assets/dfe9419c-85a2-4cb9-88c2-3c286503c427" />)

## What it does

- Predicts next-gameweek points for every player with a machine-learning model trained on three
  seasons of real match data.
- Builds your optimal 15-player squad and starting XI under the £100m budget, and picks a captain.
- Plans ahead over the next several gameweeks so you can spot who has a good run of fixtures coming.
- Surfaces the best value picks (points per £m) and low-owned differentials.
- Explains each pick in a sentence or two, written by Claude, so you know *why* it is recommended.

## Why it is interesting

- Beats a "recent form average" baseline, and handles the messy real world: players who changed
  clubs over the summer, newly promoted teams, and signings who have never played in the league all
  get sensible predictions instead of breaking the model.
- Predicts each match from only what you would actually know beforehand (form so far plus the
  upcoming opponent and venue), not hindsight.
- Ships as a real API, so the predictions can power other tools, and it measures itself with usage
  analytics and a live A/B test on how recommendations are explained.

## Built with

Python · XGBoost · FastAPI · PuLP (squad optimization) · the Claude API · a zero-build HTML
frontend. Deployed on Render.

---

<details>
<summary><b>For developers: run locally, API reference, and deploy</b></summary>

### Run locally

```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp ../.env.example .env          # add your ANTHROPIC_API_KEY (optional; falls back to a template)
export $(grep -v '^#' .env | xargs)
uvicorn main:app --reload
```

Open http://localhost:8000 for the app or http://localhost:8000/docs for the interactive API.
The first request trains the model (downloads a few season files), which takes about a minute, then
it is cached.

### API

| Method | Path | What |
|---|---|---|
| GET | `/health` | model status, next gameweek, accuracy vs baseline |
| GET | `/api/predictions?limit=&position=` | top predicted players for the next GW |
| GET | `/api/squad?budget=` | optimal 15 + starting XI + captain |
| GET | `/api/horizon?n=` | total predicted points over the next n gameweeks |
| GET | `/api/value` | best value and low-owned differentials |
| POST | `/api/player/{id}/explain` | Claude explanation (A/B variant chosen per session) |
| POST | `/api/feedback` | thumbs up/down that feeds the A/B test |
| GET | `/api/stats` | usage and A/B results |

### Deploy to Render (free)

1. Push this repo to GitHub (contents at the root: `backend/`, `frontend/`, `render.yaml`).
2. In Render: **New → Blueprint**, select the repo. It reads `render.yaml` automatically.
3. Add `ANTHROPIC_API_KEY` as a secret env var on the web service (or omit it for the fallback).
4. After the first deploy, put your app URL in the `WARM_URL` env var of the `fpl-warm` cron so the
   free instance stays awake and the model stays fresh.

Free-tier notes: the instance sleeps after ~15 min idle and wakes in ~1 min. It has 512MB RAM; if a
build runs out of memory, set `FPL_HIST_SEASONS=2024-25,2025-26` to train on two seasons.

### How it fits together

`engine.py` fetches live FPL data plus the historical archive, trains the model once and caches it,
and produces predictions, squads and the fixture-run plan. `main.py` serves those over the API and
hosts the frontend. `llm.py` writes the explanations. `analytics.py` logs usage and runs the A/B
test (concise vs detailed explanations, compared by thumbs-up rate).

</details>

## Model

The prediction model started life as a notebook
(`Predicting_EPL_Player_Performance_2026-27.ipynb`, included here) and is refactored into
`backend/engine.py` for serving.
