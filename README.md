# FPL Predictor

A shipped, AI-native Fantasy Premier League tool. It predicts each player's points for the
upcoming gameweek, builds the optimal squad under budget, plans over a multi-week fixture run,
surfaces value picks and differentials, and explains every recommendation in plain English with
Claude. It exposes all of this as a public API and measures itself with usage analytics and a live
A/B test.

Built on top of an XGBoost model trained on three seasons of real match data, with team-strength,
recent-form, fixture and price features. Handles transfers, promoted clubs and players new to the
league via cold-start priors.

## What's inside

```
fpl-app/
  backend/
    engine.py        model + data pipeline, trains once and caches to disk
    main.py          FastAPI app (the public API) and static frontend host
    llm.py           Claude "why this pick" explanation layer (+ no-key fallback)
    analytics.py     SQLite event logging + the A/B test
    requirements.txt
  frontend/
    index.html       single-page UI (no build step)
  render.yaml        one-click Render deploy (web service + warm cron)
  .env.example
```

## The four product ingredients

1. **Deployed for real users** — one-command Render deploy, public URL.
2. **AI layer** — Claude turns model output into a plain-english recommendation (`llm.py`).
3. **Public API** — the FastAPI service itself; see the endpoints below and interactive docs at `/docs`.
4. **Analytics + one A/B test** — every session is bucketed into a *concise* or *detailed*
   explanation style; thumbs-up rates per variant are tracked and exposed at `/api/stats`.

## API

| Method | Path | What |
|---|---|---|
| GET | `/health` | model status, next gameweek, MAE vs baseline |
| GET | `/api/predictions?limit=&position=` | top predicted players for the next GW |
| GET | `/api/squad?budget=` | optimal 15 + starting XI + captain |
| GET | `/api/horizon?n=` | total predicted points over the next n gameweeks |
| GET | `/api/value` | best value (points per £m) and low-owned differentials |
| GET | `/api/player/{id}` | one player's row + features |
| POST | `/api/player/{id}/explain` | Claude explanation (A/B variant chosen by session) |
| POST | `/api/feedback` | `{player_id, helpful}` thumbs up/down for the A/B test |
| GET | `/api/stats` | usage + A/B results |
| POST | `/api/refresh` | retrain now (optionally protected by `FPL_REFRESH_TOKEN`) |

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
2. In Render: **New → Blueprint**, point it at the repo. It reads `render.yaml`.
3. Add your `ANTHROPIC_API_KEY` as a secret env var on the web service (leave blank to use the
   templated fallback).
4. After the first deploy, copy your `https://<app>.onrender.com` URL into the `WARM_URL` env var of
   the `fpl-warm` cron so the free instance stays awake and the model stays fresh.

Notes for the free tier: the instance sleeps after ~15 min idle and takes ~1 min to wake (the warm
cron avoids this). It has 512MB RAM; if a build runs out of memory, set
`FPL_HIST_SEASONS=2024-25,2025-26` to train on two seasons instead of three.

## The A/B test

Hypothesis: detailed explanations earn more trust (higher thumbs-up rate) than one-liners.
Each session is deterministically assigned a variant, explanations are rendered in that style, and
`/api/stats` reports the helpful rate for each. Swap the variant styles in `llm.py` to test other
hypotheses.
