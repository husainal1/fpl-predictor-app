"""
FPL Predictor API (FastAPI).

This IS the public developer-facing API (ingredient 3). It serves the trained
model's predictions, the optimal squad, the multi-week horizon, value picks, and
Claude-written explanations, and it records analytics + an A/B test.

Run locally:   uvicorn main:app --reload
Docs:          http://localhost:8000/docs
"""
import os
import uuid

import numpy as np
import pandas as pd
from fastapi import FastAPI, Request, Response, Body, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

import engine
import llm
import analytics

app = FastAPI(title="FPL Predictor API", version="1.0",
              description="Predicted FPL points, optimal squads, and AI pick explanations.")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend")
SID_COOKIE = "fpl_sid"


@app.on_event("startup")
def _startup():
    analytics.init_db()


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _records(df: pd.DataFrame, cols=None):
    if df is None or df.empty:
        return []
    d = df[cols] if cols else df
    out = []
    for _, r in d.iterrows():
        row = {}
        for k, v in r.items():
            if isinstance(v, (np.integer,)):
                v = int(v)
            elif isinstance(v, (np.floating,)):
                v = None if pd.isna(v) else round(float(v), 2)
            elif isinstance(v, (np.bool_,)):
                v = bool(v)
            row[k] = v
        out.append(row)
    return out


def _sid(request: Request, response: Response) -> str:
    sid = request.cookies.get(SID_COOKIE)
    if not sid:
        sid = uuid.uuid4().hex
        response.set_cookie(SID_COOKIE, sid, max_age=60 * 60 * 24 * 90, samesite="lax")
    return sid


PRED_COLS = ["player_id", "web_name", "team_name", "position", "price",
             "pred_points", "owned_pct", "opp_name", "is_home"]

# Official club badges + team codes, keyed off live FPL team data so they stay
# correct as teams change. The code powers both the badge and the kit image.
# Fetched once and cached.
_CRESTS = {}
_CODES = {}


def _team_maps():
    global _CRESTS, _CODES
    if _CRESTS:
        return _CRESTS, _CODES
    try:
        b = engine.get_json(f"{engine.BASE}/bootstrap-static/")
        _CRESTS = {t["name"]: f"https://resources.premierleague.com/premierleague/badges/50/t{t['code']}.png"
                   for t in b.get("teams", [])}
        _CODES = {t["name"]: t["code"] for t in b.get("teams", [])}
    except Exception:  # noqa: BLE001
        _CRESTS = {"__tried__": None}
    return _CRESTS, _CODES


def _add_crest(players):
    cm, codes = _team_maps()
    for p in players:
        p["crest"] = cm.get(p.get("team_name"))
        p["team_code"] = codes.get(p.get("team_name"))
    return players


# --------------------------------------------------------------------------- #
# endpoints
# --------------------------------------------------------------------------- #
@app.get("/health")
def health():
    st = engine.get_state()
    return {"status": "ok", "next_gw": st.next_gw, "season_started": st.season_started,
            "built_at": st.built_at, "metrics": st.metrics}


@app.get("/api/predictions")
def predictions(limit: int = 50, position: str = None):
    st = engine.get_state()
    d = st.predictions
    if position:
        d = d[d["position"] == position.upper()]
    players = _add_crest(_records(d.head(limit), PRED_COLS))
    fn = dict(zip(st.players_now["player_id"], st.players_now["full_name"]))
    for p in players:
        p["full_name"] = fn.get(p.get("player_id"), p.get("web_name"))
    return {"next_gw": st.next_gw, "players": players}


@app.get("/api/squad")
def squad(budget: float = engine.SQUAD_BUDGET):
    st = engine.get_state()
    sq = st.squad(budget=budget)
    cap = sq[sq["is_cap"] == 1]["web_name"]
    return {
        "next_gw": st.next_gw,
        "cost": round(float(sq["price"].sum()), 1),
        "captain": (cap.iloc[0] if len(cap) else None),
        "predicted_xi_points": round(float(sq[sq["is_start"] == 1]["pred_points"].sum()), 2),
        "squad": _add_crest(_records(sq, PRED_COLS + ["is_start", "is_cap"])),
    }


@app.get("/api/horizon")
def horizon(n: int = 5, limit: int = 30):
    st = engine.get_state()
    run = st.horizon(n=n)
    return {"from_gw": st.next_gw, "weeks": n,
            "players": _add_crest(_records(run.head(limit),
                                ["player_id", "web_name", "team_name", "position",
                                 "price", "games", "horizon_points", "per_game"]))}


@app.get("/api/forecast")
def forecast(weeks: int = 5):
    st = engine.get_state()
    weeks = max(1, min(6, weeks))
    gws, players = st.forecast(n=weeks)
    players = _add_crest(players)
    fn = dict(zip(st.players_now["player_id"], st.players_now["full_name"]))
    for p in players:
        p["full_name"] = fn.get(p.get("player_id"), p.get("web_name"))
    return {"from_gw": st.next_gw, "gws": gws, "players": players}


@app.get("/api/team/{entry_id}")
def team(entry_id: int):
    st = engine.get_state()
    try:
        data = st.import_team(entry_id)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception:  # noqa: BLE001
        raise HTTPException(404, "Couldn't load that team ID — double-check it and try again.")
    data["squad"] = _add_crest(data["squad"])
    return data


@app.get("/api/solve/{entry_id}")
def solve(entry_id: int, ft: int = 1, hit_budget: int = 0):
    st = engine.get_state()
    try:
        data = st.solve(entry_id, ft=ft, hit_budget=hit_budget)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception:  # noqa: BLE001
        raise HTTPException(404, "Couldn't solve that team — double-check the ID and try again.")
    data["squad"] = _add_crest(data["squad"])
    for m in data.get("moves", []):
        _add_crest([m["out"], m["in"]])
    if data.get("captain"):
        _add_crest([data["captain"]])
    if data.get("vice"):
        _add_crest([data["vice"]])
    data["verdict"] = llm.explain_reckoning(data).get("text", "")
    return data


@app.get("/api/chip/{entry_id}")
def chip(entry_id: int, chip: str = "wildcard"):
    st = engine.get_state()
    try:
        data = st.chip_team(entry_id, chip=chip)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception:  # noqa: BLE001
        raise HTTPException(404, "Couldn't build that chip team.")
    data["squad"] = _add_crest(data["squad"])
    if data.get("captain"):
        _add_crest([data["captain"]])
    return data


_PLAN_CACHE = {}


@app.get("/api/plan/{entry_id}")
def plan(entry_id: int, weeks: int = 3, ft: int = 1):
    import time as _t
    weeks = max(1, min(6, weeks))
    key = (entry_id, weeks, ft)
    hit = _PLAN_CACHE.get(key)
    if hit and (_t.time() - hit[0]) < 1800:
        return hit[1]
    st = engine.get_state()
    try:
        data = st.plan(entry_id, weeks=weeks, ft=ft)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception:  # noqa: BLE001
        raise HTTPException(404, "Couldn't build a plan for that team.")
    _PLAN_CACHE[key] = (_t.time(), data)
    return data


_STATS_CACHE = {"at": 0.0, "data": None}


@app.get("/api/playerstats")
def playerstats():
    import time as _t
    if _STATS_CACHE["data"] is not None and (_t.time() - _STATS_CACHE["at"]) < 1800:
        return _STATS_CACHE["data"]
    b = engine.get_json(f"{engine.BASE}/bootstrap-static/")
    tname = {t["id"]: t["name"] for t in b.get("teams", [])}
    posmap = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}

    def fnum(e, k):
        try:
            return round(float(e.get(k) or 0), 2)
        except (TypeError, ValueError):
            return 0.0

    def inum(e, k):
        try:
            return int(float(e.get(k) or 0))
        except (TypeError, ValueError):
            return 0

    out = []
    for e in b.get("elements", []):
        pos = posmap.get(e.get("element_type"))
        if pos is None:
            continue
        out.append({
            "player_id": e["id"],
            "web_name": e.get("web_name"),
            "full_name": (str(e.get("first_name", "")) + " " + str(e.get("second_name", ""))).strip(),
            "team_name": tname.get(e.get("team")),
            "position": pos,
            "price": round((e.get("now_cost") or 0) / 10.0, 1),
            "total_points": inum(e, "total_points"),
            "goals": inum(e, "goals_scored"),
            "assists": inum(e, "assists"),
            "xg": fnum(e, "expected_goals"),
            "xa": fnum(e, "expected_assists"),
            "xgi": fnum(e, "expected_goal_involvements"),
            "minutes": inum(e, "minutes"),
            "bonus": inum(e, "bonus"),
            "defcon": inum(e, "defensive_contribution"),
            "owned_pct": fnum(e, "selected_by_percent"),
        })
    out.sort(key=lambda r: r["total_points"], reverse=True)
    result = {"players": _add_crest(out)}
    _STATS_CACHE["data"] = result
    _STATS_CACHE["at"] = _t.time()
    return result


@app.get("/api/value")
def value():
    st = engine.get_state()
    best, diffs = st.value_and_differentials()
    return {"best_value": _records(best), "differentials": _records(diffs)}


@app.get("/api/player/{pid}")
def player(pid: int):
    st = engine.get_state()
    p = st.player(pid)
    if p is None:
        raise HTTPException(404, "player not found")
    return {k: (None if (isinstance(v, float) and pd.isna(v)) else v)
            for k, v in p.items() if not isinstance(v, (list, dict))}


@app.post("/api/player/{pid}/explain")
def explain(pid: int, request: Request, response: Response):
    st = engine.get_state()
    p = st.player(pid)
    if p is None:
        raise HTTPException(404, "player not found")
    sid = _sid(request, response)
    variant = analytics.variant_for(sid)          # A/B: concise vs detailed
    result = llm.explain(p, variant=variant)
    analytics.log_event(sid, "explain", player_id=pid, variant=variant,
                        meta={"source": result["source"]})
    return {"player_id": pid, "web_name": p.get("web_name"), **result}


@app.post("/api/feedback")
def feedback(request: Request, response: Response, body: dict = Body(...)):
    sid = _sid(request, response)
    pid = body.get("player_id")
    helpful = bool(body.get("helpful"))
    variant = analytics.variant_for(sid)
    analytics.log_feedback(sid, pid, variant, helpful)
    return {"ok": True, "variant": variant}


@app.get("/api/stats")
def stats():
    return analytics.stats()


@app.post("/api/refresh")
def refresh(request: Request):
    token = os.environ.get("FPL_REFRESH_TOKEN")
    if token and request.headers.get("x-refresh-token") != token:
        raise HTTPException(401, "bad refresh token")
    st = engine.get_state(force_refresh=True)
    return {"ok": True, "built_at": st.built_at, "metrics": st.metrics}


@app.get("/")
def home():
    index = os.path.join(FRONTEND_DIR, "index.html")
    if os.path.exists(index):
        return FileResponse(index)
    return JSONResponse({"message": "FPL Predictor API. See /docs."})
