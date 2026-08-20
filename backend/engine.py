"""
FPL prediction engine (serve-ready).

Refactored from the validated Colab notebook into importable functions plus an
on-disk cache so the API trains once (and refreshes on a schedule) instead of
retraining on every request.

Public surface used by the API:
    get_state(force_refresh=False) -> EngineState
    EngineState.predictions            # next-GW predicted points per player
    EngineState.squad()                # optimal 15 + XI + captain
    EngineState.horizon(n)             # summed predicted points over next n GWs
    EngineState.value_and_differentials()
    EngineState.player(pid)            # one player's row + context (for explanations)
"""
from __future__ import annotations

import os
import time
import pickle
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd
import requests
# NB: sklearn / xgboost / pulp are imported lazily inside the functions that need
# them, so the module (and its pure data logic) imports with only pandas+numpy.

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
BASE = "https://fantasy.premierleague.com/api"
ARCHIVE = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data"
# Seasons of match history to train on. Configurable via env so you can drop to
# 2 seasons if a small (512MB) free-tier instance runs low on memory.
HIST_SEASONS = [s.strip() for s in
                os.environ.get("FPL_HIST_SEASONS", "2023-24,2024-25,2025-26").split(",") if s.strip()]
CURRENT_SEASON = os.environ.get("FPL_CURRENT_SEASON", "2026-27")

LAGS = (1, 2, 3)
WINDOWS = (3, 5)
SQUAD_BUDGET = 100.0

POS_MAP = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}
VALID_POS = ["GK", "DEF", "MID", "FWD"]

BASE_COLS = ["total_points", "minutes", "goals_scored", "assists", "ict_index",
             "creativity", "influence", "threat", "expected_goals",
             "expected_assists", "expected_goal_involvements"]

FEATURES = ([f"{c}_lag{L}" for c in BASE_COLS for L in LAGS] +
            [f"{c}_roll{W}_mean" for c in BASE_COLS for W in WINDOWS] +
            ["played_last", "played_last3_pct", "team_attack_z", "team_defence_z",
             "opp_attack_z", "opp_defence_z", "attack_vs_oppdef", "def_vs_oppatt",
             "is_home", "price", "month"] + [f"is_{p}" for p in VALID_POS])

FORM_COLS = ([f"{c}_lag{L}" for c in BASE_COLS for L in LAGS] +
             [f"{c}_roll{W}_mean" for c in BASE_COLS for W in WINDOWS] +
             ["played_last", "played_last3_pct"])

KEEP = ["total_points", "minutes", "goals_scored", "assists", "ict_index",
        "creativity", "influence", "threat", "expected_goals", "expected_assists",
        "expected_goal_involvements", "expected_goals_conceded", "team_h_score",
        "team_a_score", "was_home", "kickoff_time", "round"]

CACHE_PATH = os.environ.get("FPL_CACHE_PATH", "/tmp/fpl_engine_cache.pkl")
CACHE_TTL_SECONDS = int(os.environ.get("FPL_CACHE_TTL", 6 * 60 * 60))  # 6h


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def norm_name(s) -> str:
    if not isinstance(s, str):
        return ""
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    return " ".join(s.lower().split())


def get_json(url, retries=5, sleep=0.6):
    last = None
    for i in range(retries):
        try:
            r = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
            if r.status_code == 200:
                return r.json()
            last = f"HTTP {r.status_code}"
        except Exception as e:  # noqa: BLE001
            last = str(e)
        time.sleep(sleep * (i + 1))
    raise RuntimeError(f"Failed to fetch {url}: {last}")


def _z(x):
    x = pd.to_numeric(x, errors="coerce")
    sd = x.std(ddof=0)
    return (x - x.mean()) / sd if sd and not np.isnan(sd) else x * 0.0


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
def fetch_current(session_get=get_json):
    """Live current-season data from the FPL API."""
    boot = session_get(f"{BASE}/bootstrap-static/")
    elements = pd.DataFrame(boot["elements"])
    teams_raw = pd.DataFrame(boot["teams"])
    events = pd.DataFrame(boot["events"])
    fixtures = pd.DataFrame(session_get(f"{BASE}/fixtures/"))
    return elements, teams_raw, events, fixtures


def load_archive_season(season):
    df = pd.read_csv(f"{ARCHIVE}/{season}/gws/merged_gw.csv", encoding="utf-8-sig")
    for c in KEEP:
        if c not in df.columns:
            df[c] = np.nan
    out = df[["name", "position", "team", "opponent_team", "value"] + KEEP].copy()
    out["season"] = season
    out["name_key"] = out["name"].map(norm_name)
    return out


def load_archive(opp_name_by_season):
    frames = []
    for s in HIST_SEASONS:
        d = load_archive_season(s)
        d["opp_name"] = [opp_name_by_season.get((s, int(o)), None) if pd.notna(o) else None
                         for o in d["opponent_team"]]
        frames.append(d)
    hist = pd.concat(frames, ignore_index=True)
    hist["position"] = hist["position"].replace({"GKP": "GK", "AM": "MID"})
    return hist[hist["position"].isin(VALID_POS)].copy()


# --------------------------------------------------------------------------- #
# Feature engineering (identical logic to the validated notebook)
# --------------------------------------------------------------------------- #
def _team_strength_fn(hist, cur_strength):
    h = hist.copy()
    h["gf"] = np.where(h["was_home"] == True, h["team_h_score"], h["team_a_score"])  # noqa: E712
    h["ga"] = np.where(h["was_home"] == True, h["team_a_score"], h["team_h_score"])  # noqa: E712
    tm = (h.dropna(subset=["team", "gf", "ga"])
            .drop_duplicates(["season", "team", "round"])
            .groupby(["season", "team"]).agg(gf=("gf", "mean"), ga=("ga", "mean")).reset_index())
    tm["attack_z"] = tm.groupby("season")["gf"].transform(lambda x: (x - x.mean()) / (x.std(ddof=0) or 1))
    tm["defence_z"] = tm.groupby("season")["ga"].transform(lambda x: -(x - x.mean()) / (x.std(ddof=0) or 1))
    hist_strength = {(r["season"], r["team"]): (float(r["attack_z"]), float(r["defence_z"]))
                     for _, r in tm.iterrows()}

    def team_strength(season, team_name):
        if season == CURRENT_SEASON and team_name in cur_strength:
            return cur_strength[team_name]
        if (season, team_name) in hist_strength:
            return hist_strength[(season, team_name)]
        if team_name in cur_strength:
            return cur_strength[team_name]
        return (0.0, 0.0)

    return team_strength


def add_features(df, team_strength):
    df = df.copy()
    df["kickoff_time"] = pd.to_datetime(df["kickoff_time"], errors="coerce", utc=True)
    df = df.sort_values(["name_key", "kickoff_time", "season", "round"]).reset_index(drop=True)
    g = df.groupby("name_key", group_keys=False)
    for col in BASE_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        for L in LAGS:
            df[f"{col}_lag{L}"] = g[col].shift(L)
        for W in WINDOWS:
            df[f"{col}_roll{W}_mean"] = g[col].shift(1).rolling(W).mean()
    df["played_last"] = g["minutes"].shift(1).fillna(0).gt(0).astype(int)
    df["played_last3_pct"] = g["minutes"].shift(1).rolling(3).apply(lambda x: np.mean(x > 0), raw=True)
    st = df.apply(lambda r: team_strength(r["season"], r["team"]), axis=1)
    ost = df.apply(lambda r: team_strength(r["season"], r["opp_name"]), axis=1)
    df["team_attack_z"], df["team_defence_z"] = zip(*st)
    df["opp_attack_z"], df["opp_defence_z"] = zip(*ost)
    df["attack_vs_oppdef"] = df["team_attack_z"] - df["opp_defence_z"]
    df["def_vs_oppatt"] = df["team_defence_z"] - df["opp_attack_z"]
    df["is_home"] = (df["was_home"] == True).astype(int)  # noqa: E712
    for p in VALID_POS:
        df[f"is_{p}"] = (df["position"] == p).astype(int)
    df["price"] = pd.to_numeric(df["value"], errors="coerce") / 10.0
    df["month"] = df["kickoff_time"].dt.month.fillna(0).astype(int)
    return df


def latest_form(df):
    df = df.copy()
    df["kickoff_time"] = pd.to_datetime(df["kickoff_time"], errors="coerce", utc=True)
    df = df.sort_values(["name_key", "kickoff_time", "season", "round"])
    key = df["name_key"]
    for col in BASE_COLS:
        s = pd.to_numeric(df[col], errors="coerce")
        for L in LAGS:
            df[f"{col}_lag{L}"] = s.groupby(key).shift(L - 1)
        for W in WINDOWS:
            df[f"{col}_roll{W}_mean"] = s.groupby(key).transform(lambda x: x.rolling(W).mean())
    mins = pd.to_numeric(df["minutes"], errors="coerce")
    df["played_last"] = (mins > 0).astype(int)
    df["played_last3_pct"] = (mins > 0).groupby(key).transform(lambda x: x.rolling(3).mean())
    return df.groupby("name_key").tail(1)


# --------------------------------------------------------------------------- #
# Engine state
# --------------------------------------------------------------------------- #
@dataclass
class EngineState:
    built_at: float
    next_gw: int
    season_started: bool
    predictions: pd.DataFrame
    players_now: pd.DataFrame
    fixtures: pd.DataFrame
    team_name_now: dict
    cur_strength: dict
    recent: pd.DataFrame
    model: object
    metrics: dict = field(default_factory=dict)

    # ---- served computations ------------------------------------------------
    def _team_strength(self, team):
        return self.cur_strength.get(team, (0.0, 0.0))

    def squad(self, budget=SQUAD_BUDGET):
        return optimize_squad(self.predictions, budget=budget)

    def value_and_differentials(self, min_pts=3.0, diff_max_own=10.0, diff_min_pts=3.5):
        d = self.predictions.copy()
        d["pts_per_million"] = (d["pred_points"] / d["price"]).round(2)
        elig = d[d["pred_points"] > 0]
        best = elig[elig["pred_points"] >= min_pts].sort_values("pts_per_million", ascending=False)
        diffs = elig[(elig["owned_pct"] < diff_max_own) & (elig["pred_points"] >= diff_min_pts)] \
            .sort_values("pred_points", ascending=False)
        cols = ["player_id", "web_name", "team_name", "position", "price",
                "pred_points", "pts_per_million", "owned_pct"]
        return best[cols].head(20), diffs[cols].head(20)

    def horizon(self, n=5):
        return horizon_run(self, n)

    def forecast(self, n=5):
        return forecast_run(self, n)

    def player(self, pid):
        row = self.predictions[self.predictions["player_id"] == pid]
        return None if row.empty else row.iloc[0].to_dict()


# --------------------------------------------------------------------------- #
# Optimizer (PuLP)
# --------------------------------------------------------------------------- #
def optimize_squad(pred_df, budget=SQUAD_BUDGET, max_per_club=3, squad=(2, 5, 5, 3), starters=11):
    from pulp import LpProblem, LpMaximize, LpVariable, lpSum, value, LpBinary, PULP_CBC_CMD
    d = pred_df.reset_index(drop=True).copy()
    P = range(len(d))
    pick = LpVariable.dicts("pick", P, cat=LpBinary)
    start = LpVariable.dicts("start", P, cat=LpBinary)
    cap = LpVariable.dicts("cap", P, cat=LpBinary)
    prob = LpProblem("fpl", LpMaximize)
    prob += lpSum((start[i] + cap[i]) * d.loc[i, "pred_points"] for i in P)
    prob += lpSum(pick[i] for i in P) == sum(squad)
    prob += lpSum(pick[i] * d.loc[i, "price"] for i in P) <= budget
    prob += lpSum(start[i] for i in P) == starters
    prob += lpSum(cap[i] for i in P) == 1
    need = dict(zip(VALID_POS, squad))
    xi_min = {"GK": 1, "DEF": 3, "MID": 2, "FWD": 1}
    xi_max = {"GK": 1, "DEF": 5, "MID": 5, "FWD": 3}
    for pos in VALID_POS:
        idx = [i for i in P if d.loc[i, "position"] == pos]
        prob += lpSum(pick[i] for i in idx) == need[pos]
        prob += lpSum(start[i] for i in idx) >= xi_min[pos]
        prob += lpSum(start[i] for i in idx) <= xi_max[pos]
    for club in d["team_name"].unique():
        idx = [i for i in P if d.loc[i, "team_name"] == club]
        prob += lpSum(pick[i] for i in idx) <= max_per_club
    for i in P:
        prob += start[i] <= pick[i]
        prob += cap[i] <= start[i]
    prob.solve(PULP_CBC_CMD(msg=0))
    d["in_squad"] = [int(value(pick[i]) or 0) for i in P]
    d["is_start"] = [int(value(start[i]) or 0) for i in P]
    d["is_cap"] = [int(value(cap[i]) or 0) for i in P]
    sq = d[d["in_squad"] == 1].sort_values(["is_start", "position"], ascending=[False, True])
    return sq


# --------------------------------------------------------------------------- #
# Prediction row builder + horizon
# --------------------------------------------------------------------------- #
def _pred_row(p, opp, home, team_strength, r_idx, gw=None):
    ta, td = team_strength(CURRENT_SEASON, p["team_name"])
    oa, od = team_strength(CURRENT_SEASON, opp)
    row = {
        "player_id": p["player_id"], "web_name": p["web_name"], "team_name": p["team_name"],
        "position": p["position"], "price": p["price"], "is_home": home,
        "team_attack_z": ta, "team_defence_z": td, "opp_attack_z": oa, "opp_defence_z": od,
        "attack_vs_oppdef": ta - od, "def_vs_oppatt": td - oa,
        "month": datetime.now(timezone.utc).month, "opp_name": opp,
    }
    if gw is not None:
        row["gw"] = gw
    for pos in VALID_POS:
        row[f"is_{pos}"] = int(p["position"] == pos)
    if p["name_key"] in r_idx.index:
        rr = r_idx.loc[p["name_key"]]
        rr = rr.iloc[0] if isinstance(rr, pd.DataFrame) else rr
        for c in FORM_COLS:
            if c in rr.index:
                row[c] = rr[c]
        row["is_new"] = 0
    else:
        row["is_new"] = 1
    return row


def _opp_map(fixtures, team_name_now, gw):
    fx = fixtures[fixtures["event"] == gw]
    m = {}
    for _, r in fx.iterrows():
        h, a = team_name_now.get(r["team_h"]), team_name_now.get(r["team_a"])
        if h:
            m.setdefault(h, []).append((a, 1))
        if a:
            m.setdefault(a, []).append((h, 0))
    return m


def _avail_mult(chance, status):
    """Availability multiplier from FPL data. Uses the graded chance_of_playing
    percentage when present (0-100); otherwise treats suspended/injured/unavailable
    as out and everyone else as fully available."""
    try:
        return max(0.0, min(1.0, float(chance) / 100.0))
    except (TypeError, ValueError):
        return 0.0 if status in ("s", "i", "u", "n") else 1.0


def predict_gw(players_now, fixtures, team_name_now, team_strength, recent, model, gw,
               statuses, chances=None, damp=True):
    chances = chances or {}
    r_idx = recent.set_index("name_key")
    opp = _opp_map(fixtures, team_name_now, gw)
    rows = []
    for _, p in players_now.iterrows():
        for o, hm in opp.get(p["team_name"], []):
            rows.append(_pred_row(p, o, hm, team_strength, r_idx, gw=gw))
    if not rows:
        return pd.DataFrame()
    pf = pd.DataFrame(rows)
    for f in FEATURES:
        if f not in pf.columns:
            pf[f] = 0.0
    pf[FEATURES] = pf[FEATURES].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    pf["pred_points"] = model.predict(pf[FEATURES])
    # Scale by FPL availability -- only in-season, since pre-season flags are
    # unreliable (e.g. a World Cup knock). In-season, use FPL's graded
    # chance_of_playing percentage so it self-corrects as FPL updates.
    if damp:
        pf["pred_points"] = [pp * _avail_mult(chances.get(pid), statuses.get(pid, "a"))
                             for pp, pid in zip(pf["pred_points"], pf["player_id"])]
    return pf


def horizon_run(state: "EngineState", n=5):
    statuses = dict(zip(state.players_now["player_id"],
                        state.players_now.get("status", pd.Series(["a"] * len(state.players_now)))))
    chances = dict(zip(state.players_now["player_id"],
                       state.players_now.get("chance", pd.Series([None] * len(state.players_now)))))
    team_strength = _make_current_only_strength(state.cur_strength)
    gws = list(range(state.next_gw, min(state.next_gw + n, 39)))
    frames = [predict_gw(state.players_now, state.fixtures, state.team_name_now,
                         team_strength, state.recent, state.model, g, statuses,
                         chances=chances, damp=True) for g in gws]
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame()
    allgw = pd.concat(frames, ignore_index=True)
    run = (allgw.groupby(["player_id", "web_name", "team_name", "position", "price"])
           .agg(games=("gw", "nunique"), horizon_points=("pred_points", "sum")).reset_index())
    run["per_game"] = (run["horizon_points"] / run["games"]).round(2)
    return run.sort_values("horizon_points", ascending=False)


def forecast_run(state: "EngineState", n=5):
    """Per-gameweek predicted points for every current player over the next n GWs.

    Returns (gws, players): gws is the list of gameweek numbers, players is a list
    of dicts sorted by total descending, each with a gw_points list aligned to gws
    (double gameweeks are summed within a GW, blanks are 0). Availability damping
    is applied so injured/departed players don't top the table.
    """
    statuses = dict(zip(state.players_now["player_id"],
                        state.players_now.get("status", pd.Series(["a"] * len(state.players_now)))))
    chances = dict(zip(state.players_now["player_id"],
                       state.players_now.get("chance", pd.Series([None] * len(state.players_now)))))
    team_strength = _make_current_only_strength(state.cur_strength)
    gws = list(range(state.next_gw, min(state.next_gw + n, 39)))
    frames = [predict_gw(state.players_now, state.fixtures, state.team_name_now,
                         team_strength, state.recent, state.model, g, statuses,
                         chances=chances, damp=True) for g in gws]
    frames = [f for f in frames if not f.empty]
    if not frames:
        return gws, []
    allgw = pd.concat(frames, ignore_index=True)
    per = allgw.groupby(["player_id", "gw"])["pred_points"].sum()
    meta = state.predictions.drop_duplicates("player_id").set_index("player_id")
    out = []
    for pid, grp in per.groupby(level=0):
        if pid not in meta.index:
            continue
        by_gw = grp.droplevel(0)
        row_pts = [round(float(by_gw.get(g, 0.0)), 2) for g in gws]
        m = meta.loc[pid]
        own = m["owned_pct"] if "owned_pct" in m.index else 0.0
        out.append({
            "player_id": int(pid),
            "web_name": m["web_name"], "team_name": m["team_name"],
            "position": m["position"], "price": round(float(m["price"]), 1),
            "owned_pct": round(float(own), 1) if pd.notna(own) else 0.0,
            "gw_points": row_pts,
            "total": round(sum(row_pts), 2),
        })
    out.sort(key=lambda r: r["total"], reverse=True)
    return gws, out


def _make_current_only_strength(cur_strength):
    def team_strength(season, team_name):
        return cur_strength.get(team_name, (0.0, 0.0))
    return team_strength


# --------------------------------------------------------------------------- #
# Build / train
# --------------------------------------------------------------------------- #
def build_state(session_get=get_json) -> EngineState:
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import mean_absolute_error
    from xgboost import XGBRegressor

    elements, teams_raw, events, fixtures = fetch_current(session_get)

    finished_gws = int(events["finished"].sum())
    next_row = events[events["is_next"]]
    cur_row = events[events["is_current"]]
    current_gw = int(cur_row["id"].iloc[0]) if len(cur_row) else 0
    next_gw = int(next_row["id"].iloc[0]) if len(next_row) else (current_gw + 1 if current_gw else 1)
    season_started = finished_gws > 0

    team_name_now = dict(zip(teams_raw["id"], teams_raw["name"]))
    ts = teams_raw.copy()
    ts["attack"] = ts[["strength_attack_home", "strength_attack_away"]].mean(axis=1)
    ts["defence"] = ts[["strength_defence_home", "strength_defence_away"]].mean(axis=1)
    ts["attack_z"] = _z(ts["attack"])
    ts["defence_z"] = _z(ts["defence"])
    cur_strength = {r["name"]: (float(r["attack_z"]), float(r["defence_z"])) for _, r in ts.iterrows()}

    players_now = pd.DataFrame({
        "player_id": elements["id"],
        "web_name": elements["web_name"],
        "full_name": elements["first_name"] + " " + elements["second_name"],
        "team_name": elements["team"].map(team_name_now),
        "position": elements["element_type"].map(POS_MAP),
        "price": elements["now_cost"] / 10.0,
        "status": elements["status"],
        "chance": pd.to_numeric(elements.get("chance_of_playing_next_round"), errors="coerce"),
        "owned_pct": pd.to_numeric(elements["selected_by_percent"], errors="coerce"),
    })
    players_now = players_now[players_now["position"].isin(VALID_POS)].copy()
    # Drop players FPL flags as gone for the season. Status "u" means a permanent
    # transfer out of the league or a loan away (e.g. Digne, "Has joined PSG
    # permanently") - they will not feature for a PL club, so they should never
    # appear in picks or the squad. Injuries/doubts/suspensions (i/d/s) are kept,
    # since those players still play; their minutes risk is handled by damping.
    players_now = players_now[players_now["status"] != "u"].copy()
    players_now["name_key"] = players_now["full_name"].map(norm_name)

    # opponent id -> name per season, from the archive master list
    mtl = pd.read_csv(f"{ARCHIVE}/master_team_list.csv")
    opp_name_by_season = {(str(s), int(t)): n for s, t, n in
                          zip(mtl["season"], mtl["team"], mtl["team_name"])}

    hist_archive = load_archive(opp_name_by_season)

    # live current-season match history, if any games have been played
    hist = hist_archive
    if season_started:
        live_rows = []
        pos_now = dict(zip(players_now["player_id"], players_now["position"]))
        name_now = dict(zip(players_now["player_id"], players_now["full_name"]))
        price_now = dict(zip(players_now["player_id"], players_now["price"]))
        for pid in players_now["player_id"]:
            try:
                j = session_get(f"{BASE}/element-summary/{pid}/")
                h = pd.DataFrame(j.get("history", []))
                if h.empty:
                    continue
                h["name"] = name_now.get(pid)
                h["position"] = pos_now.get(pid)
                h["team"] = None
                h["value"] = price_now.get(pid, np.nan) * 10
                h["season"] = CURRENT_SEASON
                h["opp_name"] = h["opponent_team"].map(team_name_now)
                live_rows.append(h)
            except Exception:  # noqa: BLE001
                pass
        if live_rows:
            live = pd.concat(live_rows, ignore_index=True)
            for c in KEEP:
                if c not in live.columns:
                    live[c] = np.nan
            live["name_key"] = live["name"].map(norm_name)
            keep_cols = ["name", "position", "team", "opponent_team", "value",
                         "season", "name_key", "opp_name"] + KEEP
            hist = pd.concat([hist_archive, live[keep_cols]], ignore_index=True)

    team_strength = _team_strength_fn(hist, cur_strength)

    # features + train
    fe = add_features(hist, team_strength)
    fe["y"] = pd.to_numeric(fe["total_points"], errors="coerce")
    train = fe.dropna(subset=["y", "minutes_lag1", "total_points_lag1"]).copy()
    X = train[FEATURES].fillna(0.0)
    y = train["y"].astype(float)
    groups = train["name_key"]
    baseline = train["total_points_roll3_mean"].fillna(train["total_points_lag1"]).fillna(0)

    gkf = GroupKFold(n_splits=5)
    oof = np.zeros(len(train))
    for tr, va in gkf.split(X, y, groups):
        m = XGBRegressor(n_estimators=600, learning_rate=0.05, max_depth=6, subsample=0.8,
                         colsample_bytree=0.8, min_child_weight=5, random_state=42,
                         n_jobs=-1, tree_method="hist")
        m.fit(X.iloc[tr], y.iloc[tr], verbose=False)
        oof[va] = m.predict(X.iloc[va])
    metrics = {
        "rows": int(len(train)),
        "model_mae": float(mean_absolute_error(y, oof)),
        "baseline_mae": float(mean_absolute_error(y, baseline)),
    }

    model = XGBRegressor(n_estimators=800, learning_rate=0.04, max_depth=6, subsample=0.9,
                         colsample_bytree=0.9, min_child_weight=5, random_state=42,
                         n_jobs=-1, tree_method="hist")
    model.fit(X, y, verbose=False)

    # next-GW predictions
    recent = latest_form(hist)
    statuses = dict(zip(players_now["player_id"], players_now["status"]))
    chances = dict(zip(players_now["player_id"], players_now["chance"]))
    cur_only = _make_current_only_strength(cur_strength)
    # Always apply the availability penalty, including pre-season. FPL's injury and
    # suspension flags are accurate before kickoff, so an injured player (e.g. Araujo,
    # unknown return) should be zeroed now, not just once the season starts. The graded
    # multiplier leaves fully available players (status "a") completely untouched.
    pred = predict_gw(players_now, fixtures, team_name_now, cur_only, recent, model, next_gw,
                      statuses, chances=chances, damp=True)
    own = dict(zip(players_now["player_id"], players_now["owned_pct"]))
    pred["owned_pct"] = pred["player_id"].map(own).fillna(0.0)
    pred = pred.sort_values("pred_points", ascending=False).reset_index(drop=True)

    return EngineState(
        built_at=time.time(), next_gw=next_gw, season_started=season_started,
        predictions=pred, players_now=players_now, fixtures=fixtures,
        team_name_now=team_name_now, cur_strength=cur_strength, recent=recent,
        model=model, metrics=metrics,
    )


# --------------------------------------------------------------------------- #
# Cache
# --------------------------------------------------------------------------- #
_STATE: Optional[EngineState] = None


def get_state(force_refresh=False) -> EngineState:
    """Return a cached EngineState, refreshing from disk / API when stale."""
    global _STATE
    if _STATE is not None and not force_refresh and (time.time() - _STATE.built_at) < CACHE_TTL_SECONDS:
        return _STATE
    if not force_refresh and os.path.exists(CACHE_PATH):
        try:
            with open(CACHE_PATH, "rb") as f:
                cached = pickle.load(f)
            if (time.time() - cached.built_at) < CACHE_TTL_SECONDS:
                _STATE = cached
                return _STATE
        except Exception:  # noqa: BLE001
            pass
    _STATE = build_state()
    try:
        with open(CACHE_PATH, "wb") as f:
            pickle.dump(_STATE, f)
    except Exception:  # noqa: BLE001
        pass
    return _STATE


if __name__ == "__main__":
    st = get_state(force_refresh=True)
    print("built. next GW:", st.next_gw, "| metrics:", st.metrics)
    print(st.predictions[["web_name", "team_name", "position", "price", "pred_points"]].head(10))
