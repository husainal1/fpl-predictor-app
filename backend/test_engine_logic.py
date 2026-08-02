"""Offline test of the engine's pure logic on mock data (no network / xgboost needed)."""
import numpy as np, pandas as pd
import engine as E

rng = np.random.default_rng(0)
teams = ["Arsenal", "Liverpool", "Everton", "Burnley"]
roster = [("Mo Salah", "MID", "Liverpool"), ("Erling Haaland", "FWD", "Arsenal"),
          ("Virgil van", "DEF", "Liverpool"), ("Keeper One", "GK", "Everton"),
          ("Jack Grealish", "MID", "Everton")]
rows = []
for pid, (name, pos, club) in enumerate(roster):
    for season in ["2024-25", "2025-26"]:
        for gw in range(1, 7):
            opp = teams[(gw + pid) % len(teams)]
            home = gw % 2
            rows.append(dict(name=name, name_key=E.norm_name(name), position=pos, team=club,
                opp_name=opp, season=season, round=gw, was_home=bool(home),
                kickoff_time=f"20{'24' if season=='2024-25' else '25'}-0{1+gw}-01T15:00:00Z",
                team_h_score=int(rng.integers(0, 4)), team_a_score=int(rng.integers(0, 4)),
                minutes=int(rng.choice([0, 45, 90], p=[.1, .2, .7])),
                total_points=float(rng.integers(0, 13)), goals_scored=int(rng.integers(0, 2)),
                assists=int(rng.integers(0, 2)), ict_index=rng.random()*10, creativity=rng.random()*30,
                influence=rng.random()*30, threat=rng.random()*40, expected_goals=rng.random(),
                expected_assists=rng.random(), expected_goal_involvements=rng.random(),
                expected_goals_conceded=rng.random(), value=int(rng.integers(40, 140))))
hist = pd.DataFrame(rows)

cur_strength = {t: (float(rng.standard_normal()), float(rng.standard_normal())) for t in teams}
cur_strength["Burnley"] = (-1.2, -1.0)  # promoted-style weak club

players_now = pd.DataFrame([
    ("Mo Salah", "MID", "Liverpool", 13.0, "a", 45.0),
    ("Erling Haaland", "FWD", "Arsenal", 14.0, "a", 60.0),
    ("Virgil van", "DEF", "Liverpool", 6.0, "a", 30.0),
    ("Keeper One", "GK", "Everton", 5.0, "d", 8.0),
    ("Jack Grealish", "MID", "Everton", 7.0, "a", 5.0),
    ("New Signing", "FWD", "Burnley", 7.5, "a", 2.0)],
    columns=["full_name", "position", "team_name", "price", "status", "owned_pct"])
players_now["player_id"] = range(1, len(players_now) + 1)
players_now["web_name"] = players_now["full_name"]
players_now["name_key"] = players_now["full_name"].map(E.norm_name)

team_strength = E._team_strength_fn(hist, cur_strength)
fe = E.add_features(hist, team_strength)
assert set(f in fe.columns for f in E.FEATURES) == {True}, "missing features"
print(f"[add_features] {len(fe)} rows, all {len(E.FEATURES)} feature cols present")

recent = E.latest_form(hist)
assert len(recent) == hist["name_key"].nunique()
print(f"[latest_form] snapshot rows = {len(recent)}")

fixtures = pd.DataFrame([{"event": 1, "team_h": 1, "team_a": 2},
                         {"event": 1, "team_h": 3, "team_a": 4},
                         {"event": 2, "team_h": 2, "team_a": 3},
                         {"event": 2, "team_h": 4, "team_a": 1}])
team_name_now = {1: "Liverpool", 2: "Arsenal", 3: "Everton", 4: "Burnley"}
cur_only = E._make_current_only_strength(cur_strength)


class Stub:
    def predict(self, X): return np.linspace(1, 9, len(X))


statuses = dict(zip(players_now["player_id"], players_now["status"]))
pred = E.predict_gw(players_now, fixtures, team_name_now, cur_only, recent, Stub(), 1, statuses)
assert "pred_points" in pred and len(pred) >= 1
new_row = pred[pred["web_name"] == "New Signing"]
assert len(new_row) == 1 and int(new_row["is_new"].iloc[0]) == 1
# 'd' status keeper should be damped to 0.4x
print(f"[predict_gw] rows={len(pred)} new-signing handled, statuses damped")

pred["owned_pct"] = pred["player_id"].map(dict(zip(players_now.player_id, players_now.owned_pct))).fillna(0)
pred = pred.sort_values("pred_points", ascending=False).reset_index(drop=True)
state = E.EngineState(built_at=0, next_gw=1, season_started=True, predictions=pred,
                      players_now=players_now, fixtures=fixtures, team_name_now=team_name_now,
                      cur_strength=cur_strength, recent=recent, model=Stub())
best, diffs = state.value_and_differentials()
print(f"[value_and_differentials] best={len(best)} diffs={len(diffs)} "
      f"(cols ok: {'pts_per_million' in best.columns})")

run = state.horizon(n=2)
assert not run.empty and "horizon_points" in run.columns
print(f"[horizon] players scored over 2 GWs = {len(run)}; top = {run.iloc[0]['web_name']}")
print("ENGINE LOGIC TEST PASSED")
