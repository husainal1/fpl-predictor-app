"""
Claude explanation layer (ingredient 2: the AI-native credential).

Given a player's model output and context, produce a plain-english take on why
they are (or aren't) a good FPL pick this week. The A/B test varies the style:
  - "concise"  : one punchy sentence
  - "detailed" : two or three sentences with the reasoning

Falls back to a templated explanation when no ANTHROPIC_API_KEY is set, so the
app is fully functional for local dev and demos without a key.
"""
import os

MODEL = os.environ.get("FPL_LLM_MODEL", "claude-haiku-4-5-20251001")


def _facts(player: dict) -> str:
    def g(k, d="?"):
        v = player.get(k, d)
        return d if v is None else v
    parts = [
        f"Name: {g('web_name')}",
        f"Position: {g('position')}",
        f"Club: {g('team_name')}",
        f"Price: {g('price')}m",
        f"Model predicted points next match: {round(float(g('pred_points', 0)), 2)}",
        f"Ownership: {g('owned_pct', 0)}%",
        f"Home game: {'yes' if g('is_home', 0) else 'no'}",
        f"Recent points (last game): {g('total_points_lag1', 'n/a')}",
        f"Recent minutes (last game): {g('minutes_lag1', 'n/a')}",
        f"Team attack strength (z): {round(float(g('team_attack_z', 0)), 2)}",
        f"Opponent defence strength (z): {round(float(g('opp_defence_z', 0)), 2)}",
    ]
    return "\n".join(parts)


def _verdict(pts: float) -> str:
    if pts >= 5:
        return "one of the strongest picks this week"
    if pts >= 4:
        return "a strong pick"
    if pts >= 3:
        return "a solid, sensible pick"
    if pts >= 2:
        return "a fringe pick with rotation or minutes risk"
    return "a weak pick this week"


def _prompt(player: dict, variant: str) -> str:
    pts = float(player.get("pred_points", 0) or 0)
    verdict = _verdict(pts)
    style = ("Answer in ONE punchy sentence."
             if variant == "concise" else
             "Answer in two or three sentences, explaining the key reasons.")
    return (
        "You are an assistant inside a Fantasy Premier League tool. The model rates this "
        f"player as {verdict}. Using ONLY the facts below, explain the main reasons that "
        "support that rating (form, fixture, value). Stay consistent with the rating: for a "
        "player the model rates highly, make the positive case and mention at most one genuine "
        "risk briefly, without leading with the word 'risky' or overstating the downside. Do "
        "not contradict the model's verdict. Do not invent stats. Do not use em dashes.\n\n"
        f"{_facts(player)}\n\n{style}"
    )


def _fallback(player: dict, variant: str) -> str:
    name = player.get("web_name", "This player")
    pts = round(float(player.get("pred_points", 0) or 0), 1)
    price = player.get("price", "?")
    home = "at home" if player.get("is_home", 0) else "away"
    own = player.get("owned_pct", 0)
    one = f"{name} projects for about {pts} pts {home} at {price}m, a {'strong' if pts >= 5 else 'modest'} option this week."
    if variant == "concise":
        return one
    tail = (f" Ownership is {own}%, so he's {'a popular pick' if (own or 0) >= 15 else 'a potential differential'}. "
            f"The model likes the {'matchup and recent form' if pts >= 5 else 'price more than the ceiling'} here.")
    return one + tail


def _reckoning_facts(r: dict) -> str:
    cap = r.get("captain") or {}
    lines = [
        f"Upcoming gameweek: GW{r.get('next_gw')}",
        f"Free transfers available: {r.get('free_transfers')}",
        f"Recommended transfers: {r.get('transfers_made')} "
        f"({r.get('paid_transfers')} paid, costing {r.get('hit_cost')} points)",
        f"Recommended captain: {cap.get('web_name', '?')} "
        f"(projected {cap.get('pred_points', '?')} next gameweek)",
    ]
    for m in r.get("moves", []):
        lines.append(f"Transfer out {m['out']['web_name']} for {m['in']['web_name']} "
                     f"(+{m['gain']} projected over the horizon)")
    lines.append(f"Net projected gain after any hits: {r.get('net_gain')} points "
                 f"over the next {r.get('horizon_weeks')} gameweeks")
    for c in r.get("chips", []):
        lines.append(f"Chip {c['chip']}: {c['note']}")
    return "\n".join(lines)


def _reckoning_fallback(r: dict) -> str:
    cap = r.get("captain") or {}
    n = r.get("transfers_made", 0)
    if n == 0:
        move = "Hold your team — no transfer beats it this week."
    else:
        parts = [f"{m['out']['web_name']} out for {m['in']['web_name']}" for m in r.get("moves", [])]
        hit = r.get("hit_cost", 0)
        tail = f" for a net {r.get('net_gain')} points after the {hit}-point hit." if hit else \
               f", projected to add {r.get('net_gain')} points."
        move = f"Make {n} move{'s' if n > 1 else ''}: " + ", ".join(parts) + tail
    capname = cap.get("web_name", "your best starter")
    return f"Captain {capname}. {move}"


def explain_reckoning(rec: dict) -> dict:
    """A short, brand-voice write-up of the solver's plan. Falls back with no API key."""
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        return {"text": _reckoning_fallback(rec), "source": "fallback"}
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=key)
        prompt = (
            "You are LokiFPL's analyst — sharp, a little mischievous, but genuinely useful. "
            "Using ONLY the facts below, write a 2 to 4 sentence verdict on this manager's "
            "gameweek: who to captain, the transfer plan and whether any hit is worth it, and a "
            "word on chips if relevant. Be decisive and consistent with the numbers. Do not invent "
            "stats. Do not use em dashes.\n\n" + _reckoning_facts(rec)
        )
        msg = client.messages.create(model=MODEL, max_tokens=220,
                                     messages=[{"role": "user", "content": prompt}])
        text = "".join(getattr(b, "text", "") for b in msg.content).strip()
        return {"text": text or _reckoning_fallback(rec), "source": "claude"}
    except Exception as e:  # noqa: BLE001
        print("LLM_ERROR", type(e).__name__, str(e)[:300], flush=True)
        return {"text": _reckoning_fallback(rec), "source": "fallback"}


def _profile_facts(prof: dict) -> str:
    h = prof.get("header", {}) or {}
    s = prof.get("season", {}) or {}
    proj = prof.get("projection") or {}
    pos = h.get("position", "?")

    def g(d, k, default="n/a"):
        v = d.get(k)
        return default if v is None else v

    lines = [
        f"Name: {g(h, 'web_name')}",
        f"Position: {pos}",
        f"Club: {g(h, 'team_name')}",
        f"Price: {g(h, 'price')}m",
        f"Ownership: {g(h, 'owned_pct')}%",
        f"Availability: status={prof.get('status', '?')}, news={prof.get('news') or 'none'}",
        f"Games played this season: {prof.get('games_played', 0)} (starts: {g(s, 'starts')})",
        f"Points this season: {g(s, 'total_points')} (per game: {g(s, 'points_per_game')}, form: {g(s, 'form')})",
        f"Minutes: {g(s, 'minutes')}",
        f"Goals: {g(s, 'goals_scored')}, Assists: {g(s, 'assists')}",
        f"Expected: xG {g(s, 'expected_goals')}, xA {g(s, 'expected_assists')}, xGI {g(s, 'expected_goal_involvements')}",
        f"Defensive contribution (DefCon) total: {g(s, 'defensive_contribution')}",
        f"Clean sheets: {g(s, 'clean_sheets')}, xG conceded: {g(s, 'expected_goals_conceded')}",
        f"Bonus: {g(s, 'bonus')}, ICT index: {g(s, 'ict_index')}",
        f"Recent points (last games, oldest to newest): {prof.get('form_last5_points') or 'n/a'}",
    ]

    # Per-90 rates (fair comparison across differing minutes).
    p90 = prof.get("per90") or {}
    if any(v is not None for v in p90.values()):
        lines.append(f"Per 90: xG {g(p90, 'xg')}, xA {g(p90, 'xa')}, "
                     f"xGI {g(p90, 'xgi')}, DefCon {g(p90, 'defcon')}")

    # Value.
    if prof.get("value") is not None:
        lines.append(f"Value: {prof.get('value')} points per million")

    # Set-piece and penalty duty (order 1 = first-choice taker).
    sp = prof.get("set_pieces") or {}
    roles = []
    if sp.get("pens") == 1:
        roles.append("first-choice penalty taker")
    elif sp.get("pens") == 2:
        roles.append("second penalty taker")
    if sp.get("fk") == 1:
        roles.append("takes direct free kicks")
    if sp.get("corners") == 1:
        roles.append("takes corners")
    lines.append("Set-piece duty: " + (", ".join(roles) if roles else "no set-piece duty"))

    # Price and transfer momentum.
    pr = prof.get("price") or {}
    if pr:
        ce = pr.get("change_event")
        move = "rising" if (ce or 0) > 0 else ("falling" if (ce or 0) < 0 else "steady")
        lines.append(f"Price momentum: {move} this week (change {ce}m), "
                     f"net transfers {pr.get('net_transfers')} this gameweek")

    if proj:
        opp = proj.get("opp")
        where = "home" if proj.get("home") else "away"
        z = proj.get("opp_def_z")
        if z is None:
            strength = ""
        elif z >= 0.6:
            strength = ", a strong defence"
        elif z <= -0.6:
            strength = ", a weak defence"
        else:
            strength = ", an average defence"
        lines.append(f"Model projection next gameweek (GW{proj.get('next_gw')}): "
                     f"{proj.get('pred_points')} points vs {opp} ({where}{strength})")
    fx = prof.get("upcoming") or []
    if fx:
        runs = ", ".join(f"GW{f.get('gw')} {f.get('opp')} "
                         f"({'H' if f.get('home') else 'A'}, FDR {f.get('difficulty')})"
                         for f in fx[:5])
        lines.append(f"Upcoming fixtures (with difficulty 1 easy to 5 hard): {runs}")
    return "\n".join(lines)


def explain_profile(prof: dict) -> dict:
    """A richer, analyst-style read on a single player for the profile view.

    Covers form and underlying numbers, minutes/role security, the fixture run,
    value and defensive contribution, plus one genuine risk. Falls back to a
    templated note with no API key. Returns {'text': str, 'source': ...}."""
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        return {"text": _fallback(
            {**prof.get("header", {}),
             "pred_points": (prof.get("projection") or {}).get("pred_points", 0),
             "is_home": (prof.get("projection") or {}).get("home", 0)},
            "detailed"), "source": "fallback"}
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=key)
        prompt = (
            "You are LokiFPL's analyst, writing for a Fantasy Premier League manager who is smart "
            "but not a stats expert and is deciding whether to buy, hold, captain or avoid this "
            "player. Using ONLY the facts below, write a tight 5 to 6 sentence read that TEACHES "
            "the numbers as you reason, so the reader understands WHY each stat matters. Keep it "
            "concise: cover only what actually changes the decision, no filler. Priorities:\n"
            "- Compare xG and xA to his actual goals and assists to flag overperformance (may cool "
            "off) or underperformance (may be due).\n"
            "- Use the ICT index as a plain-English gauge of how involved he is in his team's "
            "attack, and say if it is high or low for his position.\n"
            "- If he has set-piece or penalty duty, say so early, because it raises his ceiling; if "
            "he has none, do not dwell on it.\n"
            "- Use defensive contribution (DefCon) to describe his points floor, especially for "
            "defenders and midfielders.\n"
            "- Judge minutes and starts for game-time security, and read the fixture run (difficulty "
            "1 easy to 5 hard), including how strong the next opponent's defence is.\n"
            "- If the price is clearly rising or falling this week, mention it as a timing nudge.\n"
            "- Use price, value (points per million) and ownership to frame him as essential, a "
            "solid pick, or a differential.\n"
            "End with a clear verdict (buy, hold, captain option, or avoid) and the single biggest "
            "risk. Be decisive, quote the key numbers, and do not invent stats not given. Do not "
            "use em dashes.\n\n" + _profile_facts(prof)
        )
        msg = client.messages.create(model=MODEL, max_tokens=440,
                                     messages=[{"role": "user", "content": prompt}])
        text = "".join(getattr(b, "text", "") for b in msg.content).strip()
        return {"text": text or _fallback(prof.get("header", {}), "detailed"), "source": "claude"}
    except Exception as e:  # noqa: BLE001
        print("LLM_ERROR", type(e).__name__, str(e)[:300], flush=True)
        return {"text": _fallback(prof.get("header", {}), "detailed"), "source": "fallback"}


def explain(player: dict, variant: str = "concise") -> dict:
    """Return {'text': str, 'variant': str, 'source': 'claude'|'fallback'}."""
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        return {"text": _fallback(player, variant), "variant": variant, "source": "fallback"}
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=key)
        msg = client.messages.create(
            model=MODEL,
            max_tokens=160,
            messages=[{"role": "user", "content": _prompt(player, variant)}],
        )
        text = "".join(getattr(b, "text", "") for b in msg.content).strip()
        return {"text": text or _fallback(player, variant), "variant": variant, "source": "claude"}
    except Exception as e:  # noqa: BLE001 - never let the explainer break the request
        print("LLM_ERROR", type(e).__name__, str(e)[:300], flush=True)
        return {"text": _fallback(player, variant), "variant": variant, "source": "fallback"}
