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
