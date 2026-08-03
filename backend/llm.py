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


def _prompt(player: dict, variant: str) -> str:
    style = ("Answer in ONE punchy sentence."
             if variant == "concise" else
             "Answer in two or three sentences, explaining the key reasons.")
    return (
        "You are an assistant inside a Fantasy Premier League tool. Using ONLY the "
        "facts below, explain to a manager whether this player is a good pick for the "
        "upcoming gameweek and why. Be specific about form, fixture and value. Do not "
        "invent stats. Do not use em dashes.\n\n"
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
