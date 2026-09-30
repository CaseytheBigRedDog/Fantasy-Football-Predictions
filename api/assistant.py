"""
assistant.py - the "who should I start?" assistant.

It is GROUNDED in your own projections: every number it says comes from the weekly
predictions CSV written by predict_week.py. Two modes:

  * "claude"  - if ANTHROPIC_API_KEY is set, Claude answers your question in plain English,
                but it can only get numbers by calling the two lookup tools below, which read
                your CSV. It is told not to use any outside knowledge about stats or injuries.
  * "rules"   - with no key (or if the Claude call fails), a simple built-in parser finds the
                players you named and compares them. No AI, no cost, always works.

Either way the players and numbers used are returned, so the website shows exactly what the
answer was based on.
"""
import json
import os
import re

import projections as pj

MODEL = os.environ.get("ASSISTANT_MODEL", "claude-sonnet-5-5")
MAX_TOOL_ROUNDS = 5
CLOSE_CALL = 1.0  # points: under this gap we call it a toss-up

POSITION_WORDS = {
    "qb": "QB", "quarterback": "QB", "quarterbacks": "QB", "qbs": "QB",
    "rb": "RB", "running": "RB", "rbs": "RB", "runningback": "RB",
    "wr": "WR", "receiver": "WR", "receivers": "WR", "wrs": "WR",
    "te": "TE", "tight": "TE", "tes": "TE",
}

SHOWN_FIELDS = [
    "player_id", "player", "position", "team", "opponent", "is_home", "total_line",
    "floor", "median", "ceiling", "expected", "expected_if_active", "p_play", "status", "depth",
]


def _slim(row):
    return {k: row.get(k) for k in SHOWN_FIELDS}


def describe(row):
    where = "at" if row.get("is_home") == 0 else "vs"
    text = (
        f"{row['player']} ({row['position']}, {row['team']} {where} {row['opponent']}): "
        f"expected {row['expected']:.1f}, floor {row['floor']:.1f}, "
        f"median {row['median']:.1f}, ceiling {row['ceiling']:.1f}"
    )
    return text


def injury_note(row):
    p_play = row.get("p_play")
    status = row.get("status") or ""
    if p_play is not None and p_play < 0.99:
        label = status if status else "uncertain"
        return (
            f"{row['player']} is listed {label} (about {p_play:.0%} chance to play), "
            f"so the expected number is already shrunk; if he plays it is "
            f"{row['expected_if_active']:.1f}."
        )
    if status:  # practice-only flag: shown, but it does not change the numbers
        return f"{row['player']} has a practice flag ({status}); it is shown but does not change the projection."
    return None


def compare(rows):
    """Rank players by expected points and write a plain-English verdict."""
    ranked = sorted(rows, key=lambda r: r["expected"], reverse=True)
    best, second = ranked[0], ranked[1]
    gap = best["expected"] - second["expected"]
    lines = []

    if gap < CLOSE_CALL:
        contenders = [r for r in ranked if best["expected"] - r["expected"] < CLOSE_CALL]
        names = [r["player"] for r in contenders]
        joined = names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]
        spread = best["expected"] - contenders[-1]["expected"]
        lines.append(f"Toss-up: {joined} are within {spread:.1f} points of each other.")

        by_floor = max(contenders, key=lambda r: r["floor"])
        by_ceiling = max(contenders, key=lambda r: r["ceiling"])
        extras = []
        if by_floor["player_id"] != best["player_id"]:
            extras.append(
                f"{by_floor['player']} has the higher floor ({by_floor['floor']:.1f}), "
                f"the safer pick if you are the favorite this week"
            )
        if by_ceiling["player_id"] != best["player_id"]:
            extras.append(
                f"{by_ceiling['player']} has the higher ceiling ({by_ceiling['ceiling']:.1f}), "
                f"the bigger swing if you are the underdog"
            )
        if extras:
            lines.append(f"Lean {best['player']} on the numbers, but " + "; ".join(extras) + ".")
        else:
            lines.append(
                f"Lean {best['player']}: he is also highest in floor and ceiling among them."
            )
    else:
        lines.append(f"Start {best['player']}. The projections favor him by {gap:.1f} points.")

    lines.append("Projections (full PPR):")
    lines.extend("- " + describe(r) for r in ranked)
    for r in ranked:
        note = injury_note(r)
        if note:
            lines.append("Heads up: " + note)
    return "\n".join(lines), ranked


def _position_in(question):
    for word in re.findall(r"[a-z]+", question.lower()):
        if word in POSITION_WORDS:
            return POSITION_WORDS[word]
    return None


def answer_rules(question):
    """No-AI fallback: find the named players and compare them."""
    rows, ambiguous = pj.find_players_in_text(question)

    if ambiguous and len(rows) < 2:
        parts = []
        for last, options in ambiguous.items():
            names = ", ".join(f"{o['player']} ({o['position']}, {o['team']})" for o in options)
            parts.append(f"More than one player matches '{last}': {names}. Please use the full name.")
        return " ".join(parts), []

    if len(rows) >= 2:
        text, ranked = compare(rows)
        for last, options in ambiguous.items():  # a name we could not pin down: say so, don't drop it silently
            names = ", ".join(f"{o['player']} ({o['position']}, {o['team']})" for o in options)
            text += f"\nNote: '{last}' matches more than one player ({names}), so I left it out. Use the full name to include one."
        return text, ranked

    if len(rows) == 1:
        r = rows[0]
        text = "I only found one player in your question, so there is nothing to compare yet.\n"
        text += describe(r)
        note = injury_note(r)
        if note:
            text += "\nHeads up: " + note
        text += "\nName a second player (for example 'Start " + r["player"] + " or ...?')."
        return text, rows

    position = _position_in(question)
    if position or re.search(r"\b(best|top)\b", question.lower()):
        count = re.search(r"\b(?:top|best)\s+(\d+)\b", question.lower())
        n = max(1, min(int(count.group(1)), 25)) if count else 5
        top_rows = pj.top(position=position, n=n)
        label = position or "players overall"
        text = f"Top {len(top_rows)} {label} by expected points this week:\n"
        text += "\n".join(f"{i}. " + describe(r) for i, r in enumerate(top_rows, 1))
        return text, top_rows

    return (
        "I could not find any players in your question. Try something like "
        "'Should I start Gibbs or Chase?' or 'Who are the top 5 running backs?'. "
        "(Full natural-language answers turn on when an ANTHROPIC_API_KEY is set.)",
        [],
    )


# ---------------------------------------------------------------- Claude mode

SYSTEM_PROMPT = """You are a fantasy football start/sit assistant for one person's own prediction model.
Scoring is full PPR. Week {week} of {season}.

Rules:
- Every number you state MUST come from your tools (lookup_players, top_players). Never use outside
  knowledge for stats, injuries, news, matchups, or rankings, and never guess a number.
- If a player is not found, say so plainly and suggest a spelling. If a name is ambiguous, ask which one.
- 'expected' is the average outcome (compare this first), 'median' is the typical game, 'floor' is a
  realistic bad game (10th percentile), 'ceiling' is a great game (90th percentile).
  'p_play' and 'status' show injury risk; expected already includes it, expected_if_active does not.
- Give a clear recommendation first, then 2-4 short sentences of reasoning. If two players are within
  about 1 point of expected, call it a toss-up and explain the floor/ceiling tradeoff.
- Mention that you only see the model's projections (no breaking news) when it matters.
- Keep answers under about 150 words. Plain text, no tables."""

TOOLS = [
    {
        "name": "lookup_players",
        "description": "Look up the projections for players by name. Accepts full names, last names or "
                       "slightly misspelled names. Returns matching rows (several rows = ambiguous).",
        "input_schema": {
            "type": "object",
            "properties": {"names": {"type": "array", "items": {"type": "string"}}},
            "required": ["names"],
        },
    },
    {
        "name": "top_players",
        "description": "List the highest-projected players this week by expected points, optionally "
                       "filtered by position (QB, RB, WR, TE) and/or team abbreviation (e.g. DET).",
        "input_schema": {
            "type": "object",
            "properties": {
                "position": {"type": "string", "enum": ["QB", "RB", "WR", "TE"]},
                "team": {"type": "string"},
                "n": {"type": "integer", "description": "How many players (default 10, max 25)."},
            },
        },
    },
]


def _run_tool(name, args, seen):
    if name == "lookup_players":
        result = {}
        for typed in args.get("names", [])[:10]:
            rows = [_slim(r) for r in pj.lookup(typed)]
            if len(rows) == 1:  # only show confident matches on the website
                seen[rows[0]["player_id"]] = rows[0]
            result[typed] = rows if rows else "not found"
        return result
    if name == "top_players":
        n = max(1, min(int(args.get("n") or 10), 25))
        rows = [_slim(r) for r in pj.top(args.get("position"), args.get("team"), n)]
        for r in rows:
            seen[r["player_id"]] = r
        return rows
    return {"error": f"unknown tool {name}"}


def answer_claude(question):
    import anthropic  # imported here so the rules mode works even without the package

    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
    info = pj.meta()
    system = SYSTEM_PROMPT.format(week=info["week"], season=info["season"])
    messages = [{"role": "user", "content": question}]
    seen = {}

    for _ in range(MAX_TOOL_ROUNDS):
        response = client.messages.create(
            model=MODEL, max_tokens=700, system=system, tools=TOOLS, messages=messages
        )
        if response.stop_reason != "tool_use":
            text = "".join(b.text for b in response.content if b.type == "text").strip()
            return text, list(seen.values())

        messages.append({"role": "assistant", "content": response.content})
        results = []
        for block in response.content:
            if block.type == "tool_use":
                output = _run_tool(block.name, block.input, seen)
                results.append(
                    {"type": "tool_result", "tool_use_id": block.id, "content": json.dumps(output)}
                )
        messages.append({"role": "user", "content": results})

    return "Sorry, that question needed too many lookups. Try asking about fewer players.", list(seen.values())


def ask(question):
    """Returns {"answer": str, "players": [rows used], "mode": "claude" | "rules", "note": str|None}."""
    question = question.strip()
    note = None
    if os.environ.get("ANTHROPIC_API_KEY"):
        try:
            text, players = answer_claude(question)
            return {"answer": text, "players": players, "mode": "claude", "note": None}
        except Exception as error:  # network, bad key, rate limit... fall back instead of failing
            note = f"Claude was unavailable ({type(error).__name__}), so this used the built-in comparison."
    text, players = answer_rules(question)
    return {"answer": text, "players": [_slim(p) for p in players], "mode": "rules", "note": note}
