"""
add_ask_feature.py

Adds a "Start/Sit" question box to the public website by patching build_site.py.

Run it ONCE, from the project folder (the one that contains build_site.py):

    python add_ask_feature.py
    python build_site.py

Then open docs/index.html in your browser to try it before you publish anything.

What it changes in build_site.py (a backup is saved first as build_site.py.bak_ask):
  * adds a small extra field ("eia", expected points if the player is active) to the page data
  * adds a "Start/Sit" section and a nav link
  * adds the JavaScript that answers questions like "Should I start X or Y?" and
    "Show me the top 5 wide receivers", using only the projections already on the page.
    It runs in the visitor's browser: no AI, no server, nothing is sent anywhere.

It is safe to run twice: it detects that the feature is already there and does nothing.
If it cannot find the exact spots it needs, it stops without changing anything.
"""
import shutil
import sys
from pathlib import Path

TARGET = Path("build_site.py")
MARK = "ASK-LOGIC-START"

# ------------------------------------------------------------------ pieces to insert

CSS = r'''/* Start/Sit question box */
#askform{display:flex;gap:8px;margin:0 0 10px}
#askq{flex:1;min-width:0;font:inherit;padding:9px 12px;border:1px solid var(--line);border-radius:8px;background:var(--bg);color:var(--text)}
#askgo{font:inherit;padding:9px 18px;border:1px solid var(--accent);border-radius:8px;background:var(--accent);color:#fff;cursor:pointer}
.chips{display:flex;flex-wrap:wrap;gap:8px;margin:0 0 12px}
.chips button{font:inherit;font-size:.88rem;padding:5px 12px;border:1px solid var(--line);border-radius:999px;background:var(--bg);color:var(--text);cursor:pointer}
.chips button:hover{background:var(--head)}
#askans{white-space:pre-wrap;margin:12px 0 0;padding:12px 14px;border-radius:8px;background:var(--head);overflow-wrap:anywhere}
#askans:empty{display:none}
@media (max-width:640px){#askform{flex-direction:column}}'''

NAV = '<a href="#ask">Start/Sit</a>\n'

SECTION = r'''<!-- Start/Sit question box -->
<section id="ask">
<h2>Start/sit questions</h2>
<div class="card">
<form id="askform" autocomplete="off">
<input id="askq" type="text" placeholder="Ask a question, e.g. Should I start one player or another?" aria-label="Ask a start/sit question" maxlength="200">
<button id="askgo" type="submit">Ask</button>
</form>
<div class="chips" id="askchips"></div>
<div id="askans" role="status" aria-live="polite"></div>
<p class="legend">Ask who to start, or for the top players at a position. Answers come straight from the projections on this page and are worked out in your browser: there is no AI, and nothing you type is sent anywhere. Players within 1 point of expected are flagged as a toss-up. The model has no breaking news, so check injury reports before you lock your lineup.</p>
</div>
</section>
'''

JS = r'''/* ASK-LOGIC-START  Start/Sit questions. Pure functions over DATA (no page access), so they can be tested on their own. */
const ASK = (function () {
  const SUFFIXES = ["jr", "sr", "ii", "iii", "iv", "v"];
  const POSITION_WORDS = {
    qb: "QB", quarterback: "QB", quarterbacks: "QB", qbs: "QB",
    rb: "RB", running: "RB", rbs: "RB", runningback: "RB",
    wr: "WR", receiver: "WR", receivers: "WR", wrs: "WR",
    te: "TE", tight: "TE", tes: "TE"
  };
  const CLOSE_CALL = 1.0;   // points: under this gap we call it a toss-up
  const f1 = n => Number(n).toFixed(1);
  const isSuffix = w => SUFFIXES.includes(w);

  // "Ja'Marr Chase" -> "jamarr chase"
  function normalize(text) {
    return String(text).normalize("NFKD").replace(/[^\x00-\x7F]/g, "").toLowerCase()
      .replace(/['.]/g, "").replace(/[^a-z0-9]+/g, " ").trim();
  }
  function bareName(key) { return key.split(" ").filter(w => w && !isSuffix(w)).join(" "); }
  function lastName(key) {
    const parts = key.split(" ").filter(w => w && !isSuffix(w));
    return parts.length ? parts[parts.length - 1] : "";
  }
  const PLAYERS = DATA.map(d => {
    const key = normalize(d.player);
    return Object.assign({}, d, { key: key, bare: bareName(key), last: lastName(key) });
  });

  function maxBy(rows, field) {
    let best = rows[0];
    for (const r of rows) if (r[field] > best[field]) best = r;
    return best;
  }

  function describe(r) {
    const where = r.home === 0 ? "at" : "vs";
    return `${r.player} (${r.pos}, ${r.team} ${where} ${r.opp}): expected ${f1(r.expected)}, floor ${f1(r.floor)}, median ${f1(r.median)}, ceiling ${f1(r.ceiling)}`;
  }

  function injuryNote(r) {
    const status = r.status || "";
    if (r.p < 0.99) {
      const label = status ? status : "uncertain";
      return `${r.player} is listed ${label} (about ${Math.round(r.p * 100)}% chance to play), so the expected number is already shrunk; if he plays it is ${f1(r.eia)}.`;
    }
    if (status) return `${r.player} has a practice flag (${status}); it is shown but does not change the projection.`;
    return null;
  }

  function compare(rows) {
    const ranked = rows.slice().sort((a, b) => b.expected - a.expected);
    const best = ranked[0], second = ranked[1];
    const gap = best.expected - second.expected;
    const lines = [];
    if (gap < CLOSE_CALL) {
      const contenders = ranked.filter(r => best.expected - r.expected < CLOSE_CALL);
      const names = contenders.map(r => r.player);
      const joined = names.length === 1 ? names[0] : names.slice(0, -1).join(", ") + " and " + names[names.length - 1];
      const spread = best.expected - contenders[contenders.length - 1].expected;
      lines.push(`Toss-up: ${joined} are within ${f1(spread)} points of each other.`);
      const byFloor = maxBy(contenders, "floor");
      const byCeiling = maxBy(contenders, "ceiling");
      const extras = [];
      if (byFloor.id !== best.id) extras.push(`${byFloor.player} has the higher floor (${f1(byFloor.floor)}), the safer pick if you are the favorite this week`);
      if (byCeiling.id !== best.id) extras.push(`${byCeiling.player} has the higher ceiling (${f1(byCeiling.ceiling)}), the bigger swing if you are the underdog`);
      if (extras.length) lines.push(`Lean ${best.player} on the numbers, but ` + extras.join("; ") + ".");
      else lines.push(`Lean ${best.player}: he is also highest in floor and ceiling among them.`);
    } else {
      lines.push(`Start ${best.player}. The projections favor him by ${f1(gap)} points.`);
    }
    lines.push("Projections (full PPR):");
    ranked.forEach(r => lines.push("- " + describe(r)));
    ranked.forEach(r => { const n = injuryNote(r); if (n) lines.push("Heads up: " + n); });
    return { text: lines.join("\n"), players: ranked };
  }

  // Finds players named inside a sentence. Returns the confident matches and any
  // ambiguous last names (two different Smiths) so we can ask which one was meant.
  function findPlayersInText(text) {
    const norm = normalize(text);
    const padded = " " + norm + " ";
    const matched = [];
    const usedLast = new Set();
    for (const p of PLAYERS) {
      if (padded.includes(" " + p.key + " ") || padded.includes(" " + p.bare + " ")) {
        matched.push(p.id);
        usedLast.add(p.last);
      }
    }
    const words = new Set(norm.split(" ").filter(Boolean));
    const groups = new Map();
    for (const p of PLAYERS) {
      if (!groups.has(p.last)) groups.set(p.last, []);
      groups.get(p.last).push(p);
    }
    const ambiguous = {};
    for (const last of Array.from(groups.keys()).sort()) {
      const group = groups.get(last);
      if (last.length < 3 || !words.has(last) || usedLast.has(last)) continue;
      if (group.length === 1) matched.push(group[0].id);
      else ambiguous[last] = group.slice().sort((a, b) => b.expected - a.expected).slice(0, 5);
    }
    const ids = Array.from(new Set(matched));
    const rows = ids.map(id => PLAYERS.find(p => p.id === id));
    return { rows: rows, ambiguous: ambiguous };
  }

  function positionIn(question) {
    const words = question.toLowerCase().match(/[a-z]+/g) || [];
    for (const w of words) if (Object.prototype.hasOwnProperty.call(POSITION_WORDS, w)) return POSITION_WORDS[w];
    return null;
  }

  function top(position, n) {
    return PLAYERS.filter(p => !position || p.pos === position).sort((a, b) => b.expected - a.expected).slice(0, n);
  }

  function answer(question) {
    const q = String(question).trim();
    const found = findPlayersInText(q);
    const rows = found.rows, ambiguous = found.ambiguous;
    const ambKeys = Object.keys(ambiguous);

    if (ambKeys.length && rows.length < 2) {
      const parts = ambKeys.map(last => `More than one player matches '${last}': ` +
        ambiguous[last].map(o => `${o.player} (${o.pos}, ${o.team})`).join(", ") + ". Please use the full name.");
      return { text: parts.join(" "), players: [] };
    }
    if (rows.length >= 2) {
      const result = compare(rows);
      for (const last of ambKeys) {   // a name we could not pin down: say so, don't drop it silently
        const names = ambiguous[last].map(o => `${o.player} (${o.pos}, ${o.team})`).join(", ");
        result.text += `\nNote: '${last}' matches more than one player (${names}), so I left it out. Use the full name to include one.`;
      }
      return result;
    }
    if (rows.length === 1) {
      const r = rows[0];
      let text = "I only found one player in your question, so there is nothing to compare yet.\n" + describe(r);
      const note = injuryNote(r);
      if (note) text += "\nHeads up: " + note;
      text += `\nName a second player (for example 'Start ${r.player} or ...?').`;
      return { text: text, players: rows };
    }
    const position = positionIn(q);
    if (position || /\b(best|top)\b/.test(q.toLowerCase())) {
      const count = q.toLowerCase().match(/\b(?:top|best)\s+(\d+)\b/);
      const n = count ? Math.max(1, Math.min(parseInt(count[1], 10), 25)) : 5;
      const rowsTop = top(position, n);
      const label = position || "players overall";
      const text = `Top ${rowsTop.length} ${label} by expected points this week:\n` +
        rowsTop.map((r, i) => `${i + 1}. ` + describe(r)).join("\n");
      return { text: text, players: rowsTop };
    }
    return {
      text: "I could not find any players in your question. Try something like 'Should I start Player A or Player B?' or 'Show me the top 5 wide receivers.'",
      players: []
    };
  }

  return { answer: answer, normalize: normalize };
})();
/* ASK-LOGIC-END */

/* Start/Sit box: wires the form to ASK.answer. The answer is shown as plain text (never as HTML). */
(function () {
  const form = $("#askform"), input = $("#askq"), out = $("#askans"), chips = $("#askchips");
  function run(text) {
    const q = text.trim();
    if (!q) return;
    out.textContent = ASK.answer(q).text;
  }
  form.addEventListener("submit", e => { e.preventDefault(); run(input.value); });

  // Example questions are built from this week's data, so they always name real, current players.
  const wr = DATA.filter(d => d.pos === "WR").sort((a, b) => a.prank - b.prank);
  const examples = [];
  if (wr.length > 9) {
    examples.push(`Should I start ${wr[8].player} or ${wr[9].player}?`);
    input.placeholder = `Ask a question, e.g. Should I start ${wr[8].player} or ${wr[9].player}?`;
  }
  examples.push("Show me the top 5 wide receivers", "Who are the top 3 running backs?");
  chips.innerHTML = examples.map(t => `<button type="button">${esc(t)}</button>`).join("");
  chips.querySelectorAll("button").forEach(b => b.addEventListener("click", () => {
    input.value = b.textContent;
    run(b.textContent);
  }));
})();'''

# (exact text to find, how to change it). Each anchor must appear exactly once.
EDITS = [
    # 1. page data: add expected-if-active so injury notes can show it
    ('"p": round(float(r["p_play"]), 2),',
     '"p": round(float(r["p_play"]), 2),\n'
     '    "eia": round(float(r["expected_if_active"]) if "expected_if_active" in pred.columns else float(r["expected"]), 1),'),
    # 2. styles
    ('.rmv:hover{text-decoration:underline}',
     '.rmv:hover{text-decoration:underline}\n' + CSS),
    # 3. nav link (before My Team)
    ('<a href="#myteam">My Team</a>',
     NAV + '<a href="#myteam">My Team</a>'),
    # 4. the section itself (before the My Team section)
    ('<!-- NEW: My Fantasy Team tab -->\n<section id="myteam">',
     SECTION + '\n<!-- NEW: My Fantasy Team tab -->\n<section id="myteam">'),
    # 5. the JavaScript (before the nav script)
    ('/* NEW: left-hand nav',
     JS + '\n\n/* NEW: left-hand nav'),
]


def main():
    if not TARGET.exists():
        sys.exit("build_site.py not found here. Run this from the project folder "
                 "(the one that contains build_site.py).")
    text = TARGET.read_text(encoding="utf-8")
    if MARK in text:
        print("The Start/Sit feature is already in build_site.py. Nothing to do.")
        return

    for anchor, _ in EDITS:
        count = text.count(anchor)
        if count != 1:
            sys.exit(f"Could not safely patch build_site.py: expected to find this exactly once "
                     f"but found it {count} times:\n    {anchor[:70]!r}\n"
                     "Nothing was changed. Send me this message and I will adjust the patch.")

    shutil.copyfile(TARGET, "build_site.py.bak_ask")
    for anchor, new in EDITS:
        text = text.replace(anchor, new, 1)
    TARGET.write_text(text, encoding="utf-8")
    print("Done. A backup was saved as build_site.py.bak_ask.")
    print("Next: run  python build_site.py  and open docs/index.html in your browser.")


if __name__ == "__main__":
    main()
