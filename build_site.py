"""
build_site.py

Builds the public website for GitHub Pages from your latest files:

    docs/index.html              the whole site (one self-contained page, no external services)
    docs/predictions_latest.csv  the current projections, for download
    docs/.nojekyll               tells GitHub Pages to serve the files as they are

It reads:
    predictions_<season>_week<N>.csv   the newest one (the _baseline copy is ignored)
    scored_<season>_week<N>.csv        the newest one, for the "last week's results" section
    accuracy_log.csv                   for the track record table and chart

Run it after predict_week.py (and after score_predictions.py on Tuesdays):

    python build_site.py

Then publish with:  git add .   git commit -m "Update site"   git push
"""
import glob
import html
import json
import os
import re
import shutil
import sys
from datetime import datetime

import numpy as np
import pandas as pd

# ---------------------------------------------------------------
# Settings you can edit
# ---------------------------------------------------------------
SITE_TITLE = "Fantasy Football Projections"
REPO_URL = "https://github.com/CaseytheBigRedDog/Fantasy-Football-Predictions"
AUTHOR = ""                        # optional: your name, shown in the footer (leave "" to hide)
SHOW_FANTASYPROS_COMPARISON = True # set False to hide the expert-consensus comparison table
OUT_DIR = "docs"

# Backtest results shown on the page. These are static: update them by hand after you
# retrain and the numbers in the pipeline output change.
BACKTEST = {
    "season": 2025,
    "models": [("Naive baseline (player's last 3 games)", 5.207), ("Ridge regression", 4.807),
               ("Gradient boosting", 4.795), ("XGBoost", 4.767)],
    "position_models": [("QB", 6.424), ("RB", 4.704), ("WR", 4.535), ("TE", 3.642)],
    "position_overall": 4.624,
    "fantasypros": [("QB", 0.412, 0.475), ("RB", 0.656, 0.694), ("WR", 0.568, 0.598), ("TE", 0.553, 0.568)],
    "fantasypros_n": 4104,
}

esc = html.escape


def find_latest(pattern, regex):
    found = []
    for fn in glob.glob(pattern):
        m = re.search(regex, fn)
        if m:
            found.append(((int(m.group(1)), int(m.group(2))), fn))
    return max(found) if found else ((None, None), None)


# ---------------------------------------------------------------
# 1. Projections
# ---------------------------------------------------------------
(season, week), pred_path = find_latest("predictions_*_week*.csv", r"predictions_(\d{4})_week(\d+)\.csv$")
if pred_path is None:
    sys.exit("No predictions file found. Run predict_week.py first.")

pred = pd.read_csv(pred_path)
rank_col = "expected" if "expected" in pred.columns else "median"
if rank_col == "median":
    pred["expected"] = pred["median"]
pred["is_home"] = pred["is_home"].fillna(0)
has_inj = "status" in pred.columns
pred["status"] = pred["status"].fillna("").astype(str) if has_inj else ""
pred["depth"] = pred["depth"].fillna("").astype(str) if "depth" in pred.columns else ""
pred["p_play"] = pred["p_play"].fillna(1.0) if "p_play" in pred.columns else 1.0
pred["pos_rank"] = pred.groupby("position")["expected"].rank(ascending=False, method="first").astype(int)

records = [{
    "player": r["player"], "pos": r["position"], "team": r["team"], "opp": r["opponent"],
    "home": int(r["is_home"]), "floor": round(float(r["floor"]), 1), "median": round(float(r["median"]), 1),
    "expected": round(float(r["expected"]), 1), "ceiling": round(float(r["ceiling"]), 1),
    "prank": int(r["pos_rank"]), "status": r["status"], "depth": r["depth"], "p": round(float(r["p_play"]), 2),
} for _, r in pred.iterrows()]
n_out = int((pred["p_play"] <= 0.02).sum())
n_q = int(pred["status"].str.startswith("Questionable").sum())
n_prov = int(pred["status"].str.startswith("Practice").sum())
if has_inj and (n_out + n_q + n_prov):
    injury_note = (f'<p class="legend"><strong>Injury report:</strong> {n_out} player{"s" if n_out != 1 else ""} listed Out or Doubtful '
                   f'(shown at 0.0 and sorted to the bottom; choose "all" to see them), {n_q} Questionable (projection reduced by the '
                   f'chance they don\'t play), and {n_prov} with practice flags only (shown, but not applied, because early-week practice '
                   f'status is unreliable).</p>')
else:
    injury_note = ""
if has_inj:
    notice_html = ('<strong>Official injury designations are included</strong> (Out, Doubtful, Questionable), but teams publish them late in '
                   'the week, so early-week projections may not reflect them, and late scratches and breaking news are not included. '
                   'This is a hobby project for learning; it is not betting or financial advice.')
    how_inj = ('<li><strong>Injuries and depth charts:</strong> an official designation changes a projection by the historical chance that '
               'players with that designation (and practice status) actually played. Depth-chart slots are shown as context only, because '
               'the data source changed in 2025 and there is too little history to learn from.</li>')
    limit_inj = ('<li>Injury designations arrive late in the week, and late scratches and news are unknowable in advance. '
                 'Depth charts are context only; the model does not learn from them.</li>')
else:
    notice_html = ('<strong>Injuries and inactives are not included.</strong> A player who got hurt still gets a projection based on his '
                   'recent form. This is a hobby project for learning; it is not betting or financial advice.')
    how_inj = ""
    limit_inj = "<li>No injury, inactive or depth-chart information (the biggest gap).</li>"
data_json = json.dumps(records, separators=(",", ":")).replace("</", "<\\/")

# ---------------------------------------------------------------
# 2. Track record (from accuracy_log.csv)
# ---------------------------------------------------------------
def line_chart(weeks, series):
    """Small inline SVG line chart. series = [(label, values, css_class)]."""
    W, H, L, R, T, B = 640, 250, 46, 16, 16, 44
    allv = [v for _, vals, _ in series for v in vals]
    lo, hi = min(allv), max(allv)
    pad = max((hi - lo) * 0.25, 0.3)
    lo, hi = max(lo - pad, 0), hi + pad
    x = lambda i: L + (W - L - R) * (i / max(len(weeks) - 1, 1))
    y = lambda v: T + (H - T - B) * (1 - (v - lo) / (hi - lo))
    parts = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Average error by week, model vs naive baseline" class="chart">']
    for k in range(5):
        v = lo + (hi - lo) * k / 4
        parts.append(f'<line x1="{L}" x2="{W - R}" y1="{y(v):.1f}" y2="{y(v):.1f}" class="grid"/>')
        parts.append(f'<text x="{L - 6}" y="{y(v) + 4:.1f}" class="tick" text-anchor="end">{v:.1f}</text>')
    for i, wk in enumerate(weeks):
        parts.append(f'<text x="{x(i):.1f}" y="{H - B + 18}" class="tick" text-anchor="middle">Wk {wk}</text>')
    for label, vals, cls in series:
        pts = " ".join(f"{x(i):.1f},{y(v):.1f}" for i, v in enumerate(vals))
        parts.append(f'<polyline points="{pts}" class="line {cls}"/>')
        for i, v in enumerate(vals):
            parts.append(f'<circle cx="{x(i):.1f}" cy="{y(v):.1f}" r="3.5" class="dot {cls}"><title>{esc(label)}, week {weeks[i]}: {v:.2f}</title></circle>')
    lx = L
    for label, _, cls in series:
        parts.append(f'<rect x="{lx}" y="{H - 14}" width="12" height="4" class="key {cls}"/>')
        parts.append(f'<text x="{lx + 18}" y="{H - 8}" class="tick">{esc(label)}</text>')
        lx += 250
    parts.append("</svg>")
    return "".join(parts)


track_html = ('<p class="muted">Graded results appear here after each week\'s games are final. '
              'Tracking starts with Week 3 of the 2026 season; every prediction is committed to '
              'GitHub before kickoff.</p>')
if os.path.exists("accuracy_log.csv"):
    log = pd.read_csv("accuracy_log.csv")
    allrows = log[log["position"] == "ALL"].sort_values(["season", "week"])
    allrows = allrows[allrows["season"] == season] if (allrows["season"] == season).any() else allrows
    if len(allrows):
        n_tot = allrows["n"].sum()
        wavg = lambda c: float((allrows[c] * allrows["n"]).sum() / n_tot)
        mae_m, mae_n, inr = wavg("model_mae"), wavg("naive_mae"), wavg("in_range")
        better = "better than" if mae_m < mae_n else "not better than"
        track_html = (
            f'<p>Season to date ({len(allrows)} graded week{"s" if len(allrows) != 1 else ""}, '
            f'{int(n_tot):,} player-weeks): the projections\' typical miss is <strong>{mae_m:.2f} points</strong>, '
            f'{better} the naive last-3-games guess ({mae_n:.2f}), and <strong>{inr:.0%}</strong> of results '
            f'landed inside the floor-to-ceiling range (target 80%).</p>')
        rows = []
        for _, r in allrows.iterrows():
            win = "Yes" if r["model_mae"] < r["naive_mae"] else "No"
            rows.append(f'<tr><td>Week {int(r["week"])}</td><td>{int(r["n"]):,}</td><td>{r["model_mae"]:.2f}</td>'
                        f'<td>{r["naive_mae"]:.2f}</td><td>{win}</td><td>{r["in_range"]:.0%}</td></tr>')
        track_html += ('<div class="scroll"><table class="plain"><thead><tr><th>Week</th><th>Players</th>'
                       '<th>Model error</th><th>Naive error</th><th>Beat naive?</th><th>In range</th></tr></thead>'
                       f'<tbody>{"".join(rows)}</tbody></table></div>')
        if len(allrows) >= 2:
            track_html += line_chart([int(w) for w in allrows["week"]], [
                ("Model", [float(v) for v in allrows["model_mae"]], "c1"),
                ("Naive last-3-games guess", [float(v) for v in allrows["naive_mae"]], "c2")])
        track_html += '<p class="muted small">Error = average points off per player-week (lower is better).</p>'

# ---------------------------------------------------------------
# 3. Last graded week
# ---------------------------------------------------------------
last_html = '<p class="muted">Last week\'s results appear here once a week has been graded.</p>'
(ls, lw), scored_path = find_latest("scored_*_week*.csv", r"scored_(\d{4})_week(\d+)\.csv$")
if scored_path:
    sc = pd.read_csv(scored_path)
    sc = sc[sc["status"] == "scored"].copy()
    prior_path = f"predictions_{ls}_week{lw}.csv"
    if os.path.exists(prior_path):
        pp = pd.read_csv(prior_path)
        if "expected" in pp.columns:
            sc = sc.merge(pp[["player", "position", "team", "expected"]], on=["player", "position", "team"], how="left")
    if "expected" not in sc.columns:
        sc["expected"] = sc["median"]
    sc["expected"] = sc["expected"].fillna(sc["median"])
    if len(sc):
        inside = ((sc["actual"] >= sc["floor"]) & (sc["actual"] <= sc["ceiling"])).mean()
        top = sc.sort_values("expected", ascending=False).head(20)
        rows = []
        for _, r in top.iterrows():
            if r["actual"] > r["ceiling"]:
                tag, cls = "Above ceiling", "hi"
            elif r["actual"] < r["floor"]:
                tag, cls = "Below floor", "lo"
            else:
                tag, cls = "In range", "ok"
            rows.append(f'<tr><td>{esc(str(r["player"]))}</td><td>{esc(str(r["position"]))}</td>'
                        f'<td>{esc(str(r["team"]))} vs {esc(str(r["opponent"]))}</td>'
                        f'<td>{r["floor"]:.1f} &ndash; {r["ceiling"]:.1f}</td><td>{r["expected"]:.1f}</td>'
                        f'<td><strong>{r["actual"]:.1f}</strong></td><td class="{cls}">{tag}</td></tr>')
        last_html = (f'<p>Week {lw}, {ls}: the 20 highest-projected players. Across all {len(sc):,} graded '
                     f'players, <strong>{inside:.0%}</strong> landed inside their floor-to-ceiling range.</p>'
                     '<div class="scroll"><table class="plain"><thead><tr><th>Player</th><th>Pos</th><th>Matchup</th>'
                     '<th>Range</th><th>Expected</th><th>Actual</th><th>Result</th></tr></thead>'
                     f'<tbody>{"".join(rows)}</tbody></table></div>')

# ---------------------------------------------------------------
# 4. Backtest tables (static)
# ---------------------------------------------------------------
b = BACKTEST
naive = b["models"][0][1]
bt_rows = "".join(f"<tr><td>{esc(n)}</td><td>{v:.3f}</td></tr>" for n, v in b["models"])
bt_rows += f"<tr><td>Position-specific models (overall)</td><td>{b['position_overall']:.3f}</td></tr>"
xgb = b["models"][-1][1]
backtest_html = (
    f'<p>Trained on 2013&ndash;2023, tuned on 2024, tested on the full <strong>{b["season"]}</strong> season '
    f'(about 5,200 player-weeks) the model never saw. Average error, in fantasy points per player-week:</p>'
    f'<div class="scroll"><table class="plain narrow"><thead><tr><th>Model</th><th>Average error</th></tr></thead>'
    f'<tbody>{bt_rows}</tbody></table></div>'
    f'<p>That is about <strong>{(naive - xgb) / naive:.0%}</strong> better than the naive guess for the main model and '
    f'<strong>{(naive - b["position_overall"]) / naive:.0%}</strong> for the position-specific models. Across 2023&ndash;2025, '
    f'about 80% of results landed inside the 10th&ndash;90th percentile range, as intended.</p>')
if SHOW_FANTASYPROS_COMPARISON:
    fp_rows = "".join(f"<tr><td>{p}</td><td>{a:.3f}</td><td>{f:.3f}</td></tr>" for p, a, f in b["fantasypros"])
    backtest_html += (
        f'<h3>Versus expert consensus</h3><p>How well each ranks players against actual results (rank correlation, '
        f'{b["season"]} season, {b["fantasypros_n"]:,} matched player-weeks; higher is better). The benchmark is '
        f'<a href="https://www.fantasypros.com/">FantasyPros</a> expert consensus rankings, used only as a yardstick.</p>'
        f'<div class="scroll"><table class="plain narrow"><thead><tr><th>Position</th><th>This model</th>'
        f'<th>Expert consensus</th></tr></thead><tbody>{fp_rows}</tbody></table></div>'
        f'<p><strong>The model does not beat the experts</strong>: they are ahead at every position. One reason is '
        f'timing: their rankings reflect late injury news, and this model has no injury information at all.</p>')

# ---------------------------------------------------------------
# 5. The page
# ---------------------------------------------------------------
TEMPLATE = r'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>%%TITLE%%</title>
<meta name="description" content="Weekly PPR fantasy football projections with floor, median, ceiling and expected points, plus a public accuracy track record. Built with Python, pandas and XGBoost.">
<meta property="og:title" content="%%TITLE%%">
<meta property="og:description" content="Weekly fantasy football projections with honest accuracy tracking. Built with Python, pandas and XGBoost.">
<meta property="og:type" content="website">
<style>
:root{--bg:#f7f8fa;--card:#fff;--text:#1a2332;--muted:#5a6678;--line:#e2e6ec;--accent:#1d5fd1;--accent2:#8792a6;--ok:#1b7f3b;--hi:#0b6bcb;--lo:#b3261e;--warn:#a15c00;--bar:#c9d8f5;--head:#eef1f6}
@media (prefers-color-scheme:dark){:root{--bg:#0f141c;--card:#171e29;--text:#e7ebf2;--muted:#9aa6b8;--line:#2a3444;--accent:#6ea0ff;--accent2:#7f8ba0;--ok:#5fd17f;--hi:#7db4ff;--lo:#ff8a80;--warn:#f0b35a;--bar:#2b4272;--head:#1d2633}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--text);font:16px/1.55 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
a{color:var(--accent)}
header{background:var(--card);border-bottom:1px solid var(--line)}
.wrap{max-width:1080px;margin:0 auto;padding:0 18px}
header .wrap{padding-top:26px;padding-bottom:22px}
h1{margin:0 0 4px;font-size:1.9rem;letter-spacing:-.01em}
h2{margin:0 0 10px;font-size:1.35rem}
h3{margin:22px 0 6px;font-size:1.05rem}
.sub{margin:0;color:var(--muted)}
.notice{margin:18px 0 0;padding:12px 14px;border:1px solid var(--line);border-left:4px solid var(--lo);border-radius:8px;background:var(--bg);font-size:.95rem}
section{margin:26px 0}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:18px}
.muted{color:var(--muted)} .small{font-size:.86rem}
.controls{display:flex;flex-wrap:wrap;gap:10px;align-items:center;margin:0 0 12px}
.controls input,.controls select{font:inherit;padding:7px 10px;border:1px solid var(--line);border-radius:8px;background:var(--bg);color:var(--text)}
.pills{display:flex;gap:6px;flex-wrap:wrap}
.pills button{font:inherit;padding:6px 14px;border:1px solid var(--line);border-radius:999px;background:var(--bg);color:var(--text);cursor:pointer}
.pills button[aria-pressed="true"]{background:var(--accent);border-color:var(--accent);color:#fff}
.scroll{overflow-x:auto}
table{border-collapse:collapse;width:100%}
th,td{padding:8px 10px;text-align:right;white-space:nowrap;border-bottom:1px solid var(--line)}
th:nth-child(-n+3),td:nth-child(-n+3){text-align:left}
thead th{background:var(--head);position:sticky;top:0;font-size:.85rem}
th button{all:unset;cursor:pointer;font-weight:600}
th button:focus-visible,.pills button:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
th[aria-sort="ascending"] button::after{content:" \25B2";font-size:.7em}
th[aria-sort="descending"] button::after{content:" \25BC";font-size:.7em}
tbody tr:hover{background:var(--head)}
.plain th,.plain td{text-align:left}
.plain th:not(:first-child),.plain td:not(:first-child){text-align:right}
.plain.narrow{max-width:520px}
#proj td:nth-child(2){font-weight:600}
.rb{position:relative;display:inline-block;width:150px;height:10px;background:var(--head);border-radius:5px;vertical-align:middle}
.rr{position:absolute;top:0;height:10px;background:var(--bar);border-radius:5px}
.rm{position:absolute;top:-2px;width:2px;height:14px;background:var(--text)}
.re{position:absolute;top:0;width:10px;height:10px;margin-left:-5px;border-radius:50%;background:var(--accent);border:2px solid var(--card);box-sizing:content-box;top:-2px}
.legend{margin:10px 0 0;color:var(--muted);font-size:.86rem}
td.ok{color:var(--ok)} td.hi{color:var(--hi)} td.lo{color:var(--lo)}
.chart{width:100%;max-width:680px;height:auto;margin-top:12px}
.chart .grid{stroke:var(--line)} .chart .tick{fill:var(--muted);font-size:11px}
.chart .line{fill:none;stroke-width:2.5} .chart .c1{stroke:var(--accent);fill:var(--accent)} .chart .c2{stroke:var(--accent2);fill:var(--accent2)}
.chart .line.c1,.chart .line.c2{fill:none}
ul{padding-left:20px} li{margin:5px 0}
footer{margin:34px 0 40px;color:var(--muted);font-size:.9rem;overflow-wrap:anywhere}
.count{color:var(--muted);font-size:.9rem;margin-left:auto}
.mob{display:none;font-weight:400;font-size:.78rem;color:var(--muted)}
.dpt{color:var(--muted);font-weight:400;font-size:.78rem;margin-left:6px}
.tag{display:block;font-weight:500;font-size:.78rem}
.t-out{color:var(--lo)} .t-q{color:var(--warn)} .t-p{color:var(--muted)}
@media (max-width:640px){
  th:nth-child(3),td:nth-child(3),th:nth-child(4),td:nth-child(4),th:nth-child(5),td:nth-child(5),th:nth-child(7),td:nth-child(7),th:nth-child(8),td:nth-child(8){display:none}
  .mob{display:block}
  #proj th,#proj td{padding:8px 5px;font-size:.92rem}
  #proj td:nth-child(2){white-space:normal;min-width:104px}
  .card{padding:14px 10px}
  h1{font-size:1.55rem}
}
</style>
</head>
<body>
<header><div class="wrap">
<h1>%%HEADING%%</h1>
<p class="sub">Weekly PPR projections for QB, RB, WR and TE &middot; %%WEEKLABEL%% &middot; updated %%UPDATED%%</p>
<p class="notice">%%NOTICE%%</p>
</div></header>
<main class="wrap">

<section id="projections">
<h2>%%WEEKLABEL%% projections</h2>
<div class="card">
<div class="controls">
<div class="pills" role="group" aria-label="Filter by position">
<button data-pos="ALL" aria-pressed="true">All</button><button data-pos="QB" aria-pressed="false">QB</button><button data-pos="RB" aria-pressed="false">RB</button><button data-pos="WR" aria-pressed="false">WR</button><button data-pos="TE" aria-pressed="false">TE</button>
</div>
<input id="q" type="search" placeholder="Search player or team" aria-label="Search player or team">
<label class="muted small">Show <select id="limit"><option value="25">top 25</option><option value="50" selected>top 50</option><option value="100">top 100</option><option value="9999">all</option></select></label>
<span class="count" id="count"></span>
</div>
<div class="scroll"><table id="proj">
<thead><tr>
<th aria-sort="none"><button data-k="prank">Rank</button></th>
<th aria-sort="none"><button data-k="player">Player</button></th>
<th aria-sort="none"><button data-k="opp">Matchup</button></th>
<th aria-sort="none"><button data-k="floor">Floor</button></th>
<th aria-sort="none"><button data-k="median">Median</button></th>
<th aria-sort="none"><button data-k="expected">Expected</button></th>
<th aria-sort="none"><button data-k="ceiling">Ceiling</button></th>
<th>Range</th>
</tr></thead>
<tbody></tbody></table></div>
<p class="legend">Bar = floor to ceiling &middot; tick = median (the typical game) &middot; dot = expected (the average, comparable with ESPN-style projections). Floor, median and ceiling are the 10th, 50th and 90th percentile outcomes. Rank is within position, by expected points. Small grey label after a name = depth-chart slot (WR1 = first-string receiver); a colored note under a name = injury-report status. Click a column heading to sort. On a phone, the table shows expected points, with the matchup and floor-to-ceiling range under each name; the CSV has everything. <a href="predictions_latest.csv" download>Download the CSV</a>.</p>
%%INJNOTE%%
</div>
</section>

<section id="track"><h2>Track record</h2><div class="card">%%TRACK%%</div></section>
<section id="last"><h2>Last week&rsquo;s results</h2><div class="card">%%LAST%%</div></section>
<section id="backtest"><h2>How accurate is it?</h2><div class="card">%%BACKTEST%%</div></section>

<section id="how"><h2>How it works</h2><div class="card">
<ol>
<li><strong>Data:</strong> free public data from <a href="https://github.com/nflverse/nflverse-data">nflverse</a> (weekly stats, schedules with Vegas lines, snap counts).</li>
<li><strong>Features</strong> use only what was known before kickoff, computed separately for each player: recent and recency-weighted averages of production and usage, snap share, opponent strength, Vegas spread and total, home/away and rest.</li>
<li><strong>Models</strong> are gradient-boosted trees (XGBoost) evaluated on time-ordered splits, so a season is never predicted using its own future.</li>
<li><strong>Ranges</strong> (floor, median, ceiling) are calibrated so that about 80% of results should land between the floor and the ceiling; the backtest checks this on seasons the model never saw.</li>
%%HOWINJ%%
<li><strong>Grading:</strong> every prediction is committed to GitHub before the games, then scored against actual results each week.</li>
</ol>
<h3>What building it taught me</h3>
<ul>
<li><strong>Hot streaks cool off, and the model accounts for it:</strong> players whose last five games were at least 1.5x their career average scored only about 10 points in their next game.</li>
<li><strong>Heavily used receivers were not under-rated:</strong> those with a 25%+ target share and 60%+ snap share were projected at 14.8 points and scored 14.9.</li>
<li><strong>A bug I found and fixed:</strong> rolling averages were leaking one player&rsquo;s stats into the next player&rsquo;s first few games. Fixing it cut the model&rsquo;s error from 4.87 to 4.81 points.</li>
<li><strong>A benchmark I got wrong at first:</strong> the expert rankings were matched to the wrong week, which had made the model look competitive. Correcting it showed the experts are ahead.</li>
<li><strong>Medians are not averages:</strong> the typical game runs 1&ndash;2 points below the average for most players, so the page shows both.</li>
</ul>
<h3>Limitations</h3>
<ul>
<li>PPR scoring only, and one full test season so far.</li>
<li>A player&rsquo;s team is taken from his most recent game, so off-season moves are not handled.</li>
<li>Rookies and players with few games get rougher ranges.</li>
%%LIMITINJ%%
</ul>
</div></section>

</main>
<footer><div class="wrap">
Code and full write-up: <a href="%%REPO%%">%%REPO%%</a>. Data: nflverse (weekly stats, schedules, snap counts); expert-consensus benchmark: FantasyPros.
%%AUTHOR%%
</div></footer>
<script>
const DATA = %%DATA%%;
const state = {pos:"ALL", q:"", k:"expected", dir:-1, limit:50};
const $ = s => document.querySelector(s);
const esc = s => String(s).replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const maxV = Math.max(...DATA.map(r => r.ceiling), 1);
const pct = v => (Math.min(v, maxV) / maxV * 100).toFixed(1);

function render() {
  let rows = DATA.filter(r => (state.pos === "ALL" || r.pos === state.pos) &&
    (!state.q || (r.player + " " + r.team + " " + r.opp).toLowerCase().includes(state.q)));
  const total = rows.length;
  rows.sort((a, b) => {
    const x = a[state.k], y = b[state.k];
    const c = typeof x === "string" ? x.localeCompare(y) : x - y;
    return (c * state.dir) || (b.expected - a.expected);
  });
  rows = rows.slice(0, state.limit);
  $("#proj tbody").innerHTML = rows.map(r => `<tr>
    <td>${esc(r.pos)}${r.prank}</td>
    <td>${esc(r.player)}${r.depth ? `<span class="dpt" title="Depth chart slot">${esc(r.depth)}</span>` : ""}${r.status ? `<span class="tag ${/^(Out|Doubtful)/.test(r.status) ? "t-out" : /^Questionable/.test(r.status) ? "t-q" : "t-p"}">${esc(r.status)}</span>` : ""}<span class="mob">${esc(r.team)} ${r.home ? "vs" : "@"} ${esc(r.opp)} &middot; range ${r.floor.toFixed(1)}&ndash;${r.ceiling.toFixed(1)}</span></td>
    <td>${esc(r.team)} ${r.home ? "vs" : "@"} ${esc(r.opp)}</td>
    <td>${r.floor.toFixed(1)}</td><td>${r.median.toFixed(1)}</td><td><strong>${r.expected.toFixed(1)}</strong></td><td>${r.ceiling.toFixed(1)}</td>
    <td><span class="rb" role="img" aria-label="Range ${r.floor.toFixed(1)} to ${r.ceiling.toFixed(1)}, median ${r.median.toFixed(1)}, expected ${r.expected.toFixed(1)}">
      <span class="rr" style="left:${pct(r.floor)}%;width:${(pct(r.ceiling) - pct(r.floor)).toFixed(1)}%"></span>
      <span class="rm" style="left:${pct(r.median)}%"></span><span class="re" style="left:${pct(r.expected)}%"></span></span></td>
  </tr>`).join("");
  $("#count").textContent = `Showing ${rows.length} of ${total}`;
  document.querySelectorAll("#proj thead th").forEach(th => {
    const b = th.querySelector("button");
    if (b) th.setAttribute("aria-sort", b.dataset.k === state.k ? (state.dir > 0 ? "ascending" : "descending") : "none");
  });
}

document.querySelectorAll(".pills button").forEach(b => b.addEventListener("click", () => {
  state.pos = b.dataset.pos;
  document.querySelectorAll(".pills button").forEach(x => x.setAttribute("aria-pressed", x === b ? "true" : "false"));
  render();
}));
document.querySelectorAll("#proj thead button").forEach(b => b.addEventListener("click", () => {
  const k = b.dataset.k;
  if (state.k === k) state.dir = -state.dir;
  else { state.k = k; state.dir = (k === "player" || k === "opp" || k === "prank") ? 1 : -1; }
  render();
}));
$("#q").addEventListener("input", e => { state.q = e.target.value.trim().toLowerCase(); render(); });
$("#limit").addEventListener("change", e => { state.limit = +e.target.value; render(); });
render();
</script>
</body>
</html>
'''

author_html = f"<br>Built by {esc(AUTHOR)}." if AUTHOR else ""
page = (TEMPLATE
        .replace("%%NOTICE%%", notice_html)
        .replace("%%INJNOTE%%", injury_note)
        .replace("%%HOWINJ%%", how_inj)
        .replace("%%LIMITINJ%%", limit_inj)
        .replace("%%TITLE%%", esc(SITE_TITLE))
        .replace("%%HEADING%%", esc(SITE_TITLE))
        .replace("%%WEEKLABEL%%", f"{season} Week {week}")
        .replace("%%UPDATED%%", datetime.now().strftime("%b %d, %Y").replace(" 0", " "))
        .replace("%%TRACK%%", track_html)
        .replace("%%LAST%%", last_html)
        .replace("%%BACKTEST%%", backtest_html)
        .replace("%%REPO%%", esc(REPO_URL))
        .replace("%%AUTHOR%%", author_html)
        .replace("%%DATA%%", data_json))

os.makedirs(OUT_DIR, exist_ok=True)
with open(os.path.join(OUT_DIR, "index.html"), "w", encoding="utf-8") as f:
    f.write(page)
open(os.path.join(OUT_DIR, ".nojekyll"), "w").close()
shutil.copyfile(pred_path, os.path.join(OUT_DIR, "predictions_latest.csv"))

print(f"Built {OUT_DIR}/index.html from {pred_path} ({len(records)} players).")
print(f"  Track record: {'yes' if 'Season to date' in track_html else 'not yet (no graded weeks)'}"
      f" | Last week's results: {'yes' if scored_path and 'Across all' in last_html else 'not yet'}")
print(f"Open {os.path.join(OUT_DIR, 'index.html')} in your browser to preview it.")
