"use client";

import { useEffect, useMemo, useState } from "react";
import { api } from "./api";
import Assistant from "./components/Assistant";
import PlayerTable from "./components/PlayerTable";

const POSITIONS = ["ALL", "QB", "RB", "WR", "TE"];

export default function Home() {
  const [meta, setMeta] = useState(null);
  const [rows, setRows] = useState([]);
  const [error, setError] = useState("");
  const [position, setPosition] = useState("ALL");
  const [search, setSearch] = useState("");
  const [selected, setSelected] = useState([]);
  const [result, setResult] = useState(null);

  useEffect(() => {
    Promise.all([api("/api/meta"), api("/api/projections?limit=1000")])
      .then(([m, r]) => {
        setMeta(m);
        setRows(r);
      })
      .catch((e) => setError(e.message));
  }, []);

  const visible = useMemo(() => {
    const needle = search.trim().toLowerCase();
    return rows.filter(
      (r) =>
        (position === "ALL" || r.position === position) &&
        (!needle || r.player.toLowerCase().includes(needle) || r.team.toLowerCase() === needle)
    );
  }, [rows, position, search]);

  function toggle(id) {
    setSelected((current) =>
      current.includes(id) ? current.filter((x) => x !== id) : current.length < 8 ? [...current, id] : current
    );
  }

  async function compareSelected() {
    try {
      const query = selected.map((id) => `ids=${encodeURIComponent(id)}`).join("&");
      const data = await api(`/api/compare?${query}`);
      setResult({ title: "Comparison of selected players", mode: "rules", note: null, ...data });
      window.scrollTo({ top: 0, behavior: "smooth" });
    } catch (e) {
      setError(e.message);
    }
  }

  return (
    <main>
      <header>
        <h1>Fantasy Football Predictions</h1>
        <p className="sub">
          {meta ? `Week ${meta.week}, ${meta.season} · full PPR · ${meta.players} players` : "Loading…"}
        </p>
      </header>

      {error && <p className="error">{error}</p>}

      <Assistant result={result} setResult={setResult} meta={meta} />

      <section className="card">
        <h2>Projections</h2>
        <div className="controls">
          <div className="tabs">
            {POSITIONS.map((p) => (
              <button key={p} className={p === position ? "tab active" : "tab"} onClick={() => setPosition(p)}>
                {p}
              </button>
            ))}
          </div>
          <input
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="Search a player or team (e.g. DET)"
            aria-label="Search"
          />
          <button disabled={selected.length < 2} onClick={compareSelected}>
            Compare selected ({selected.length})
          </button>
          {selected.length > 0 && (
            <button className="link" onClick={() => setSelected([])}>
              Clear
            </button>
          )}
        </div>
        <PlayerTable rows={visible} selected={selected} onToggle={toggle} />
        <p className="hint">
          Expected = average outcome. Floor / Median / Ceiling = 10th / 50th / 90th percentile. The bar
          shows floor to ceiling, and the tick marks Expected. Click a column to sort; tick two or more
          players to compare them.
        </p>
      </section>

      <footer>
        Projections come from my own model, which trails FantasyPros expert rankings in backtests. It has no
        breaking news, so check injury reports before locking your lineup.
      </footer>
    </main>
  );
}
