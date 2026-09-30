"use client";

import { useMemo, useState } from "react";
import { fmt } from "../api";

const SCALE_MAX = 36; // points: the right edge of the floor-to-ceiling bars

function RangeBar({ row }) {
  const pct = (v) => `${Math.max(0, Math.min(100, (v / SCALE_MAX) * 100))}%`;
  return (
    <div className="range" title={`Floor ${fmt(row.floor)} · Median ${fmt(row.median)} · Ceiling ${fmt(row.ceiling)}`}>
      <div className="range-fill" style={{ left: pct(row.floor), width: `calc(${pct(row.ceiling)} - ${pct(row.floor)})` }} />
      <div className="range-mark" style={{ left: pct(row.expected) }} />
    </div>
  );
}

const COLUMNS = [
  { key: "player", label: "Player" },
  { key: "position", label: "Pos" },
  { key: "team", label: "Team" },
  { key: "opponent", label: "Opp" },
  { key: "expected", label: "Expected", numeric: true },
  { key: "floor", label: "Floor", numeric: true },
  { key: "median", label: "Median", numeric: true },
  { key: "ceiling", label: "Ceiling", numeric: true },
];

// A table of projections. With `selected` + `onToggle` it shows checkboxes (for comparing).
export default function PlayerTable({ rows, selected, onToggle, compact = false }) {
  const [sort, setSort] = useState({ key: "expected", dir: "desc" });
  const selectable = Boolean(onToggle);

  const sorted = useMemo(() => {
    if (compact) return rows;
    const copy = [...rows];
    copy.sort((a, b) => {
      const x = a[sort.key];
      const y = b[sort.key];
      const order = typeof x === "number" ? x - y : String(x).localeCompare(String(y));
      return sort.dir === "asc" ? order : -order;
    });
    return copy;
  }, [rows, sort, compact]);

  function clickHeader(key) {
    setSort((s) => (s.key === key ? { key, dir: s.dir === "asc" ? "desc" : "asc" } : { key, dir: key === "player" ? "asc" : "desc" }));
  }

  return (
    <div className="table-wrap">
      <table>
        <thead>
          <tr>
            {selectable && <th aria-label="Select" />}
            {COLUMNS.map((c) => (
              <th
                key={c.key}
                className={c.numeric ? "num" : ""}
                onClick={compact ? undefined : () => clickHeader(c.key)}
                style={compact ? undefined : { cursor: "pointer" }}
              >
                {c.label}
                {!compact && sort.key === c.key ? (sort.dir === "asc" ? " ▲" : " ▼") : ""}
              </th>
            ))}
            <th>Range</th>
            <th>Status</th>
          </tr>
        </thead>
        <tbody>
          {sorted.map((r) => (
            <tr key={r.player_id} className={selectable && selected.includes(r.player_id) ? "picked" : ""}>
              {selectable && (
                <td>
                  <input
                    type="checkbox"
                    checked={selected.includes(r.player_id)}
                    onChange={() => onToggle(r.player_id)}
                    aria-label={`Select ${r.player}`}
                  />
                </td>
              )}
              <td>{r.player}</td>
              <td>{r.position}</td>
              <td>{r.team}</td>
              <td>
                {r.is_home === 0 ? "at " : "vs "}
                {r.opponent}
              </td>
              <td className="num strong">{fmt(r.expected)}</td>
              <td className="num">{fmt(r.floor)}</td>
              <td className="num">{fmt(r.median)}</td>
              <td className="num">{fmt(r.ceiling)}</td>
              <td>
                <RangeBar row={r} />
              </td>
              <td>{r.status || (r.p_play !== null && r.p_play < 0.99 ? `${Math.round(r.p_play * 100)}% to play` : "")}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
