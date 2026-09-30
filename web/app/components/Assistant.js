"use client";

import { useState } from "react";
import { api } from "../api";
import PlayerTable from "./PlayerTable";

const EXAMPLES = [
  "Should I start Gibbs or Chase?",
  "Who are the top 5 running backs?",
  "Flex: Bijan Robinson, Kenneth Walker or Christian McCaffrey?",
];

// Shows the answer to one question and the numbers it was based on.
export default function Assistant({ result, setResult, meta }) {
  const [question, setQuestion] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  async function ask(text) {
    const q = (text ?? question).trim();
    if (!q) return;
    setQuestion(q);
    setLoading(true);
    setError("");
    try {
      const data = await api("/api/assistant", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: q }),
      });
      setResult({ title: q, ...data });
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <section className="card">
      <h2>Who should I start?</h2>
      <form
        className="ask"
        onSubmit={(e) => {
          e.preventDefault();
          ask();
        }}
      >
        <input
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          placeholder="Ask about your players, e.g. “Should I start Gibbs or Chase?”"
          maxLength={500}
          aria-label="Your question"
        />
        <button type="submit" disabled={loading}>
          {loading ? "Thinking…" : "Ask"}
        </button>
      </form>

      <div className="chips">
        {EXAMPLES.map((example) => (
          <button key={example} type="button" className="chip" onClick={() => ask(example)}>
            {example}
          </button>
        ))}
      </div>

      {meta && meta.assistant_mode === "rules" && (
        <p className="hint">
          Running without an AI key, so answers use the built-in comparison. Add
          ANTHROPIC_API_KEY to api/.env for free-form questions.
        </p>
      )}
      {error && <p className="error">{error}</p>}

      {result && (
        <div className="answer">
          <div className="answer-head">
            <strong>{result.title}</strong>
            <span className="badge">{result.mode === "claude" ? "Claude + your projections" : "Built-in comparison"}</span>
          </div>
          <p className="answer-text">{result.answer}</p>
          {result.note && <p className="hint">{result.note}</p>}
          {result.players && result.players.length > 0 && (
            <>
              <p className="hint">Numbers from your model behind this answer:</p>
              <PlayerTable rows={result.players} compact />
            </>
          )}
        </div>
      )}
    </section>
  );
}
