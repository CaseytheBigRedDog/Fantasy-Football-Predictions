"""
Quick self-check for the projection lookups and the assistant's built-in comparison.

Run from the api/ folder:   python test_logic.py
"""
import assistant
import projections as pj


def check(label, condition):
    print(("PASS  " if condition else "FAIL  ") + label)
    if not condition:
        raise SystemExit(1)


info = pj.meta()
print("Using", info["file"], "-", info["players"], "players\n")

check("loads a non-empty table", info["players"] > 100)
check("newest file is chosen (not a baseline copy)", "baseline" not in info["file"])

check("exact name", pj.lookup("Ja'Marr Chase")[0]["player"] == "Ja'Marr Chase")
check("no apostrophe / lowercase", pj.lookup("jamarr chase")[0]["player"] == "Ja'Marr Chase")
check("last name only", pj.lookup("Gibbs")[0]["player"] == "Jahmyr Gibbs")
check("suffix optional", pj.lookup("Kenneth Walker")[0]["player"].startswith("Kenneth Walker"))
check("typo tolerated", pj.lookup("Bijan Robinsen")[0]["player"] == "Bijan Robinson")
check("unknown name -> empty", pj.lookup("Zzzz Qqqq") == [])

top = pj.top(position="RB", n=5)
check("top RBs sorted and filtered", len(top) == 5 and all(r["position"] == "RB" for r in top)
      and top[0]["expected"] >= top[-1]["expected"])
check("search by text", all("walker" in r["player"].lower() for r in pj.search(q="walker")))

rows, _ = pj.find_players_in_text("Should I start Gibbs or Chase this week?")
check("finds two players in a sentence", {r["player"] for r in rows} == {"Jahmyr Gibbs", "Ja'Marr Chase"})

result = assistant.ask("Should I start Gibbs or Chase?")
check("assistant picks the higher expected player",
      result["answer"].startswith("Start Ja'Marr Chase") or "Toss-up" in result["answer"])
check("assistant returns the numbers it used", len(result["players"]) == 2)

result = assistant.ask("Who are the top 3 running backs?")
check("position question -> top list of the size asked", result["answer"].startswith("Top 3 RB"))

result = assistant.ask("What is the weather?")
check("unrelated question -> helpful message, no players", result["players"] == [])

print("\nAll checks passed.")
