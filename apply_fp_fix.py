"""
apply_fp_fix.py

Applies the FantasyPros week-alignment fix found by check_fp_alignment.py:

  * 04_fantasypros_comparison.py: shifts each scrape's week label back by one
    week (the weekly scrapes happen on Fridays, after that week's Thursday
    game, so the old "next week" label was a week late), and corrects the
    coverage note (the archive starts in 2023, not 2019).
  * score_predictions.py: sets FP_WEEK_OFFSET = -1.

Run ONCE from your Fantasy-Football-Predictions folder:
    python apply_fp_fix.py
Then rerun the comparison:
    python 04_fantasypros_comparison.py

To undo: git checkout 04_fantasypros_comparison.py score_predictions.py
"""
import sys

# ---------------- 04_fantasypros_comparison.py ----------------
name04 = "04_fantasypros_comparison.py"
with open(name04, encoding="utf-8", newline="") as f:
    s = f.read()

if "FP_WEEK_OFFSET" in s:
    print(f"{name04} already has the fix. Nothing changed there.")
else:
    anchor = 'fp_mapped["week"] = fp_mapped["week"].astype(int)'
    if s.count(anchor) != 1:
        print(f"Could not find the expected line in {name04}; no changes made.")
        sys.exit(1)
    nl = "\r\n" if "\r\n" in s else "\n"
    fix = nl.join([
        anchor,
        "",
        "# The scrapes happen on Fridays, AFTER that week's Thursday game, so the",
        "# 'next week' label above is one week late. check_fp_alignment.py measured",
        "# this: shifting back one week lines the rankings up with the right games.",
        "FP_WEEK_OFFSET = -1",
        'fp_mapped["week"] = fp_mapped["week"] + FP_WEEK_OFFSET',
        'fp_mapped = fp_mapped[fp_mapped["week"] >= 1]',
    ])
    s = s.replace(anchor, fix, 1)
    s = s.replace("Coverage: weekly PPR positional rankings, 2019-2025.",
                  "Coverage: weekly PPR positional rankings, 2023-present.")
    with open(name04, "w", encoding="utf-8", newline="") as f:
        f.write(s)
    print(f"{name04}: applied the -1 week shift.")

# ---------------- score_predictions.py ----------------
name_sc = "score_predictions.py"
with open(name_sc, encoding="utf-8", newline="") as f:
    s = f.read()

if "FP_WEEK_OFFSET = 0" in s:
    s = s.replace("FP_WEEK_OFFSET = 0", "FP_WEEK_OFFSET = -1", 1)
    with open(name_sc, "w", encoding="utf-8", newline="") as f:
        f.write(s)
    print(f"{name_sc}: FP_WEEK_OFFSET set to -1.")
elif "FP_WEEK_OFFSET = -1" in s:
    print(f"{name_sc} already uses -1. Nothing changed there.")
else:
    print(f"Could not find FP_WEEK_OFFSET in {name_sc}; check it by hand.")

print("\nNext: python 04_fantasypros_comparison.py")
