"""
update_split.py

Moves the train / validation / test years forward by one season.

    Before: train through 2022, validate on 2023, test on 2024
    After:  train through 2023, validate on 2024, test on 2025

Edits these files in place:
    03_train_model.py
    03b_position_quantile_models.py
    04_fantasypros_comparison.py

Run it ONCE from your Fantasy-Football-Predictions folder:
    python update_split.py

It refuses to run a second time (so it can't shift the years twice).
To undo it, use:  git checkout 03_train_model.py 03b_position_quantile_models.py 04_fantasypros_comparison.py
"""
import re
import sys

FILES = [
    "03_train_model.py",
    "03b_position_quantile_models.py",
    "04_fantasypros_comparison.py",
]
SHIFT = {"2022": "2023", "2023": "2024", "2024": "2025"}

# Safety check: has the split already been moved?
with open("03_train_model.py", encoding="utf-8", newline="") as f:
    if 'df["season"] == 2025' in f.read():
        print("The split already looks updated (found a 2025 test set in "
              "03_train_model.py). Nothing changed.")
        sys.exit(0)

for name in FILES:
    with open(name, encoding="utf-8", newline="") as f:
        text = f.read()

    # The old docstring said 2010; the data actually starts in 2013.
    text = text.replace("train on 2010-2022", "train on 2013-2022")

    # One pass, so a year is never shifted twice.
    new_text, n = re.subn(r"2022|2023|2024", lambda m: SHIFT[m.group()], text)

    with open(name, "w", encoding="utf-8", newline="") as f:
        f.write(new_text)
    print(f"{name}: shifted {n} year references")

print("\nCheck the split lines in 03_train_model.py:")
with open("03_train_model.py", encoding="utf-8") as f:
    for line in f:
        if "_mask = df" in line:
            print("  " + line.rstrip())
