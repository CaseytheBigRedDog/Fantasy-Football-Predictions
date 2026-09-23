"""
adopt_new_features.py

Switches the newer features ON for the models, but only run this AFTER
compare_features.py says adoption looks worthwhile.

What it does:
  * 03_train_model.py and 03b_position_quantile_models.py: adds two lines that
    pull in the extra feature columns (they add nothing while the switch is off).
  * new_features.py: sets USE_NEW_FEATURES = True.

Run once from your Fantasy-Football-Predictions folder:
    python adopt_new_features.py
Then retrain everything:
    python refresh_data.py

To undo: set USE_NEW_FEATURES = False in new_features.py (the two added lines
in 03 and 03b are harmless when it is False).
"""
import sys

ANCHOR = '    "is_home", "rest_days", "def_pts_allowed_r5",\n]'

for name in ["03_train_model.py", "03b_position_quantile_models.py"]:
    with open(name, encoding="utf-8", newline="") as f:
        s = f.read()
    if "extra_feature_cols" in s:
        print(f"{name}: already set up.")
        continue
    nl = "\r\n" if "\r\n" in s else "\n"
    anchor = ANCHOR.replace("\n", nl)
    if s.count(anchor) != 1:
        print(f"Could not find the expected feature list in {name}; no changes made.")
        sys.exit(1)
    hook = nl.join([
        anchor,
        "from new_features import extra_feature_cols",
        "feature_cols += extra_feature_cols(df)",
    ])
    with open(name, "w", encoding="utf-8", newline="") as f:
        f.write(s.replace(anchor, hook, 1))
    print(f"{name}: new features hooked in.")

with open("new_features.py", encoding="utf-8", newline="") as f:
    s = f.read()
if "USE_NEW_FEATURES = False" in s:
    with open("new_features.py", "w", encoding="utf-8", newline="") as f:
        f.write(s.replace("USE_NEW_FEATURES = False", "USE_NEW_FEATURES = True", 1))
    print("new_features.py: switch set to ON.")
else:
    print("new_features.py: switch is already ON (or was edited by hand).")

print("\nNext: python refresh_data.py")
