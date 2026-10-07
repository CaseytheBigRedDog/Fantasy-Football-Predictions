import os
from datetime import datetime
import pandas as pd
import nflreadpy as nfl

os.makedirs("redzone_data", exist_ok=True)
out_path = "redzone_data/redzone_weekly.csv"

now = datetime.now()
last = now.year if now.month >= 9 else now.year - 1
first = int(pd.read_parquet("stats_clean.parquet", columns=["season"])["season"].min())

old = pd.read_csv(out_path) if os.path.exists(out_path) else pd.DataFrame()
have = set(old["season"].unique()) if len(old) else set()
to_get = [s for s in range(first, last + 1) if s not in have or s == last]
print(f"Red zone data: downloading seasons {to_get}")

cols = ["season", "week", "season_type", "yardline_100", "pass_attempt",
        "rush_attempt", "receiver_player_id", "rusher_player_id"]
pieces = []
for season in to_get:
    print(f"  downloading {season}...")
    try:
        pbp = nfl.load_pbp(season).select(cols).to_pandas()
    except Exception as e:
        print(f"  [skip] {season}: {e}")
        continue
    pbp = pbp[(pbp["season_type"] == "REG") & (pbp["yardline_100"] <= 20)]
    rec = (pbp[(pbp["pass_attempt"] == 1) & pbp["receiver_player_id"].notna()]
           .groupby(["receiver_player_id", "season", "week"]).size()
           .reset_index(name="rz_targets")
           .rename(columns={"receiver_player_id": "player_id"}))
    car = (pbp[(pbp["rush_attempt"] == 1) & pbp["rusher_player_id"].notna()]
           .groupby(["rusher_player_id", "season", "week"]).size()
           .reset_index(name="rz_carries")
           .rename(columns={"rusher_player_id": "player_id"}))
    pieces.append(rec.merge(car, on=["player_id", "season", "week"], how="outer"))

if len(old):
    old = old[~old["season"].isin(to_get)]
    pieces.append(old)

out = pd.concat(pieces, ignore_index=True).fillna(0)
out[["season", "week"]] = out[["season", "week"]].astype(int)
out["rz_touches"] = out["rz_targets"] + out["rz_carries"]
out.to_csv(out_path, index=False)
print(f"Saved {len(out):,} rows to {out_path}")
