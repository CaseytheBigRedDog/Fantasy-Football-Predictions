import sqlite3
from pathlib import Path
import pandas as pd

DB_PATH = "fantasy.db"
DATA_DIR = Path("data")

conn = sqlite3.connect(DB_PATH)

for csv_file in DATA_DIR.glob("*.csv"):
    table_name = csv_file.stem.lower().replace("-", "_").replace(" ", "_")
    df = pd.read_csv(csv_file)
    df.to_sql(table_name, conn, if_exists="replace", index=False)
    print(f"Loaded {csv_file.name} -> table '{table_name}' ({len(df)} rows)")

conn.close()
print("Done. Database saved as", DB_PATH)