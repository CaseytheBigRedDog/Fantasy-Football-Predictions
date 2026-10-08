import sqlite3
import sys
import pandas as pd

# Usage: python run_query.py sql/your_file.sql
sql_file = sys.argv[1] if len(sys.argv) > 1 else "sql/01_rolling_features.sql"

conn = sqlite3.connect("fantasy.db")
query = open(sql_file).read()
df = pd.read_sql_query(query, conn)
print(df.head(40).to_string())
conn.close()