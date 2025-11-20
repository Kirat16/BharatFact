import pandas as pd
import os

paths = [
    "data/bharatfakenewskosh (3).xlsx",
    "data/ISOT/True.csv",
    "data/ISOT/Fake.csv",
    "data/gossipcop/gossipcop_fake.csv",
    "data/gossipcop/gossipcop_real.csv",
    "data/politifact/politifact_fake.csv",
    "data/politifact/politifact_real.csv"
]

for p in paths:
    if os.path.exists(p):
        print("\n======", p, "======")
        df = pd.read_csv(p) if p.endswith(".csv") else pd.read_excel(p)
        print("Columns:", df.columns.tolist())
        print(df.head(3))
    else:
        print("\n[MISSING]", p)
