import os
import numpy as np
import pandas as pd

DATA_PATH = "data/example_comments_copy3.csv"
base_path, _ = os.path.splitext(DATA_PATH)
OUTPUT_PATH = f"{base_path}_cleaned.csv"

df = pd.read_csv(DATA_PATH, sep=";")
df = df.drop(columns=["name", "email", "pnr"], errors="ignore")

# Remove null, empty, and whitespace-only comments
df["comment"] = df["comment"].replace(r"^\s*$", np.nan, regex=True)
df = df.dropna(subset=["comment"]).reset_index(drop=True)

output_dir = os.path.dirname(OUTPUT_PATH)
if output_dir:
    os.makedirs(output_dir, exist_ok=True)

df.to_csv(OUTPUT_PATH, sep=";", index=False)
print(f"Saved {len(df)} rows to {OUTPUT_PATH}")
