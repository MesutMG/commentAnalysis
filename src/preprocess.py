import os
import numpy as np
import pandas as pd


class Preprocessor:
    def __init__(self):
        pass

    def process(self, df: pd.DataFrame):
        df = df.drop(columns=["name", "email", "pnr"], errors="ignore")
        
        # Remove null, empty, and whitespace-only comments
        df["comment"] = df["comment"].replace(r"^\s*$", np.nan, regex=True)
        df = df.dropna(subset=["comment"]).reset_index(drop=True)
        return df
