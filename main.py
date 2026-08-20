import os
import pandas as pd
from src.filter import Filter
from src.preprocess import Preprocessor
from src.analyze import Analyzer

preprocessor = Preprocessor()
filter = Filter()
analyzer = Analyzer()

DATA_PATH = "data/example_comments_copy3.csv"
base_path, _ = os.path.splitext(DATA_PATH)
df = pd.read_csv(DATA_PATH, sep=";")

#remove email and pnr columns
df = preprocessor.process(df)

#apply inappropriate word filter
df_clean, df_flagged = filter.apply_filter(df)

#analyze with llm
df = analyzer.analyze(df_clean)


