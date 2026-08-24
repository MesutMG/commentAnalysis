import json
import os
import pandas as pd
from src.analyze import Analyzer
from src.filter import Filter
from src.preprocess import Preprocessor

CONFIG_PATH = "config.json"
with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

DATA_PATH = config.get("data_path", "data/example_comments.csv")
OUTPUT_PATH = config.get("output_path", "output.csv")
base_output, ext = os.path.splitext(OUTPUT_PATH)
OUTPUT_PATH_CLEAN = f"{base_output}_cleaned{ext}"
LLM_ROWS = 100

# 1. Load Data
try:
    df = pd.read_csv(DATA_PATH, sep=";", encoding="utf-8")
except UnicodeDecodeError:
    df = pd.read_csv(DATA_PATH, sep=";", encoding="windows-1254")

# 2. Preprocess (Drop PII & empty comments)
preprocessor = Preprocessor()
df = preprocessor.process(df)

# 3. Static Filter
comment_filter = Filter()
df_clean, df_flagged = comment_filter.filter(df)

# 4. LLM Analysis on Clean Rows
analyzer = Analyzer()
df_analyzed = analyzer.run_pipeline(df_clean, LLM_ROWS)

# 5. Populate Default Fields for Statically Flagged Rows
if not df_flagged.empty:
    df_flagged["degerlendirme"] = "negatif"
    df_flagged["kategori"] = df_flagged["flag_category"].str.lower()
    df_flagged["llm_uygunsuz"] = True
    df_flagged["llm_sebep"] = "Statik kural motoru tarafından engellendi."

# 6. Combine Datasets
df_final = pd.concat([df_analyzed, df_flagged], ignore_index=True)


# 7. Star Rating Correction Logic
def correct_star_rating(row: pd.Series) -> float:
    try:
        star = float(row["star"])
    except (ValueError, TypeError):
        return row.get("star", 1.0)

    sentiment = str(row.get("degerlendirme", "")).lower()

    if sentiment == "negatif" and star == 5.0:
        return 3.0
    if sentiment == "pozitif" and star in [1.0, 2.0]:
        return 4.0

    return star


df_final["star"] = df_final.apply(correct_star_rating, axis=1)

# 8. Column Configurations
TARGET_COLUMNS_ALL = [
    "id",
    "comment",
    "star",
    "is_flagged",
    "degerlendirme",
    "kategori",
    "llm_uygunsuz",
    "llm_sebep",
]

TARGET_COLUMNS_CLEAN = [
    "id",
    "comment",
    "star",
    "degerlendirme",
    "kategori",
]

# 9. Export All Records (Auditing / Debug file)
df_all_export = (
    df_final[TARGET_COLUMNS_ALL].sort_values(by="id").reset_index(drop=True)
)
df_all_export.to_csv(OUTPUT_PATH, sep=";", index=False, encoding="utf-8")

# 10. Filter and Export Only Valid Records (Neither statically nor LLM flagged)
clean_mask = (~df_final["is_flagged"]) & (df_final["llm_uygunsuz"] == False)
df_clean_export = (
    df_final[clean_mask][TARGET_COLUMNS_CLEAN]
    .sort_values(by="id")
    .reset_index(drop=True)
)
df_clean_export.to_csv(
    OUTPUT_PATH_CLEAN, sep=";", index=False, encoding="utf-8"
)

print(f"Exported total records ({len(df_all_export)}) -> {OUTPUT_PATH}")
print(f"Exported approved clean records ({len(df_clean_export)}) -> {OUTPUT_PATH_CLEAN}")
