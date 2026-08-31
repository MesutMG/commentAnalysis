import json
import os
import pandas as pd
import asyncio
import aiohttp
from tqdm.asyncio import tqdm_asyncio
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
LLM_ROWS = None
# Controls how many requests hit the LLM at the exact same time
CONCURRENCY_LIMIT = 2

preprocessor = Preprocessor()
comment_filter = Filter()
analyzer = Analyzer()

async def process_llm_batch(df_subset: pd.DataFrame):
    sem = asyncio.Semaphore(CONCURRENCY_LIMIT)

    async def bounded_analyze(session, comment):
        # The semaphore prevents sending all 100 requests at once
        async with sem:
            return await analyzer.analyze_comment_async(session, str(comment))

    async with aiohttp.ClientSession() as session:
        tasks = [bounded_analyze(session, comment) for comment in df_subset["comment"]]
        # tqdm_asyncio wraps gather to keep your progress bar functional
        results = await tqdm_asyncio.gather(*tasks, desc="LLM Inference")
        
    return results

async def main():
    try:
        df = pd.read_csv(DATA_PATH, sep=";", encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(DATA_PATH, sep=";", encoding="windows-1254")

    df = preprocessor.process(df)
    df_clean, df_flagged = comment_filter.filter(df)
    
    df_subset = df_clean.head(LLM_ROWS).copy() if LLM_ROWS else df_clean.copy()

    # Wait for the async LLM batch to finish
    llm_results = await process_llm_batch(df_subset)

    # Extract validated Pydantic properties straight into dataframe columns
    df_subset["degerlendirme"] = [res.degerlendirme for res in llm_results]
    df_subset["kategori"] = [res.kategori for res in llm_results]
    df_subset["llm_uygunsuz"] = [res.llm_uygunsuz for res in llm_results]
    df_subset["llm_sebep"] = [res.llm_sebep for res in llm_results]

    if not df_flagged.empty:
        df_flagged["degerlendirme"] = "negatif"
        df_flagged["kategori"] = df_flagged["flag_category"].str.lower()
        df_flagged["llm_uygunsuz"] = True
        df_flagged["llm_sebep"] = "Statik kural motoru tarafından engellendi."

    df_final = pd.concat([df_subset, df_flagged], ignore_index=True)

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

    TARGET_COLUMNS_ALL = [
        "id",
        "comment",
        "star",
        "is_flagged",
        "degerlendirme", 
        "kategori",
        "llm_uygunsuz",
        "llm_sebep"
    ]
    TARGET_COLUMNS_CLEAN = [
        "id",
        "comment",
        "star",
        "degerlendirme",
        "kategori"
    ]

    df_all_export = df_final[TARGET_COLUMNS_ALL].sort_values(by="id").reset_index(drop=True)
    df_all_export.to_csv(OUTPUT_PATH, sep=";", index=False, encoding="utf-8")

    clean_mask = (~df_final["is_flagged"]) & (df_final["llm_uygunsuz"] == False)
    df_clean_export = df_final[clean_mask][TARGET_COLUMNS_CLEAN].sort_values(by="id").reset_index(drop=True)
    df_clean_export.to_csv(OUTPUT_PATH_CLEAN, sep=";", index=False, encoding="utf-8")

    print(f"Exported total records ({len(df_all_export)}) -> {OUTPUT_PATH}")
    print(f"Exported approved clean records ({len(df_clean_export)}) -> {OUTPUT_PATH_CLEAN}")

if __name__ == "__main__":
    asyncio.run(main())
