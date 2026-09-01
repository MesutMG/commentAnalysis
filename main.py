import io
import json
import os
import asyncio
import aiohttp
import pandas as pd
from typing import Optional
from tqdm.asyncio import tqdm_asyncio
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from src.analyze import Analyzer
from src.filter import Filter
from src.preprocess import Preprocessor

# --- CONFIGURATION ---
CONFIG_PATH = "config.json"
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config = json.load(f)

DATA_PATH = config.get("data_path", "data/example_comments.csv")
OUTPUT_PATH = config.get("output_path", "output.csv")
base_output, ext = os.path.splitext(OUTPUT_PATH)
OUTPUT_PATH_CLEAN = f"{base_output}_cleaned{ext}"

LLM_ROWS = config.get("llm_rows", None)
CONCURRENCY_LIMIT = config.get("concurrency_limit", 2)

preprocessor = Preprocessor()
comment_filter = Filter()
analyzer = Analyzer()

# --- FASTAPI APP ---
app = FastAPI(title="Comment Analysis API")

class SingleCommentRequest(BaseModel):
    comment: str
    star: Optional[float] = 1.0


# --- CORE PIPELINE LOGIC ---
async def process_llm_batch(df_subset: pd.DataFrame):
    sem = asyncio.Semaphore(CONCURRENCY_LIMIT)
    timeout = aiohttp.ClientTimeout(total=180)

    async def bounded_analyze(session, comment):
        async with sem:
            return await analyzer.analyze_comment_async(session, str(comment))

    async with aiohttp.ClientSession(timeout=timeout) as session:
        tasks = [bounded_analyze(session, comment) for comment in df_subset["comment"]]
        # tqdm_asyncio wraps gather to keep your progress bar functional
        results = await tqdm_asyncio.gather(*tasks, desc="LLM Inference")
        
    return results

def correct_star_rating(star: float, sentiment: str) -> float:
    sentiment = str(sentiment).lower()
    if sentiment == "negatif" and star == 5.0:
        return 3.0
    if sentiment == "pozitif" and star in [1.0, 2.0]:
        return 4.0
    return star

async def process_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Processes a dataframe through the entire pipeline and returns full and clean dataframes."""
    df = preprocessor.process(df)
    df_clean, df_flagged = comment_filter.filter(df)
    
    if LLM_ROWS and len(df_clean) > LLM_ROWS:
        df_subset = df_clean.sample(n=LLM_ROWS, random_state=42).copy().reset_index(drop=True)
    else:
        df_subset = df_clean.copy().reset_index(drop=True)

    # Wait for the async LLM batch to finish
    llm_results = await process_llm_batch(df_subset)

    # Extract validated Pydantic properties straight into dataframe columns
    df_subset["degerlendirme"] = [res.degerlendirme for res in llm_results]
    df_subset["kategori"] = [res.kategori for res in llm_results]
    df_subset["llm_uygunsuz"] = [res.llm_uygunsuz for res in llm_results]
    df_subset["llm_sebep"] = [res.llm_sebep for res in llm_results]

    if not df_flagged.empty:
        df_flagged = df_flagged.copy()
        df_flagged["degerlendirme"] = "negatif"
        df_flagged["kategori"] = df_flagged["flag_category"].astype(str).str.lower()
        df_flagged["llm_uygunsuz"] = True
        df_flagged["llm_sebep"] = "Statik kural motoru tarafından engellendi."

    df_final = pd.concat([df_subset, df_flagged], ignore_index=True)

    df_final["star"] = df_final.apply(
        lambda row: correct_star_rating(
            float(row.get("star", 1.0)) if pd.notnull(row.get("star")) else 1.0, 
            row.get("degerlendirme", "")
        ), axis=1
    )

    TARGET_COLUMNS_ALL = ["id", "comment", "star", "is_flagged", "degerlendirme", "kategori", "llm_uygunsuz", "llm_sebep"]
    TARGET_COLUMNS_CLEAN = ["id", "comment", "star", "degerlendirme", "kategori"]

    # Ensure columns exist even if empty
    for col in TARGET_COLUMNS_ALL:
        if col not in df_final.columns:
            df_final[col] = None

    df_all_export = df_final[TARGET_COLUMNS_ALL].sort_values(by="id", na_position='first').reset_index(drop=True)
    
    clean_mask = (~df_final["is_flagged"]) & (df_final["llm_uygunsuz"] == False)
    df_clean_export = df_final[clean_mask][TARGET_COLUMNS_CLEAN].sort_values(by="id", na_position='first').reset_index(drop=True)

    return df_all_export, df_clean_export


# --- API ENDPOINTS ---
@app.post("/analyzeComment/")
async def analyze_comment_endpoint(request: SingleCommentRequest):
    # Analyzes a single comment without going through Pandas batch processing
    comment = request.comment
    
    # 1. Check static filter
    is_flagged, category = comment_filter.check_comment(comment)
    
    if is_flagged:
        sentiment = "negatif"
        star_corrected = correct_star_rating(request.star, sentiment)
        return {
            "comment": comment,
            "star": star_corrected,
            "is_flagged": True,
            "degerlendirme": sentiment,
            "kategori": category.lower(),
            "llm_uygunsuz": True,
            "llm_sebep": "Statik kural motoru tarafından engellendi."
        }

    # 2. Run LLM
    timeout = aiohttp.ClientTimeout(total=180)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        res = await analyzer.analyze_comment_async(session, comment)

    star_corrected = correct_star_rating(request.star, res.degerlendirme)

    return {
        "comment": comment,
        "star": star_corrected,
        "is_flagged": False,
        "degerlendirme": res.degerlendirme,
        "kategori": res.kategori,
        "llm_uygunsuz": res.llm_uygunsuz,
        "llm_sebep": res.llm_sebep
    }

@app.post("/analyzeWholeCsv/")
async def analyze_whole_csv_endpoint(file: UploadFile = File(...)):
    # Processes a csv file and returns the analyzed dataset as JSON
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are allowed.")

    content = await file.read()
    
    try:
        df = pd.read_csv(io.BytesIO(content), sep=";", encoding="utf-8")
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(io.BytesIO(content), sep=";", encoding="windows-1254")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"File reading error: {str(e)}")

    if "comment" not in df.columns:
        raise HTTPException(status_code=400, detail="CSV must contain a 'comment' column.")

    df_all_export, _ = await process_dataframe(df)
    
    # Return as list of JSON records
    return JSONResponse(content=df_all_export.fillna("").to_dict(orient="records"))


# --- CLI MODE ---
async def run_cli():
    try:
        df = pd.read_csv(DATA_PATH, sep=";", encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(DATA_PATH, sep=";", encoding="windows-1254")

    df_all_export, df_clean_export = await process_dataframe(df)

    df_all_export.to_csv(OUTPUT_PATH, sep=";", index=False, encoding="utf-8")
    df_clean_export.to_csv(OUTPUT_PATH_CLEAN, sep=";", index=False, encoding="utf-8")

    print(f"Exported total records ({len(df_all_export)}) -> {OUTPUT_PATH}")
    print(f"Exported approved clean records ({len(df_clean_export)}) -> {OUTPUT_PATH_CLEAN}")

if __name__ == "__main__":
    # If run directly via `python3 main.py`, execute the CLI mode.
    # If run via Uvicorn, this block is ignored and the FastAPI app takes over.
    asyncio.run(run_cli())