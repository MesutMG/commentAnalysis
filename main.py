import json
import os
import asyncio
import pandas as pd
from typing import Any, Dict, Optional, Union, List
from fastapi import FastAPI, HTTPException
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


class CommentItem(BaseModel):
    id: Optional[Union[int, str]] = None
    comment: str
    star: Optional[float] = 5.0
    name: Optional[str] = None
    email: Optional[str] = None
    pnr: Optional[str] = None


class CommentAnalysisRequest(BaseModel):
    comments: List[CommentItem]
    max_llm_rows: Optional[int] = None


def correct_star_rating(row: pd.Series) -> float:
    try:
        star = float(row.get("star", 1.0))
    except (ValueError, TypeError):
        return 1.0

    sentiment = str(row.get("degerlendirme", "")).lower()

    if sentiment == "negatif" and star == 5.0:
        return 3.0
    if sentiment == "pozitif" and star in [1.0, 2.0]:
        return 4.0

    return star


def run_comment_pipeline_sync(
    comments_data: List[Dict[str, Any]], max_rows: Optional[int] = None
) -> Dict[str, Any]:
    if not comments_data:
        return {"total": 0, "approved_count": 0, "records": []}

    df = pd.DataFrame(comments_data)

    # 1. Preprocess (PII & Empty drop)
    preprocessor = Preprocessor()
    df = preprocessor.process(df)

    if df.empty:
        return {"total": 0, "approved_count": 0, "records": []}

    # 2. Static Filter
    comment_filter = Filter()
    df_clean, df_flagged = comment_filter.filter(df)

    # 3. LLM Analysis on Clean Rows
    analyzer = Analyzer()
    df_analyzed = analyzer.run_pipeline(df_clean, max_rows=max_rows)

    # 4. Fill defaults for statically flagged rows
    if not df_flagged.empty:
        df_flagged["degerlendirme"] = "negatif"
        df_flagged["kategori"] = df_flagged["flag_category"].str.lower()
        df_flagged["llm_uygunsuz"] = True
        df_flagged["llm_sebep"] = "Statik kural motoru tarafından engellendi."

    # prevent missing columns to be NaN in clean rows
    if not df_analyzed.empty:
        df_analyzed["is_flagged"] = False
        df_analyzed["flag_category"] = None

    # 5. Combine Datasets
    df_final = pd.concat([df_analyzed, df_flagged], ignore_index=True)

    # 6. Star Rating Correction
    if "star" in df_final.columns:
        df_final["star"] = df_final.apply(correct_star_rating, axis=1)

    # 7. Clean mask
    clean_mask = (~df_final["is_flagged"].fillna(False)) & (
        df_final["llm_uygunsuz"].fillna(False) == False
    )

    # keeps the nan as none for pandas
    df_final = df_final.where(pd.notnull(df_final), None)

    all_records = df_final.to_dict(orient="records")
    approved_records = [r for r in all_records if r.get("id") in df_final[clean_mask]["id"].values]

    return {
        "success": True,
        "total_count": len(all_records),
        "approved_count": len(approved_records),
        "approved_comments": approved_records,
        "all_comments": all_records,
    }


# ==============================================================================
# Example Request Body (JSON):
# {
#   "max_llm_rows": 50,  // Optional (int or null)
#   "comments": [
#     {
#       "id": 101,                                       // Optional (int | string)
#       "comment": "Otobüs çok temizdi.",                // Required (string)
#       "star": 5.0,                                     // Optional (default: 5.0)
#       "name": "Ahmet Yılmaz",                          // Optional (string)
#       "email": "ahmet@example.com",                    // Optional (string)
#       "pnr": "PNR12345"                                // Optional (string)
#     },
#     {
#       "id": 102,
#       "comment": "Şoför çok kabaydı ve araç pisti.",
#       "star": 1.0
#     }
#   ]
# }
#
# Example Usage in Laravel:
# $response = Http::post('http://localhost:6161/comments/analyze', [
#     'max_llm_rows' => 50, // optional
#     'comments' => [
#         [
#             'id' => $comment->id,
#             'comment' => $comment->body,
#             'star' => $comment->rating,
#             'name' => $comment->user_name, // optional
#             'email' => $comment->user_email, // optional
#             'pnr' => $comment->pnr // optional
#         ]
#     ]
# ]);

@app.post("/comments/analyze")
async def analyze_comments(request: CommentAnalysisRequest):
    try:
        data = [c.model_dump() for c in request.comments]
        # Run synchronous CPU/network heavy pipeline in background thread
        result = await asyncio.to_thread(
            run_comment_pipeline_sync, data, request.max_llm_rows
        )
        return result
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500, detail=f"Comment analysis failed: {str(e)}"
        )