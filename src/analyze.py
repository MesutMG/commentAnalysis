import json
import os
import aiohttp
from pydantic import BaseModel, Field, ValidationError

CONFIG_PATH = "config.json"
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config = json.load(f)

OLLAMA_URL = config.get("ollama_url", "http://localhost:11434/api/generate")
MODEL_NAME = config.get("model_name", "qwen2.5:3b")


# Pydantic Validation Schema
class AnalysisResult(BaseModel):
    degerlendirme: str = Field(default="notr")
    kategori: str = Field(default="diger")
    llm_uygunsuz: bool = Field(default=False)
    llm_sebep: str | None = Field(default=None)


class Analyzer:
    def __init__(self):
        self.PROMPT_TEMPLATE = """Aşağıdaki müşteri yorumunu analiz et ve yalnızca JSON döndür:
Yorum: "{comment}"

JSON formatı:
{{
  "degerlendirme": "pozitif" | "notr" | "negatif",
  "kategori": "dakiklik" | "servis" | "personel" | "temizlik" | "spam" | "diger",
  "llm_uygunsuz": false,
  "llm_sebep": null
}}
"""

    async def analyze_comment_async(
        self, session: aiohttp.ClientSession, comment: str
    ) -> AnalysisResult:
        payload = {
            "model": MODEL_NAME,
            "prompt": self.PROMPT_TEMPLATE.format(comment=comment),
            "stream": False,
            "format": "json",
            "options": {"temperature": 0.1},
        }

        # Generous timeout for local inference
        req_timeout = aiohttp.ClientTimeout(total=180)

        try:
            async with session.post(OLLAMA_URL, json=payload, timeout=req_timeout) as response:
                response.raise_for_status()
                result = await response.json()
                raw_text = result.get("response", "").strip()

                # Clean markdown wrapping if present
                if raw_text.startswith("```"):
                    raw_text = raw_text.strip("`")
                    if raw_text.startswith("json"):
                        raw_text = raw_text[4:].strip()

                res = AnalysisResult.model_validate_json(raw_text)

                # Reset placeholder strings if flagged false
                if not res.llm_uygunsuz:
                    res.llm_sebep = None
                return res

        except Exception as e:
            err_detail = f"{type(e).__name__}: {e}" if str(e) else type(e).__name__
            print(f"\n[Analyzer Error on '{comment[:20]}...']: {err_detail}")
            return AnalysisResult(
                degerlendirme="hata",
                kategori="hata",
                llm_uygunsuz=False,
                llm_sebep=err_detail,
            )
