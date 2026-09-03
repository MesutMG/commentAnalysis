import json
import os
import asyncio
import aiohttp
import pandas as pd
from pydantic import BaseModel, Field, ValidationError

CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "config.json")
config = {}
if os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = json.load(f)

OLLAMA_URL = os.getenv(
    "OLLAMA_BASE_URL", 
    config.get("ollama_url", "http://localhost:11434/api/generate")
)
if not OLLAMA_URL.endswith("/api/generate"):
    OLLAMA_URL = f"{OLLAMA_URL.rstrip('/')}/api/generate"
    
MODEL_NAME = os.getenv("OLLAMA_MODEL",config.get("model_name", "qwen2.5:3b"))


# Pydantic Validation Schema
class AnalysisResult(BaseModel):
    degerlendirme: str = Field(default="notr")
    kategori: str = Field(default="diger")
    llm_uygunsuz: bool = Field(default=False)
    llm_sebep: str | None = Field(default=None)


class Analyzer:
    def __init__(self):
        self.PROMPT_TEMPLATE = """Sen bir müşteri yorumu moderasyon ve analiz motorusun.
            Aşağıdaki yorumu titizlikle analiz et ve SADECE istenen JSON formatında yanıt ver.

            Kurallar:
            1. "degerlendirme":
            - "pozitif": Memnuniyet, övgü, tavsiye, teşekkür.
            - "negatif": Arıza, teknik sorun (klima, aşınmış lastik, ses vb.), kaza riski, gecikme, pislik, kötü muamele, şikayet.
            - "notr": Salt durum tespiti, duygu içermeyen ifadeler.
            (DİKKAT: Cümle kibar olsa bile araç arızası, güvenlik riski veya eksik hizmet kesinlikle "negatif" olarak sınıflandırılmalıdır.)

            2. "llm_uygunsuz" (true/false):
            - TRUE olmalı: Çalışan/personel adı vererek hedef gösterme (örn: "personel Hakan", "danışmadaki Ali"), şahsi tehdit ("mahkemeye vereceğim", "peşini bırakmayacağım"), şantaj veya doğrudan taciz.
            - FALSE olmalı: Sert ve olumsuz olsa dahi yalnızca firmaya veya araca yönelik genel hizmet/ürün eleştirileri ("araç berbattı", "servis rezaletti").

            3. "kategori":
            - "dakiklik", "servis", "personel", "temizlik", "spam", "diger" seçeneklerinden en uygun olanı.

            Yorum: "{comment}"

            Döndürülecek JSON Şeması:
            {{
            "degerlendirme": "pozitif" | "notr" | "negatif",
            "kategori": "dakiklik" | "servis" | "personel" | "temizlik" | "spam" | "diger",
            "llm_uygunsuz": false,
            "llm_sebep": null
            }}
            """

    async def analyze_comment_async(self, session: aiohttp.ClientSession, comment: str) -> AnalysisResult:
        payload = {
            "model": MODEL_NAME,
            "prompt": self.PROMPT_TEMPLATE.format(comment=comment),
            "stream": False,
            "format": "json",
            "options": {"temperature": 0.1},
            "keep_alive": "5m",
        }
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
                llm_uygunsuz=True,
                llm_sebep=err_detail,
            )

    async def process_dataframe_async(self, df: pd.DataFrame, max_rows: int | None = None) -> pd.DataFrame:
        df_target = df.iloc[:max_rows].copy() if max_rows else df.copy()

        if df_target.empty or "comment" not in df_target.columns:
            return df_target

        # Limit concurrent calls so local Ollama doesn't overload or timeout
        semaphore = asyncio.Semaphore(4)

        async with aiohttp.ClientSession() as session:
            async def run_task(text: str):
                async with semaphore:
                    return await self.analyze_comment_async(session, text)

            tasks = [run_task(str(comment)) for comment in df_target["comment"]]

            try:
                from tqdm.asyncio import tqdm_asyncio
                results = await tqdm_asyncio.gather(*tasks, desc="LLM Analysis")
            except ImportError:
                results = await asyncio.gather(*tasks)

        df_target["degerlendirme"] = [r.degerlendirme for r in results]
        df_target["kategori"] = [r.kategori for r in results]
        df_target["llm_uygunsuz"] = [r.llm_uygunsuz for r in results]
        df_target["llm_sebep"] = [r.llm_sebep for r in results]

        return df_target

    def run_pipeline(self, df: pd.DataFrame, max_rows: int | None = None) -> pd.DataFrame:
        if df.empty:
            return df
        return asyncio.run(self.process_dataframe_async(df, max_rows=max_rows))