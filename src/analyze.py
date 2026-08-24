import json
import os
import pandas as pd
import requests
from tqdm import tqdm

CONFIG_PATH = "config.json"
with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

OLLAMA_URL = config.get("ollama_url", "http://localhost:11434/api/generate")
MODEL_NAME = config.get("model_name", "qwen2.5:3b")


class Analyzer:
    def __init__(self):
        self.PROMPT_TEMPLATE = """Aşağıdaki müşteri yorumunu analiz et.
        Yorum: "{comment}"

        YALNIZCA aşağıdaki JSON formatında ve Türkçe değerlerle yanıt ver:
        {{
        "degerlendirme": "pozitif" | "notr" | "negatif",
        "kategori": "servis" | "dakiklik" | "personel" | "temizlik" | "spam" | "diger",
        "llm_uygunsuz": true | false,
        "llm_sebep": "Uygunsuzluk veya şikayet nedeni (Türkçe) ya da null"
        }}
        """

    def analyze_comment(self, comment: str) -> dict:
        payload = {
            "model": MODEL_NAME,
            "prompt": self.PROMPT_TEMPLATE.format(comment=comment),
            "stream": False,
            "format": "json",
            "options": {
                "temperature": 0.1  # Low temperature ensures deterministic output
            },
        }

        try:
            response = requests.post(OLLAMA_URL, json=payload, timeout=30)
            response.raise_for_status()
            result = response.json()
            return json.loads(result["response"])
        except Exception as e:
            return {
                "degerlendirme": "hata",
                "kategori": "hata",
                "llm_uygunsuz": False,
                "llm_sebep": str(e),
            }

    def run_pipeline(self, df: pd.DataFrame, max_rows: int = None) -> pd.DataFrame:
        df_subset = df.head(max_rows).copy() if max_rows else df.copy()

        degerlendirmeler = []
        kategoriler = []
        uygunsuzluklar = []
        sebepler = []

        for comment in tqdm(df_subset["comment"], desc="LLM Inference"):
            res = self.analyze_comment(str(comment))
            degerlendirmeler.append(res.get("degerlendirme", "notr"))
            kategoriler.append(res.get("kategori", "diger"))
            uygunsuzluklar.append(res.get("llm_uygunsuz", False))
            sebepler.append(res.get("llm_sebep", None))

        df_subset["degerlendirme"] = degerlendirmeler
        df_subset["kategori"] = kategoriler
        df_subset["llm_uygunsuz"] = uygunsuzluklar
        df_subset["llm_sebep"] = sebepler

        return df_subset
