# Static swear word checks
import os
import re
import pandas as pd


class Filter:
    def __init__(self):
        self.FILTER_RULES = {
            "PROFANITY": [
                r"\b(sik[a-z]*)\b",
                r"\b(amk[a-z]*)\b",
                r"\b(aq[a-z]*)\b",
                r"\b(amq[a-z]*)\b",
                r"\b(amc[ıi]k[a-z]*)\b",
                r"\b([oö]r[oö]s[a-z]*)\b",
                r"\b(pi[cç][a-z]*)\b",
                r"\b(yarr[a-z]*)\b",
            ],
            "INSULT": [
                r"\b(gerzek|ger[ıi]zekal[ıi][a-z]*)\b",
                r"\b(salak[a-z]*)\b",
                r"\b(aptal[a-z]*)\b",
                r"\b(k[oö]pek[a-z]*)\b",
                r"\b(ahmak[a-z]*)\b",
                r"\b(k[üu]stah[a-z]*\b)",
            ],
            "OTHER": [
                r"\b(ter[oö]r[iı]st[a-z]*)",
                r"\b(mafya[a-z]*)",
            ],
        }

        # Pre-compiling patterns once upfront avoids this overhead during filtering.
        self.COMPILED_RULES = {}
        for category, patterns in self.FILTER_RULES.items():
            pattern_list = []
            for pattern in patterns:
                pattern_list.append(re.compile(pattern, re.IGNORECASE))
            self.COMPILED_RULES[category] = pattern_list

    def normalize_turkish(self, text: str) -> str:
        if not isinstance(text, str):
            return ""
        text = text.replace("İ", "i").replace("I", "ı")
        return text.lower().strip()


    def check_comment(self, comment: str) -> tuple[bool, str | None]:
        text = self.normalize_turkish(comment)

        for category, patterns in self.COMPILED_RULES.items():
            for pattern in patterns:
                if pattern.search(text):
                    return True, category

        return False, None


    def apply_filter(self, df: pd.DataFrame) -> pd.DataFrame:
        is_flagged_list = []
        flag_category_list = []

        for comment in df["comment"]:
            is_flagged, category = self.check_comment(comment)
            is_flagged_list.append(is_flagged)
            flag_category_list.append(category)

        df["is_flagged"] = is_flagged_list
        df["flag_category"] = flag_category_list
        return df


    def filter(self, df: pd.DataFrame):
        df = self.apply_filter(df)

        df_clean_for_llm = df[~df["is_flagged"]]
        df_flagged = df[df["is_flagged"]]

        print(f"Clean rows: {len(df_clean_for_llm)} | Flagged rows: {len(df_flagged)}")

        return (df_clean_for_llm, df_flagged)
