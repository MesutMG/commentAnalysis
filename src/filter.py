import re
import pandas as pd
from tqdm import tqdm


class Filter:
    def __init__(self):
        self.FILTER_RULES = {
            "PROFANITY": [
                r"\b(s[iı]k(?!ıntı|inti|let|ke)(er[a-z]*|tir[a-z]*|ik[a-z]*|ey[iı]m[a-z]*|t[iı][a-z]*|m[eı][a-z]*|i[sş][a-z]*))\b",
                r"\b(amk|aq|amq)\b",
                r"\b(amc[ıi]k[a-z]*)\b",
                r"\b(o[rro]+spu[a-z]*)\b",
                r"\b(piç|piçler|piçin|piçlik|piçtir|pic|picler)\b",
                r"\b(yarr[a-z]*)\b",
            ],
            "INSULT": [
                r"\b(gerzek|ger[ıi]zekal[ıi][a-z]*)\b",
                r"\b(salak[a-z]*)\b",
                r"\b(aptal[a-z]*)\b",
                r"\b(ahmak[a-z]*)\b",
                r"\b(k[üu]stah[a-z]*)\b",
                r"\b(şerefsiz[a-z]*)\b",
                r"\b(haysiyetsiz[a-z]*)\b",
                r"\b(köpek[a-z]*)\b",
                r"\b(adi[a-z]*)\b",
            ],
            "FRAUD_AND_LEGAL": [
                r"\b(doland[ıi]r[ıi]c[ıi][a-z]*)\b",
                r"\b(sahtek[aâ]r[a-z]*)\b",
                r"\b(h[ıi]rs[ıi]z[a-z]*)\b",
                r"\b(vurguncu[a-z]*)\b",
                r"\b(soyguncu[a-z]*)\b",
                r"\b(gasp[a-z]*)\b",
                r"\b(tefeci[a-z]*)\b",
            ],
            "PHONE_NUMBER": [
                r"(?:\+?90\s*|\b0)?5\d{2}[\s.-]?\d{3}[\s.-]?\d{2}[\s.-]?\d{2}\b",
                r"(?:\+?90\s*|\b0)?[1-9]\d{2}[\s.-]?\d{3}[\s.-]?\d{4}\b",
                r"(?:\+|00)(?:49|44|359|40|1|91)[\s.-]?(?:\(0\)[\s.-]*)?(?:\d[\s.-]?){7,12}\b",
            ],
            "LINKS_AND_DOMAINS": [
                r"https?:\/\/[^\s]+",
                r"\bwww\.[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b",
                r"\b[a-zA-Z0-9.-]+\.(?:com|net|org|com\.tr|gen\.tr|info|biz|site|online|store|xyz)\b(?:\/[^\s]*)?",
            ],
            "SPAM": [
                r"^(test|testing|deneme)(\s*\d+)*$",
                r"^(.)\1{4,}$",  # 'aaaaa', '.....' gibi anlamsız karakter tekrarları
            ],

            "OTHER": [
                r"\b(ter[oö]r[iı]st[a-z]*)",
                r"\b(mafya[a-z]*)",
                r"\b(r[üu][sş]vet[a-z]*)",
            ],
        }

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

        # Flag comments with <= 2 words
        if len(text.split()) <= 2:
            return True, "SHORT_COMMENT"

        # Static regex checks
        for category, patterns in self.COMPILED_RULES.items():
            for pattern in patterns:
                if pattern.search(text):
                    return True, category

        return False, None

    def apply_filter(self, df: pd.DataFrame) -> pd.DataFrame:
        is_flagged_list = []
        flag_category_list = []

        # Added tqdm progress bar here
        for comment in tqdm(df["comment"], desc="Keyword Filtering"):
            is_flagged, category = self.check_comment(comment)
            is_flagged_list.append(is_flagged)
            flag_category_list.append(category)

        df["is_flagged"] = is_flagged_list
        df["flag_category"] = flag_category_list
        return df

    def filter(self, df: pd.DataFrame):
        df = self.apply_filter(df)
        df_clean = df[~df["is_flagged"]].reset_index(drop=True)
        df_flagged = df[df["is_flagged"]].reset_index(drop=True)
        print(f"Clean rows: {len(df_clean)} | Flagged rows: {len(df_flagged)}")
        return df_clean, df_flagged
