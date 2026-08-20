# Static swear word checks
import re
import pandas as pd

DATA_PATH = "data/example_comments_copy3_cleaned.csv"

FILTER_RULES = {
    "PROFANITY": [
        r"\b(s[iı]k[a-z]*)\b",
        r"\b(amk|aq|amq)\b",
        r"\b(orospu[a-z]*)\b",
        r"\b(pi[cç][a-z]*)\b",
        r"\b(yarra[a-z]*)\b",
    ],
    "INSULT": [
        r"\b(gerizekal[ıi][a-z]*)\b",
        r"\b(salak[a-z]*)\b",
        r"\b(aptal[a-z]*)\b",
        r"\b(k[oö]pek[a-z]*)\b",
        r"\b(ahmak[a-z]*)\b",
    ],
}

# Pre-compiling patterns once upfront avoids this overhead during filtering.
COMPILED_RULES = {}
for category, patterns in FILTER_RULES.items():
    pattern_list = []
    for pattern in patterns:
        pattern_list.append(re.compile(pattern, re.IGNORECASE))
    COMPILED_RULES[category] = pattern_list


def normalize_turkish(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.replace("İ", "i").replace("I", "ı")
    return text.lower().strip()


def check_comment(comment: str) -> tuple[bool, str | None]:
    text = normalize_turkish(comment)

    for category, patterns in COMPILED_RULES.items():
        for pattern in patterns:
            if pattern.search(text):
                return True, category

    return False, None


def apply_filter(df: pd.DataFrame) -> pd.DataFrame:
    is_flagged_list = []
    flag_category_list = []

    for comment in df["comment"]:
        is_flagged, category = check_comment(comment)
        is_flagged_list.append(is_flagged)
        flag_category_list.append(category)

    df["is_flagged"] = is_flagged_list
    df["flag_category"] = flag_category_list
    return df


if __name__ == "__main__":
    df = pd.read_csv(DATA_PATH, sep=";")
    df = apply_filter(df)

    df_clean_for_llm = df[~df["is_flagged"]]
    df_flagged = df[df["is_flagged"]]

    print(f"Clean rows: {len(df_clean_for_llm)} | Flagged rows: {len(df_flagged)}")