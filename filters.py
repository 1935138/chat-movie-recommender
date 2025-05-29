
"""
Filtering helpers for movie metadata.
"""

import pandas as pd
from typing import Set

def filter_by_information(df, conditions):
    df = df.copy()

    if bool(conditions.get("actor")):
        df = df[df["actor"].astype(str).str.contains(str(conditions["actor"]), na=False)]

    if bool(conditions.get("director")):
        df = df[df["director"].astype(str).str.contains(str(conditions["director"]), na=False)]

    if bool(conditions.get("cp_name")):
        df = df[df["cp_name"].astype(str).str.contains(str(conditions["cp_name"]), na=False)]

    if bool(conditions.get("target_age")):
        df["target_age"] = pd.to_numeric(df["target_age"], errors="coerce")
        df = df[df["target_age"] <= conditions["target_age"]]

    if bool(conditions.get("national_name")):
        df = df[df["national_name"] == conditions["national_name"]]

    return df

def get_content_score(row, user_meta):
    score = 0
    for category, user_keywords in user_meta.items():
        content_keywords = str(row.get(category, "")).split(",")
        matched = set(user_keywords) & set(k.strip() for k in content_keywords)
        score += len(matched)
    return score


# 최근 추천된 콘텐츠 제목 저장용 집합
previous_recommend_titles: Set[str] = set()


# ✅ 추천 콘텐츠 필터링 함수
