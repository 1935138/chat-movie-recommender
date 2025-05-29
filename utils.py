"""
Utility functions for conversation flow and text processing.
"""

import re
from typing import List, Tuple

import pandas as pd


def normalize_title(title):
    # (더빙), (자막), [극장판] 등 제거
    title = re.sub(r'^[\(\[]?(더빙|자막|극장판)[\)\]]?\s*', '', title)
    return title.strip()

# ✅ 최종 추천 응답 생성 함수 (재추천 대응)
def is_follow_up_question(user_input: str, previous_titles: List[str]) -> bool:
    pattern = r"(이 중에|이중에|여기서|영화들 중에|영화 중에|추천받은 영화 중에|추천한 영화 중에|알려준 영화 중에|방금 추천한)"
    if re.search(pattern, user_input, re.IGNORECASE):
        return True
    for title in previous_titles:
        if title in user_input:
            return True
    return False

# ✅ 재추천 판단 함수
def is_retry_request(user_input: str) -> Tuple[bool, str]:
    exclude_keywords = ["제외", "빼고", "빼줘", "빼서", "뺀", "제외하고", "빼고 추천"]
    if any(kw in user_input for kw in exclude_keywords):
        return True, "제외"
    if "다시 추천" in user_input or "다른 영화" in user_input:
        return True, "결합"
    return False, ""

# ✅ 정보형 QA 관련 필요한 함수들

# 1. 추천 요청 판단
def is_recommendation_request(user_input: str) -> bool:
    rec_keywords = ["추천해줘", "추천해", "볼만한", "비슷한 영화", "유사한 영화", "영화 알려줘",'보고싶어','추천해줄래']
    return any(keyword in user_input for keyword in rec_keywords)

# 2. 제작 정보 관련 추천 판단
def is_movie_info_related(user_input: str) -> bool:
    info_keywords = ["배우", "감독", "출연", "제작", "줄거리", "내용"]
    return any(keyword in user_input for keyword in info_keywords)

# ✅ 제작 정보 기반 필터링
def filter_by_movie_info(query: str, df: pd.DataFrame) -> pd.DataFrame:
    info_columns = ["actor", "director", "description", "cp_name"]

    # 1. 검색 키워드 추출
    match = re.search(r"(.*?)(이|가)?\s*(출연|감독|제작|나오는|줄거리|내용).*?(영화)?(추천해줘|알려줘)?", query)
    if not match:
        print("❌ 제작 정보 관련 키워드 매칭 실패")
        return pd.DataFrame()

    keyword = match.group(1).strip()
    print(f"🔎 추출된 검색 키워드: {keyword}")

    # 2. 각 컬럼별로 키워드 포함 여부 필터링
    filtered_df = df[
        df.apply(lambda row: any(keyword in str(row[col]) for col in info_columns), axis=1)
    ]

    print(f"🔍 필터링된 결과 수: {len(filtered_df)}개")

    return filtered_df

# ✅ 유사 추천 여부 판단
def is_similar_recommendation(user_input: str) -> bool:
    return bool(re.search(r"(비슷한|유사한).*영화", user_input))

# ✅ fallback 추천 함수
def truncate_document(text: str, limit=1200) -> str:
    return text if len(text) <= limit else text[:limit] + "..."


