
"""
Data loading and preprocessing utilities.
"""

import pandas as pd
from os import path
from langchain.schema import Document

# ===== Excel settings (can be adjusted) =====
excel_file = 'data/movie_data.xlsx'
sheet_input = 'movie_meta_info_update_20250519'
sheet_output = 'movie_combined_docu'

def make_document(row):
    return f"""제목: "{row['title']}"
감독/연출: "{row['director']}"
출연/배우: "{row['actor']}"
제작/배급사: "{row['cp_name']}"
평점: "{row['rating']}"
러닝타임(분): "{row['running_time']}"
줄거리: "{row['description']}"
메타:
- 주제: "{row['Subject']}"
- 장르: "{row['genre']}"
- 감정: "{row['Emotion']}"
- 분위기: "{row['atmosphere']}"
- 캐릭터: "{row['character_A']}"
- 판타지적 요소: "{row['character_B']}"
- 직업적 요소: "{row['character_C']}"
- 사랑 요소: "{row['love']}"
- 가족 요소: "{row['family']}"
- 범죄 요소: "{row['criminal']}"
- 사회 요소: "{row['social_culture']}"
- 자연 요소: "{row['natural_science']}"
- 배경 요소: "{row['background']}"
- 종교 요소: "{row['religion']}"
- 영화 스타일: "{row['style']}"
"""

df = pd.read_excel(excel_file, sheet_name=sheet_input).fillna('')

# document 컬럼 생성
df['document'] = df.apply(make_document, axis=1)

def load_dataframe(file_path: str = excel_file, sheet: str = sheet_input):
    """
    Load the movie metadata spreadsheet into a pandas DataFrame.
    """
    if not path.exists(file_path):
        raise FileNotFoundError(f"엑셀 파일을 찾을 수 없습니다: {file_path}")
    df = pd.read_excel(file_path, sheet_name=sheet).fillna('')
    # 새 컬럼 'document' 생성
    df["document"] = df.apply(make_document, axis=1)
    return df

def build_documents(df):
    """Convert dataframe rows into LangChain-compatible Document objects."""
    docs = [
        Document(page_content=row['document'], metadata={"title": row['title']})
        for _, row in df.iterrows()
    ]

    return docs
