"""
SQLite helper functions for user interaction logging.
"""

import sqlite3
from contextlib import contextmanager

@contextmanager
def get_db(db_path="movie_recommendation.db"):
    conn = sqlite3.connect(db_path, check_same_thread=False)
    try:
        yield conn
    finally:
        conn.close()

def init_db(db_path="movie_recommendation.db"):
    with get_db() as conn:
        cursor = conn.cursor()

        cursor.executescript("""
        -- 유저 정보
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_name TEXT UNIQUE NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        -- 유저 입력 히스토리 (예: "무서운 영화 추천해줘")
        CREATE TABLE IF NOT EXISTS user_interactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            user_input TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        );

        -- 추천된 영화에 대한 유저의 선택/비선택 피드백
        CREATE TABLE IF NOT EXISTS user_feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            interaction_id INTEGER NOT NULL,
            movie_title TEXT NOT NULL,
            is_selected BOOLEAN DEFAULT 0,
            is_disliked BOOLEAN DEFAULT 0,
            feedback_text TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (interaction_id) REFERENCES user_interactions(id)
        );

        -- 배우/장르/스타일 등 싫어하는 요소 저장
        CREATE TABLE IF NOT EXISTS user_dislikes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            category TEXT NOT NULL,  -- 예: actor, genre, title, etc.
            value TEXT NOT NULL,     -- 예: 류승룡, 공포, 곡성
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        );

        -- 실제 추천된 영화 로그 저장 (재추천 방지 등)
        CREATE TABLE IF NOT EXISTS recommendation_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            interaction_id INTEGER NOT NULL,
            movie_title TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (interaction_id) REFERENCES user_interactions(id)
        );
        """)

        conn.commit()
        conn.close()
    print("✅ 데이터베이스 초기화 완료 (5개 테이블 생성됨)")




def get_or_create_user_id(user_name: str, db_path: str = "movie_recommendation.db") -> int:
    with get_db() as conn:
        cursor = conn.cursor()

        # 1. user_name 존재하는지 확인
        cursor.execute("SELECT id FROM users WHERE user_name = ?", (user_name,))
        row = cursor.fetchone()

        if row:
            user_id = row[0]
        else:
            # 2. 없으면 새로 생성
            cursor.execute("INSERT INTO users (user_name) VALUES (?)", (user_name,))
            conn.commit()
            user_id = cursor.lastrowid

        conn.close()
        return user_id

def create_interaction(user_id: int, user_input: str, db_path: str = "movie_recommendation.db") -> int:
    """사용자 입력 저장 후 interaction_id 반환"""
    with get_db() as conn:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO user_interactions (user_id, user_input) VALUES (?, ?)",
            (user_id, user_input)
        )
        conn.commit()
        interaction_id = cursor.lastrowid
        conn.close()
        return interaction_id

def log_recommendations(interaction_id: int, titles: list[str], db_path: str = "movie_recommendation.db"):
    """추천 영화 목록 recommendation_logs 테이블에 저장"""
    with get_db() as conn:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        for title in titles:
            cursor.execute(
                "INSERT INTO recommendation_logs (interaction_id, movie_title) VALUES (?, ?)",
                (interaction_id, title)
            )
        conn.commit()
        conn.close()

def get_previous_recommendations(user_id: int, db_path: str = "movie_recommendation.db") -> list:
    """해당 유저의 과거 추천된 영화 목록 반환"""
    with get_db() as conn:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT rl.movie_title
            FROM recommendation_logs rl
            JOIN user_interactions ui ON rl.interaction_id = ui.id
            WHERE ui.user_id = ?
        """, (user_id,))
        titles = [row[0] for row in cursor.fetchall()]
        conn.close()
    return titles

def save_feedback(interaction_id: int, movie_title: str, is_selected: bool, is_disliked: bool, feedback_text: str = "", db_path: str = "movie_recommendation.db"):
    """유저의 피드백 (선택 or 싫어요 등) 저장"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO user_feedback (interaction_id, movie_title, is_selected, is_disliked, feedback_text)
            VALUES (?, ?, ?, ?, ?)
        """, (interaction_id, movie_title, is_selected, is_disliked, feedback_text))
        conn.commit()
        conn.close()

def add_user_dislike(user_id: int, category: str, value: str, db_path: str = "movie_recommendation.db"):
    """사용자가 싫어하는 요소 (배우, 장르 등) 저장"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO user_dislikes (user_id, category, value)
            VALUES (?, ?, ?)
        """, (user_id, category, value))
        conn.commit()
        conn.close()

def get_user_dislikes(user_id: int, db_path: str = "movie_recommendation.db") -> list[tuple[str, str]]:
    """해당 유저가 저장한 싫어하는 요소 목록 반환"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT category, value
            FROM user_dislikes
            WHERE user_id = ?
        """, (user_id,))
        dislikes = cursor.fetchall()
        conn.close()
        return dislikes

def show_user_dislikes(user_id: int, db_path: str = "movie_recommendation.db"):
    """특정 유저가 싫어한다고 표시한 요소들을 보기 좋게 출력"""
    with get_db() as conn:
        cursor = conn.cursor()

        cursor.execute("""
            SELECT category, value, created_at
            FROM user_dislikes
            WHERE user_id = ?
            ORDER BY created_at DESC
        """, (user_id,))

        rows = cursor.fetchall()
        conn.close()

    if not rows:
        print(f"⚠️ 사용자 ID {user_id}는 아직 싫어하는 요소를 등록하지 않았습니다.")
        return

    print(f"📋 사용자 ID {user_id}의 싫어하는 요소 목록:")
    for i, (category, value, created_at) in enumerate(rows, 1):
        print(f"{i}. ❌ [{category}] {value} (등록 시점: {created_at})")

def show_user_feedback(user_id: int, db_path: str = "movie_recommendation.db"):
    """특정 유저의 피드백 기록만 보기 좋게 출력"""
    with get_db() as conn:
        cursor = conn.cursor()

        cursor.execute("""
            SELECT uf.id, u.user_name, ui.user_input, uf.movie_title,
                uf.is_selected, uf.is_disliked, uf.feedback_text, ui.created_at
            FROM user_feedback uf
            JOIN user_interactions ui ON uf.interaction_id = ui.id
            JOIN users u ON ui.user_id = u.id
            WHERE u.id = ?
            ORDER BY uf.id ASC
        """, (user_id,))

        rows = cursor.fetchall()
        conn.close()

        if not rows:
            print(f"⚠️ 사용자 ID {user_id}에 대한 피드백이 없습니다.")
            return

        print(f"📋 사용자 ID {user_id}의 피드백 기록")
        for row in rows:
            fid, uname, question, title, selected, disliked, comment, time = row
            print(f"🧾 Feedback ID: {fid}")
            print(f"👤 사용자: {uname}")
            print(f"🗣️ 질문: {question}")
            print(f"🎬 영화: {title}")
            print(f"✅ 선택됨: {bool(selected)} / ❌ 싫어요: {bool(disliked)}")
            print(f"💬 의견: {comment}")
            print(f"🕒 시간: {time}")
            print("-" * 50)

def get_feedback_by_user_id(user_id: int, db_path: str = "movie_recommendation.db"):
    """특정 유저의 피드백 기록만 보기 좋게 출력"""
    with get_db() as conn:
        cursor = conn.cursor()

        cursor.execute("""
            SELECT uf.id, u.user_name, ui.user_input, uf.movie_title,
                uf.is_selected, uf.is_disliked, uf.feedback_text, ui.created_at
            FROM user_feedback uf
            JOIN user_interactions ui ON uf.interaction_id = ui.id
            JOIN users u ON ui.user_id = u.id
            WHERE u.id = ?
            ORDER BY uf.id ASC
        """, (user_id,))

        rows = cursor.fetchall()
        conn.close()

        return rows


def apply_user_filters(df, user_id, selected_title=None):
    disliked_items = get_user_dislikes(user_id)
    previous_titles = get_previous_recommendations(user_id)

    if selected_title:
        previous_titles.append(selected_title)

    disliked_titles = [value for cat, value in disliked_items if cat == "title"]
    filtered_df = df[~df["title"].isin(previous_titles + disliked_titles)]

    for category, value in disliked_items:
        if category != "title" and category in filtered_df.columns:
            col_series = filtered_df[category].fillna("").astype(str)
            filtered_df = filtered_df[~col_series.str.contains(re.escape(value), na=False)]

    print("📌 필터링된 영화 수:", len(filtered_df))
    return filtered_df

init_db()