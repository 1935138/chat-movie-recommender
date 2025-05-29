import streamlit as st
from langchain.schema import Document
from langchain.vectorstores import FAISS

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By

import time
from typing import List
import os
from pathlib import Path
from datetime import datetime

from database import *
from utils import *
from recommender import *
from vector_db import build_vectorstore, build_qa_chain
from data_loader import load_dataframe
from config import model_name, embedding_model_name
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain



st.set_page_config(
    page_title="영화 추천 시스템",
    layout="centered",
)
# -----------------------------------------------------------------------------
# 0. 스타일 로드 -----------------------------------------------------------------
# -----------------------------------------------------------------------------
def load_css():
    """Streamlit 앱에 사용자 정의 CSS 스타일을 로드하고 적용합니다."""
    with open('static/styles.css', encoding="utf-8") as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# 앱 시작 시 CSS 로드
load_css()

def set_branch(branch_key: str):
    """현재 실행 분기를 기록하고 Sidebar를 즉시 갱신합니다."""
    st.session_state.branch = branch_key
    update_sidebar()

# -----------------------------------------------------------------------------
# 0. 모델·데이터 초기화 ---------------------------------------------------------
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner="초기화 중…")
def initialize_models():
    embedding_model = OpenAIEmbeddings(model=embedding_model_name)
    llm = ChatOpenAI(model_name=model_name, temperature=0.3)
    df = load_dataframe()[:100]
    vectorstore_global = build_vectorstore(
        df["document"].apply(lambda x: Document(page_content=x, metadata={})).tolist()
    )
    qa_chain = build_qa_chain(vectorstore_global)
    return embedding_model, llm, df, qa_chain

embedding_model, llm, df, qa_chain = initialize_models()

keyword_columns: List[str] = [
    "Emotion", "Subject", "atmosphere", "background", "character_A", "character_B", "character_C",
    "criminal", "family", "genre", "love", "natural_science", "religion", "social_culture", "style",
]

# -----------------------------------------------------------------------------
# 썸네일 관련 헬퍼 --------------------------------------------------------------
# -----------------------------------------------------------------------------

def get_wavve_thumbnail_cached(movieid):
    cache = st.session_state.thumbnail_cache
    if movieid in cache:
        return cache[movieid]

    try:
        url = f"https://www.wavve.com/player/movie?movieid={movieid}"
        chrome_options = Options()
        chrome_options.add_argument('--headless')
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        driver = webdriver.Chrome(options=chrome_options)
        driver.get(url)
        time.sleep(2)
        img_tag = driver.find_element(By.CSS_SELECTOR, ".detail-view-box .thumb-box .picture-area img")
        img_url = img_tag.get_attribute("src")
        driver.quit()
        if img_url:
            cache[movieid] = img_url
            return img_url
    except Exception as e:
        print("썸네일 크롤링 실패:", e)
    return "static/no_poster.png"



def render_recommendation_thumbnails(key_prefix: str = "", max_items: int = 3):
    """
    추천 영화에 대한 썸네일과 액션 버튼(좋아요/싫어요)을 렌더링합니다.
    `key_prefix`는 다른 컨텍스트에서 호출될 때 버튼의 고유성을 보장하기 위해 버튼 키에 추가됩니다.
    """
    # 추천 결과가 없는 경우
    if "last_recommend_df" not in st.session_state or st.session_state.last_recommend_df is None:
        st.markdown("📭 아직 추천된 영화가 없습니다.")
        return

    df_recommend = st.session_state.last_recommend_df
    rows = list(df_recommend.itertuples())[:max_items]

    # 좋아요/싫어요 상태 저장을 위한 세션 상태 딕셔너리 초기화 (없을 경우)
    if "liked_movies" not in st.session_state:
        st.session_state.liked_movies = {}
    if "disliked_movies" not in st.session_state:
        st.session_state.disliked_movies = {}

    # 추천 항목 수에 따라 동적으로 열을 생성합니다.
    cols = st.columns(len(rows))  

    for idx, row in enumerate(rows):
        with cols[idx]:
            title = row.title
            content_id = row.content_id
            # 썸네일 URL 가져오기 (캐시 사용)
            img_url = get_wavve_thumbnail_cached(content_id) if content_id else None

            if img_url:
                wavve_url = f"https://www.wavve.com/player/movie?movieid={content_id}"
                # Wavve 링크와 함께 영화 포스터 카드 렌더링
                st.markdown(
                    f"""
                    <div class='poster-card'>
                        <a href="{wavve_url}" target="_blank" class="poster-link">
                            <img src="{img_url}" alt="{title}">
                            <p>{title}</p>
                        </a>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            # 영화의 현재 좋아요/싫어요 상태를 확인합니다.
            liked = st.session_state.liked_movies.get(title, False)
            disliked = st.session_state.disliked_movies.get(title, False)

            # 좋아요/싫어요 버튼을 위한 두 개의 열을 생성합니다.
            col1, col2 = st.columns(2)

            with col1:
                # StreamlitDuplicateElementKey 오류를 방지하기 위해 각 버튼에 고유한 키를 사용합니다.
                # 키는 이제 key_prefix, idx, title, user_id를 포함하여 전역 고유성을 보장합니다.
                if st.button("👍 좋아요", key=f"{key_prefix}main_like_{idx}_{title}_{st.session_state.user_id}", disabled=liked):
                    st.session_state.liked_movies[title] = True
                    # 선택적으로, 좋아요 영화에 대한 피드백 메커니즘을 추가하거나 여기에서 데이터베이스를 업데이트할 수 있습니다.
                    print("save_feedbacksave_feedback")
                    save_feedback(
                        interaction_id=st.session_state.interaction_id,
                        movie_title=title,
                        is_selected=True,
                        is_disliked=False,
                        feedback_text=""
                    )

            with col2:
                # 각 버튼에 고유한 키를 사용합니다.
                if st.button("👎 싫어요", key=f"{key_prefix}main_dislike_{idx}_{title}_{st.session_state.user_id}", disabled=disliked):
                    try:
                        # 데이터베이스에 사용자 싫어요 목록에 영화를 추가합니다.
                        add_user_dislike(st.session_state.user_id, "title", title)
                        save_feedback(
                            interaction_id=st.session_state.interaction_id,
                            movie_title=title,
                            is_selected=False,
                            is_disliked=True,
                            feedback_text=""
                        )
                        st.session_state.disliked_movies[title] = True
                    except Exception as e:
                        st.error(f"싫어요 저장 중 오류가 발생했습니다: {str(e)}")



# 초기 로드 시 `last_recommend_df`가 존재하면 추천을 렌더링합니다.
# 이 특정 호출은 사이드바의 호출과 구별하기 위해 키 접두사가 필요합니다.
if st.session_state.get("last_recommend_df") is not None:
    render_recommendation_thumbnails(key_prefix="initial_render_")


# -----------------------------------------------------------------------------
# 1. 세션 상태 초기화 -----------------------------------------------------------
# -----------------------------------------------------------------------------
if "__initialized__" not in st.session_state:    
    st.session_state.__initialized__ = True
    st.session_state.user_name = ""
    st.session_state.user_id = None
    st.session_state.selected_title = None
    st.session_state.last_recommend_df = None
    st.session_state.last_recommend_query = None
    st.session_state.first_turn = True
    st.session_state.thumbnail_cache = {}
    st.session_state.previous_titles = set()
    st.session_state.branch = "대기"
    st.session_state.chat_history = []  # 전체 채팅 기록을 저장합니다.
    st.session_state.show_recommendations = False # 사이드바에 이전 추천을 표시하기 위한 토글
    st.session_state.interaction_id = None

# -----------------------------------------------------------------------------
# 채팅 기록 관리 함수 -----------------------------------------------------------
# -----------------------------------------------------------------------------
def add_to_chat_history(role: str, content: str, branch: str):
    """채팅 메시지를 세션 상태의 채팅 기록에 추가합니다."""
    st.session_state.chat_history.append({
        "role": role,
        "content": content,
        "branch": branch,
        "timestamp": datetime.now().isoformat()
    })


# -----------------------------------------------------------------------------
# 2. Sidebar (동적 표시) --------------------------------------------------------
# -----------------------------------------------------------------------------
sidebar_placeholder = st.sidebar.empty()
branch_labels = [
    ("첫 추천", "first"),
    ("후속 질문", "follow_up"),
    ("유사 추천", "similar"),
    ("재추천", "retry"),
    ("일반 QA", "qa"),
]

def update_sidebar():
    """
    사이드바 콘텐츠를 렌더링하고 업데이트합니다. 여기에는 사용자 정보, 분기 상태,
    이전에 추천된 영화가 포함됩니다.
    """
    sidebar_placeholder.empty()               # 이전 사이드바 콘텐츠를 지웁니다.
    container = sidebar_placeholder.container()
    container.title("✨ 감정 매칭 추천")

    if st.session_state.user_id:
        container.markdown(f"**👤 {st.session_state.user_name}**")
        
        # 브랜치가 변경될 때마다 새로운 버튼 key 생성
        for label, key in branch_labels:
            icon = "🟢" if st.session_state.branch == key else "⚪️"
            container.markdown(
                f"""
                <div class='branch-label'>
                    <span class='icon'>{icon}</span>
                    <span class='text'>{label}</span>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        container.markdown("---")
        
        # 사이드바 버튼 스타일 (캡슐화를 위해 함수 내부에 정의)
        container.markdown(
            """
            <style>
            /* 사이드바 버튼 컨테이너 스타일 */
            div[data-testid="stButton"] {
                width: 100% !important;
                display: block !important;
                text-align: center !important;
                margin: 0.3rem 0 !important;
            }
            
            /* 사이드바 버튼 스타일 */
            div[data-testid="stButton"] > button {
                width: 90% !important;
                margin: 0 auto !important;
                padding: 0.5rem 1.2rem !important;
                background-color: var(--sogang-red-lighter) !important;
                color: var(--sogang-red) !important;
                border: 1px solid var(--sogang-red) !important;
                border-radius: 8px !important;
                font-weight: 500 !important;
                display: inline-block !important;
                text-align: center !important;
                transition: all 0.2s ease !important;
                white-space: nowrap !important;
            }
            
            div[data-testid="stButton"] > button:hover {
                background-color: var(--sogang-red) !important;
                color: white !important;
            }
            
            /* 추천 목록 스타일 - 박스 제거 */
            .recommendation-item {
                padding: 0.3rem 0;
                font-size: 0.9rem;
                color: var(--text-color);
            }
            </style>
            """,
            unsafe_allow_html=True
        )
        
        # 이전에 추천된 영화 표시/숨기기 토글 버튼
        # 키는 `show_recommendations`의 현재 상태를 기반으로 하여 각 사이드바 렌더링 주기 내에서 고유하도록 합니다.
        container.button(
            "🎬 추천 받은 영화",
            key=f"toggle_recommendations_{st.session_state.show_recommendations}",
            use_container_width=True,
            type="primary" if st.session_state.show_recommendations else "secondary"
        )
        st.session_state.show_recommendations = not st.session_state.show_recommendations
            # 변경 사항을 반영하기 위해 버튼 클릭 직후 다시 실행합니다.
           # st.rerun()

        # 토글이 활성화되면 이전에 추천된 영화를 표시합니다.
        if st.session_state.show_recommendations:
            previous_titles = get_previous_recommendations(st.session_state.user_id) # 과거 추천을 가져오는 함수

            if previous_titles:
                
                container.markdown("### 이전에 추천 받은 영화")
                for i, title in enumerate(previous_titles, 1):
                    container.markdown(
                        f"""
                        <div class="recommendation-item">
                            <strong>{i}.</strong> {title}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
            else:
                container.markdown("아직 추천 받은 영화가 없어요.")

        # `last_recommend_df`가 존재하면 마지막으로 추천된 영화의 썸네일을 표시합니다.
        # 메인 콘텐츠 영역과의 충돌을 피하기 위해 여기에 고유한 접두사를 추가합니다.
        if st.session_state.get("last_recommend_df") is not None:
            container.markdown("### 🎬 마지막 추천 영화")
            render_recommendation_thumbnails(key_prefix="sidebar_") # 다른 키 접두사 사용


# 사이드바 콘텐츠를 렌더링하기 위해 update_sidebar를 호출합니다.
update_sidebar()
# -----------------------------------------------------------------------------
# 3. 사용자 이름 입력 (메인) ---------------------------------------------------- 사용자 이름 입력 (메인) ----------------------------------------------------
# -----------------------------------------------------------------------------
if st.session_state.user_id is None:
    st.markdown(
        """
        <div style="text-align: center; margin: 2rem 0;">
            <h1 style="color: var(--sogang-red); font-size: 2.5rem; margin-bottom: 1rem;">
                안녕하세요, 사만다 입니다. 🎬
            </h1>
            <p style="color: var(--text-color-light); font-size: 1.1rem; line-height: 1.6;">
                당신의 이름을 알려주시겠어요?<br>
                함께 이야기 나누며 마음에 드는 영화를 찾아볼까요? ✨
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
    name_input = st.text_input("이름을 입력해주세요", key="name_input")
    if name_input:
        st.session_state.user_name = name_input.strip()
        # 입력된 이름을 기반으로 사용자 ID를 가져오거나 생성합니다.
        st.session_state.user_id = get_or_create_user_id(st.session_state.user_name)
        st.rerun() # 새 사용자 ID로 앱을 새로 고치기 위해 다시 실행합니다.
    st.stop() # 이름이 입력될 때까지 더 이상 실행하지 않습니다.

# -----------------------------------------------------------------------------
# 4. 메인 대화 입력 -------------------------------------------------------------
# -----------------------------------------------------------------------------
# 이미지 경로 설정
STATIC_DIR = "static"
SOGANG_HAWK_AVATAR = os.path.join(STATIC_DIR, "sogang_hawk.png")
USER_AVATAR = os.path.join(STATIC_DIR, "user_avatar.png")

# 이미지 파일 존재 확인
if not os.path.exists(SOGANG_HAWK_AVATAR):
    # 기본 이모지로 대체
    SOGANG_HAWK_AVATAR = "🦅"
if not os.path.exists(USER_AVATAR):
    # 기본 이모지로 대체
    USER_AVATAR = "👤"

# 메인 채팅 로직 수정
if st.session_state.first_turn:
    welcome_message = f"안녕하세요, {st.session_state.user_name}님! ✨\n오늘 하루는 어떠셨나요? 기분 좋은 일이 있었나요?\n지금 기분이나 끌리는 분위기를 말씀해주시면 딱 맞는 영화를 골라드릴게요!😊"
    with st.chat_message("assistant", avatar=SOGANG_HAWK_AVATAR):
        st.markdown(welcome_message)
    # 환영 메시지 저장
    add_to_chat_history("assistant", welcome_message, st.session_state.branch)

# 사용자 입력 처리
user_query = st.chat_input("오늘 하루 너무 힘들었어. 스트레스가 싹 풀릴 만큼 통쾌한 액션 영화 추천 해줘")
if not user_query:
    st.stop()
else: 
    interaction_id = create_interaction(st.session_state.user_id, user_query)
    st.session_state.interaction_id = interaction_id

# 사용자 메시지 표시 및 저장
with st.chat_message("user", avatar=USER_AVATAR):
    st.markdown(user_query)
add_to_chat_history("user", user_query, st.session_state.branch)

# 봇 응답 처리 및 저장
if user_query.lower() in {"exit", "quit", "종료", "고마워 사만다"} or "사만다 고마워" in user_query:
    goodbye_message = "👋 대화를 종료합니다. 좋은 하루 되세요! 💕"
    with st.chat_message("assistant", avatar=SOGANG_HAWK_AVATAR):
        st.markdown(goodbye_message)
    add_to_chat_history("assistant", goodbye_message, st.session_state.branch)
    st.stop()

# -----------------------------------------------------------------------------
# 5‑0. 완료 처리 ---------------------------------------------------------------
# -----------------------------------------------------------------------------
if "완료" in user_query:
    if st.session_state.last_recommend_df is None or st.session_state.last_recommend_df.empty:
        error_message = "⚠️ 이전에 추천된 영화가 없습니다. 먼저 추천을 받아주세요."
        with st.chat_message("assistant", avatar=SOGANG_HAWK_AVATAR):
            st.markdown(error_message)
        add_to_chat_history("assistant", error_message, st.session_state.branch)
        st.stop()
    set_branch("complete")
    sel_title = handle_completion(user_query, st.session_state.last_recommend_df, interaction_id, st.session_state.user_id)
    if sel_title:
        st.session_state.selected_title = sel_title
        complete_message = "✅ 선택 완료! 좋은 감상 되세요."
        with st.chat_message("assistant", avatar=SOGANG_HAWK_AVATAR):
            st.markdown(complete_message)
        add_to_chat_history("assistant", complete_message, st.session_state.branch)
    st.stop()

# -----------------------------------------------------------------------------
# 5‑1. 이전 추천 기반 분기 ------------------------------------------------------
# -----------------------------------------------------------------------------
if (
    not st.session_state.first_turn and
    st.session_state.last_recommend_df is not None and
    not st.session_state.last_recommend_df.empty
):
    # (a) 후속 질문
    if is_follow_up_question(user_query, st.session_state.last_recommend_df["title"].tolist()):
        set_branch("follow_up")
        st.chat_message("assistant", avatar=SOGANG_HAWK_AVATAR).write("📌 후속 질문으로 판단됨 → 이전 추천 콘텐츠에서 검색 중…")
        local_docs = [
            Document(page_content=truncate_document(r.document), metadata={"title": r.title})
            for r in st.session_state.last_recommend_df.itertuples()
        ]
        answer = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=FAISS.from_documents(local_docs, embedding_model).as_retriever(),
            memory=None
        ).invoke({"question": user_query, "chat_history": []})["answer"]
        st.chat_message("assistant", avatar=SOGANG_HAWK_AVATAR).write(answer)
        st.stop()

    # (b) 유사 추천
    if is_similar_recommendation(user_query):
        set_branch("similar")
        df_sim = handle_similar_recommendation(
            user_query, df, st.session_state.user_id, st.session_state.selected_title, extract_user_meta, keyword_columns
        )
        if df_sim.empty:
            st.chat_message("assistant", avatar=SOGANG_HAWK_AVATAR).write("죄송해요, 유사한 콘텐츠를 찾지 못했어요.")
            st.stop()
        resp = generate_recommendation_response(user_query, df_sim, st.session_state.user_name)
        st.chat_message("assistant", avatar=SOGANG_HAWK_AVATAR).write(resp)
        log_recommendations(interaction_id, df_sim["title"].tolist())
        st.session_state.previous_titles.update(df_sim["title"].tolist())
        st.session_state.last_recommend_df = df_sim.copy()
        st.session_state.last_recommend_query = user_query

        # 메인 영역에 새 추천을 렌더링합니다.
        render_recommendation_thumbnails(key_prefix="similar_recommend_")
        st.stop()

    # (c) 재추천
    is_retry, _ = is_retry_request(user_query)
    if is_retry and st.session_state.last_recommend_query:
        set_branch("retry")
        merged_query = st.session_state.last_recommend_query
        user_meta = extract_user_meta(merged_query)
        df_ret = handle_recommendation(
            df,
            st.session_state.user_id,
            user_meta,
        )
    else:
        merged_query = user_query
        user_meta = extract_user_meta(merged_query)
        st.session_state.last_recommend_query = merged_query
        df_ret = handle_recommendation(
            df,
            st.session_state.user_id,
            user_meta,
        )

    if df_ret is None or df_ret.empty:
        st.chat_message("assistant", avatar=SOGANG_HAWK_AVATAR).write("죄송해요, 추천할 콘텐츠를 찾지 못했어요.")
        st.stop()

    resp = generate_recommendation_response(merged_query, df_ret, st.session_state.user_name, is_retry=is_retry)
    st.chat_message("assistant", avatar=SOGANG_HAWK_AVATAR).write(resp)
    log_recommendations(interaction_id, df_ret["title"].tolist())
    st.session_state.previous_titles.update(df_ret["title"].tolist())
    st.session_state.last_recommend_df = df_ret.copy()
    st.session_state.first_turn = False

    render_recommendation_thumbnails(key_prefix="retry_recommend_")
    st.stop()

# -----------------------------------------------------------------------------
# 5‑2. 첫 추천 -----------------------------------------------------------------
# -----------------------------------------------------------------------------
if st.session_state.first_turn and is_recommendation_request(user_query):
    set_branch("first")
    user_meta = extract_user_meta(user_query)
    df_first = handle_recommendation(
        df,
        st.session_state.user_id,
        user_meta,
    )
    if df_first.empty:
        st.chat_message("assistant", avatar=SOGANG_HAWK_AVATAR).write("죄송해요, 적절한 콘텐츠를 찾지 못했어요.")
        st.stop()
    resp = generate_recommendation_response(user_query, df_first, st.session_state.user_name)
    st.chat_message("assistant").write(resp)
    render_recommendation_thumbnails(df_first)
    log_recommendations(interaction_id, df_first["title"].tolist())
    
    st.session_state.previous_titles.update(df_first["title"].tolist())
    st.session_state.last_recommend_df = df_first.copy()
    st.session_state.first_turn = False

    render_recommendation_thumbnails(key_prefix="first_recommend_")

    st.stop()

# -----------------------------------------------------------------------------
# 5‑3. 일반 QA -----------------------------------------------------------------
# -----------------------------------------------------------------------------
else:
    set_branch("qa")
    answer = qa_chain.invoke({"question": user_query})["answer"]
    st.chat_message("assistant", avatar=SOGANG_HAWK_AVATAR).write(answer)



# 폰트 로드
st.markdown("""
<link href="https://api.fontshare.com/v2/css?f[]=satoshi@400,500,600&display=swap" rel="stylesheet">
<link href="https://api.fontshare.com/v2/css?f[]=clash-display@400,500,600&display=swap" rel="stylesheet">
""", unsafe_allow_html=True)



