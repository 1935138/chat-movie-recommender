import uuid

import streamlit as st
from langchain.schema import Document
from langchain_community.vectorstores import FAISS

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By

import time
from datetime import datetime

from recommender import *
from vector_db import build_vectorstore, build_qa_chain
from data_loader import load_dataframe
from config import model_name, embedding_model_name
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain

# Streamlit 페이지 기본 설정
st.set_page_config(
    page_title="영화 추천 시스템",
    layout="centered",
)

# 스타일 로드
def load_css():
    """Streamlit 앱에 사용자 정의 CSS 스타일을 로드하고 적용합니다."""
    try:
        with open('static/styles.css', encoding="utf-8") as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except FileNotFoundError:
        st.error("Error: static/styles.css 파일을 찾을 수 없습니다. CSS 파일이 올바른 위치에 있는지 확인하세요.")

# 앱 시작 시 CSS 로드 함수 호출
load_css()

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

# 초기화 함수를 호출하여 필요한 모델과 데이터를 가져옴
embedding_model, llm, df, qa_chain = initialize_models()

# 영화 추천에 사용될 키워드 컬럼 리스트 정의
keyword_columns: List[str] = [
    "Emotion", "Subject", "atmosphere", "background", "character_A", "character_B", "character_C",
    "criminal", "family", "genre", "love", "natural_science", "religion", "social_culture", "style",
]

# 사이드바의 빈 플레이스홀더를 생성
sidebar_placeholder = st.sidebar.empty()
# 대화의 분기(branch)를 나타내는 레이블과 키 정의
branch_labels = [
    ("첫 추천", "first"),
    ("후속 질문", "follow_up"),
    ("유사 추천", "similar"),
    ("재추천", "retry"),
    ("일반 QA", "qa"),
]

# 이미지 경로 설정
STATIC_DIR = "static"
SOGANG_HAWK_AVATAR = os.path.join(STATIC_DIR, "sogang_hawk.png")
USER_AVATAR = os.path.join(STATIC_DIR, "user_avatar.png")

# 이미지 파일 존재 여부 확인 및 대체 이미지(이모지) 설정
if not os.path.exists(SOGANG_HAWK_AVATAR):
    # 기본 이모지로 대체
    SOGANG_HAWK_AVATAR = "🦅"
if not os.path.exists(USER_AVATAR):
    # 기본 이모지로 대체
    USER_AVATAR = "👤"


def set_branch(branch_key: str):
    """
    현재 대화의 분기(branch)를 세션 상태에 기록하고,
    사이드바의 브랜치 상태 표시를 즉시 갱신합니다.
    """
    print("========set_branch", branch_key)
    st.session_state.branch = branch_key
    update_sidebar()

# -----------------------------------------------------------------------------
# 썸네일 관련 헬퍼 --------------------------------------------------------------
# -----------------------------------------------------------------------------
def get_wavve_thumbnail_cached(movieid):
    """
    Wavve에서 영화 썸네일 URL을 가져오고 캐시합니다.
    만약 캐시에 이미 URL이 있으면 캐시된 값을 반환합니다.
    """
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



def render_recommendation_thumbnails(df_recommend, key_prefix: str = "", max_items: int = 3):
    """
    추천 영화에 대한 썸네일과 액션 버튼(좋아요/싫어요)을 렌더링합니다.
    `key_prefix`는 다른 컨텍스트에서 호출될 때 버튼의 고유성을 보장하기 위해 버튼 키에 추가됩니다.
    """
    # 추천 결과가 없는 경우
    if df_recommend is None:
        st.markdown("📭 아직 추천된 영화가 없습니다.")
        return

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

def initialization():
    """
    메소드 초기화
    """
    st.session_state.__initialized__ = True
    st.session_state.user_name = ""
    st.session_state.user_id = None
    st.session_state.selected_title = None
    st.session_state.last_recommend_df = None
    st.session_state.last_recommend_query = None
    st.session_state.first_turn = True
    st.session_state.thumbnail_cache = {}
    st.session_state.previous_titles = set()
    st.session_state.branch = ""
    st.session_state.chat_history = []  # 전체 채팅 기록을 저장합니다.
    st.session_state.show_recommendations = False  # 사이드바에 이전 추천을 표시하기 위한 토글
    st.session_state.interaction_id = None



# -----------------------------------------------------------------------------
# 1. 세션 상태 초기화 -----------------------------------------------------------
# -----------------------------------------------------------------------------
if "__initialized__" not in st.session_state:
    initialization()


# -----------------------------------------------------------------------------
# 채팅 기록 관리 함수 -----------------------------------------------------------
# -----------------------------------------------------------------------------
def add_to_chat_history(role: str, content: str, key_prefix: str = "", message_type: str = "text"):
    """
        채팅 메시지를 세션 상태의 채팅 기록에 추가합니다.
        데이터프레임도 저장할 수 있도록 확장되었습니다.
        """
    item_to_add = {
        "role": role,
        "branch": st.session_state.branch,
        "timestamp": datetime.now().isoformat(),
        "message_type": message_type
    }

    if message_type == "dataframe" and isinstance(content, pd.DataFrame):
        item_to_add["content"] = content.to_json(orient="records")
        item_to_add["key_prefix"] = key_prefix
    else:
        item_to_add["content"] = str(content)
        item_to_add["key_prefix"] = ""

    st.session_state.chat_history.append(item_to_add)

# --- 화면에 채팅 기록 표시하는 로직 ---
def display_chat_history():
    """세션 상태에 저장된 채팅 기록을 화면에 표시합니다."""
    for message in st.session_state.chat_history:
        avatar = SOGANG_HAWK_AVATAR
        if message["role"] == "user":
            avatar = USER_AVATAR

        with st.chat_message(message["role"], avatar = avatar):
            # 여기가 바로 message["content"]를 화면에 표시하는 부분입니다.
            if message["message_type"] == "dataframe":
                try:
                    df = pd.read_json(message["content"], orient="records")
                    st.write("다음과 같은 추천 결과가 있습니다:")
                    st.dataframe(df)  # 데이터프레임을 Streamlit의 dataframe으로 표시
                    render_recommendation_thumbnails(df, message["key_prefix"])
                except Exception as e:
                    st.warning(f"추천 데이터프레임을 로드하는 데 실패했습니다: {e}")
                    st.write(message["content"])  # 오류 시 원본 문자열이라도 표시
            else:  # message_type == "text"
                st.write(message["content"])  # 일반 텍스트 메시지 표시

if st.session_state.get("chat_history") is not None:
    display_chat_history()

# -----------------------------------------------------------------------------
# 2. Sidebar (동적 표시) --------------------------------------------------------
# -----------------------------------------------------------------------------
# --- 새로운 함수 1: 브랜치 상태 표시 ---
def render_branch_status(container):
    """
    사이드바 컨테이너에 브랜치 레이블과 현재 활성 브랜치 상태를 표시합니다.
    """
    for label, key in branch_labels:
        icon = "🟢" if st.session_state.branch == key else "⚪️"
        container.markdown(
            f"""
            <div class='branch-label'>
                <span class='icon'>{icon}</span>
                <span id='branch-text-{key}' class='text'>{label}</span>
            </div>
            """,
            unsafe_allow_html=True
        )
    container.markdown("---")  # 브랜치 섹션과 다음 섹션 구분


def render_previous_recommendations(container):
    """
    사이드바 컨테이너에 이전에 추천된 영화 목록을 토글 버튼과 함께 표시합니다.
    """
    container.markdown(
        """
        <style>
        /* 사이드바 버튼 컨테이너 스타일 */
        div[data-testid="stButton"] {
            width: 100% !important;
            display: block !important;
            text-align: center !important;
        }

        /* 사이드바 버튼 스타일 */
        div[data-testid="stButton"] > button {
            width: 90% !important;
            margin: 0 auto !important;
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

    if container.button(
            "🎬 추천 받은 영화",
            key=f"toggle_recommendations_{st.session_state.show_recommendations}",  # 고유 키 생성
            use_container_width=True,
            type="primary" if st.session_state.show_recommendations else "secondary"
    ):
        st.session_state.show_recommendations = not st.session_state.show_recommendations
        st.rerun()  # 상태 변경 후 Streamlit 앱을 다시 실행하여 UI 업데이트

    # 토글이 활성화되면 이전에 추천된 영화를 표시합니다.
    if st.session_state.show_recommendations:
        previous_titles = get_previous_recommendations(st.session_state.user_id)

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


# --- 메인 업데이트 함수 (두 함수를 호출) ---
def update_sidebar():
    """
    사이드바 콘텐츠를 렌더링하고 업데이트합니다. 여기에는 사용자 정보, 분기 상태,
    이전에 추천된 영화가 포함됩니다.
    """
    sidebar_placeholder.empty()  # 이전 사이드바 콘텐츠를 지웁니다.
    container = sidebar_placeholder.container()
    container.title("✨ 감정 매칭 추천")

    if st.session_state.user_id:
        container.image(
            "static/user_avatar.png",
            caption=f"{st.session_state.user_name}님"
        )

        # 분리된 함수들을 호출하여 사이드바 콘텐츠를 구성합니다.
        render_branch_status(container)

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
# 메인 채팅 로직 수정
if st.session_state.first_turn and (st.session_state.chat_history is None or st.session_state.chat_history == []):

    welcome_message = f"안녕하세요, {st.session_state.user_name}님! ✨\n오늘 하루는 어떠셨나요? 기분 좋은 일이 있었나요?\n지금 기분이나 끌리는 분위기를 말씀해주시면 딱 맞는 영화를 골라드릴게요!😊"
    with st.chat_message("assistant", avatar=SOGANG_HAWK_AVATAR):
        st.markdown(welcome_message)
    # 환영 메시지 저장
    add_to_chat_history("assistant", welcome_message)


# 사용자 입력 처리
if st.session_state.last_recommend_df is not None:
    prompt = "더 물어보고 싶은게 있을까요? 없으면 '고마워 사만다'라고 말해주세요"
else :
    prompt = "오늘 하루 너무 힘들었어. 스트레스가 싹 풀릴 만큼 통쾌한 액션 영화 추천 해줘"
user_query = st.chat_input(prompt)

if not user_query:
    st.stop()
else: 
    interaction_id = create_interaction(st.session_state.user_id, user_query)
    st.session_state.interaction_id = interaction_id

# 사용자 메시지 표시 및 저장
with st.chat_message("user", avatar=USER_AVATAR):
    st.markdown(user_query)
    add_to_chat_history("user", user_query)

# 봇 응답 처리 및 저장
if user_query.lower() in {"exit", "quit", "종료", "고마워 사만다"} or "사만다 고마워" in user_query:
    goodbye_message = "👋 대화를 종료합니다. 좋은 하루 되세요! 💕"
    with st.chat_message("assistant", avatar=SOGANG_HAWK_AVATAR):
        st.markdown(goodbye_message)
    initialization()
    st.rerun()

# -----------------------------------------------------------------------------
# 5‑0. 완료 처리 ---------------------------------------------------------------
# -----------------------------------------------------------------------------
if "완료" in user_query:
    if st.session_state.last_recommend_df is None or st.session_state.last_recommend_df.empty:
        error_message = "⚠️ 이전에 추천된 영화가 없습니다. 먼저 추천을 받아주세요."
        with st.chat_message("assistant", avatar=SOGANG_HAWK_AVATAR):
            st.markdown(error_message)
        add_to_chat_history("assistant", error_message)
        st.stop()

    set_branch("complete")
    sel_title = handle_completion(user_query, st.session_state.last_recommend_df, interaction_id, st.session_state.user_id)
    if sel_title:
        st.session_state.selected_title = sel_title
        complete_message = "✅ 선택 완료! 좋은 감상 되세요."
        with st.chat_message("assistant", avatar=SOGANG_HAWK_AVATAR):
            st.markdown(complete_message)
        add_to_chat_history("assistant", complete_message)
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
        add_to_chat_history("assistant", answer)
        st.session_state.first_turn = False
        st.stop()

    # (b) 유사 추천
    if is_similar_recommendation(user_query):
        set_branch("similar")
        df_sim = handle_similar_recommendation(
            user_query, df, st.session_state.user_id, st.session_state.selected_title, extract_user_meta, keyword_columns
        )

        if df_sim.empty:
            st.chat_message("assistant", avatar=SOGANG_HAWK_AVATAR).write("죄송해요, 유사한 콘텐츠를 찾지 못했어요.")
            add_to_chat_history("assistant", "죄송해요, 유사한 콘텐츠를 찾지 못했어요.")
            st.stop()

        resp = generate_recommendation_response(user_query, df_sim, st.session_state.user_name)

        st.chat_message("assistant", avatar=SOGANG_HAWK_AVATAR).write(resp)
        add_to_chat_history("assistant", resp)

        log_recommendations(interaction_id, df_sim["title"].tolist())

        st.session_state.previous_titles.update(df_sim["title"].tolist())
        st.session_state.last_recommend_df = df_sim.copy()
        st.session_state.last_recommend_query = user_query
        st.session_state.first_turn = False

        # 메인 영역에 새 추천을 렌더링합니다.
        render_recommendation_thumbnails(df_sim, key_prefix="similar_recommend_")
        add_to_chat_history("assistant", df_sim, "similar_recommend_", "dataframe")

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
        set_branch("follow_up")
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
        add_to_chat_history("assistant", "죄송해요, 추천할 콘텐츠를 찾지 못했어요.")
        st.stop()

    resp = generate_recommendation_response(merged_query, df_ret, st.session_state.user_name, is_retry=is_retry)

    st.chat_message("assistant", avatar=SOGANG_HAWK_AVATAR).write(resp)
    add_to_chat_history("assistant", resp)

    log_recommendations(interaction_id, df_ret["title"].tolist())

    st.session_state.previous_titles.update(df_ret["title"].tolist())
    st.session_state.last_recommend_df = df_ret.copy()
    st.session_state.first_turn = False

    render_recommendation_thumbnails(df_ret, key_prefix="retry_recommend_")
    add_to_chat_history("assistant", df_ret, "retry_recommend_", "dataframe")

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
        add_to_chat_history("assistant", "죄송해요, 적절한 콘텐츠를 찾지 못했어요.")
        st.stop()

    resp = generate_recommendation_response(user_query, df_first, st.session_state.user_name)

    st.chat_message("assistant", avatar=SOGANG_HAWK_AVATAR).write(resp)
    add_to_chat_history("assistant", resp)

    log_recommendations(interaction_id, df_first["title"].tolist())

    st.session_state.previous_titles.update(df_first["title"].tolist())
    st.session_state.last_recommend_df = df_first.copy()
    st.session_state.first_turn = False

    render_recommendation_thumbnails(df_first, key_prefix="first_recommend_")
    add_to_chat_history("assistant", df_first, "first_recommend_", "dataframe")

    st.stop()

# -----------------------------------------------------------------------------
# 5‑3. 일반 QA -----------------------------------------------------------------
# -----------------------------------------------------------------------------
else:
    set_branch("qa")
    answer = qa_chain.invoke({"question": user_query})["answer"]
    st.chat_message("assistant", avatar=SOGANG_HAWK_AVATAR).write(answer)
    add_to_chat_history("assistant", answer)
    st.session_state.first_turn = False


# 폰트 로드
st.markdown("""
<link href="https://api.fontshare.com/v2/css?f[]=satoshi@400,500,600&display=swap" rel="stylesheet">
<link href="https://api.fontshare.com/v2/css?f[]=clash-display@400,500,600&display=swap" rel="stylesheet">
""", unsafe_allow_html=True)



