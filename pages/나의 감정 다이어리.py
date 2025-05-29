import streamlit as st

from database import get_feedback_by_user_id


st.set_page_config(
    page_title="나의 감정 다이어리",
    layout="wide",
)


# ------------------------ ① 사용자 정의 CSS ------------------------ #
st.markdown("""
<style>
.feedback-card {
    background: #e3e6e5;
    border-radius: 12px;
    padding: 18px 14px;
    box-shadow: 0 3px 8px rgba(0,0,0,0.07);
    height: 270px;                     /* st.container(height=300) 과 호환 */
    display: flex;
    flex-direction: column;
    justify-content: space-between;   /* 위·아래 여백 균등 분배 */
    transition: 0.25s ease;
    color: #5f5554;
}
.feedback-card.selected {             /* ✅ 선택된 카드 강조 */
    border: 3px solid #5f5554;
}
.feedback-date      {font-size: 26px; font-weight: 700; text-align: center; margin-bottom: 6px;}
.question-text      {font-size: 15px; text-align: center;  color:#555;      margin-bottom: 2px;}
.movie-title        {font-size: 18px; font-weight: 600;   text-align: center; margin:4px 0;}
.tag                {display:inline-block; padding:2px 8px; font-size:12px; border-radius:20px; background:#E0E0E0; margin-right:4px;}
.tag-selected       {background:#4CAF50; color:#fff;}     /* 초록 배지 */
.tag-disliked       {background:#EF5350; color:#fff;}     /* 빨강 배지 */
.comment            {font-size: 14px; color:#333; word-break:break-all;}
</style>
""", unsafe_allow_html=True)

# ------------------------ ② 예시 데이터 ------------------------ #
# feedback_rows = [
#     (1, '보리차', '감동적인 가족 영화 추천해줘', '원더', 1, 0,'가족애가 돋보여 눈물이 났어요.','2025-05-25 13:51:19'),
#     (2, '보리차', '감동적인 가족 영화 추천해줘', '라라랜드', 0, 1, '음악은 좋았지만 스토리가 아쉬웠어요.','2025-05-25 13:51:19'),
#     (3, '보리차', '감동적인 가족 영화 추천해줘', '세 얼간이', 0, 1, '유쾌하면서도 감동적인 메시지가 좋았어요!','2025-05-25 13:51:19'),
#     (4, '보리차', '감동적인 가족 영화 추천해줘', '소울', 0, 0, '삶의 의미를 다시 생각하게 됐어요.','2025-05-25 13:51:19'),
#     (5, '보리차', '감동적인 가족 영화 추천해줘', '미니특공대', 0, 1, '어린이 취향이라 저에겐 맞지 않았어요.','2025-05-25 13:51:19'),
#     (6, '보리차', '감동적인 가족 영화 추천해줘', '굿 윌 헌팅', 1, 0, '잔잔하지만 깊은 울림이 있는 작품이었어요.','2025-05-25 13:51:19'),
#     # …
# ]

# 실제 데이터
user_id = st.session_state.user_id
feedback_rows = get_feedback_by_user_id(user_id)

# ------------------------ ③ 3개씩 컬럼 배치 ------------------------ #
for i in range(0, len(feedback_rows), 3):
    cols = st.columns(3, gap="small")
    for j in range(3):
        if i + j >= len(feedback_rows):
            continue

        (fid, user_name, question, movie_title,
         is_selected, is_disliked, comment, created_at) = feedback_rows[i + j]

        # 선택된 카드면 selected 클래스 추가
        card_cls = "feedback-card selected" if is_selected else "feedback-card"

        # ---------------- 컨테이너 ---------------- #
        with cols[j]:
            with st.container(height=300):
                st.markdown(f"""
<div class="{card_cls}">
    <div class="feedback-date">{created_at.split()[0]}</div>
    <div class="question-text">“{question}”</div>
    <div class="movie-title" style="font-size: 26px;">{movie_title}</div>
    <div style="text-align:center; margin-top:6px;">{"<span class='tag tag-selected'>✅ 선택됨</span>" if is_selected else ""}{"<span class='tag tag-disliked'>❌ 싫어요</span>" if is_disliked else ""}
</div>
<div class="comment">{comment}</div>
</div>
""", unsafe_allow_html=True)
