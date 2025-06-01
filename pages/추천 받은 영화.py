import streamlit as st
from database import get_previous_recommendations
from datetime import datetime

st.set_page_config(
    page_title="추천 받은 영화",
    layout="wide",
)

# 사용자 ID 확인
if 'user_id' in st.session_state and st.session_state.user_id:
    user_id = st.session_state.user_id
    feedback_rows = get_previous_recommendations(user_id)

    # 추천 제목과 추천 일시를 리스트 형식으로 표시하는 HTML 생성
    if feedback_rows:
        st.markdown("<h3>🎬 이전에 추천 받은 영화</h3>", unsafe_allow_html=True)
        html_table = "<table><thead><tr><th>추천 제목</th><th>추천 일시</th></tr></thead><tbody>"
        for row in feedback_rows:
            title = row[1]
            timestamp = row[0]
            try:
                formatted_timestamp = datetime.fromisoformat(timestamp).strftime('%Y-%m-%d %H:%M')
            except (ValueError, TypeError, AttributeError):
                formatted_timestamp = str(timestamp)
            
            html_table += f"<tr><td>{title}</td><td>{formatted_timestamp}</td></tr>"
            
        html_table += "</tbody></table>"
        st.markdown(html_table, unsafe_allow_html=True)
    else:
        st.markdown("이전에 추천 받은 영화가 없어요.")
else:
    st.markdown("<div style='text-align: center; margin: 0 auto; width: fit-content;'>", unsafe_allow_html=True)

    col1, col2 = st.columns([1, 2])

    with col1:
        st.image("static/user_avatar.png", width=150, use_container_width=False)

    with col2:
        st.markdown(
            """
            <div style="margin-top: 45px; font-size: 1.2em; text-align: left;">
                사용자 정보가 없습니다.<br>감성 매칭 추천 페이지로 이동하여 이름을 입력해주세요.
            </div>
            """, unsafe_allow_html=True
        )

    if st.button("감성매칭추천 이동하기", use_container_width=True):
        st.switch_page("app.py")

    st.markdown("</div>", unsafe_allow_html=True)


