
"""
CLI entry point for the movie recommendation service.
"""

from langchain.schema import Document
from langchain.vectorstores import FAISS

from database import *
from utils import *
from recommender import *

from vector_db import build_vectorstore, build_qa_chain
from data_loader import load_dataframe
from config import model_name, embedding_model_name
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain

from typing import List

# Global models
embedding_model = OpenAIEmbeddings(model=embedding_model_name)
llm = ChatOpenAI(model_name=model_name, temperature=0.3)

# Load data at startup
df = load_dataframe()
# Todo: 삭제
df = df[:100]
vectorstore_global = build_vectorstore(df["document"].apply(lambda x: Document(page_content=x, metadata={})).tolist())
qa_chain = build_qa_chain(vectorstore_global)

def main_chat_loop():
    user_name = input("👋 당신의 이름을 알려주세요: ").strip()
    user_id = get_or_create_user_id(user_name)
    print(f"{user_name}님 안녕하세요? 영화 추천을 시작합니다 😊")

    selected_title = None
    interaction_id = None
    last_recommend_df = None
    last_recommend_query = None
    last_recommend_type = None
    first_turn = True
    last_selected_title = None

    while True:
        query = input("\n❓ 사용자: ").strip()

        if query.lower() in ["exit", "quit", "종료"]:
            print("👋 대화를 종료합니다.")
            break

        interaction_id = create_interaction(user_id, query)

        # ✅ 완료 처리
        if "완료" in query:
            if last_recommend_df is None or last_recommend_df.empty:
                print("⚠️ 이전에 추천된 영화가 없습니다. 먼저 추천을 받아주세요.")
                continue
            if last_recommend_df is not None and not last_recommend_df.empty:
                selected_title = handle_completion(query, last_recommend_df, interaction_id, user_id)
                if selected_title:
                    break
                else:
                    continue

        # ✅ 후속 질문 처리
        if not first_turn and last_recommend_df is not None and not last_recommend_df.empty:
            if is_follow_up_question(query, last_recommend_df["title"].tolist()):
                print("📌 후속 질문으로 판단됨 → 이전 추천 콘텐츠에서 검색 중...")
                local_docs = [
                    Document(page_content=truncate_document(row["document"]), metadata={"title": row["title"]})
                    for _, row in last_recommend_df.iterrows()
                ]
                local_store = FAISS.from_documents(local_docs, embedding_model)
                local_chain = ConversationalRetrievalChain.from_llm(
                    llm=llm,
                    retriever=local_store.as_retriever(),
                    memory=None
                )
                result = local_chain.invoke({"question": query, "chat_history": []})
                print("🤖 GPT:\n", result["answer"])
                continue

            if is_similar_recommendation(query):
                print("is_similar_recommendation")
                df_recommend = handle_similar_recommendation(query, df, user_id, selected_title, extract_user_meta, keyword_columns)
                if df_recommend.empty:
                    print("🤖 GPT:\n죄송해요, 유사한 콘텐츠를 찾지 못했어요.")
                    continue
                response_text = generate_recommendation_response(query, df_recommend, user_name)
                print("🤖 GPT:\n", response_text)
                log_recommendations(interaction_id, df_recommend["title"].tolist())
                last_recommend_df = df_recommend.copy()
                last_recommend_query = query
                continue

            is_retry, retry_mode = is_retry_request(query)
            if is_retry and last_recommend_query:
                # ── 1) 재추천 분기 ──
                print(f"🔁 재추천 요청 ({retry_mode}) → 이전 쿼리로 키워드 추출: {last_recommend_query}")
                # (1) merged_query는 이전 쿼리 재사용
                merged_query = last_recommend_query
                # (2) user_meta는 직전에 저장해 둔 메타 재활용
                user_meta = extract_user_meta(merged_query)
                last_user_meta    = user_meta
                # (3) 재추천 핸들러 호출 (selected_title이 있으면 넘겨주고, 없으면 None)
                last_selected_title=df_recommend["title"].tolist()
                df_recommend = handle_recommendation(df,user_id,user_meta,  selected_title=last_selected_title   )
            else:
                # ── 2) 첫 추천 분기 ──
                merged_query = query
                # (1) query에서 한 번만 메타 추출
                user_meta = extract_user_meta(merged_query)
                # (2) 세션에 저장
                last_user_meta = user_meta
                last_recommend_query = merged_query
                # (3) 첫 추천 핸들러 호출
                df_recommend = handle_recommendation(
                    df,
                    user_id=user_id,
                    user_meta=user_meta
                )

            # ── 공통: 추천 결과 출력 및 상태 업데이트 ──
            if df_recommend is None or df_recommend.empty:
                print("🤖 GPT:\n죄송해요, 추천할 콘텐츠를 찾지 못했어요.")
                continue
            response_text = generate_recommendation_response(merged_query,df_recommend,user_name,is_retry=is_retry)
            print("🤖 GPT:\n", response_text)
            log_recommendations(interaction_id, df_recommend["title"].tolist())

            # 재추천 이후에도 last_selected_title 은
            # 사용자가 선택했을 때 별도 로직으로 업데이트해두세요.
            last_recommend_df = df_recommend.copy()
            first_turn = False

        if first_turn and is_recommendation_request(query):
            print("first question")

            # ① user_meta 한 번만 추출
            user_meta = extract_user_meta(query)
            # (필요하다면 세션에도 저장)
            last_user_meta = user_meta

            # ② 수정된 handle_recommendation 호출
            #    시그니처: handle_recommendation(df, user_id, user_meta, selected_title=None)
            df_recommend = handle_recommendation(
                df,
                user_id,
                user_meta,
                selected_title  # 첫 추천 단계이므로 보통 None
            )

            if df_recommend.empty:
                print("🤖 GPT:\n죄송해요, 적절한 콘텐츠를 찾지 못했어요.")
                continue

            response_text = generate_recommendation_response(query, df_recommend, user_name)
            print("🤖 GPT:\n", response_text)

            log_recommendations(interaction_id, df_recommend["title"].tolist())

            # ③ 상태 업데이트
            last_recommend_df    = df_recommend.copy()
            last_recommend_query = query
            first_turn           = False

            continue

        # ✅ 기타 일반 질문 처리
        else:
            print("🟠 일반 질문으로 판단 → QA 체인 실행")
            try:
                result = qa_chain.invoke({"question": query})
                print("🤖 GPT:\n", result["answer"])
            except Exception as e:
                print("❌ 토큰 초과 또는 처리 오류 → 일반 응답 불가")
                print("🤖 GPT:\n죄송해요, 해당 질문에는 답변할 수 없습니다.")

if __name__ == "__main__":
    main_chat_loop()
