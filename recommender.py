"""
Content recommendation logic.
"""

import pandas as pd
from typing import List, Dict, Any

from config import *
from filters import *
from utils import *
from database import *

import re

keyword_columns = [
    "Emotion", "Subject", "atmosphere", "background", "character_A", "character_B", "character_C",
    "criminal", "family", "genre", "love", "natural_science", "religion", "social_culture", "style"
]

# 시스템 프롬프트 전체 정의
keyword_prompt = """
당신은 메타 키워드 분류 전문가입니다. 사용자의 인풋에서 여러 카테고리의 메타 키워드를 분류하세요.
주의사항:
- 반드시 아래 키워드 목록에서만 선택하여 분류할 것.
- 아래 키워드 목록에 없는 키워드는 절대 포함하지 말 것.
- 예를 들어 "기분 전환"은 "즐거운"으로, "속상한"은 "슬픈"으로 변경하는 등 유사한 키워드를 찾아 매칭할 것.
- 유사 키워드를 찾아봤는데도 아래 키워드 목록에 포함되어 있지 않다면 반드시 생략할 것.
- 각 카테고리당 최대 5개까지만 나열하며, 같은 키워드는 반복하지 말 것.
- 키워드가 없으면 해당 카테고리는 빈 칸으로 두세요. (예: Emotion: )

Emotion (긴장감 넘치는, 몰입되는, 즐거운, 통쾌한, 짜릿한, 불안한, 충만한, 감동적인, 가슴 뭉클한, 들뜨게 하는, 긍정적인, 애타는, 화가 나는, 쓸쓸한, 설레는, 먹먹한, 불쾌한, 행복한, 슬픈, 우울한, 충격적인, 불길한, 아주 신나는, 달콤한, 멘붕, 흥겨운, 사랑스러운, 무서운, 허탈한, 놀라운, 영감을 주는, 편안한, 열광적인, 절망적인, 역겨운)

Subject (고난/역경, 불굴의 의지/열정, 고뇌/번민/내적 갈등, 성장, 의지가 되는/힘이 되는, 탐욕/부패, 우정, 환상의 팀워크, 불의의 사고, 치유/치료, 복수, 자아/자기존중/자아실현, 인간 비판, 따뜻한 위로, 설상가상/산 넘어 산, 불편한 진실, 희망/신념, 삶의 의미/인생/철학, 충격과 반전, 사회 문제, 죽음 이후 남겨진 이들, 모성애/부성애, 역사의 소용돌이, 체제 비판/사회 비판, 권력에 저항, 떡밥/밑밥/복선, 배신, 극복, 어차피 만날 운명, 공존, 억울한 누명, 삶과 죽음, 인권, 스캔들, 체제 대항/체제 저항, 시간의 족쇄, 전쟁과 평화, 선한 영향력, 내부고발, 음모론, 신앙/영성, 맥거핀, 담론)

atmosphere (흥미진진한, 진정성 있는, 액션 대폭발, 유머러스한, 재치 있는, 로맨틱한, 어두운, 심장을 서서히 옥죄는, 폭력적인, 잔잔한, 생각하게 하는, 힐링, 추악한, 눈물샘을 자극하는, 격정적인, 신비한, 기이한, 차가운, 웅장한, 잔인한, 참혹한, 얼빠진, 동기 부여되는, 싱그러운, 자극적인, 황당한, 관능적인, 숭고한, 몽환적인, 머리를 쓰는, 선정적인, 정신 착란을 일으키는, 도발적인, 한가로운, 황홀한, 외설적인, 나른한)

background (마을/동네/스몰타운, 학교/학원, 제한된 공간, 내부자들, 여름, 직장/회사, 돌아온 고향, 디스토피아, 겨울, 휴가/바캉스/여행, 감옥/교도소, 군대, 병원, 대학/캠퍼스, 농촌, 연말연시/크리스마스/홀리데이, 연예계, 포스트 아포칼립스, 방송국, 사이버펑크, 평행우주/멀티버스, 어촌, 특수부대, 기숙사, 봄, 이세계, 가을, 호텔, 정신 병원, 법원, 공항, 심해, 사후세계/천국/지옥, 미술관/아트센터, 수용소, 카지노, 무인도, 박물관, 핼러윈, 스팀펑크, 신혼여행, 밸런타인데이)

character_A (방황하는 캐릭터, 진취적인 캐릭터, 냉소적인 캐릭터, 정의로운 캐릭터, 여성 캐릭터, 십대/틴에이저/하이틴, 청춘/하이틴, 똑 부러지는 캐릭터, 유머러스한 캐릭터, 실존 인물, 중년, 사회적 약자/소수자, 반전 캐릭터, 못 말리는 캐릭터, 반항적인 캐릭터, 이민자/이방인, 소년/소녀, 능력자, 노년, 능글맞은 캐릭터, 푼수 캐릭터, 어린이, 성소수자, 내향적인 캐릭터, 자유로운 캐릭터, 츤데레 캐릭터, 무해한 캐릭터, 장애인, 매혹적인 캐릭터, 천재, 캐릭터, 30대, 멀티캐스팅, 너드 캐릭터, 수다스러운 캐릭터, 멍청한 캐릭터, 인싸 캐릭터, 남장여자/여장남자, 워커홀릭, 퇴폐미 캐릭터, 모태솔로, 댕댕이/멍뭉미 캐릭터, 서브병 캐릭터)

character_B (애니멀, 강아지/개/반려견, 괴물/크리쳐/몬스터, 로봇/사이보그, 유령/귀신, 외계인/에일리언, 슈퍼히어로, 말하는 동물, 신, 뮤턴트/돌연변이, 괴수, 새, 요정/엘프, 고양이/반려묘, 무속인/무당/영매, 악마/데블/사탄, 드래곤/용, 좀비/언데드, 물고기/어류/수중생물/해양동물, 곤충, 요괴, 거인/소인, 마녀, 다크히어로/안티히어로, 늑대, 흡혈귀/뱀파이어/드라큘라, 공룡, 상어, 늑대인간, 저승사자/사신/그림리퍼, 호랑이, 천사, 구미호, 사자, 미라/미이라, 강시, 산타 클로스)

character_C (학생, 사제/스승/선생님, 형사, 군인, 특수요원/비밀요원, 경찰/경찰관, 싱글맘/싱글대디, 의사/간호사/의료인, 블루칼라/프롤레타리아, 작가, 연예인, 예술가/아티스트, 왕실/왕/황제/왕자/공주, 검사/변호사/판사/법률가, 음악가, 암살자/킬러, 기자, 조폭, 건달/양아치, 재벌, 연쇄살인범, 사이코패스/소시오패스, 성직자, 운동선수, 백수/한량, 부패 경찰, 사기꾼, 대통령/총리/관료, 도둑/괴도, 정치인/국회 의원, 스파이, 무사/검객, 갱스터, 왕따, 강도, 발명자, 돌싱, 취준생, 일진, 공무원, 파일럿, 인플루언서, 요리사/셰프, 탐정, 미친 과학자, 비정규직, 마피아, 원주민/인디언, 고뇌하는 천재, 삼합회, 카르텔, 보디가드/경호원, 승무원, 해커, 소방관/구조원, 사무라이/닌자, 야쿠자, 조선족, 변태, 해적, 바보, 앵커/아나운서, 검투사, 바이킹, 프로파일러, 구두쇠)

criminal (추격, 범죄, 추적, 복수, 구속/속박/감금, 납치/유괴, 자살, 수사, 마약, 암살, 성폭력/성범죄, 연쇄살인, 사기, 잠입 수사/위장, 테러, 인질극, 가정폭력, 학교폭력, 실종, 방화, 해킹, 가스라이팅, 스토킹, 잠복, 사이버 불링, 불법 촬영/몰카, 데이트 폭력, 유기)

family (아들, 가족, 딸, 부부, 남매, 형제, 문제 가정, 부모, 자매, 출생의 비밀, 조부모/외조부모, 재혼 가정/새엄마/새아빠, 입양, 쌍둥이, 중년의 위기, 처가월드/시월드)

genre (드라마, 액션, 스릴러, 코미디, 멜로, 판타지, 시대물, 공포, 독립, 다큐멘터리, 기타 스타일)

love (연애/썸, 커플/연애, 성/섹스, 짝사랑, 결혼, 불륜, 재회, 헤어진 연인, 첫사랑, 삼각관계, 금지된 사랑, 동성애, 집착, 성숙한 사랑, 이혼, 연하, 성 도착증, 성 정체성, 비혼, 폴리아모리, 성인식)

natural_science (상처/트라우마, 모험, 무기, 생존, 콤플렉스/열등감, 불치병/난치병, 장애, 자동차, 무전/전화/휴대폰, 정신질환, 재난, 여행, 자연, SNS/앱/이메일, 기억상실, 중독, 잠수함/배/선박, 비행기, 우주, 시간여행, 전염병/역병/바이러스, 인공지능/AI, 질병, PTSD, 자전거/바이크, 과학, 기차, 치매/알츠하이머, 사냥, 야생, 복제 인간, 라디오, 의학/메디컬, 환경/환경보호, 타임루프, 가상현실/메타버스, 항공 우주/천문, 자폐, 우주여행, 캠핑)

religion (친구, 초자연/불가사의, 욕설/비속어, 고립, 초능력, 브로맨스, 아포칼립스/멸망, 마법, 영혼 체인지/빙의, 보석/보물, 천주교, 워맨스, 도박, 종교, 기독교, 엑소시즘/퇴마, 무속신앙, 카니발리즘, 사이비 종교, 환생/윤회, 인형, 크리스마스, 장난감, 불교, 아이돌, 마술, 이슬람, 성경, 도플갱어)

social_culture (격투/이종격투, 사회/문화, 돈, 전쟁, 음악/뮤직, 역사, 단체/기업, 라이벌, 예술, 게임/놀이/시합, 경쟁, 정치, 제국주의/침략/지배, 스포츠, 댄스/춤/무용, 육아, 세계대전/제1차 세계대전/제2차 세계대전, 인종차별, 무술/쿵푸, 음식/푸드, 경영/비즈니스, 법정, 책/도서, 요리/쿡, 반려동물, 남북관계, 건축/건축물, 나치/파시즘, 사채, 사건/사고, 공산주의/사회주의, 학문, 의류/패션, 사교육/입시, 6.25, 홀로코스트, 야구, 교육, 투자/재테크, 농구, 독립운동, 축구, 경제, 라이프, 뷰티, 바둑/장기/체스, 지식, 골프, 부동산, 낚시, 생활정보, 체험)

style (킬링타임, 로맨스, 판타지, 블록버스터, 어드벤처, SF, 미스터리, 애니메이션, 예술영화/예술성, 어린이/키즈/가족, 독특한 소재, 전기, 로맨틱 코미디, 걸 파워/걸 크러시, 풍자/블랙 코미디, 느와르, 열린 결말, 호러, 피카레스크, 고전, 버디 무비, 고어/스플래터, 무협/무협물, 뮤지컬, 일상, 하이스트/케이퍼, 실험적, 로드 무비, 재현/재구성, 다중 플롯, 컬트, 옴니버스/앤솔러지, 동화, 코즈믹 호러, 신화/전설/서사시, 단편, 희곡/연극, 스페이스 오페라, 슬래셔, 서부/웨스턴, 막장, 모큐멘터리, 성인, 먹방, 우주 탐사 SF, 관찰, 에피소드식 구성, 오디션/경연, 공연, 시트콤, 강의/교양/지식/교육, 퀴즈 쇼, 음악쇼/음악방송/콘서트, 탐사보도)

"""

def extract_user_meta(query):
    try:
        system_prompt = keyword_prompt
        response = openai.ChatCompletion.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"사용자 입력:\n{query}\n\n위 입력에서 키워드를 정리해 주세요."}
            ],
            temperature=0.3
        )
        gpt_text = response["choices"][0]["message"]["content"]
        print("\n🔍 GPT 키워드 추출 결과:\n", gpt_text)

        user_meta = {}
        for line in gpt_text.strip().splitlines():
            if ":" in line:
                key, values = line.split(":", 1)
                key = key.strip()
                user_meta[key] = [v.strip() for v in values.split(",") if v.strip()]
        return user_meta
    except Exception as e:
        print("❗ 키워드 추출 실패:", e)
        return {}

def recommend_contents(user_input, extract_user_meta, df, previous_recommend_titles=set()):
    user_meta = extract_user_meta(user_input)
    df["score"] = df.apply(lambda row: get_content_score(row, user_meta), axis=1)
    filtered_df = df[~df["title"].isin(previous_recommend_titles)]
    df_recommend = filtered_df[filtered_df["score"] > 0].sort_values(by="score", ascending=False).head(5)
    return df_recommend

def Enoung_recommend_contents(extract_user_meta, df, user_id):

    previous_titles = get_previous_recommendations(user_id)

    # 1. 싫어하는 영화 필터링
    disliked_items = get_user_dislikes(user_id)
    disliked_titles = [value for category, value in disliked_items if category == "title"]

    # 2. 필터링 적용
    filtered_df = df.copy()
    filtered_df = filtered_df[~filtered_df["title"].isin(previous_titles + disliked_titles)]

    # 3. 점수 계산
    filtered_df["score"] = filtered_df.apply(lambda row: get_content_score(row, extract_user_meta), axis=1)

    # 4. 추천 결과 반환
    df_recommend = filtered_df[filtered_df["score"] > 0].sort_values(by="score", ascending=False).head(5)
    return df_recommend



def generate_recommendation_response(user_input, df_recommend, user_name, is_retry=False):
    seen_titles = set()
    filtered_rows = []

    # 중복 제거
    for row in df_recommend.itertuples():
        norm_title = normalize_title(row.title)
        if norm_title not in seen_titles:
            seen_titles.add(norm_title)
            filtered_rows.append(row)

    # 콘텐츠 요약 생성 (제목, 줄거리, 키워드 포함)
    summaries = []
    for i, row in enumerate(filtered_rows, 1):
        summaries.append(
            f"🎬 {i}. {row.title}\n"
            f"✨ 줄거리: {row.description}\n"
            f"📌 관련 키워드: {row.Subject}, {row.Emotion}, {row.atmosphere}"
        )
    content_summary = "\n\n".join(summaries)

    # 프롬프트 템플릿 구성
    if is_retry:
        prompt = f"""
사용자 입력:
"{user_input}"

추천된 콘텐츠:
{content_summary}

---

당신은 공감형 콘텐츠 큐레이터입니다. 다음 지침을 따라 {user_name}님에게 재추천 응답을 제공해 주세요:

1. "{user_name}님, 요청하신 내용을 반영해서 다시 추천해드릴게요." 와 같은 도입 문장
2. 위 추천된 콘텐츠 리스트 중 3~5개를 골라 제목, 줄거리 요약, 관련 키워드를 자연스럽게 소개
3. 마지막에는 "{user_name}님, 이번 추천이 마음에 드셨으면 좋겠어요!" 와 같은 부드러운 마무리 멘트
4. 친구처럼 따뜻한 말투와 적절한 이모지를 사용하세요.
"""
    else:
        prompt = f"""
사용자 입력:
"{user_input}"

추천된 콘텐츠:
{content_summary}

---

당신은 공감형 콘텐츠 큐레이터입니다. 다음 지침을 따라 {user_name}님에게 따뜻하고 자연스럽게 응답해 주세요:

1. {user_name}님의 상황에 공감하는 다정한 인사말을 먼저 전하세요.
2. 위 추천된 콘텐츠 리스트에서 3~5개를 골라 제목, 줄거리 요약, 관련 키워드를 소개하세요.
3. 마지막에는 "{user_name}님을 응원합니다!"와 같은 응원 문장을 포함하세요.
4. 친구처럼 부드러운 말투와 이모지를 사용하세요.
"""

    # GPT 응답 생성
    response = openai.ChatCompletion.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "당신은 친구처럼 따뜻하게 공감해주는 콘텐츠 큐레이터입니다."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3
    )

    return response["choices"][0]["message"]["content"]



def recommend_similar_contents(user_input, extract_user_meta, df, keyword_columns):
    # 더 넓은 범위를 커버하는 정규표현식
    title_match = re.search(r'(.+?)(?:이랑|랑|과|와|같은|처럼.*?)\s*비슷한\s*영화', user_input) \
               or re.search(r'(.+?)\s*같은\s*영화', user_input) \
               or re.search(r'(.+?)\s*처럼\s*\S+\s*영화', user_input)

    if not title_match:
        return pd.DataFrame()

    title = title_match.group(1).strip()
    print(f"🎯 추출된 영화 제목: {title}")

    if title not in df["title"].values:
        print(f"⚠️ '{title}'은(는) 데이터셋에 존재하지 않습니다.")
        return pd.DataFrame()

    reference_row = df[df["title"] == title].iloc[0]
    user_meta = {col: str(reference_row[col]).split(",") for col in keyword_columns}

    df["score"] = df.apply(lambda row: get_content_score(row, user_meta), axis=1)
    return df[df["title"] != title].sort_values(by="score", ascending=False).head(5)




def recommend_by_movie_info_and_meta(user_input: str, df: pd.DataFrame, extract_user_meta) -> pd.DataFrame:
    # 1. 제작 키워드로 1차 필터링
    df_filtered = filter_by_movie_info(user_input, df)
    if df_filtered.empty:
        return pd.DataFrame()

    # 2. 메타 키워드 추출
    user_meta = extract_user_meta(user_input)

    # 3. df_filtered 내부에서 메타 키워드로 추가 필터링
    def get_meta_score(row):
        score = 0
        for category, keywords in user_meta.items():
            content_keywords = str(row.get(category, "")).split(",")
            matched = set(keywords) & set(map(str.strip, content_keywords))
            score += len(matched)
        return score

    df_filtered["score"] = df_filtered.apply(get_meta_score, axis=1)
    df_result = df_filtered[df_filtered["score"] > 0].sort_values(by="score", ascending=False)

    if df_result.empty:
        # 메타 매칭이 없다면, 제작 키워드 필터링만 적용된 것 중 평점순으로 추천
        df_result = df_filtered.sort_values(by="rating", ascending=False)

    return df_result.head(5)

# ✅ 문서 길이 자르기



def fallback_recommend_by_rating(user_meta: dict, df: pd.DataFrame, top_n=3) -> pd.DataFrame:
    keywords = [kw for values in user_meta.values() for kw in values]
    if not keywords:
        return pd.DataFrame()
    regex = '|'.join(map(re.escape, keywords))
    filtered_df = df[df.apply(lambda row: bool(re.search(regex, str(row), re.IGNORECASE)), axis=1)]
    return filtered_df.sort_values(by="rating", ascending=False).head(top_n)

# ✅ 유사 콘텐츠 추천 함수 (정규표현식 강화 버전)



def handle_recommendation(df, user_id, user_meta, selected_title=None):
    print("✅ handle_recommendation")
    if selected_title:
        if isinstance(selected_title, list):
            exclude = selected_title
        else:
            exclude = [selected_title]
    else:
        exclude = []

    # 기존 apply_user_filters로 비선호나 이전 선택 제외 후,
    filtered_df = apply_user_filters(df, user_id)
    # 추가로 exclude 리스트에 든 제목들 모두 제외
    filtered_df = filtered_df[~filtered_df["title"].isin(exclude)]
    # 1) 사용자 비선호/이전추천 제외 필터링
    #filtered_df = apply_user_filters(df, user_id, selected_title)

    # 2) 전달된 user_meta 사용 (내부에서 재추출하지 않음)
    total_keywords = sum(len(v) for v in user_meta.values())

    # 3) 키워드 충분 여부에 따라 분기
    if total_keywords >= 5:
        print("✅ 키워드가 충분하므로 유사도 기반 추천 실행")
        # recommend_contents의 시그니처도 아래처럼 바꿔주세요:
        # recommend_contents(user_meta, filtered_df, user_id)
        # return recommend_contents(user_meta, filtered_df, user_id)
        return Enoung_recommend_contents(user_meta, filtered_df, user_id)
    else:
        print("⚠️ 키워드가 부족하므로 정규표현식 기반 평점 추천 실행")
        return fallback_recommend_by_rating(user_meta, filtered_df)

# ✅ 유사 콘텐츠 추천 처리



def handle_similar_recommendation(query, df, user_id, selected_title, extract_user_meta, keyword_columns):
    print("✅ handle_similar_recommendation")
    filtered_df = apply_user_filters(df, user_id, selected_title)
    return recommend_similar_contents(query, extract_user_meta, filtered_df, keyword_columns)

# ✅ 재추천 처리 (결합/제외)



def handle_retry_recommendation(merged_query, df, user_id, selected_title, user_meta ):
    print("✅ handle_retry_recommendation")
    filtered_df = apply_user_filters(df, user_id, selected_title)
    return recommend_contents(user_input=merged_query, user_meta=user_meta, df=filtered_df, user_id=user_id)


# ✅ 완료 처리



def handle_completion(query, last_recommend_df, interaction_id, user_id):
    print("✅ handle_completion")
    possible_title = query.replace("완료", "").strip()
    if not possible_title:
        print("⚠️ 선택한 영화 제목이 없습니다. 다시 입력해주세요.")
        return None

    def normalize_title(text):
        return re.sub(r"[^\w\s가-힣]", "", text).lower()

    possible_title_clean = normalize_title(possible_title)

    matched_titles = [
        title for title in last_recommend_df["title"].tolist()
        if possible_title_clean in normalize_title(title)
    ]

    if matched_titles:
        selected_title = matched_titles[0]
        save_feedback(interaction_id, selected_title, is_selected=True, is_disliked=False)
        print(f"✅ '{selected_title}'을(를) 선택하셨습니다. 기록합니다.")
        print("👋 대화를 종료합니다. 이용해주셔서 감사합니다!")
        return selected_title
    else:
        print("🧾 추천된 영화 목록:")
        for title in last_recommend_df["title"].tolist():
            print(f"  - {title} (cleaned: {normalize_title(title)})")

        print(f"📝 사용자 입력 제목: {possible_title} (cleaned: {normalize_title(possible_title)})")
        print("⚠️ 추천된 영화 중 해당 제목이 없습니다. 다시 확인해주세요.")


