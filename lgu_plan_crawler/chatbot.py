# LG U+ 요금제 질의 챗봇 V1
import pandas as pd
import chromadb
from openai import OpenAI
import os
import json

# --- (이전과 동일한 설정 부분) ---
try:
    client = OpenAI(api_key="API_KEY")
    print("✅ OpenAI API 키가 성공적으로 로드되었습니다.")
except KeyError:
    print("❌ 'OPENAI_API_KEY' 환경 변수가 설정되지 않았습니다.")
    exit()

db_path = "chroma_db"
persistent_client = chromadb.PersistentClient(path=db_path)
collection_name = "lgu_plans_upgraded"
try:
    collection = persistent_client.get_collection(name=collection_name)
    print(f"✅ ChromaDB 클라이언트가 준비되었고, '{collection_name}' 컬렉션을 불러왔습니다.")
except ValueError:
    print(f"❌ '{collection_name}' 컬렉션을 찾을 수 없습니다. DB 셋업을 먼저 실행해주세요.")
    exit()

# 전체 데이터를 메모리에 로드 (Pandas 조건 검색용)
try:
    full_df = pd.read_csv('lgu_plans_refined.csv')
    print("✅ 조건 검색을 위해 전체 요금제 데이터를 메모리에 로드했습니다.")
except FileNotFoundError:
    print("❌ 'lgu_plans_refined.csv' 파일을 찾을 수 없습니다.")
    exit()

EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o"


# --- (generate_final_answer, setup_database 함수는 이전과 동일) ---
def generate_final_answer(query, retrieved_info):
    # ... 이전 코드와 동일 ...
    print("🤖 LLM이 검색된 정보를 바탕으로 최종 답변을 생성합니다...")
    system_prompt = """
    당신은 LG U+ 모바일 요금제 전문 상담원입니다.
    주어진 '참고 정보'를 바탕으로 사용자의 '질문'에 대해 친절하고 명확하게 답변해주세요.
    - 항상 '참고 정보'에 있는 내용만을 기반으로 답변해야 합니다. 정보를 지어내지 마세요.
    - 가격 정보는 '월 x,xxx원' 형식으로 표기해주세요.
    - 각 요금제의 핵심 특징을 잘 요약해서 설명해주세요.
    - 만약 참고 정보가 질문과 관련이 없거나 부족하다면, "죄송하지만 요청하신 정보를 찾을 수 없습니다." 라고 솔직하게 답변해주세요.
    """
    user_prompt = f"[참고 정보]\n{json.dumps(retrieved_info, indent=2, ensure_ascii=False)}\n\n[질문]\n{query}"
    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            temperature=0.2
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"❌ 답변 생성 중 오류가 발생했습니다: {e}"


# --- 검색 도구들 정의 ---

def search_plans_from_db(query, top_k=5):
    """[도구 1: 의미 검색] ChromaDB에서 의미적으로 유사한 요금제를 검색합니다."""
    print(f"\n🔍 (의미 검색) '{query}' 관련 정보를 ChromaDB에서 검색합니다...")
    query_embedding = client.embeddings.create(input=[query], model=EMBEDDING_MODEL).data[0].embedding
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)
    print(f"✅ {len(results['metadatas'][0])}개의 관련 요금제 정보를 찾았습니다.")
    return results['metadatas'][0]


def search_plans_with_pandas(operation, column, top_k=3):
    """[도구 2: 조건 검색] Pandas로 데이터를 정렬/필터링합니다."""
    print(f"\n⚙️ (조건 검색) Pandas를 사용하여 '{column}' 컬럼을 기준으로 '{operation}' 작업을 수행합니다.")

    if operation == 'max':
        result_df = full_df.sort_values(by=column, ascending=False).head(top_k)
    elif operation == 'min':
        result_df = full_df.sort_values(by=column, ascending=True).head(top_k)
    else:
        return []

    return json.loads(result_df.to_json(orient='records'))


# --- 매니저 함수 정의 ---

def chatbot_manager(query):
    """사용자 질문의 의도를 파악하고 적절한 도구를 선택하는 매니저 역할을 합니다."""
    print(f"\n🧠 매니저가 '{query}' 질문의 의도를 파악합니다...")

    # LLM을 이용한 의도 파악 프롬프트
    intent_prompt = f"""
    사용자의 질문을 분석하여 어떤 검색 유형에 해당하는지 결정하고, 필요한 정보를 JSON 형식으로 반환해줘.
    검색 유형은 'semantic' (의미 기반 검색) 또는 'structured' (조건 기반 검색) 중 하나야.

    - "제일 비싼", "가장 저렴한", "최고가" 등의 질문은 'structured' 검색이야. 'operation'은 'max' 또는 'min'으로, 'column'은 'monthly_price'로 설정해줘.
    - "데이터 제일 많은" 등의 질문도 'structured' 검색이야. 'operation'은 'max'로, 'column'은 'data_gb'로 설정해줘.
    - 그 외 "야무진", "쓸만한", "청소년용" 등 추상적인 추천 요청은 모두 'semantic' 검색이야.

    사용자 질문: "{query}"

    JSON 출력:
    """

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": intent_prompt}],
        response_format={"type": "json_object"},
        temperature=0.0
    )

    try:
        intent_result = json.loads(response.choices[0].message.content)
        search_type = intent_result.get('search_type')
        print(f"✅ 의도 파악 완료: {search_type}")

        retrieved_plans = []
        if search_type == 'structured':
            # 조건 검색 도구 사용
            operation = intent_result.get('operation')
            column = intent_result.get('column')
            retrieved_plans = search_plans_with_pandas(operation, column)
        else:  # 'semantic' 또는 미분류
            # 의미 검색 도구 사용
            retrieved_plans = search_plans_from_db(query)

        if not retrieved_plans:
            print("관련 요금제를 찾지 못했습니다.")
            return

        # 검색된 결과를 바탕으로 최종 답변 생성
        final_answer = generate_final_answer(query, retrieved_plans)

        print("\n---------- 챗봇 최종 답변 ----------")
        print(final_answer)
        print("------------------------------------")

    except (json.JSONDecodeError, KeyError) as e:
        print(f"🚨 의도 파악 결과(JSON)를 분석하는 중 오류가 발생했습니다: {e}")
        print("기본 의미 검색을 실행합니다.")
        # 실패 시 기본 의미 검색 실행
        retrieved_plans = search_plans_from_db(query)
        final_answer = generate_final_answer(query, retrieved_plans)
        print("\n---------- 챗봇 최종 답변 ----------")
        print(final_answer)
        print("------------------------------------")


# --- 메인 코드 실행 ---
if __name__ == "__main__":
    print("\n==================================================")
    print("🤖 LG U+ 요금제 상담 챗봇을 시작합니다. (v2. 하이브리드 검색)")
    print("==================================================")

    while True:
        user_query = input("질문을 입력하세요 (종료하시려면 '종료' 입력): ")
        if user_query.strip().lower() == '종료':
            print("👋 챗봇을 종료합니다. 이용해주셔서 감사합니다.")
            break
        if not user_query.strip():
            print("질문을 입력해주세요.")
            continue

        # 매니저 함수를 호출하여 챗봇 실행
        chatbot_manager(user_query)