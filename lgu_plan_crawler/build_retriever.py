# RAG retriever 생성 기능

import pandas as pd
import numpy as np
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
import os

try:
    client = OpenAI(api_key="API_KEY")
    print("✅ OpenAI API 키가 성공적으로 로드되었습니다.")
except KeyError:
    print("❌ 'OPENAI_API_KEY' 환경 변수가 설정되지 않았습니다.")
    print("API 키를 설정한 후 다시 시도해주세요.")
    exit()

# OpenAI의 최신 임베딩 모델을 지정합니다.
EMBEDDING_MODEL = "text-embedding-3-small"

# --- 2. 정제된 데이터 로드 및 검색용 텍스트 생성 ---
print("📄 정제된 CSV 파일을 로드하고 검색용 텍스트를 생성합니다.")
try:
    df = pd.read_csv('lgu_plans_refined.csv')
except FileNotFoundError:
    print("❌ 'lgu_plans_refined.csv' 파일을 찾을 수 없습니다. 이전 단계의 파일이 있는지 확인해주세요.")
    exit()

# 검색의 효율성을 위해 각 요금제의 주요 정보를 하나의 문장으로 결합합니다.
df['search_text'] = df.apply(
    lambda row: f"요금제명 {row['plan_name']}, 월 {row['monthly_price']}원, 데이터 {row['data_gb']}GB, 특징 {row['tags']}",
    axis=1
)

# --- 3. 각 요금제 텍스트를 OpenAI 모델로 임베딩 ---
print(f"🧠 OpenAI '{EMBEDDING_MODEL}' 모델로 임베딩을 시작합니다. (시간이 소요될 수 있습니다)")
try:
    # OpenAI API를 호출하여 임베딩을 생성합니다.
    response = client.embeddings.create(
        input=df['search_text'].tolist(),
        model=EMBEDDING_MODEL
    )
    plan_embeddings = np.array([item.embedding for item in response.data])
    print(f"✅ 총 {len(plan_embeddings)}개의 요금제에 대한 임베딩을 완료했습니다.")

    # 4. 생성된 벡터와 원본 데이터를 파일로 저장
    np.save('plan_embeddings_openai.npy', plan_embeddings)
    df.to_json('plan_data.json', orient='records', lines=True, force_ascii=False)
    print("💾 임베딩 벡터와 요금제 데이터를 파일로 저장했습니다. ('plan_embeddings_openai.npy', 'plan_data.json')")

except Exception as e:
    print(f"❌ OpenAI API 호출 중 오류가 발생했습니다: {e}")
    exit()


# ------------------- OpenAI 기반 검색 테스트 함수 -------------------
def find_similar_plans_openai(query, top_k=3):
    """
    사용자의 질문(query)을 받아 가장 유사한 요금제 top_k개를 찾아 반환합니다.
    """
    print(f"\n🔍 '{query}'와(과) 가장 유사한 요금제를 검색합니다...")

    try:
        plan_embeddings = np.load('plan_embeddings_openai.npy')
        plans_df = pd.read_json('plan_data.json', orient='records', lines=True)
    except FileNotFoundError:
        print("❌ 저장된 임베딩 파일을 찾을 수 없습니다. 먼저 스크립트를 실행하여 파일을 생성해주세요.")
        return

    # 사용자 질문을 OpenAI 모델로 임베딩
    query_embedding = client.embeddings.create(input=[query], model=EMBEDDING_MODEL).data[0].embedding

    # 코사인 유사도 계산
    cos_similarities = cosine_similarity([query_embedding], plan_embeddings)[0]

    # 유사도가 높은 순으로 정렬하고 상위 top_k개 인덱스 추출
    top_indices = cos_similarities.argsort()[-top_k:][::-1]

    print("\n---------- 검색 결과 ----------")
    for i, index in enumerate(top_indices):
        plan = plans_df.iloc[index]
        similarity = cos_similarities[index]
        print(f"🏅 {i + 1}순위 (유사도: {similarity:.4f})")
        print(f"  - 요금제명: {plan['plan_name']}")
        print(f"  - 월정액: {plan['monthly_price']}원")
        print(f"  - 데이터: {plan['data_gb']}GB ({plan['data_type']})")
        print(f"  - 태그: {plan['tags']}")
        print("-" * 20)

    return plans_df.iloc[top_indices]


# --- 메인 코드 실행 ---
if __name__ == "__main__":
    # --- 검색 테스트 ---
    find_similar_plans_openai("데이터 무제한 요금제 중에 제일 싼거")
    find_similar_plans_openai("청소년이 쓸만한 요금제 추천해줘")