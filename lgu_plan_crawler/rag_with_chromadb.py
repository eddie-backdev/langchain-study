# -*- coding: utf-8 -*-

import pandas as pd
import chromadb
from openai import OpenAI
import os
import json

# --- 1. OpenAI API 키 및 ChromaDB 클라이언트 설정 ---
try:
    client = OpenAI(api_key="API_KEY")
    print("✅ OpenAI API 키가 성공적으로 로드되었습니다.")
except KeyError:
    print("❌ 'OPENAI_API_KEY' 환경 변수가 설정되지 않았습니다.")
    exit()

# ChromaDB 클라이언트 설정 (데이터를 'chroma_db' 폴더에 영구 저장)
db_path = "chroma_db"
persistent_client = chromadb.PersistentClient(path=db_path)

# 컬렉션 생성 (테이블과 유사한 개념)
# get_or_create_collection: 있으면 가져오고, 없으면 생성
collection_name = "lgu_plans"
collection = persistent_client.get_or_create_collection(name=collection_name)
print(f"✅ ChromaDB 클라이언트가 준비되었고, '{collection_name}' 컬렉션을 사용합니다.")

EMBEDDING_MODEL = "text-embedding-3-small"


def setup_database(csv_file='lgu_plans_refined.csv'):
    """
    정제된 CSV 데이터를 읽어와 ChromaDB에 저장(임베딩)합니다.
    이미 데이터가 있다면 이 함수는 건너뛸 수 있습니다.
    """
    if collection.count() > 0:
        print(f"ℹ️ 데이터베이스에 이미 {collection.count()}개의 요금제 정보가 있습니다. 셋업을 건너뜁니다.")
        return

    print(f"🚚 '{csv_file}'에서 데이터를 로드하여 데이터베이스 셋업을 시작합니다.")
    df = pd.read_csv(csv_file)
    df = df.astype(str)  # 모든 데이터를 문자열로 변환 (메타데이터 저장용)

    # 검색에 사용될 텍스트 문서 생성
    documents = df.apply(
        lambda
            row: f"요금제명: {row['plan_name']}. 월정액: {row['monthly_price']}원. 데이터: {row['data_gb']}GB. 특징: {row['tags']}",
        axis=1
    ).tolist()

    # 검색 결과와 함께 반환될 메타데이터 생성
    metadatas = df.to_dict('records')

    # 각 요금제를 구분할 고유 ID 생성
    ids = [f"plan_{i}" for i in range(len(df))]

    print(f"🧠 OpenAI '{EMBEDDING_MODEL}' 모델로 임베딩 및 DB 저장을 시작합니다...")

    # OpenAI 임베딩 생성
    response = client.embeddings.create(input=documents, model=EMBEDDING_MODEL)
    embeddings = [item.embedding for item in response.data]

    # ChromaDB에 데이터 추가!
    collection.add(
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )
    print(f"✅ 총 {len(ids)}개의 요금제 정보를 ChromaDB에 성공적으로 저장했습니다.")


def search_plans_in_db(query, top_k=3):
    """
    사용자 질문(query)을 받아 ChromaDB에서 가장 유사한 요금제 top_k개를 검색합니다.
    """
    print(f"\n🔍 ChromaDB에서 '{query}'와(과) 가장 유사한 요금제를 검색합니다...")

    # 사용자 질문을 OpenAI 모델로 임베딩
    query_embedding = client.embeddings.create(input=[query], model=EMBEDDING_MODEL).data[0].embedding

    # ChromaDB에 쿼리 실행
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )

    print("\n---------- 검색 결과 ----------")
    if not results['documents'][0]:
        print("관련 요금제를 찾지 못했습니다.")
        return []

    # 검색 결과(메타데이터)를 출력
    for i, metadata in enumerate(results['metadatas'][0]):
        print(f"🏅 {i + 1}순위 (유사도 점수: {results['distances'][0][i]:.4f})")
        print(f"  - 요금제명: {metadata['plan_name']}")
        print(f"  - 월정액: {metadata['monthly_price']}원")
        print(f"  - 데이터: {metadata['data_gb']}GB ({metadata['data_type']})")
        print(f"  - 태그: {metadata['tags']}")
        print("-" * 20)

    return results['metadatas'][0]


# --- 메인 코드 실행 ---
if __name__ == "__main__":
    # 1. 데이터베이스 셋업 (최초 1회 또는 데이터 변경 시 실행)
    setup_database()

    # 2. 검색 테스트
    search_plans_in_db("데이터 무제한 요금제 중에 제일 싼거")
    search_plans_in_db("청소년이 쓸만한 요금제 추천해줘")
    search_plans_in_db("데이터는 10GB 정도만 있으면 돼")