
# lgu_plan_chatbot_langchain/01_setup_database.py

import pandas as pd
from langchain_community.document_loaders import CSVLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
import os
from dotenv import load_dotenv

# .env 파일에서 환경 변수 로드
load_dotenv()

# --- 1. 기본 설정 ---
# 정제된 데이터 파일 경로
REFINED_CSV_PATH = 'lgu_plans_refined.csv'
# ChromaDB를 저장할 디렉토리
CHROMA_DB_PATH = "chroma_db_langchain"
# OpenAI 임베딩 모델
EMBEDDING_MODEL = "text-embedding-3-small"

def setup_database():
    """
    정제된 CSV 데이터를 LangChain을 이용해 로드하고,
    OpenAI 임베딩을 거쳐 ChromaDB에 저장합니다.
    """
    # --- API 키 확인 ---
    try:
        api_key = os.environ["OPENAI_API_KEY"]
        print("✅ OpenAI API 키를 성공적으로 로드했습니다.")
    except KeyError:
        print("❌ 'OPENAI_API_KEY' 환경 변수가 설정되지 않았거나 .env 파일에 없습니다.")
        print("API 키를 설정한 후 다시 시도해주세요.")
        return

    # --- 데이터 파일 확인 ---
    if not os.path.exists(REFINED_CSV_PATH):
        print(f"❌ '{REFINED_CSV_PATH}' 파일을 찾을 수 없습니다.")
        print("데이터 크롤링 및 정제 스크립트를 먼저 실행하여 데이터를 준비해주세요.")
        return

    print(f"🚚 '{REFINED_CSV_PATH}' 파일에서 데이터를 로드합니다.")

    # --- 2. LangChain의 CSVLoader를 사용하여 데이터 로드 ---
    # 각 row를 하나의 Document 객체로 로드합니다.
    # page_content는 기본적으로 모든 컬럼을 합친 문자열이 됩니다.
    loader = CSVLoader(file_path=REFINED_CSV_PATH, encoding='utf-8')
    documents = loader.load()

    # UTF-8 인코딩 시 문제가 발생하는 surrogate characters 제거
    for doc in documents:
        doc.page_content = doc.page_content.encode('utf-8', 'surrogatepass').decode('utf-8')
        for key, value in doc.metadata.items():
            if isinstance(value, str):
                doc.metadata[key] = value.encode('utf-8', 'surrogatepass').decode('utf-8')

    print(f"📄 총 {len(documents)}개의 요금제 정보를 Document로 변환했습니다.")
    print("미리보기 (첫 번째 Document):")
    # print(documents[0])

    # --- 3. LangChain의 OpenAIEmbeddings와 Chroma를 사용하여 DB 구축 ---
    print(f"🧠 OpenAI '{EMBEDDING_MODEL}' 모델로 임베딩 및 DB 저장을 시작합니다...")

    # 임베딩 모델 초기화
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL, api_key=api_key)

    # Chroma.from_documents 함수
    # 1. documents의 각 Document에 대해 page_content를 임베딩
    # 2. 임베딩 벡터와 Document(page_content + metadata)를 DB에 저장
    # 3. DB를 영구적으로 저장할 경로 지정 (persist_directory)
    db = Chroma.from_documents(
        documents,
        embeddings,
        persist_directory=CHROMA_DB_PATH
    )

    print(f"✅ 총 {len(documents)}개의 요금제 정보를 ChromaDB ('{CHROMA_DB_PATH}')에 성공적으로 저장했습니다.")
    print("데이터베이스 셋업이 완료되었습니다.")


if __name__ == "__main__":
    setup_database()
