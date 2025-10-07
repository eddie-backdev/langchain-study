# lgu_plan_chatbot_langchain/02_chatbot_langchain.py

import pandas as pd
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain.agents import tool, AgentExecutor, create_react_agent
from langchain import hub
import os
import json
from dotenv import load_dotenv

# .env 파일에서 환경 변수 로드
load_dotenv()

# --- 1. 기본 설정 ---
CHROMA_DB_PATH = "chroma_db_langchain"
REFINED_CSV_PATH = 'lgu_plans_refined.csv'
EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o"

# --- API 키 및 DB/데이터 파일 확인 ---
try:
    api_key = os.environ["OPENAI_API_KEY"]
    print("✅ OpenAI API 키를 성공적으로 로드했습니다.")
except KeyError:
    print("❌ 'OPENAI_API_KEY' 환경 변수가 설정되지 않았거나 .env 파일에 없습니다. API 키를 설정해주세요.")
    exit()

if not os.path.exists(CHROMA_DB_PATH):
    print(f"❌ ChromaDB 디렉토리('{CHROMA_DB_PATH}')를 찾을 수 없습니다.")
    print("먼저 01_setup_database.py를 실행하여 데이터베이스를 생성해주세요.")
    exit()

if not os.path.exists(REFINED_CSV_PATH):
    print(f"❌ 원본 데이터 파일('{REFINED_CSV_PATH}')을 찾을 수 없습니다.")
    exit()

# --- 2. LangChain 컴포넌트 초기화 ---
print("🔗 LangChain 컴포넌트를 초기화합니다...")
# LLM, 임베딩, 벡터 저장소, 리트리버 초기화
llm = ChatOpenAI(model=LLM_MODEL, temperature=0, api_key=api_key)
embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL, api_key=api_key)
vector_store = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embeddings)
retriever = vector_store.as_retriever(search_kwargs={'k': 5})

# 전체 데이터 로드 (조건 검색용)
full_df = pd.read_csv(REFINED_CSV_PATH)
print("✅ 컴포넌트 초기화 완료.")


# --- 3. 에이전트가 사용할 도구(Tool) 정의 ---

@tool
def semantic_search(query: str):
    """(의미 검색) 사용자의 질문과 의미적으로 가장 유사한 요금제 정보를 찾을 때 사용합니다.
    예: '청소년에게 쓸만한 요금제', '데이터 많이 쓰는 사람에게 좋은 요금제', '가성비 좋은 요금제' 등 추상적인 질문에 사용합니다.
    """
    print(f"\n>> 도구 실행: semantic_search(query='{query}')")

    # RAG Chain 정의 (LCEL - LangChain Expression Language)
    prompt = ChatPromptTemplate.from_template(
        """당신은 LG U+ 요금제 전문 상담원입니다.
        주어진 [참고 정보]를 바탕으로 사용자의 [질문]에 대해 친절하고 명확하게 답변해주세요.
        - 항상 [참고 정보]에 있는 내용만을 기반으로 답변해야 합니다. 정보를 지어내지 마세요.
        - 가격 정보는 '월 x,xxx원' 형식으로 표기해주세요.
        - 각 요금제의 핵심 특징을 잘 요약해서 설명해주세요.
        - 만약 참고 정보가 질문과 관련이 없거나 부족하다면, "죄송하지만 요청하신 정보를 찾을 수 없습니다." 라고 솔직하게 답변해주세요.

        [참고 정보]
        {context}

        [질문]
        {question}
        """
    )

    def format_docs(docs):
        # 검색된 Document 객체들을 하나의 문자열로 합칩니다.
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # RAG 체인 실행
    result = rag_chain.invoke(query)
    return result


@tool
def structured_search(operation: str, column: str):
    """(조건 검색) 특정 조건(가장 비싼/저렴한, 가장 많은/적은)에 맞는 요금제를 찾을 때 사용합니다.
    'operation' 인자에는 'max'(최대) 또는 'min'(최소)을, 'column' 인자에는 'monthly_price'(가격) 또는 'data_gb'(데이터)를 사용합니다.
    예: '가장 비싼 요금제', '데이터가 가장 많은 요금제'
    """
    print(f"\n>> 도구 실행: structured_search(operation='{operation}', column='{column}')")

    if column not in ['monthly_price', 'data_gb'] or operation not in ['max', 'min']:
        return "잘못된 인자입니다. column은 'monthly_price' 또는 'data_gb', operation은 'max' 또는 'min' 이어야 합니다."

    ascending = True if operation == 'min' else False
    result_df = full_df.sort_values(by=column, ascending=ascending).head(3)

    # 결과를 JSON 문자열로 변환하여 LLM이 이해하기 쉽게 만듦
    result_json = result_df.to_json(orient='records', force_ascii=False)

    # 최종 답변 생성을 위해 LLM 호출
    prompt = ChatPromptTemplate.from_template(
        """당신은 LG U+ 요금제 데이터 분석가입니다.
        주어진 [요금제 데이터]를 보고, 사용자의 [질문] 의도에 맞게 결과를 요약하고 친절하게 설명해주세요.

        [요금제 데이터]
        {context}

        [질문]
        {question}
        """
    )
    chain = prompt | llm | StrOutputParser()
    # 원래 질문을 함께 전달하여 더 자연스러운 답변 생성
    original_query = f"{column}을 기준으로 {operation} 값을 가지는 요금제 찾아줘"
    result = chain.invoke({"context": result_json, "question": original_query})
    return result


# --- 4. 에이전트(Agent) 생성 및 실행 ---

tools = [semantic_search, structured_search]

# ReAct 프롬프트 가져오기 (Agent가 어떤 방식으로 생각하고 행동할지 정의한 템플릿)
# https://smith.langchain.com/hub/hwchase17/react
prompt = hub.pull("hwchase17/react")

# 에이전트 생성
agent = create_react_agent(llm, tools, prompt)

# 에이전트 실행기(Executor) 생성
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)


# --- 5. 메인 실행 루프 ---
if __name__ == "__main__":
    print("\n==================================================")
    print("🤖 LG U+ 요금제 상담 챗봇을 시작합니다. (v3. LangChain Agent)")
    print("==================================================")

    while True:
        user_query = input("질문을 입력하세요 (종료하시려면 '종료' 입력): ")
        if user_query.strip().lower() == '종료':
            print("👋 챗봇을 종료합니다. 이용해주셔서 감사합니다.")
            break
        if not user_query.strip():
            print("질문을 입력해주세요.")
            continue

        # 에이전트 실행기에 질문을 전달하여 실행
        response = agent_executor.invoke({"input": user_query})

        print("\n---------- 챗봇 최종 답변 ----------")
        print(response['output'])
        print("------------------------------------")
