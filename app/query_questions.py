# query_questions.py
import os
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "questions")
ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "question.csv"

emb = OpenAIEmbeddings(model=EMBEDDING_MODEL, api_key=OPENAI_API_KEY)
llm = ChatOpenAI(model=CHAT_MODEL, temperature=0, api_key=OPENAI_API_KEY)

def search(query: str, top_k=8, ef=128, with_vectors=True):
    qvec = emb.embed_query(query)
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    res = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=qvec,
        limit=top_k,
        with_payload=True,
        with_vectors=with_vectors,
        search_params=qm.SearchParams(hnsw_ef=ef),
    )
    return np.array(qvec), res

def mmr(qvec, hits, k=5, lam=0.5):
    if not hits or hits[0].vector is None:
        return hits[:k]
    vecs = np.stack([np.array(h.vector, dtype=float) for h in hits], axis=0)
    def cos(a,b): return float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-12))
    selected, S, rest = [], [], list(range(len(hits)))
    first = max(rest, key=lambda i: cos(qvec, vecs[i]))
    selected.append(hits[first]); S.append(first); rest.remove(first)
    while len(selected) < min(k, len(hits)) and rest:
        def score(i):
            rel = cos(qvec, vecs[i])
            div = max(cos(vecs[i], vecs[j]) for j in S) if S else 0.0
            return lam*rel - (1-lam)*div
        j = max(rest, key=score)
        selected.append(hits[j]); S.append(j); rest.remove(j)
    return selected

def build_context(hits):
    lines = []
    for h in hits:
        q = h.payload.get("question", "")
        c = h.payload.get("category", "")
        lines.append(f"[{c}] {q}")
    return "\n".join(lines)

def summarize_answer(user_q, context):
    prompt = f"""아래는 기존에 등록된 유사 질문 목록입니다. 
질문자 의도를 고려해 한 문장으로 간결하게 요약 답하세요. 모르면 모른다고 하세요.

[사용자 질문]
{user_q}

[유사 질문들]
{context}
"""
    return llm.invoke(prompt).content

if __name__ == "__main__":
    user_q = input("질문: ").strip()
    qvec, hits_raw = search(user_q, top_k=12, with_vectors=True)
    hits = mmr(qvec, hits_raw, k=5, lam=0.5)

    print("\n=== 유사 질문 Top-5 ===")
    for i, h in enumerate(hits, 1):
        print(f"{i}. ({h.payload.get('category','')}) {h.payload.get('question','')}"
              f"  [score={h.score:.4f}]")

    ctx = build_context(hits)
    ans = summarize_answer(user_q, ctx)
    print("\n=== 요약 답변 ===")
    print(ans)
