import io
import os
import uuid
import numpy as np
import pandas as pd
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from qdrant_client.http import models as qm
from dotenv import load_dotenv

# 기존 공용 (임베딩/LLM/Qdrant/설정)
from .common import emb, llm, qdr, ensure_collection, COLLECTION_NAME

load_dotenv()
ensure_collection()  # 서버 기동 시 컬렉션 준비

app = FastAPI(title="Qdrant-only QA API", version="1.0.0")

# (선택) 로컬/프론트 테스트용 CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

# ---------- 스키마 ----------
class QuestionItem(BaseModel):
    question: str
    category: str

class IngestRequest(BaseModel):
    items: List[QuestionItem] = Field(..., description="question, category 배열")

class QueryRequest(BaseModel):
    query: str
    top_k: int = Field(5, ge=1, le=50)
    use_mmr: bool = True
    with_sources: bool = True

class Hit(BaseModel):
    score: float
    question: str
    category: str

class QueryResponse(BaseModel):
    answer: str
    hits: Optional[List[Hit]] = None

# ---------- 유틸 ----------
def _mmr(qvec: np.ndarray, hits, k=5, lam=0.5):
    if not hits or hits[0].vector is None:
        return hits[:k]
    vecs = np.stack([np.array(h.vector, dtype=float) for h in hits], axis=0)

    def cos(a, b): return float(np.dot(a, b) / (np.linalg.norm(a)*np.linalg.norm(b)+1e-12))
    selected, S, rest = [], [], list(range(len(hits)))

    # 첫 후보: 쿼리와 가장 유사
    first = max(rest, key=lambda i: cos(qvec, vecs[i]))
    selected.append(hits[first]); S.append(first); rest.remove(first)

    # 이후: MMR 점수 최대
    while len(selected) < min(k, len(hits)) and rest:
        def score(i):
            rel = cos(qvec, vecs[i])
            div = max(cos(vecs[i], vecs[j]) for j in S) if S else 0.0
            return lam*rel - (1-lam)*div
        j = max(rest, key=score)
        selected.append(hits[j]); S.append(j); rest.remove(j)
    return selected

def _build_context(hits) -> str:
    lines = [f"[{h.payload.get('category','')}] {h.payload.get('question','')}" for h in hits]
    return "\n".join(lines)

# ---------- 엔드포인트 ----------
@app.get("/health")
def health():
    return {"ok": True, "collection": COLLECTION_NAME}

@app.post("/ingest/json")
def ingest_json(req: IngestRequest):
    points = []
    for it in req.items:
        q = it.question.strip()
        if not q:
            continue
        vec = emb.embed_query(q)
        points.append(qm.PointStruct(
            id=str(uuid.uuid4()),
            vector=vec,
            payload={"question": q, "category": it.category.strip()}
        ))
    if not points:
        raise HTTPException(400, "no valid items")
    qdr.upsert(collection_name=COLLECTION_NAME, points=points)
    return {"upserted": len(points)}

@app.post("/ingest/csv")
async def ingest_csv(file: UploadFile = File(...)):
    # CSV 컬럼: question, category
    content = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception:
        # 엑셀에서 저장한 CSV가 MS949일 수 있음 → 백업 파서
        df = pd.read_csv(io.BytesIO(content), encoding="utf-8", on_bad_lines="skip")

    need = {"question", "category"}
    if not need.issubset(df.columns):
        raise HTTPException(400, f"CSV must have columns: {need}")

    df["question"] = df["question"].astype(str).str.strip()
    df["category"] = df["category"].astype(str).str.strip()
    df = df[df["question"] != ""].dropna(subset=["question"])
    if df.empty:
        raise HTTPException(400, "no valid rows")

    # 배치 임베딩으로 속도 ↑
    questions = df["question"].tolist()
    cats = df["category"].tolist()
    vectors = emb.embed_documents(questions)

    points = []
    for i in range(len(questions)):
        points.append(qm.PointStruct(
            id=str(uuid.uuid4()),
            vector=vectors[i],
            payload={"question": questions[i], "category": cats[i]}
        ))

    qdr.upsert(collection_name=COLLECTION_NAME, points=points)
    return {"upserted": len(points)}

@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    qvec = emb.embed_query(req.query)
    hits = qdr.search(
        collection_name=COLLECTION_NAME,
        query_vector=qvec,
        limit=max(req.top_k*2, req.top_k+4),
        with_payload=True,
        with_vectors=req.use_mmr,           # MMR 쓰면 벡터 필요
        search_params=qm.SearchParams(hnsw_ef=128),
    )
    picks = _mmr(np.array(qvec), hits, k=req.top_k) if req.use_mmr else hits[:req.top_k]
    ctx = _build_context(picks)

    prompt = f"""아래 유사 질문 목록을 참고해 사용자 질문에 간결히 답하세요. 
모르면 모른다고 하세요.

[사용자 질문]
{req.query}

[유사 질문들]
{ctx}
"""
    ans = llm.invoke(prompt).content
    out = QueryResponse(answer=ans)
    if req.with_sources:
        out.hits = [Hit(score=h.score,
                        question=h.payload.get("question",""),
                        category=h.payload.get("category",""))
                    for h in picks]
    return out
