# ingest_questions.py
import os, uuid
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm
from langchain_openai import OpenAIEmbeddings

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "questions")
ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "question.csv"

# text-embedding-3-small = 1536차원
EMBED_DIM = 1536

def ensure_collection(qdr: QdrantClient):
    names = [c.name for c in qdr.get_collections().collections]
    if COLLECTION_NAME not in names:
        qdr.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=qm.VectorParams(size=EMBED_DIM, distance=qm.Distance.COSINE),
        )

def main():
    # CSV 읽기 (컬럼: question, category)
    df = pd.read_csv(CSV_PATH)
    if not {"question", "category"}.issubset(set(df.columns)):
        raise ValueError("CSV must have columns: question, category")

    # 전처리(공백 제거, 결측 제거)
    df["question"] = df["question"].astype(str).str.strip()
    df["category"] = df["category"].astype(str).str.strip()
    df = df[df["question"] != ""].dropna(subset=["question"])

    qdr = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    ensure_collection(qdr)

    emb = OpenAIEmbeddings(model=EMBEDDING_MODEL, api_key=OPENAI_API_KEY)

    points = []
    for _, row in df.iterrows():
        q = row["question"]
        cat = row["category"]
        vec = emb.embed_query(q)

        points.append(
            qm.PointStruct(
                id=str(uuid.uuid4()),
                vector=vec,
                payload={
                    "question": q,
                    "category": cat,
                }
            )
        )

    if points:
        qdr.upsert(collection_name=COLLECTION_NAME, points=points)
        print(f"Upserted {len(points)} questions into '{COLLECTION_NAME}'")
    else:
        print("No valid rows to upsert.")

if __name__ == "__main__":
    main()
