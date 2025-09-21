import re
import sqlite3
import chromadb
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
from typing import List, Optional

# --- Configuration ---
DB_FILE = "chunks.db"
CHROMA_PATH = "chroma_db"
CHROMA_COLLECTION = "safety_docs"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# --- Reranking Parameters ---
RERANK_ALPHA = 0.5   # Weight for vector search score
ABSTAIN_THRESHOLD = 0.4  # Minimum score to generate an answer

# --- Global Objects ---
app = FastAPI(
    title="Mini RAG Q&A Service",
    description="A simple Q&A service over industrial safety documents."
)

# Load embedding model
print("Loading embedding model...")
model = SentenceTransformer(EMBEDDING_MODEL, device="cpu")
print("✅ Model loaded.")

# Connect to Chroma + SQLite
print("Connecting to databases...")
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = chroma_client.get_collection(name=CHROMA_COLLECTION)
db_conn = sqlite3.connect(DB_FILE, check_same_thread=False)
db_conn.row_factory = sqlite3.Row
print("✅ Database connections established.")


# --- Pydantic Models ---
class QuestionRequest(BaseModel):
    q: str = Field(..., description="The question to ask.", example="What is a safety function?")
    k: int = Field(5, description="Number of context chunks to return.", example=5)
    mode: str = Field("reranked", description="Search mode: 'baseline' or 'reranked'.", example="reranked")


class Context(BaseModel):
    source_id: str
    title: str
    content: str
    score: float


class AnswerResponse(BaseModel):
    answer: Optional[str]
    abstain_reason: Optional[str]
    contexts: List[Context]
    reranker_used: bool


# --- Helper Functions ---
def normalize_scores(items: List[dict], score_key: str) -> List[dict]:
    """Normalizes scores in a list of dicts to a 0-1 range."""
    scores = [item[score_key] for item in items if item.get(score_key) is not None]
    if not scores:
        return items

    min_score, max_score = min(scores), max(scores)
    if max_score == min_score:
        for item in items:
            if item.get(score_key) is not None:
                item[f"norm_{score_key}"] = 1.0
        return items

    for item in items:
        if item.get(score_key) is not None:
            item[f"norm_{score_key}"] = (item[score_key] - min_score) / (max_score - min_score)
    return items


def sanitize_for_fts(text: str) -> str:
    """
    Prepares user input for SQLite FTS5 MATCH:
    - Escapes quotes
    - Removes problematic symbols like /, :, *, etc.
    - Wraps in quotes so it's treated as a phrase
    """
    text = text.replace('"', '""')
    text = re.sub(r'[/:*^]', ' ', text)  # strip FTS special chars
    return f'"{text.strip()}"'


# --- API Endpoint ---
@app.post("/ask", response_model=AnswerResponse)
def ask_question(request: QuestionRequest):
    """Receives a question and returns a grounded answer with citations."""

    if request.mode not in ["baseline", "reranked"]:
        raise HTTPException(status_code=400, detail="Mode must be 'baseline' or 'reranked'")

    # Encode question
    query_embedding = model.encode(request.q, normalize_embeddings=True).tolist()

    # Get top N candidates from Chroma
    num_candidates = request.k * 3
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=num_candidates
    )

    candidates = []
    if results["ids"]:
        for i, chunk_id in enumerate(results["ids"][0]):
            candidates.append({
                "id": chunk_id,
                "content": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "vector_score": 1 - results["distances"][0][i]
            })

    if not candidates:
        return AnswerResponse(
            answer=None,
            abstain_reason="No relevant documents found.",
            contexts=[],
            reranker_used=False
        )

    reranker_used = (request.mode == "reranked")

    if reranker_used:
        cursor = db_conn.cursor()
        fts_query_phrase = sanitize_for_fts(request.q)

        for cand in candidates:
            sql_query = """
                SELECT bm25(chunks_fts) AS bm25_score
                FROM chunks_fts
                WHERE rowid = (SELECT rowid FROM chunks WHERE id = ?)
                  AND content MATCH ?
            """
            cursor.execute(sql_query, (cand["id"], fts_query_phrase))
            res = cursor.fetchone()
            cand["fts_score"] = -res["bm25_score"] if res else 0.0

        candidates = normalize_scores(candidates, "vector_score")
        candidates = normalize_scores(candidates, "fts_score")

        for cand in candidates:
            cand["final_score"] = (
                (RERANK_ALPHA * cand.get("norm_vector_score", 0)) +
                ((1 - RERANK_ALPHA) * cand.get("norm_fts_score", 0))
            )

        final_results = sorted(candidates, key=lambda x: x["final_score"], reverse=True)[:request.k]

    else:  # Baseline mode
        for cand in candidates:
            cand["final_score"] = cand["vector_score"]
        final_results = candidates[:request.k]

    # Build contexts
    contexts = [
        Context(
            source_id=res["metadata"]["source_id"],
            title=res["metadata"]["title"],
            content=res["content"],
            score=res["final_score"]
        ) for res in final_results
    ]

    # Decide whether to answer or abstain
    answer = None
    abstain_reason = None
    if contexts:
        top_context = contexts[0]
        if top_context.score >= ABSTAIN_THRESHOLD:
            answer = top_context.content
        else:
            abstain_reason = f"Top result score ({top_context.score:.2f}) is below the threshold of {ABSTAIN_THRESHOLD}."

    return AnswerResponse(
        answer=answer,
        abstain_reason=abstain_reason,
        contexts=contexts,
        reranker_used=reranker_used
    )


@app.on_event("shutdown")
def shutdown_event():
    """Close the database connection on application shutdown."""
    db_conn.close()
    print("Database connection closed.")
