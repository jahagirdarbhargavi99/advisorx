import json
import os
from pathlib import Path
from typing import Optional

import chromadb
import vertexai
from vertexai.language_models import TextEmbeddingModel
from chromadb import Documents, EmbeddingFunction, Embeddings
from dotenv import load_dotenv

load_dotenv()

PROJECT_ID = os.getenv("GCP_PROJECT_ID", "advisorx-demo")
LOCATION   = os.getenv("GCP_LOCATION", "us-central1")
CHROMA_DIR = Path(__file__).parent.parent / "chroma_db"

# ── initialize once at module load ─────────────────────────────────────────
vertexai.init(project=PROJECT_ID, location=LOCATION)
_embed_model = TextEmbeddingModel.from_pretrained("text-embedding-004")


class VertexEmbeddingFunction(EmbeddingFunction):
    def __call__(self, input: Documents) -> Embeddings:
        all_embeddings = []
        for i in range(0, len(input), 5):
            batch = input[i:i+5]
            results = _embed_model.get_embeddings(batch)
            all_embeddings.extend([r.values for r in results])
        return all_embeddings


_client     = chromadb.PersistentClient(path=str(CHROMA_DIR))
_embed_fn   = VertexEmbeddingFunction()


def _get_collection(name: str):
    return _client.get_collection(name, embedding_function=_embed_fn)


# ── public functions ───────────────────────────────────────────────────────

def search_courses(query: str, n_results: int = 5, dept: Optional[str] = None) -> list[dict]:
    """Search the course catalog by semantic query."""
    col = _get_collection("courses")
    where = {"dept": dept} if dept else None
    results = col.query(
        query_texts=[query],
        n_results=min(n_results, col.count()),
        where=where,
        include=["documents", "metadatas", "distances"]
    )
    output = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ):
        output.append({
            "course_id": meta["course_id"],
            "dept": meta["dept"],
            "level": meta["level"],
            "credits": meta["credits"],
            "prereqs": json.loads(meta["prereqs"]),
            "relevance_score": round(1 - dist, 3),
            "document": doc,
        })
    return output


def search_degree_requirements(query: str, n_results: int = 3) -> list[dict]:
    """Search degree requirement categories by semantic query."""
    col = _get_collection("degree_requirements")
    results = col.query(
        query_texts=[query],
        n_results=min(n_results, col.count()),
        include=["documents", "metadatas", "distances"]
    )
    output = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ):
        output.append({
            "category": meta["category"],
            "required_credits": meta["required_credits"],
            "required_courses": json.loads(meta.get("required_courses", "[]")),
            "relevance_score": round(1 - dist, 3),
            "document": doc,
        })
    return output


def search_policies(query: str, n_results: int = 4) -> list[dict]:
    """Search academic policies by semantic query."""
    col = _get_collection("policies")
    results = col.query(
        query_texts=[query],
        n_results=min(n_results, col.count()),
        include=["documents", "metadatas", "distances"]
    )
    output = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ):
        output.append({
            "source": meta["source"],
            "policy_name": meta["policy_name"],
            "chunk": meta["chunk"],
            "relevance_score": round(1 - dist, 3),
            "content": doc,
        })
    return output


if __name__ == "__main__":
    print("=== Course search: machine learning ===")
    results = search_courses("machine learning and AI", n_results=3)
    for r in results:
        print(f"  {r['course_id']} — score {r['relevance_score']}")

    print("\n=== Degree requirements: math ===")
    results = search_degree_requirements("mathematics requirements", n_results=2)
    for r in results:
        print(f"  {r['category']} — score {r['relevance_score']}")

    print("\n=== Policy search: course waiver ===")
    results = search_policies("how to waive a required course", n_results=3)
    for r in results:
        print(f"  [{r['policy_name']}] score {r['relevance_score']}")
        print(f"  {r['content'][:120]}...")
