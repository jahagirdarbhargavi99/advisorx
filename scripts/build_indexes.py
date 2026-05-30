import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

PROJECT_ID = os.getenv("GCP_PROJECT_ID", "advisorx-demo")
LOCATION   = os.getenv("GCP_LOCATION", "us-central1")
DATA_DIR   = Path(__file__).parent.parent / "data"
CHROMA_DIR = Path(__file__).parent.parent / "chroma_db"

import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
from google.cloud import aiplatform
from vertexai.language_models import TextEmbeddingModel
import vertexai

# ── initialize Vertex AI ───────────────────────────────────────────────────
print("Initializing Vertex AI...")
vertexai.init(project=PROJECT_ID, location=LOCATION)
embed_model = TextEmbeddingModel.from_pretrained("text-embedding-004")

# ── custom embedding function for ChromaDB ─────────────────────────────────
class VertexEmbeddingFunction(EmbeddingFunction):
    def __call__(self, input: Documents) -> Embeddings:
        # batch in groups of 5 to stay within rate limits
        all_embeddings = []
        for i in range(0, len(input), 5):
            batch = input[i:i+5]
            results = embed_model.get_embeddings(batch)
            all_embeddings.extend([r.values for r in results])
        return all_embeddings

# ── initialize ChromaDB ────────────────────────────────────────────────────
print("Initializing ChromaDB...")
client = chromadb.PersistentClient(path=str(CHROMA_DIR))
embedding_fn = VertexEmbeddingFunction()

# ── helper ─────────────────────────────────────────────────────────────────
def upsert_collection(name, documents, metadatas, ids):
    col = client.get_or_create_collection(
        name=name,
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"}
    )
    batch_size = 5
    for i in range(0, len(documents), batch_size):
        col.upsert(
            documents=documents[i:i+batch_size],
            metadatas=metadatas[i:i+batch_size],
            ids=ids[i:i+batch_size],
        )
    print(f"  ✓ {name}: {col.count()} documents indexed")
    return col

# ── 1. course catalog ──────────────────────────────────────────────────────
print("\nIndexing course catalog...")
courses = json.loads((DATA_DIR / "courses.json").read_text())

course_docs, course_meta, course_ids = [], [], []
for c in courses:
    prereq_names = ", ".join(c["prereqs"]) if c["prereqs"] else "None"
    doc = f"""Course: {c['id']} — {c['name']}
Department: {c['dept']}
Level: {c['level']}-level
Credits: {c['credits']}
Prerequisites: {prereq_names}
Description: {c['description']}"""
    course_docs.append(doc)
    course_meta.append({
        "course_id": c["id"],
        "dept": c["dept"],
        "level": c["level"],
        "credits": c["credits"],
        "prereqs": json.dumps(c["prereqs"])
    })
    course_ids.append(f"course_{c['id']}")

upsert_collection("courses", course_docs, course_meta, course_ids)

# ── 2. degree requirements ─────────────────────────────────────────────────
print("\nIndexing degree requirements...")
degree = json.loads((DATA_DIR / "degree_requirements.json").read_text())

req_docs, req_meta, req_ids = [], [], []
for i, cat in enumerate(degree["categories"]):
    required = cat.get("required_courses", [])
    choose   = cat.get("choose_from", [])
    doc = f"""Degree Requirement Category: {cat['name']}
Program: {degree['program']}
Required Credits: {cat['required_credits']}
Required Courses: {', '.join(required) if required else 'None'}
Elective Options: {', '.join(choose) if choose else 'N/A'}
Description: {cat['description']}"""
    req_docs.append(doc)
    req_meta.append({
        "category": cat["name"],
        "required_credits": cat["required_credits"],
        "required_courses": json.dumps(required),
    })
    req_ids.append(f"req_{i}_{cat['name'].replace(' ','_')}")

summary_doc = f"""Degree Program: {degree['program']}
Total Credits Required: {degree['total_credits_required']}
Minimum GPA Required: {degree['min_gpa']}
Categories: {', '.join(c['name'] for c in degree['categories'])}
This is a Bachelor of Science degree in Computer Science requiring 120 total credits."""
req_docs.append(summary_doc)
req_meta.append({"category": "summary", "required_credits": degree["total_credits_required"], "required_courses": "[]"})
req_ids.append("req_summary")

upsert_collection("degree_requirements", req_docs, req_meta, req_ids)

# ── 3. policy documents ────────────────────────────────────────────────────
print("\nIndexing policy documents...")
policy_dir = DATA_DIR / "policy_docs"

pol_docs, pol_meta, pol_ids = [], [], []
for policy_file in sorted(policy_dir.glob("*.txt")):
    content = policy_file.read_text()
    paragraphs = [p.strip() for p in content.split("\n\n") if p.strip() and len(p.strip()) >= 30]
    for j, para in enumerate(paragraphs):
        pol_docs.append(para)
        pol_meta.append({
            "source": policy_file.name,
            "policy_name": policy_file.stem.replace("_", " ").title(),
            "chunk": j
        })
        pol_ids.append(f"policy_{policy_file.stem}_{j}")

upsert_collection("policies", pol_docs, pol_meta, pol_ids)

# ── summary ────────────────────────────────────────────────────────────────
print("\n" + "="*50)
print("RAG indexes built successfully.")
print(f"ChromaDB stored at: {CHROMA_DIR}")
print("Collections:")
for col_name in ["courses", "degree_requirements", "policies"]:
    col = client.get_collection(col_name, embedding_function=embedding_fn)
    print(f"  - {col_name}: {col.count()} documents")
