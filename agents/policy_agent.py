"""
Policy Agent
============
Prompt pattern : RAG with citation enforcement + Chain-of-Thought
Responsibility : Answer academic policy questions by retrieving relevant
                 policy chunks and reasoning from them with citations.
                 Includes course existence pre-check to prevent hallucination.
"""

import json
import os
import re
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

PROJECT_ID = os.getenv("GCP_PROJECT_ID", "advisorx-demo")
LOCATION   = os.getenv("GCP_LOCATION",   "us-central1")
MODEL      = os.getenv("GEMINI_MODEL",   "gemini-2.5-flash")

client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)

POLICY_SYSTEM_PROMPT = """You are the Policy Agent for AdvisorX.

You answer academic policy questions by reasoning strictly from retrieved policy documents.

Rules:
1. ALWAYS cite the policy document you are drawing from using [Policy: document_name]
2. NEVER make up policy details not present in the retrieved context
3. If the policy documents do not contain enough information, say so explicitly
4. Use Chain-of-Thought: first identify the relevant policy, then reason through it, then give the answer
5. If a student may not qualify, explain exactly what they would need to do

Format your response as JSON:
{
  "question": "...",
  "policy_sources": ["source1", "source2"],
  "reasoning": "step by step reasoning from the policy text",
  "answer": "clear direct answer to the question",
  "action_steps": ["step 1", "step 2"],
  "caveats": ["any important exceptions or conditions"],
  "confidence": "high/medium/low"
}"""

POLICY_USER_TEMPLATE = """Answer this academic policy question for a student.

STUDENT CONTEXT:
- Student ID: {student_id}
- Standing: {standing}
- GPA: {gpa}
- Major: {major}

QUESTION: {question}

RETRIEVED POLICY DOCUMENTS:
{policy_context}

Reason through the policy carefully and provide a complete answer with citations."""


def check_course_exists(course_id: str) -> bool:
    """Check if a course exists in the database."""
    from tools.db_tool import get_course
    return get_course(course_id) is not None


def extract_course_ids_from_query(query: str) -> list:
    """Extract course IDs like CS999, MATH101 from a query."""
    return re.findall(r'\b[A-Z]{2,4}\d{3}\b', query.upper())


def run_policy_agent(profile: dict, question: str) -> dict:
    from tools.rag_tool import search_policies

    # pre-check: flag any course IDs that don't exist in the catalog
    mentioned_courses = extract_course_ids_from_query(question)
    nonexistent = [c for c in mentioned_courses if not check_course_exists(c)]
    if nonexistent:
        return {
            "question": question,
            "policy_sources": [],
            "reasoning": f"Course(s) {nonexistent} not found in the course catalog.",
            "answer": f"Course {', '.join(nonexistent)} does not exist in the course catalog. Please check the course ID and try again.",
            "action_steps": ["Verify the course ID with the registrar or course catalog"],
            "caveats": [],
            "confidence": "high",
            "retrieved_sources": [],
        }

    # retrieve relevant policy chunks
    policy_results = search_policies(question, n_results=5)

    policy_context = "\n\n---\n\n".join([
        f"[Policy: {r['policy_name']}] (relevance: {r['relevance_score']})\n{r['content']}"
        for r in policy_results
    ])

    msg = POLICY_USER_TEMPLATE.format(
        student_id=profile.get("student_id", "unknown"),
        standing=profile.get("standing", "unknown"),
        gpa=profile.get("gpa", "unknown"),
        major=profile.get("major", "unknown"),
        question=question,
        policy_context=policy_context,
    )

    response = client.models.generate_content(
        model=MODEL,
        contents=msg,
        config=types.GenerateContentConfig(
            system_instruction=POLICY_SYSTEM_PROMPT,
            temperature=0.1,
            max_output_tokens=2048,
            response_mime_type="application/json",
        ),
    )

    try:
        result = json.loads(response.text)
        result["retrieved_sources"] = [r["source"] for r in policy_results]
        return result
    except json.JSONDecodeError as e:
        return {
            "question": question,
            "error": str(e),
            "answer": response.text,
            "confidence": "low",
        }


if __name__ == "__main__":
    from rich.console import Console
    from rich.panel import Panel
    from rich.json import JSON
    from agents.intake_agent import run_intake_agent

    console = Console()

    profile = run_intake_agent("STU1000")
    profile.pop("_reasoning", None)

    tests = [
        "Can I take CS999 Advanced Quantum Computing next semester?",
        "Can I waive CS101 if I took AP Computer Science with a score of 5?",
        "What happens if my GPA drops below 2.0?",
        "How many credits do I need to take to be considered full-time?",
    ]

    for q in tests:
        console.print(f"\n[bold yellow]Q: {q}[/bold yellow]")
        result = run_policy_agent(profile, q)
        console.print(Panel(
            JSON(json.dumps(result, indent=2)),
            title="Policy Answer",
            border_style="blue"
        ))
