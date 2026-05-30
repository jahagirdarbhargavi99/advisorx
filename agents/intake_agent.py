"""
Intake & Profile Agent
======================
Prompt pattern : ReAct (Reason + Act)
Responsibility : Fetch student record from SQLite, reason about
                 completeness, return a validated profile dict.
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

INTAKE_SYSTEM_PROMPT = """You are the Intake Agent for AdvisorX, an AI academic advising system.

Your job is to analyze a student's raw database record and produce a clean,
validated profile summary that other agents will use.

You reason step by step using ReAct format:
- Thought: reason about what you observe
- Action: decide what to check or extract
- Observation: note what you find
- Final Answer: structured summary

Rules:
- Always identify if any critical fields are missing or suspicious
- Flag if GPA is below 2.0 (academic probation risk)
- Flag if credits_earned seems inconsistent with standing
- Note any advisor flags in the record
- Be concise — other agents depend on your output

Output your Final Answer as a JSON object inside <profile> tags."""

INTAKE_USER_TEMPLATE = """Analyze this student record and produce a validated profile.

STUDENT RECORD:
{student_record}

Use ReAct reasoning, then output the validated profile as JSON inside <profile> tags.

The JSON must include:
- student_id, full_name, email, major, standing, gpa
- credits_earned
- completed_course_ids (list)
- risk_flags (list of any concerns)
- advisor_notes
- data_quality ("complete" or "incomplete" with reason)
"""

def run_intake_agent(student_id: str) -> dict:
    from tools.db_tool import get_student

    raw = get_student(student_id)
    if not raw:
        return {"error": f"Student {student_id} not found in database."}

    record_str = json.dumps(raw, indent=2)
    user_msg   = INTAKE_USER_TEMPLATE.format(student_record=record_str)

    response = client.models.generate_content(
        model=MODEL,
        contents=user_msg,
        config=types.GenerateContentConfig(
            system_instruction=INTAKE_SYSTEM_PROMPT,
            temperature=0.1,
            max_output_tokens=1024,
        ),
    )

    raw_text = response.text
    profile  = _extract_profile(raw_text, raw)
    profile["_reasoning"] = raw_text
    return profile


def _extract_profile(text: str, fallback: dict) -> dict:
    match = re.search(r"<profile>(.*?)</profile>", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            pass
    return {
        "student_id": fallback["student_id"],
        "full_name": f"{fallback['first_name']} {fallback['last_name']}",
        "email": fallback["email"],
        "major": fallback["major"],
        "standing": fallback["standing"],
        "gpa": fallback["gpa"],
        "credits_earned": fallback["credits_earned"],
        "completed_course_ids": [c["course_id"] for c in fallback["completed_courses"]],
        "risk_flags": ["GPA below 2.0 — academic probation risk"] if fallback["gpa"] < 2.0 else [],
        "advisor_notes": fallback["advisor_notes"],
        "data_quality": "complete",
    }


if __name__ == "__main__":
    from rich.console import Console
    from rich.panel import Panel
    from rich.json import JSON

    console = Console()
    console.print("\n[bold cyan]Running Intake Agent for STU1000...[/bold cyan]\n")

    profile = run_intake_agent("STU1000")

    if "_reasoning" in profile:
        console.print(Panel(profile["_reasoning"], title="ReAct Reasoning Trace", border_style="yellow"))
        del profile["_reasoning"]

    console.print(Panel(JSON(json.dumps(profile, indent=2)), title="Validated Student Profile", border_style="green"))
