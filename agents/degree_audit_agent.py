"""
Degree Audit Agent
==================
Prompt pattern : Chain-of-Thought with XML-tagged structured output
Responsibility : Check student progress against all degree requirements,
                 identify gaps, return structured audit result.
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

AUDIT_SYSTEM_PROMPT = """You are the Degree Audit Agent for AdvisorX.

Perform a degree audit for a Computer Science BS student.
Check every requirement category.

You MUST output a JSON object between <audit> and </audit> tags.
Before the tags, show brief reasoning for each category.

JSON structure:
{
  "student_id": "...",
  "total_credits_required": 120,
  "credits_earned": ...,
  "credits_remaining": ...,
  "overall_gpa": ...,
  "gpa_requirement_met": true/false,
  "categories": [
    {
      "name": "...",
      "required_credits": ...,
      "earned_credits": ...,
      "status": "complete/in_progress/not_started",
      "completed_courses": [...],
      "missing_courses": [...],
      "notes": "..."
    }
  ],
  "on_track_to_graduate": true/false,
  "estimated_semesters_remaining": ...,
  "critical_gaps": [...]
}"""

AUDIT_USER_TEMPLATE = """Audit this student's degree progress.

STUDENT PROFILE:
{profile}

DEGREE REQUIREMENTS:
{requirements}

For each category briefly note completed vs missing courses, then output the full audit JSON between <audit></audit> tags."""


def run_degree_audit_agent(profile: dict) -> dict:
    req_path = (
        __import__("pathlib").Path(__file__).parent.parent / "data" / "degree_requirements.json"
    )
    full_requirements = json.loads(req_path.read_text())
    requirements_str  = json.dumps(full_requirements, indent=2)
    profile_str       = json.dumps(profile, indent=2)

    user_msg = AUDIT_USER_TEMPLATE.format(
        profile=profile_str,
        requirements=requirements_str,
    )

    response = client.models.generate_content(
        model=MODEL,
        contents=user_msg,
        config=types.GenerateContentConfig(
            system_instruction=AUDIT_SYSTEM_PROMPT,
            temperature=0.1,
            max_output_tokens=8192,
        ),
    )

    raw_text = response.text
    audit    = _extract_audit(raw_text, profile)
    audit["_reasoning"] = raw_text
    return audit


def _extract_audit(text: str, profile: dict) -> dict:
    # strategy 1: <audit>...</audit>
    match = re.search(r"<audit>\s*(.*?)\s*</audit>", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # strategy 2: <audit> then grab to last }
    match2 = re.search(r"<audit>\s*(\{.*)", text, re.DOTALL)
    if match2:
        candidate = match2.group(1).strip()
        last_brace = candidate.rfind("}")
        if last_brace != -1:
            try:
                return json.loads(candidate[:last_brace+1])
            except json.JSONDecodeError:
                pass

    # strategy 3: any JSON block with student_id
    json_match = re.search(r"\{[\s\S]*\"student_id\"[\s\S]*\}", text)
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except json.JSONDecodeError:
            pass

    # fallback
    return {
        "student_id": profile.get("student_id"),
        "credits_earned": profile.get("credits_earned", 0),
        "credits_remaining": 120 - profile.get("credits_earned", 0),
        "overall_gpa": profile.get("gpa"),
        "gpa_requirement_met": profile.get("gpa", 0) >= 2.0,
        "categories": [],
        "on_track_to_graduate": False,
        "critical_gaps": ["Audit parsing failed — manual review needed"],
    }


if __name__ == "__main__":
    from rich.console import Console
    from rich.panel import Panel
    from rich.json import JSON
    from agents.intake_agent import run_intake_agent

    console = Console()
    console.print("\n[bold cyan]Running Degree Audit Agent for STU1000...[/bold cyan]\n")

    profile = run_intake_agent("STU1000")
    profile.pop("_reasoning", None)

    audit = run_degree_audit_agent(profile)

    # print raw response for debugging
    raw = audit.pop("_reasoning", "")
    console.print(Panel(
        raw[-2000:] if len(raw) > 2000 else raw,
        title="Raw LLM Response (last 2000 chars)", border_style="yellow"
    ))

    console.print(Panel(JSON(json.dumps(audit, indent=2)), title="Degree Audit Result", border_style="green"))
