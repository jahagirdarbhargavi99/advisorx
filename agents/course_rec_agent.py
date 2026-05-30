"""
Course Recommendation Agent
============================
Prompt pattern : Few-shot prompting with worked examples
Responsibility : Recommend next-semester courses based on audit gaps,
                 prerequisites, and student interests.
"""

import json
import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

PROJECT_ID = os.getenv("GCP_PROJECT_ID", "advisorx-demo")
LOCATION   = os.getenv("GCP_LOCATION",   "us-central1")
MODEL      = os.getenv("GEMINI_MODEL",   "gemini-2.5-flash")

client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)

# ── few-shot examples ──────────────────────────────────────────────────────
FEW_SHOT_EXAMPLES = """
EXAMPLE 1 — Good recommendation:
Student: Junior, GPA 3.2, completed CS201, CS202, MATH201, MATH202
Query: What should I take next semester?
Recommendation:
{
  "course_id": "CS301",
  "name": "Algorithms",
  "priority": "high",
  "rationale": "All prerequisites (CS201, MATH201) are satisfied. This is a required CS Core course and a gateway to 400-level electives. Taking it now keeps the student on track for senior year.",
  "prereqs_met": true,
  "category": "CS Core"
}

EXAMPLE 2 — Bad recommendation (do NOT do this):
Student: Freshman, GPA 2.1, completed CS101 only
Query: What should I take next semester?
BAD recommendation: CS402 Machine Learning
Why it's bad: CS402 requires CS301 and MATH301, neither of which the student has completed.
The student would be blocked from enrolling. Always verify prerequisites are fully met.

EXAMPLE 3 — Good recommendation for at-risk student:
Student: Sophomore, GPA 1.9 (academic probation), completed CS101, MATH101
Query: What should I take next semester?
Recommendation:
{
  "course_id": "CS201",
  "name": "Data Structures",
  "priority": "high",
  "rationale": "Prerequisites met (CS101). This is the next required core course. Given the student is on academic probation, recommend a lighter load of 3-4 courses maximum. CS201 is essential to make degree progress while managing GPA recovery.",
  "prereqs_met": true,
  "category": "CS Core",
  "gpa_warning": "Student is on academic probation. Limit to 12-13 credits this semester."
}
"""

SYSTEM_PROMPT = f"""You are the Course Recommendation Agent for AdvisorX.

You recommend the best courses for a student to take next semester using few-shot reasoning.
Always verify prerequisites are fully met before recommending a course.
Prioritize required courses over electives.
Consider GPA when recommending course load.

Here are examples of good and bad recommendations:
{FEW_SHOT_EXAMPLES}

Always return a JSON object with this structure:
{{
  "student_id": "...",
  "recommended_courses": [
    {{
      "course_id": "...",
      "name": "...",
      "credits": 0,
      "priority": "high/medium/low",
      "rationale": "...",
      "prereqs_met": true,
      "category": "CS Core/CS Electives/Mathematics/General Education/Free Elective"
    }}
  ],
  "total_recommended_credits": 0,
  "load_assessment": "appropriate/heavy/light",
  "next_semester_notes": "...",
  "warnings": []
}}"""


def run_course_rec_agent(profile: dict, audit: dict) -> dict:
    from tools.db_tool import get_all_courses
    from tools.rag_tool import search_courses

    all_courses   = get_all_courses()
    completed_ids = set(profile.get("completed_course_ids", []))

    # find courses with all prereqs met
    eligible = []
    for c in all_courses:
        if c["course_id"] in completed_ids:
            continue
        prereqs_met = all(p in completed_ids for p in c["prereqs"])
        if prereqs_met:
            eligible.append(c)

    # get audit gaps for context
    missing_by_cat = {}
    for cat in audit.get("categories", []):
        if cat.get("missing_courses"):
            missing_by_cat[cat["name"]] = cat["missing_courses"]

    eligible_str = " | ".join(
        f"{c['course_id']}: {c['name']} ({c['credits']}cr)"
        for c in eligible
    )

    msg = f"""Recommend next semester courses for this student.

Student: {profile.get('student_id')}, {profile.get('standing')}, GPA {profile.get('gpa')}
Risk flags: {profile.get('risk_flags', [])}
Completed courses: {sorted(completed_ids)}
Missing required courses by category: {json.dumps(missing_by_cat)}
Eligible courses (prereqs fully met): {eligible_str}

Recommend 4-5 courses for next semester following the few-shot examples.
Verify each course has all prerequisites in the completed list."""

    response = client.models.generate_content(
        model=MODEL,
        contents=msg,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.1,
            max_output_tokens=4096,
            response_mime_type="application/json",
        ),
    )

    try:
        return json.loads(response.text)
    except json.JSONDecodeError as e:
        return {
            "student_id": profile.get("student_id"),
            "error": str(e),
            "recommended_courses": [],
        }


if __name__ == "__main__":
    from rich.console import Console
    from rich.panel import Panel
    from rich.json import JSON
    from agents.intake_agent import run_intake_agent
    from agents.degree_audit_agent import run_degree_audit_agent

    console = Console()
    console.print("\n[bold cyan]Running Course Recommendation Agent for STU1000...[/bold cyan]\n")

    profile = run_intake_agent("STU1000")
    profile.pop("_reasoning", None)

    audit = run_degree_audit_agent(profile)
    audit.pop("_reasoning", None)

    recs = run_course_rec_agent(profile, audit)

    console.print(Panel(
        JSON(json.dumps(recs, indent=2)),
        title="Course Recommendations",
        border_style="green"
    ))
