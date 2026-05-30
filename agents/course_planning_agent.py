"""
Course Planning Agent
=====================
Prompt pattern : Tree-of-Thoughts (ToT)
Responsibility : Evaluate planning branches, then generate the degree
                 plan semester by semester.
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

TOT_REASONING_PROMPT = """You are the Course Planning Agent for AdvisorX.

Evaluate 3 planning strategies for this CS student:
- Branch A: Fastest path (18 credits/semester, prereqs first)
- Branch B: Balanced path (15 credits/semester, steady pace)
- Branch C: Interest-aligned path (AI/ML track prioritized)

For each branch score: feasibility (0-10), gpa_protection (0-10), career_alignment (0-10), total.
End your response with exactly: WINNER: Branch [A/B/C]
Keep total response under 250 words."""


def _generate_semester(client, model, student_id, completed_ids, all_courses, sem_num, season, year, strategy_desc):
    """Generate one semester plan using compact course representation."""
    available = [
        f"{c['course_id']}: {c['name']} ({c['credits']}cr) needs:{c['prereqs']}"
        for c in all_courses
        if c["course_id"] not in completed_ids
    ]

    msg = (
        f"Plan semester {sem_num} ({season}, {year}) for CS student {student_id}.\n"
        f"Strategy: {strategy_desc}\n"
        f"Completed: {sorted(completed_ids)}\n"
        f"Available: {' | '.join(available[:20])}\n\n"
        f"Pick 4-5 courses whose prereqs are ALL in the completed list.\n"
        f'Return JSON: {{"semester":{sem_num},"season":"{season}","year":"{year}",'
        f'"courses":[{{"course_id":"X","name":"Y","credits":3,"rationale":"reason"}}],'
        f'"total_credits":15,"semester_notes":"brief note"}}'
    )

    response = client.models.generate_content(
        model=model,
        contents=msg,
        config=types.GenerateContentConfig(
            temperature=0.1,
            max_output_tokens=4096,
            response_mime_type="application/json",
        ),
    )
    return json.loads(response.text)


def run_course_planning_agent(profile: dict, audit: dict) -> dict:
    from tools.db_tool import get_all_courses

    all_courses   = get_all_courses()
    completed_ids = set(profile.get("completed_course_ids", []))

    # ── call 1: ToT branch evaluation ─────────────────────────────────────
    reasoning_msg = (
        f"Student: {profile.get('standing')}, GPA {profile.get('gpa')}, "
        f"{profile.get('credits_earned')} credits earned. "
        f"Completed: {list(completed_ids)}. "
        f"Credits remaining: {120 - profile.get('credits_earned', 0)}."
    )

    reasoning_response = client.models.generate_content(
        model=MODEL,
        contents=reasoning_msg,
        config=types.GenerateContentConfig(
            system_instruction=TOT_REASONING_PROMPT,
            temperature=0.3,
            max_output_tokens=600,
        ),
    )
    reasoning_text = reasoning_response.text

    # extract winner
    strategy_key = "B"
    winner_match = re.search(r"WINNER:\s*Branch\s*([ABC])", reasoning_text, re.IGNORECASE)
    if winner_match:
        strategy_key = winner_match.group(1).upper()

    strategy_descs = {
        "A": "fastest path, 17-18 credits/semester, prioritize prerequisites",
        "B": "balanced path, 14-15 credits/semester, mix core and gen ed",
        "C": "AI/ML track, fast-track prerequisites for CS402 Machine Learning",
    }
    strategy_desc = strategy_descs[strategy_key]

    # extract branch scores
    branch_scores = {}
    for branch in ["A", "B", "C"]:
        nums = re.findall(r"\d+", re.sub(
            r".*?" + f"Branch {branch}" + r"(.*?)(?=Branch [ABC]|WINNER|$)",
            r"\1", reasoning_text, flags=re.IGNORECASE | re.DOTALL
        ))
        scores = [int(n) for n in nums if 0 <= int(n) <= 10][:3]
        while len(scores) < 3:
            scores.append(0)
        branch_scores[branch] = {
            "feasibility": scores[0], "gpa_protection": scores[1],
            "career_alignment": scores[2],
            "total": round(sum(scores) / 3, 1)
        }

    # ── generate each semester individually ───────────────────────────────
    schedule = [
        (1, "Fall",   "Year 1"), (2, "Spring", "Year 1"),
        (3, "Fall",   "Year 2"), (4, "Spring", "Year 2"),
        (5, "Fall",   "Year 3"), (6, "Spring", "Year 3"),
        (7, "Fall",   "Year 4"), (8, "Spring", "Year 4"),
    ]

    semesters         = []
    running_completed = set(completed_ids)

    for sem_num, season, year in schedule:
        total_so_far = profile.get("credits_earned", 0) + sum(
            s.get("total_credits", 0) for s in semesters
        )
        if total_so_far >= 108:
            break

        try:
            sem = _generate_semester(
                client, MODEL, profile.get("student_id"),
                running_completed, all_courses, sem_num, season, year, strategy_desc
            )
            for course in sem.get("courses", []):
                running_completed.add(course["course_id"])
            semesters.append(sem)
        except Exception as e:
            semesters.append({
                "semester": sem_num, "season": season, "year": year,
                "error": str(e), "courses": [], "total_credits": 0,
            })

    total_new  = sum(s.get("total_credits", 0) for s in semesters)
    total_sems = len(semesters)

    return {
        "student_id":              profile.get("student_id"),
        "selected_branch":         strategy_key,
        "branch_scores":           branch_scores,
        "selection_rationale":     f"Branch {strategy_key} selected: {strategy_desc}.",
        "semesters":               semesters,
        "total_semesters":         total_sems,
        "projected_total_credits": profile.get("credits_earned", 0) + total_new,
        "graduation_target":       f"{schedule[min(total_sems-1,7)][1]} {schedule[min(total_sems-1,7)][2]}",
        "key_milestones": [
            "Complete CS core sequence by Year 2",
            "Begin 400-level electives in Year 3",
            "Complete Senior Capstone (CS401) in Year 4",
        ],
        "_reasoning": reasoning_text,
    }


if __name__ == "__main__":
    from rich.console import Console
    from rich.panel import Panel
    from rich.json import JSON
    from agents.intake_agent import run_intake_agent
    from agents.degree_audit_agent import run_degree_audit_agent

    console = Console()
    console.print("\n[bold cyan]Running Course Planning Agent for STU1000...[/bold cyan]\n")

    profile = run_intake_agent("STU1000")
    profile.pop("_reasoning", None)

    audit = run_degree_audit_agent(profile)
    audit.pop("_reasoning", None)

    plan = run_course_planning_agent(profile, audit)

    if "_reasoning" in plan:
        console.print(Panel(
            plan["_reasoning"],
            title="Tree-of-Thoughts Branch Evaluation",
            border_style="yellow"
        ))
        del plan["_reasoning"]

    console.print(Panel(
        JSON(json.dumps(plan, indent=2)),
        title="Optimal Degree Plan",
        border_style="green"
    ))
# patch applied — semester 8 handled via fallback in run_course_planning_agent
