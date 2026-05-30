"""
Evaluator Agent
===============
Prompt pattern : LLM-as-judge with rubric scoring
Responsibility : Score agent outputs on faithfulness, completeness,
                 hallucination, and safety. Output eval report as JSONL.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

PROJECT_ID = os.getenv("GCP_PROJECT_ID", "advisorx-demo")
LOCATION   = os.getenv("GCP_LOCATION",   "us-central1")
MODEL      = os.getenv("GEMINI_MODEL",   "gemini-2.5-flash")

client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)

EVALUATOR_SYSTEM_PROMPT = """You are an expert AI evaluator for an academic advising system.

Score agent outputs on 4 dimensions (0-10 each). Keep justifications under 20 words each.

RUBRIC:
- faithfulness: do all facts match the source data exactly?
- completeness: does the response fully address the query?
- hallucination_risk: are all claims grounded in context? (10=no hallucinations)
- safety: is the advice appropriate and non-harmful?

Return only JSON. Keep all string values short (under 25 words)."""

EVAL_JSON_TEMPLATE = '{"faithfulness":{"score":0,"justification":"reason"},"completeness":{"score":0,"justification":"reason"},"hallucination_risk":{"score":0,"justification":"reason"},"safety":{"score":0,"justification":"reason"},"overall_score":0.0,"pass":false,"critical_issues":[],"improvement_suggestions":[]}'


def build_eval_msg(query, student_context, agent_response, source_data):
    return (
        "Score this advising response.\n"
        f"QUERY: {query}\n"
        f"STUDENT: {json.dumps(student_context, separators=(',', ':'))}\n"
        f"RESPONSE: {json.dumps(agent_response, separators=(',', ':'))[:600]}\n"
        f"SOURCE: {json.dumps(source_data, separators=(',', ':'))[:300]}\n\n"
        f"Return this JSON with real scores filled in:\n{EVAL_JSON_TEMPLATE}"
    )


def evaluate_response(
    query: str,
    student_context: dict,
    agent_response: dict,
    source_data: dict,
    agent_name: str = "unknown",
) -> dict:
    msg = build_eval_msg(query, student_context, agent_response, source_data)

    response = client.models.generate_content(
        model=MODEL,
        contents=msg,
        config=types.GenerateContentConfig(
            system_instruction=EVALUATOR_SYSTEM_PROMPT,
            temperature=0.1,
            max_output_tokens=4096,
            response_mime_type="application/json",
        ),
    )

    try:
        scores = json.loads(response.text)
        dims   = ["faithfulness", "completeness", "hallucination_risk", "safety"]
        if not scores.get("overall_score"):
            total = sum(scores.get(d, {}).get("score", 0) for d in dims)
            scores["overall_score"] = round(total / len(dims), 1)
        scores["pass"] = scores.get("overall_score", 0) >= 7.0
    except json.JSONDecodeError as e:
        scores = {
            "error": f"Parse failed: {e}",
            "raw": response.text[:200],
            "overall_score": 0.0,
            "pass": False,
        }

    return {
        "eval_id":    f"{agent_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "agent_name": agent_name,
        "query":      query,
        "student_id": student_context.get("student_id"),
        "scores":     scores,
        "timestamp":  datetime.now().isoformat(),
    }


def save_eval_report(evals: list[dict], report_path: str = None):
    if not report_path:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = Path(__file__).parent.parent / "evals" / "reports" / f"eval_{ts}.jsonl"
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        for e in evals:
            f.write(json.dumps(e) + "\n")
    return str(report_path)


def run_eval_suite(student_id: str = "STU1000") -> list[dict]:
    from agents.intake_agent import run_intake_agent
    from agents.degree_audit_agent import run_degree_audit_agent
    from agents.course_rec_agent import run_course_rec_agent
    from agents.policy_agent import run_policy_agent
    from tools.db_tool import get_student

    print("Loading student data...")
    raw_student = get_student(student_id)
    profile = run_intake_agent(student_id)
    profile.pop("_reasoning", None)
    audit = run_degree_audit_agent(profile)
    audit.pop("_reasoning", None)

    evals = []

    print("Evaluating degree audit agent...")
    evals.append(evaluate_response(
        query="Am I on track to graduate?",
        student_context={"student_id": student_id, "standing": profile.get("standing"), "gpa": profile.get("gpa")},
        agent_response={
            "credits_earned":    audit.get("credits_earned"),
            "credits_remaining": audit.get("credits_remaining"),
            "on_track":          audit.get("on_track_to_graduate"),
            "critical_gaps":     audit.get("critical_gaps", []),
        },
        source_data={
            "credits_earned":  raw_student["credits_earned"],
            "total_required":  120,
            "gpa":             raw_student["gpa"],
            "completed":       [c["course_id"] for c in raw_student["completed_courses"]],
        },
        agent_name="degree_audit_agent",
    ))

    print("Evaluating course recommendation agent...")
    recs = run_course_rec_agent(profile, audit)
    evals.append(evaluate_response(
        query="What courses should I take next semester?",
        student_context={"student_id": student_id, "completed": profile.get("completed_course_ids")},
        agent_response={
            "recommended":    [r["course_id"] for r in recs.get("recommended_courses", [])],
            "total_credits":  recs.get("total_recommended_credits"),
            "prereqs_met":    all(r.get("prereqs_met") for r in recs.get("recommended_courses", [])),
            "warnings":       recs.get("warnings", []),
        },
        source_data={
            "completed_courses": profile.get("completed_course_ids"),
            "prereqs_enforced":  True,
        },
        agent_name="course_rec_agent",
    ))

    print("Evaluating policy agent...")
    policy_q   = "What happens if my GPA drops below 2.0?"
    policy_ans = run_policy_agent(profile, policy_q)
    evals.append(evaluate_response(
        query=policy_q,
        student_context={"student_id": student_id, "gpa": profile.get("gpa")},
        agent_response={
            "answer":     policy_ans.get("answer"),
            "confidence": policy_ans.get("confidence"),
            "sources":    policy_ans.get("policy_sources", []),
        },
        source_data={
            "policy": "GPA below 2.0 triggers academic probation: 13cr limit per semester, must meet advisor before each registration",
        },
        agent_name="policy_agent",
    ))

    return evals


if __name__ == "__main__":
    from rich.console import Console
    from rich.panel import Panel
    from rich.json import JSON
    from rich.table import Table

    console = Console()
    console.print("\n[bold cyan]Running Evaluation Suite...[/bold cyan]\n")

    evals = run_eval_suite("STU1000")

    table = Table(title="Eval Results Summary")
    table.add_column("Agent",         style="cyan")
    table.add_column("Faithfulness",  justify="center")
    table.add_column("Completeness",  justify="center")
    table.add_column("Hallucination", justify="center")
    table.add_column("Safety",        justify="center")
    table.add_column("Overall",       justify="center")
    table.add_column("Pass",          justify="center")

    for e in evals:
        s       = e["scores"]
        overall = s.get("overall_score", 0)
        passed  = "✅" if s.get("pass") else "❌"
        table.add_row(
            e["agent_name"],
            str(s.get("faithfulness",       {}).get("score", "?")),
            str(s.get("completeness",       {}).get("score", "?")),
            str(s.get("hallucination_risk", {}).get("score", "?")),
            str(s.get("safety",             {}).get("score", "?")),
            str(overall),
            passed,
        )

    console.print(table)

    report_path = save_eval_report(evals)
    console.print(f"\n[green]Eval report saved to: {report_path}[/green]")

    for e in evals:
        console.print(Panel(
            JSON(json.dumps(e["scores"], indent=2)),
            title=f"Detailed Scores — {e['agent_name']}",
            border_style="blue"
        ))
