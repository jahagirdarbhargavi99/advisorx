"""
Orchestrator Agent
==================
Framework     : LangGraph stateful graph
Responsibility: Decompose student queries, route to sub-agents,
                manage conversation state, assemble final response.
"""

import json
import os
from typing import TypedDict, Annotated
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
import operator

load_dotenv()

# ── state schema ───────────────────────────────────────────────────────────
class AdvisorState(TypedDict):
    student_id:   str
    query:        str
    profile:      dict
    audit:        dict
    plan:         dict
    recs:         dict
    policy_answer: dict
    intent:       str
    final_response: dict
    errors:       Annotated[list, operator.add]


# ── intent classifier ──────────────────────────────────────────────────────
def classify_intent(state: AdvisorState) -> AdvisorState:
    """Classify the student query using keyword matching — faster and more reliable
    than LLM classification for short single-label tasks."""

    query = state["query"].lower()

    # course_plan keywords
    if any(k in query for k in [
        "4 year", "four year", "multi year", "long term", "plan my",
        "degree plan", "semester by semester", "all semesters", "graduation plan"
    ]):
        return {**state, "intent": "course_plan"}

    # course_rec keywords
    if any(k in query for k in [
        "next semester", "this semester", "register", "registration",
        "sign up", "enroll", "what should i take", "what courses should",
        "recommend", "courses to take", "take next", "this fall", "this spring",
        "upcoming semester"
    ]):
        return {**state, "intent": "course_rec"}

    # policy keywords
    if any(k in query for k in [
        "policy", "policies", "waive", "waiver", "probation", "dismissal",
        "appeal", "hold", "drop", "withdraw", "withdrawal", "deadline",
        "rule", "regulation", "allowed", "permitted", "exception",
        "gpa drop", "below 2", "full time", "part time", "overload",
        "transfer credit", "ap credit"
    ]):
        return {**state, "intent": "policy"}

    # degree_audit keywords
    if any(k in query for k in [
        "on track", "graduate", "graduation", "credits", "requirements",
        "missing", "completed", "progress", "audit", "degree", "how many",
        "what do i need", "still need", "remaining", "finish"
    ]):
        return {**state, "intent": "degree_audit"}

    # default
    return {**state, "intent": "degree_audit"}


# ── intake node ────────────────────────────────────────────────────────────
def run_intake(state: AdvisorState) -> AdvisorState:
    from agents.intake_agent import run_intake_agent
    profile = run_intake_agent(state["student_id"])
    profile.pop("_reasoning", None)
    return {**state, "profile": profile}


# ── degree audit node ──────────────────────────────────────────────────────
def run_audit(state: AdvisorState) -> AdvisorState:
    from agents.degree_audit_agent import run_degree_audit_agent
    audit = run_degree_audit_agent(state["profile"])
    audit.pop("_reasoning", None)
    return {**state, "audit": audit}


# ── course rec node ────────────────────────────────────────────────────────
def run_recs(state: AdvisorState) -> AdvisorState:
    from agents.course_rec_agent import run_course_rec_agent
    recs = run_course_rec_agent(state["profile"], state["audit"])
    return {**state, "recs": recs}


# ── course plan node ───────────────────────────────────────────────────────
def run_plan(state: AdvisorState) -> AdvisorState:
    from agents.course_planning_agent import run_course_planning_agent
    plan = run_course_planning_agent(state["profile"], state["audit"])
    plan.pop("_reasoning", None)
    return {**state, "plan": plan}


# ── policy node ────────────────────────────────────────────────────────────
def run_policy(state: AdvisorState) -> AdvisorState:
    from agents.policy_agent import run_policy_agent
    policy_answer = run_policy_agent(state["profile"], state["query"])
    return {**state, "policy_answer": policy_answer}


# ── response assembler ─────────────────────────────────────────────────────
def assemble_response(state: AdvisorState) -> AdvisorState:
    """Assemble final response based on intent."""
    intent  = state.get("intent", "degree_audit")
    profile = state.get("profile", {})
    audit   = state.get("audit", {})

    base = {
        "student_id":   state["student_id"],
        "query":        state["query"],
        "intent":       intent,
        "student_name": profile.get("full_name", "Student"),
        "standing":     profile.get("standing"),
        "gpa":          profile.get("gpa"),
    }

    if intent == "degree_audit":
        base["audit_summary"] = {
            "credits_earned":    audit.get("credits_earned"),
            "credits_remaining": audit.get("credits_remaining"),
            "on_track":          audit.get("on_track_to_graduate"),
            "critical_gaps":     audit.get("critical_gaps", []),
            "categories":        audit.get("categories", []),
        }
    elif intent == "course_rec":
        recs = state.get("recs", {})
        base["recommendations"] = recs.get("recommended_courses", [])
        base["total_credits"]   = recs.get("total_recommended_credits")
        base["load_assessment"] = recs.get("load_assessment")
        base["warnings"]        = recs.get("warnings", [])
    elif intent == "course_plan":
        plan = state.get("plan", {})
        base["selected_branch"]    = plan.get("selected_branch")
        base["semesters"]          = plan.get("semesters", [])
        base["graduation_target"]  = plan.get("graduation_target")
        base["key_milestones"]     = plan.get("key_milestones", [])
    elif intent == "policy":
        pa = state.get("policy_answer", {})
        base["policy_answer"]  = pa.get("answer")
        base["action_steps"]   = pa.get("action_steps", [])
        base["caveats"]        = pa.get("caveats", [])
        base["policy_sources"] = pa.get("policy_sources", [])
        base["confidence"]     = pa.get("confidence")

    return {**state, "final_response": base}


# ── routing logic ──────────────────────────────────────────────────────────
def route_by_intent(state: AdvisorState) -> str:
    intent = state.get("intent", "degree_audit")
    routes = {
        "degree_audit": "run_audit",
        "course_rec":   "run_audit",
        "course_plan":  "run_audit",
        "policy":       "run_policy",
    }
    return routes.get(intent, "run_audit")


def route_after_audit(state: AdvisorState) -> str:
    intent = state.get("intent", "degree_audit")
    if intent == "course_rec":
        return "run_recs"
    elif intent == "course_plan":
        return "run_plan"
    return "assemble_response"


# ── build graph ────────────────────────────────────────────────────────────
def build_advisor_graph():
    graph = StateGraph(AdvisorState)

    graph.add_node("classify_intent",    classify_intent)
    graph.add_node("run_intake",         run_intake)
    graph.add_node("run_audit",          run_audit)
    graph.add_node("run_recs",           run_recs)
    graph.add_node("run_plan",           run_plan)
    graph.add_node("run_policy",         run_policy)
    graph.add_node("assemble_response",  assemble_response)

    graph.set_entry_point("classify_intent")
    graph.add_edge("classify_intent", "run_intake")
    graph.add_conditional_edges("run_intake", route_by_intent)
    graph.add_conditional_edges("run_audit",  route_after_audit)
    graph.add_edge("run_recs",    "assemble_response")
    graph.add_edge("run_plan",    "assemble_response")
    graph.add_edge("run_policy",  "assemble_response")
    graph.add_edge("assemble_response", END)

    return graph.compile()


def run_advisor(student_id: str, query: str) -> dict:
    graph = build_advisor_graph()
    result = graph.invoke({
        "student_id": student_id,
        "query":      query,
        "profile":    {},
        "audit":      {},
        "plan":       {},
        "recs":       {},
        "policy_answer": {},
        "intent":     "",
        "final_response": {},
        "errors":     [],
    })
    return result["final_response"]


if __name__ == "__main__":
    from rich.console import Console
    from rich.panel import Panel
    from rich.json import JSON

    console = Console()

    test_queries = [
        ("STU1000", "Am I on track to graduate?"),
        ("STU1000", "What courses should I take next semester?"),
        ("STU1000", "What happens if my GPA drops below 2.0?"),
    ]

    for student_id, query in test_queries:
        console.print(f"\n[bold cyan]Student: {student_id} | Query: {query}[/bold cyan]\n")
        response = run_advisor(student_id, query)
        console.print(Panel(
            JSON(json.dumps(response, indent=2)),
            title=f"Orchestrator Response — {response.get('intent', 'unknown')}",
            border_style="magenta"
        ))
