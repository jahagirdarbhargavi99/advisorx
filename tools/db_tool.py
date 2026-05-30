import json
import sqlite3
from pathlib import Path
from typing import Optional

DB_PATH = Path(__file__).parent.parent / "data" / "advisorx.db"


def get_student(student_id: str) -> dict | None:
    """Fetch a student record by ID including completed courses."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    student = cur.execute(
        "SELECT * FROM students WHERE student_id = ?", (student_id,)
    ).fetchone()

    if not student:
        conn.close()
        return None

    courses = cur.execute(
        """SELECT sc.course_id, sc.grade, c.name, c.credits, c.dept, c.level
           FROM student_courses sc
           JOIN courses c ON sc.course_id = c.course_id
           WHERE sc.student_id = ?""",
        (student_id,)
    ).fetchall()

    conn.close()

    return {
        "student_id": student["student_id"],
        "first_name": student["first_name"],
        "last_name": student["last_name"],
        "email": student["email"],
        "major": student["major"],
        "standing": student["standing"],
        "gpa": student["gpa"],
        "credits_earned": student["credits_earned"],
        "advisor_notes": student["advisor_notes"],
        "completed_courses": [
            {
                "course_id": c["course_id"],
                "name": c["name"],
                "credits": c["credits"],
                "dept": c["dept"],
                "level": c["level"],
                "grade": c["grade"],
            }
            for c in courses
        ],
    }


def get_all_students() -> list[dict]:
    """Fetch all student IDs and names."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    rows = cur.execute(
        "SELECT student_id, first_name, last_name, standing, gpa FROM students"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_course(course_id: str) -> dict | None:
    """Fetch a single course by ID."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    row = cur.execute(
        "SELECT * FROM courses WHERE course_id = ?", (course_id,)
    ).fetchone()
    conn.close()
    if not row:
        return None
    return {
        "course_id": row["course_id"],
        "name": row["name"],
        "credits": row["credits"],
        "dept": row["dept"],
        "level": row["level"],
        "prereqs": json.loads(row["prereqs"]),
        "description": row["description"],
    }


def get_all_courses() -> list[dict]:
    """Fetch all courses."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    rows = cur.execute("SELECT * FROM courses").fetchall()
    conn.close()
    return [
        {
            "course_id": r["course_id"],
            "name": r["name"],
            "credits": r["credits"],
            "dept": r["dept"],
            "level": r["level"],
            "prereqs": json.loads(r["prereqs"]),
            "description": r["description"],
        }
        for r in rows
    ]


def check_prerequisites(student_id: str, course_id: str) -> dict:
    """Check if a student meets prerequisites for a course."""
    student = get_student(student_id)
    course = get_course(course_id)

    if not student:
        return {"eligible": False, "reason": f"Student {student_id} not found."}
    if not course:
        return {"eligible": False, "reason": f"Course {course_id} not found."}

    completed_ids = {c["course_id"] for c in student["completed_courses"]}
    missing = [p for p in course["prereqs"] if p not in completed_ids]

    if missing:
        return {
            "eligible": False,
            "missing_prereqs": missing,
            "reason": f"Missing prerequisites: {', '.join(missing)}"
        }
    return {"eligible": True, "missing_prereqs": [], "reason": "All prerequisites met."}


if __name__ == "__main__":
    # quick smoke test
    s = get_student("STU1000")
    print(f"Student: {s['first_name']} {s['last_name']} — {s['standing']} — GPA {s['gpa']}")
    print(f"Completed courses: {[c['course_id'] for c in s['completed_courses']]}")
    prereq = check_prerequisites("STU1000", "CS301")
    print(f"CS301 prereq check: {prereq}")
