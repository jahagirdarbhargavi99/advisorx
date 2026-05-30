import json
import os
import random
import sqlite3
from pathlib import Path

random.seed(42)

# ── paths ──────────────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).parent.parent / "data"
DB_PATH  = DATA_DIR / "advisorx.db"
DATA_DIR.mkdir(parents=True, exist_ok=True)
(DATA_DIR / "policy_docs").mkdir(parents=True, exist_ok=True)

# ── 1. course catalog ──────────────────────────────────────────────────────
COURSES = [
    # CS core
    {"id":"CS101","name":"Intro to Programming","credits":3,"dept":"CS","level":100,"prereqs":[],"description":"Fundamentals of programming using Python. Variables, loops, functions, and basic data structures."},
    {"id":"CS201","name":"Data Structures","credits":3,"dept":"CS","level":200,"prereqs":["CS101"],"description":"Arrays, linked lists, stacks, queues, trees, and graphs. Algorithm complexity analysis."},
    {"id":"CS301","name":"Algorithms","credits":3,"dept":"CS","level":300,"prereqs":["CS201","MATH201"],"description":"Sorting, searching, dynamic programming, greedy algorithms, and NP-completeness."},
    {"id":"CS401","name":"Senior Capstone","credits":3,"dept":"CS","level":400,"prereqs":["CS301"],"description":"Culminating project integrating knowledge from all CS coursework."},
    {"id":"CS202","name":"Computer Organization","credits":3,"dept":"CS","level":200,"prereqs":["CS101"],"description":"Assembly language, memory hierarchy, CPU architecture, and I/O systems."},
    {"id":"CS302","name":"Operating Systems","credits":3,"dept":"CS","level":300,"prereqs":["CS202","CS201"],"description":"Processes, threads, memory management, file systems, and concurrency."},
    {"id":"CS303","name":"Database Systems","credits":3,"dept":"CS","level":300,"prereqs":["CS201"],"description":"Relational model, SQL, query optimization, transactions, and NoSQL systems."},
    {"id":"CS304","name":"Software Engineering","credits":3,"dept":"CS","level":300,"prereqs":["CS201"],"description":"Software development lifecycle, design patterns, testing, and agile methods."},
    {"id":"CS402","name":"Machine Learning","credits":3,"dept":"CS","level":400,"prereqs":["CS301","MATH301"],"description":"Supervised and unsupervised learning, neural networks, model evaluation."},
    {"id":"CS403","name":"Computer Networks","credits":3,"dept":"CS","level":400,"prereqs":["CS302"],"description":"TCP/IP, routing, network security, and distributed systems fundamentals."},
    {"id":"CS404","name":"Computer Vision","credits":3,"dept":"CS","level":400,"prereqs":["CS402"],"description":"Image processing, feature extraction, CNNs, and object detection."},
    {"id":"CS405","name":"Natural Language Processing","credits":3,"dept":"CS","level":400,"prereqs":["CS402"],"description":"Text processing, language models, transformers, and NLP applications."},
    {"id":"CS305","name":"Programming Languages","credits":3,"dept":"CS","level":300,"prereqs":["CS201"],"description":"Language paradigms, type systems, compilers, and language design."},
    {"id":"CS306","name":"Computer Security","credits":3,"dept":"CS","level":300,"prereqs":["CS302"],"description":"Cryptography, network security, vulnerability analysis, and secure coding."},
    {"id":"CS406","name":"Cloud Computing","credits":3,"dept":"CS","level":400,"prereqs":["CS303","CS302"],"description":"Cloud architecture, containerization, microservices, and DevOps practices."},
    {"id":"CS407","name":"Distributed Systems","credits":3,"dept":"CS","level":400,"prereqs":["CS403"],"description":"Consensus algorithms, replication, fault tolerance, and distributed databases."},
    # Math
    {"id":"MATH101","name":"Calculus I","credits":4,"dept":"MATH","level":100,"prereqs":[],"description":"Limits, derivatives, and integrals of single-variable functions."},
    {"id":"MATH102","name":"Calculus II","credits":4,"dept":"MATH","level":100,"prereqs":["MATH101"],"description":"Integration techniques, sequences, series, and polar coordinates."},
    {"id":"MATH201","name":"Discrete Mathematics","credits":3,"dept":"MATH","level":200,"prereqs":["MATH101"],"description":"Logic, sets, relations, graph theory, combinatorics, and proof techniques."},
    {"id":"MATH202","name":"Linear Algebra","credits":3,"dept":"MATH","level":200,"prereqs":["MATH102"],"description":"Vectors, matrices, linear transformations, eigenvalues, and eigenvectors."},
    {"id":"MATH301","name":"Probability & Statistics","credits":3,"dept":"MATH","level":300,"prereqs":["MATH202"],"description":"Probability theory, distributions, statistical inference, and regression."},
    {"id":"MATH302","name":"Numerical Methods","credits":3,"dept":"MATH","level":300,"prereqs":["MATH202","CS101"],"description":"Numerical solutions to mathematical problems using computational methods."},
    # Electives
    {"id":"CS450","name":"AI Ethics","credits":3,"dept":"CS","level":400,"prereqs":["CS402"],"description":"Ethical implications of AI systems, bias, fairness, and policy considerations."},
    {"id":"CS451","name":"Robotics","credits":3,"dept":"CS","level":400,"prereqs":["CS402"],"description":"Robot kinematics, perception, planning, and control systems."},
    {"id":"CS452","name":"Blockchain & Cryptography","credits":3,"dept":"CS","level":400,"prereqs":["CS306"],"description":"Cryptographic protocols, blockchain architecture, and decentralized applications."},
    {"id":"CS453","name":"Human-Computer Interaction","credits":3,"dept":"CS","level":400,"prereqs":["CS304"],"description":"User-centered design, usability testing, and interface prototyping."},
    {"id":"CS350","name":"Game Development","credits":3,"dept":"CS","level":300,"prereqs":["CS201","CS202"],"description":"Game engine architecture, graphics programming, and interactive systems."},
    {"id":"CS351","name":"Mobile App Development","credits":3,"dept":"CS","level":300,"prereqs":["CS304"],"description":"iOS and Android development, mobile UI patterns, and app deployment."},
    # General education
    {"id":"ENG101","name":"English Composition","credits":3,"dept":"ENG","level":100,"prereqs":[],"description":"Academic writing, argumentation, and research skills."},
    {"id":"ENG201","name":"Technical Writing","credits":3,"dept":"ENG","level":200,"prereqs":["ENG101"],"description":"Writing for technical audiences including documentation and reports."},
    {"id":"PHYS101","name":"Physics I","credits":4,"dept":"PHYS","level":100,"prereqs":["MATH101"],"description":"Mechanics, kinematics, dynamics, energy, and momentum."},
    {"id":"PHYS102","name":"Physics II","credits":4,"dept":"PHYS","level":100,"prereqs":["PHYS101","MATH102"],"description":"Electricity, magnetism, waves, and optics."},
    {"id":"SOC101","name":"Introduction to Sociology","credits":3,"dept":"SOC","level":100,"prereqs":[],"description":"Social structures, institutions, culture, and socialization."},
    {"id":"PSY101","name":"Introduction to Psychology","credits":3,"dept":"PSY","level":100,"prereqs":[],"description":"Biological bases of behavior, perception, learning, and social psychology."},
]

# ── 2. degree requirements ─────────────────────────────────────────────────
DEGREE_REQUIREMENTS = {
    "program": "Bachelor of Science in Computer Science",
    "total_credits_required": 120,
    "min_gpa": 2.0,
    "categories": [
        {
            "name": "CS Core",
            "required_credits": 24,
            "required_courses": ["CS101","CS201","CS202","CS301","CS302","CS303","CS304","CS401"],
            "description": "Foundational computer science courses required of all CS majors."
        },
        {
            "name": "CS Electives",
            "required_credits": 15,
            "choose_from": ["CS402","CS403","CS404","CS405","CS305","CS306","CS406","CS407","CS450","CS451","CS452","CS453","CS350","CS351"],
            "min_courses": 5,
            "description": "Advanced CS electives — at least 3 must be 400-level."
        },
        {
            "name": "Mathematics",
            "required_credits": 17,
            "required_courses": ["MATH101","MATH102","MATH201","MATH202","MATH301"],
            "description": "Mathematics foundation required for CS theory and ML coursework."
        },
        {
            "name": "General Education",
            "required_credits": 18,
            "required_courses": ["ENG101","ENG201","PHYS101","PHYS102"],
            "choose_from": ["SOC101","PSY101"],
            "min_choose": 1,
            "description": "Broad liberal arts education including writing, physics, and social sciences."
        },
        {
            "name": "Free Electives",
            "required_credits": 12,
            "description": "Any courses not already counted toward other categories."
        }
    ]
}

# ── 3. student records ─────────────────────────────────────────────────────
FIRST_NAMES = ["Alex","Jordan","Morgan","Taylor","Casey","Riley","Cameron","Avery","Quinn","Skylar","Dakota","Reese","Peyton","Finley","Rowan","Sage","Blake","Drew","Emery","Parker"]
LAST_NAMES  = ["Smith","Johnson","Williams","Brown","Jones","Garcia","Miller","Davis","Martinez","Wilson","Anderson","Thomas","Jackson","White","Harris","Thompson","Moore","Martin","Lee","Walker"]
MAJORS      = ["Computer Science"] * 18 + ["Computer Science (Transfer)","Computer Science (Honors)"]
STANDINGS   = ["Freshman","Sophomore","Junior","Senior"]

def random_gpa():
    return round(random.uniform(2.1, 4.0), 2)

def generate_completed_courses(standing):
    pools = {
        "Freshman": ["CS101","MATH101","ENG101"],
        "Sophomore": ["CS101","CS201","CS202","MATH101","MATH102","MATH201","ENG101","PHYS101"],
        "Junior": ["CS101","CS201","CS202","CS301","CS302","CS303","MATH101","MATH102","MATH201","MATH202","MATH301","ENG101","ENG201","PHYS101","PHYS102"],
        "Senior": ["CS101","CS201","CS202","CS301","CS302","CS303","CS304","MATH101","MATH102","MATH201","MATH202","MATH301","ENG101","ENG201","PHYS101","PHYS102","CS402","CS403"],
    }
    base = pools[standing].copy()
    # randomly drop 1-2 courses to create audit gaps
    drop_count = random.randint(1, 2)
    for _ in range(drop_count):
        if base:
            base.pop(random.randint(0, len(base) - 1))
    return base

students = []
for i in range(20):
    standing = random.choice(STANDINGS)
    completed = generate_completed_courses(standing)
    credits_earned = sum(c["credits"] for c in COURSES if c["id"] in completed)
    student = {
        "student_id": f"STU{1000 + i}",
        "first_name": FIRST_NAMES[i],
        "last_name": LAST_NAMES[i],
        "email": f"{FIRST_NAMES[i].lower()}.{LAST_NAMES[i].lower()}@university.edu",
        "major": MAJORS[i],
        "standing": standing,
        "gpa": random_gpa(),
        "credits_earned": credits_earned,
        "completed_courses": completed,
        "advisor_notes": random.choice([
            "Student is on track.",
            "Needs to meet with advisor before registration.",
            "Interested in AI/ML track.",
            "Planning to apply to grad school.",
            "Working part-time, taking lighter course load.",
            ""
        ])
    }
    students.append(student)

# ── 4. policy documents ────────────────────────────────────────────────────
POLICIES = {
    "academic_standing.txt": """
ACADEMIC STANDING POLICY

Good Standing: Students must maintain a cumulative GPA of 2.0 or higher.

Academic Probation: Students whose cumulative GPA falls below 2.0 are placed on academic probation.
- Probationary students may not register for more than 13 credit hours per semester.
- Probationary students must meet with their academic advisor before each registration period.
- Students on probation for two consecutive semesters may be subject to academic dismissal.

Academic Dismissal: Students dismissed for academic reasons may appeal to the Academic Standards Committee within 30 days.

Dean's List: Students who earn a semester GPA of 3.5 or higher while completing at least 12 credit hours are placed on the Dean's List.
""",

    "course_waiver_policy.txt": """
COURSE WAIVER AND SUBSTITUTION POLICY

A course waiver exempts a student from a required course without receiving credit for it.
A course substitution allows a different course to fulfill a requirement.

Eligibility:
- Students may petition for a waiver if they have demonstrated equivalent competency through prior coursework, AP/IB credit, or professional experience.
- Waivers for CS Core courses require department chair approval.
- Waivers for General Education requirements require dean approval.

AP Credit Policy:
- AP Computer Science A (score 4-5): Waives CS101, grants 3 credits.
- AP Calculus BC (score 4-5): Waives MATH101 and MATH102, grants 8 credits.
- AP Statistics (score 4-5): May substitute for MATH301 with advisor approval.

Transfer Credit Policy:
- Transfer credits from accredited institutions are evaluated by the registrar.
- Courses with grade C or higher are accepted.
- Up to 60 transfer credits may be applied toward the degree.
- CS major courses transferred from non-ABET-accredited programs require department review.

Process:
1. Student submits Petition for Course Waiver/Substitution form to the registrar.
2. Advisor endorses the petition with written justification.
3. Department chair reviews and approves or denies within 15 business days.
4. Decision is final unless appealed to the Academic Standards Committee.
""",

    "registration_policy.txt": """
REGISTRATION AND ENROLLMENT POLICY

Credit Hour Limits:
- Standard load: 12-18 credit hours per semester.
- Overload (19+ credits) requires a GPA of 3.0 or higher and advisor approval.
- Minimum full-time status: 12 credit hours.
- Part-time status: fewer than 12 credit hours.

Registration Holds:
- Academic hold: placed when GPA falls below 2.0. Requires advisor meeting to lift.
- Financial hold: placed for outstanding balances. Contact bursar to resolve.
- Immunization hold: placed if health records are incomplete.

Prerequisite Enforcement:
- Prerequisites are enforced at registration. Students who lack prerequisites will be dropped from the course.
- Prerequisite waivers may be granted by the course instructor with written justification.
- Concurrent enrollment in a prerequisite course requires instructor approval.

Late Registration:
- Students may add courses through the end of the second week of classes.
- A late registration fee of $50 applies after the first week.
- Dropping a course after the 10th week results in a W grade on the transcript.
""",

    "graduation_policy.txt": """
GRADUATION REQUIREMENTS AND POLICY

To be eligible for graduation with a Bachelor of Science in Computer Science, students must:
1. Complete all required courses with a grade of C or higher in each CS Core course.
2. Earn a minimum cumulative GPA of 2.0.
3. Complete a minimum of 120 credit hours.
4. Complete at least 30 of the final 36 credit hours in residence at the university.
5. File a graduation application by the deadline (typically the semester prior to graduation).
6. Complete the Senior Capstone (CS401) in the final year.

Latin Honors:
- Cum Laude: cumulative GPA 3.5 - 3.69
- Magna Cum Laude: cumulative GPA 3.7 - 3.89
- Summa Cum Laude: cumulative GPA 3.9 or higher

Double Major:
- Students pursuing a double major must satisfy all requirements for both programs.
- Shared courses may count toward both majors with advisor approval.
- A minimum of 18 unique upper-division credits are required for the second major.

Minor Requirements:
- A minor requires completion of at least 18 credit hours in the minor field.
- At least 9 hours must be upper-division (300-400 level).
- A GPA of 2.0 or higher in minor courses is required.
"""
}

# ── 5. write everything to disk ────────────────────────────────────────────
# JSON files
with open(DATA_DIR / "courses.json", "w") as f:
    json.dump(COURSES, f, indent=2)

with open(DATA_DIR / "degree_requirements.json", "w") as f:
    json.dump(DEGREE_REQUIREMENTS, f, indent=2)

with open(DATA_DIR / "students.json", "w") as f:
    json.dump(students, f, indent=2)

# Policy text files
for filename, content in POLICIES.items():
    with open(DATA_DIR / "policy_docs" / filename, "w") as f:
        f.write(content.strip())

# ── 6. build SQLite database ───────────────────────────────────────────────
conn = sqlite3.connect(DB_PATH)
cur  = conn.cursor()

cur.executescript("""
DROP TABLE IF EXISTS students;
DROP TABLE IF EXISTS courses;
DROP TABLE IF EXISTS student_courses;

CREATE TABLE students (
    student_id      TEXT PRIMARY KEY,
    first_name      TEXT NOT NULL,
    last_name       TEXT NOT NULL,
    email           TEXT NOT NULL,
    major           TEXT NOT NULL,
    standing        TEXT NOT NULL,
    gpa             REAL NOT NULL,
    credits_earned  INTEGER NOT NULL,
    advisor_notes   TEXT
);

CREATE TABLE courses (
    course_id   TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    credits     INTEGER NOT NULL,
    dept        TEXT NOT NULL,
    level       INTEGER NOT NULL,
    prereqs     TEXT,
    description TEXT
);

CREATE TABLE student_courses (
    student_id  TEXT NOT NULL,
    course_id   TEXT NOT NULL,
    status      TEXT NOT NULL DEFAULT 'completed',
    grade       TEXT,
    PRIMARY KEY (student_id, course_id),
    FOREIGN KEY (student_id) REFERENCES students(student_id),
    FOREIGN KEY (course_id)  REFERENCES courses(course_id)
);
""")

for s in students:
    cur.execute("""
        INSERT INTO students VALUES (?,?,?,?,?,?,?,?,?)
    """, (s["student_id"], s["first_name"], s["last_name"], s["email"],
          s["major"], s["standing"], s["gpa"], s["credits_earned"], s["advisor_notes"]))
    for cid in s["completed_courses"]:
        grade = random.choice(["A","A","A-","B+","B","B-","C+","C"])
        cur.execute("INSERT INTO student_courses VALUES (?,?,?,?)",
                    (s["student_id"], cid, "completed", grade))

for c in COURSES:
    cur.execute("INSERT INTO courses VALUES (?,?,?,?,?,?,?)",
                (c["id"], c["name"], c["credits"], c["dept"], c["level"],
                 json.dumps(c["prereqs"]), c["description"]))

conn.commit()
conn.close()

print("✓ Generated 20 student records")
print("✓ Generated 35 courses")
print(f"✓ Saved courses.json, degree_requirements.json, students.json")
print(f"✓ Saved 4 policy documents to data/policy_docs/")
print(f"✓ Built SQLite database at data/advisorx.db")
print("\nData generation complete.")
