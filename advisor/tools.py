"""Deterministic academic tools over the read-only SQLite knowledge base.

Every tool returns {"result": ..., "sources": [citation, ...]}; the graph numbers sources as [S#]
so the model can only cite evidence a tool actually returned.
"""
import datetime as dt
import difflib
import json
import re
import sqlite3
from functools import lru_cache

from . import config

BATCHES = (2022, 2023, 2024, 2025, 2026)
PROGRESSION_RULE = "Student Handbook §12.1 Table 3 (Progression Criteria), p.34"
PREREQ_RULE = "Student Handbook §2.14 Course Pre-Requisites, p.18"
REREGISTER_RULE = "Student Handbook §2.9 (no re-registration for passed courses), p.17"
FAIL_RULE = "Student Handbook §8.13.3 'F' and 'FA' Grades, pp.25-26"
REGISTRATION_RULE = "Student Handbook §2.10 (conditions to register), p.17"
CALENDAR_RULE = "Student Handbook §1.2–1.3 Academic Calendar, p.15"
MAX_ROWS = 60


@lru_cache(maxsize=1)
def _con():
    con = sqlite3.connect(f"file:{config.DB_PATH.as_posix()}?mode=ro", uri=True, check_same_thread=False)
    con.row_factory = sqlite3.Row
    return con


def q(sql, *args):
    return [dict(r) for r in _con().execute(sql, args).fetchall()]


def today():
    return dt.date.today()


def cell_source(r):
    return f"Semester_Spread_Structures_Sept_2026.xlsx › {r['sheet']} › {r['cells']}"


def minor_source(r):
    return f"MinorCoursesforBTech_Students.xlsx › {r['sheet']} › {r['cells']}"


def ltp(r):
    return f"{r['L']}-{r['T']}-{r['P']}" if None not in (r["L"], r["T"], r["P"]) else None


def norm_code(s):
    return re.sub(r"\s+", "", s or "").upper()


def resolve_course(text):
    """Course code or (fuzzy) title -> list of codes. Titles without a code (TBA) resolve to themselves."""
    code = norm_code(text)
    if re.fullmatch(r"[A-Z]{3,4}\d{3}", code):
        return [code]
    key = re.sub(r"[^a-z0-9 ]+", " ", (text or "").lower()).strip()
    titles = q("SELECT DISTINCT code, title FROM options WHERE title IS NOT NULL UNION "
               "SELECT DISTINCT code, title FROM minors WHERE title IS NOT NULL UNION "
               "SELECT code, canonical_title FROM courses")
    exact = {t["code"] or t["title"] for t in titles if re.sub(r"[^a-z0-9 ]+", " ", t["title"].lower()).strip() == key}
    if exact:
        return sorted(exact)
    contains = {t["code"] or t["title"] for t in titles if key and key in t["title"].lower()}
    if contains:
        return sorted(contains)
    close = difflib.get_close_matches(key, [t["title"].lower() for t in titles], n=3, cutoff=0.75)
    return sorted({t["code"] or t["title"] for t in titles if t["title"].lower() in close})


ALIASES = {"ml": "machine learning", "dl": "deep learning", "nlp": "natural language processing", "iot": "internet of things",
           "oop": "object oriented programming", "oops": "object oriented programming", "eda": "exploratory data analysis",
           "dbms": "databases management", "os": "operating systems", "ai": "artificial intelligence", "dsa": "data structures",
           "toc": "theoretical computer science", "mlops": "mlops model deployment", "coa": "microprocessors and computer architecture"}


def _key(s):
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", (s or "").lower())).strip()


@lru_cache(maxsize=1)
def _catalog():
    rows = q("SELECT code, title FROM options WHERE code_status='ok' UNION SELECT code, canonical_title FROM courses "
             "UNION SELECT code, title FROM minors WHERE code IS NOT NULL")
    cat = {}
    for r in rows:
        cat.setdefault(_key(r["title"]), set()).add(r["code"])
    return cat


def course_mentions(text):
    """Deterministically resolve course titles / codes / common abbreviations in a student's message -> {mention: [codes]}."""
    t = f" {_key(text)} "
    for short, full in ALIASES.items():
        t = t.replace(f" {short} ", f" {_key(full)} ")
    found = {}
    for title in sorted(_catalog(), key=len, reverse=True):  # longest first: "advanced topics in deep learning" before "deep learning"
        if len(title) >= 4 and f" {title} " in t and not any(title in f for f in found):
            found[title] = sorted(_catalog()[title])
    for c in re.findall(r"\b([A-Za-z]{3,4})\s?(\d{3})\b", text):
        found[f"{c[0]}{c[1]}".upper()] = [f"{c[0]}{c[1]}".upper()]
    return found


def issues_for(codes=(), batches=(), severities=("warning",), categories=None):
    rows = q("SELECT * FROM issues")
    out = []
    for i in rows:
        if i["severity"] not in severities and not (categories and i["category"] in categories):
            continue
        if codes and i["code"] not in codes:
            continue
        if batches and i["batch"] is not None and i["batch"] not in batches:
            continue
        out.append({"issue": i["title"], "detail": i["detail"], "category": i["category"], "severity": i["severity"]})
    return out


def _term(batch, semester):
    r = q("SELECT * FROM terms WHERE batch=? AND semester=?", batch, semester)
    return r[0]["label"] if r else None


# ------------------------------------------------------------------ calendar
def academic_calendar(batch: int | None = None, semester: int | None = None, month: int | None = None, year: int | None = None):
    """Map batch/semester <-> calendar months. With month+year: which semester each batch is in. Defaults to today."""
    sources = [CALENDAR_RULE, "Derived: batch year = year of admission (Semester 1 starts Jul/Aug of that year)"]
    if batch and semester:
        return {"result": {"batch": batch, "semester": semester, "term": _term(batch, semester)}, "sources": sources}
    d = today()
    month, year = month or (d.month if not batch else None), year or (d.year if not batch else None)
    if batch and not month:
        return {"result": {"batch": batch, "semesters": [r["label"] for r in q("SELECT label FROM terms WHERE batch=? ORDER BY semester", batch)],
                           "current": _position(batch, d.month, d.year)}, "sources": sources}
    batches = [batch] if batch else list(BATCHES)
    return {"result": {"as_of": f"{year}-{month:02d}", "positions": {b: _position(b, month, year) for b in batches}}, "sources": sources}


def _position(batch, month, year):
    if month in (8, 9, 10, 11, 12) or month == 7:
        sem = 2 * (year - batch) + 1
        term = "Odd Semester (Jul/Aug–Dec)" + (" – July may still be the Summer Term (Jun–Jul)" if month == 7 else "")
    elif month in (1, 2, 3, 4, 5):
        sem = 2 * (year - 1 - batch) + 2
        term = "Even Semester (Jan–Apr/May)"
    else:  # June
        sem = 2 * (year - 1 - batch) + 2.5
        term = "Summer break / optional Summer Term (Jun–Jul)"
    if sem < 1:
        return f"not yet admitted (batch {batch} starts Jul/Aug {batch})"
    if sem > 8:
        return f"beyond semester 8 – batch {batch} completes the normal 4-year programme in Apr/May {batch + 4}"
    if sem != int(sem):
        return f"{term}, after semester {int(sem - 0.5)}"
    return f"semester {int(sem)} – {term}"


# ------------------------------------------------------------------ courses
def lookup_courses(batch: int | None = None, semester: int | None = None, course: str | None = None,
                   basket: str | None = None, min_credits: float | None = None, max_credits: float | None = None,
                   lecture: int | None = None, tutorial: int | None = None, practical: int | None = None,
                   month: int | None = None, year: int | None = None, include_placeholders: bool = True):
    """Filter the programme course plan. Any combination of filters; month+year resolve to the batch's semester."""
    where, args = ["1=1"], []
    if batch:
        where.append("o.batch=?"); args.append(batch)
    if month and year and batch:
        pos = _position(batch, month, year)
        m = re.match(r"semester (\d)", pos)
        if not m:
            return {"result": {"note": f"Batch {batch} in {year}-{month:02d}: {pos}"}, "sources": [CALENDAR_RULE]}
        semester = int(m.group(1))
    if semester:
        where.append("o.semester=?"); args.append(semester)
    if course:
        codes = resolve_course(course)
        if not codes:
            return {"result": {"note": f"No course matching '{course}' in the curriculum data."}, "sources": []}
        where.append(f"(o.code IN ({','.join('?' * len(codes))}) OR o.title IN ({','.join('?' * len(codes))}))"); args += codes * 2
    if basket:
        b = basket.upper().strip()
        where.append("(o.basket_code=? OR lower(o.basket_name) LIKE ?)"); args += [b, f"%{basket.lower()}%"]
    for col, val in (("o.L", lecture), ("o.T", tutorial), ("o.P", practical)):
        if val is not None:
            where.append(f"{col}=?"); args.append(val)
    if min_credits is not None:
        where.append("o.C>=?"); args.append(min_credits)
    if max_credits is not None:
        where.append("o.C<=?"); args.append(max_credits)
    if not include_placeholders:
        where.append("o.code_status!='slot'")
    rows = q(f"""SELECT o.*, t.months, t.calendar_year FROM options o JOIN terms t ON t.batch=o.batch AND t.semester=o.semester
                 WHERE {' AND '.join(where)} ORDER BY o.batch, o.semester, o.basket_code""", *args)
    sources, out = [], []
    for r in rows[:MAX_ROWS]:
        sources.append(cell_source(r))
        out.append({"batch": r["batch"], "semester": r["semester"], "when": f"{r['months']} {r['calendar_year']}",
                    "code": r["code"] or f"({r['code_status']})", "title": r["title"], "basket": f"{r['basket_code']} {r['basket_name']}",
                    "slot": r["slot"], "L-T-P": ltp(r), "credits": r["C"], "prerequisites": r["prereq_text"], "src": len(sources)})
    result = {"count": len(rows), "rows": out}
    if len(rows) > MAX_ROWS:
        result["truncated"] = f"showing {MAX_ROWS} of {len(rows)}; add filters (batch/semester) to narrow"
    if batch and semester and not course:
        slots = q("SELECT C FROM offerings WHERE batch=? AND semester=?", batch, semester)
        result["semester_total_credits"] = sum(s["C"] or 0 for s in slots)
    codes = {r["code"] for r in rows if r["code"]}
    result["data_notes"] = issues_for(codes, [batch] if batch else (), categories=("batch_variant", "ltp_in_title", "course_code_missing"))[:12]
    return {"result": result, "sources": sources}


def course_details(course: str):
    """Everything about one course across all batches (+ minors): per-batch semester/L-T-P/credits/prereqs, variants, dependants."""
    codes = resolve_course(course)
    if not codes:
        return {"result": {"note": f"No course matching '{course}' in the curriculum data."}, "sources": []}
    out, sources = [], []
    for code in codes[:3]:
        rows = q("SELECT o.*, t.months, t.calendar_year FROM options o JOIN terms t ON t.batch=o.batch AND t.semester=o.semester "
                 "WHERE o.code=? OR (o.code IS NULL AND o.title=?) OR (o.code_status='placeholder' AND o.title=?) ORDER BY o.batch", code, code, code)
        mins = q("SELECT * FROM minors WHERE code=? OR title=? ORDER BY batch", code, code)
        info = q("SELECT * FROM courses WHERE code=?", code)
        per_batch = []
        for r in rows:
            sources.append(cell_source(r))
            per_batch.append({"batch": r["batch"], "semester": r["semester"], "when": f"{r['months']} {r['calendar_year']}",
                              "title": r["title"], "basket": f"{r['basket_code']} {r['basket_name']}", "L-T-P": ltp(r),
                              "credits": r["C"], "prerequisites": r["prereq_text"], "src": len(sources)})
        minor_rows = []
        for m in mins:
            sources.append(minor_source(m))
            minor_rows.append({"minor": m["minor"], "batch": m["batch"], "semester": m["semester"], "title": m["title"],
                               "L-T-P": ltp(m), "credits": m["C"], "prerequisites": m["prereq_text"], "src": len(sources)})
        needed_by = q("SELECT DISTINCT batch, code, title FROM options WHERE prereq_groups LIKE ? ORDER BY batch", f'%"{code}"%')
        out.append({"code": code, "canonical_title": info[0]["canonical_title"] if info else (rows[0]["title"] if rows else code),
                    "not_offered_to_batches": [b for b in BATCHES if b not in {r["batch"] for r in rows}] if rows else None,
                    "programme_plan": per_batch, "minor_listings": minor_rows,
                    "is_prerequisite_for": [f"{n['code']} {n['title']} (batch {n['batch']})" for n in needed_by][:20],
                    "data_notes": issues_for([code], severities=("warning", "info"))[:15]})
    return {"result": out if len(out) > 1 else out[0], "sources": sources}


def prerequisite_chain(course: str, batch: int):
    """Full prerequisite tree for a course within one batch's plan (semester of each prerequisite, missing/unknown flags)."""
    codes = resolve_course(course)
    if not codes:
        return {"result": {"note": f"No course matching '{course}'."}, "sources": []}
    plan = {r["code"]: r for r in q("SELECT * FROM options WHERE batch=? AND code IS NOT NULL", batch)}
    known = {r["code"] for r in q("SELECT code FROM courses")}
    sources = []

    def node(code, depth, seen):
        r = plan.get(code)
        if not r:
            return {"code": code, "status": "not in this batch's plan" if code in known else "unknown code (not in any curriculum data)"}
        sources.append(cell_source(r))
        n = {"code": code, "title": r["title"], "semester": r["semester"], "src": len(sources), "prerequisites_text": r["prereq_text"]}
        groups = json.loads(r["prereq_groups"] or "[]")
        if groups and depth < 6 and code not in seen:
            n["requires"] = [{"any_of": [node(g, depth + 1, seen | {code}) if re.fullmatch(r"[A-Z]{3,4}\d{3}", g) else {"text": g} for g in grp]}
                             if len(grp) > 1 else (node(grp[0], depth + 1, seen | {code}) if re.fullmatch(r"[A-Z]{3,4}\d{3}", grp[0]) else {"text": grp[0]})
                             for grp in groups]
        return n
    return {"result": {"batch": batch, "tree": node(codes[0], 0, frozenset()), "data_notes": issues_for([codes[0]], [batch])},
            "sources": sources}


def semester_plan(batch: int, semester: int):
    """All course slots of one batch-semester with totals and calendar months."""
    offs = q("SELECT * FROM offerings WHERE batch=? AND semester=? ORDER BY basket_code, id", batch, semester)
    if not offs:
        return {"result": {"note": f"No plan for batch {batch} semester {semester} (batches 2022-2026, semesters 1-8)."}, "sources": []}
    sources, rows = [], []
    for off in offs:
        if off["n_options"] == 0:
            continue
        opts = q("SELECT * FROM options WHERE offering_id=? ORDER BY option_no", off["id"])
        sources.append(cell_source(off))
        rows.append({"basket": f"{off['basket_code']} {off['basket_name']}", "slot": off["slot"],
                     "options": [{"code": o["code"] or f"({o['code_status']})", "title": o["title"], "L-T-P": ltp(o),
                                  "prerequisites": o["prereq_text"]} for o in opts],
                     "credits": off["C"], "choose_one": len(opts) > 1, "src": len(sources)})
    return {"result": {"batch": batch, "semester": semester, "term": _term(batch, semester),
                       "total_credits": sum(o["C"] or 0 for o in offs), "slots": rows,
                       "data_notes": issues_for(batches=[batch], categories=("prereq_same_or_later_semester",))},
            "sources": sources + [CALENDAR_RULE]}


def credit_structure(batch: int):
    """Basket-wise credit requirements (University Core, Foundation, ... ) for a batch, with source disagreements."""
    rows = q("SELECT * FROM baskets WHERE batch=? ORDER BY basket_code", batch)
    if not rows:
        return {"result": {"note": f"No credit structure for batch {batch}."}, "sources": []}
    sources, out = [], []
    for r in rows:
        sources.append(r["struct_cell"] or r["semspread_cell"])
        vals = {r["fixed_min_semspread"], r["credits_struct"], r["credits_computed"]} - {None}
        out.append({"basket": f"{r['basket_code']} {r['basket_name']}", "credits": r["credits_struct"] if r["credits_struct"] is not None else r["credits_computed"],
                    "sources_disagree": None if len(vals) == 1 else {"Struct sheet": r["credits_struct"], "sum of course credits": r["credits_computed"],
                                                                     "semester-spread Fixed Min": r["fixed_min_semspread"]},
                    "src": len(sources)})
    return {"result": {"batch": batch, "total_credits_for_degree": sum(o["credits"] for o in out), "baskets": out}, "sources": sources}


def minor_info(minor: str | None = None, batch: int | None = None):
    """Minor programmes: courses, semesters, L-T-P, credits, prerequisites and TBA credits per batch."""
    names = [r["minor"] for r in q("SELECT DISTINCT minor FROM minor_batches")]
    if minor:
        match = [n for n in names if minor.lower().replace(" minor", "").strip() in n.lower()]
        if not match:
            return {"result": {"note": f"No minor named '{minor}'. Minors in the data: {', '.join(names)}"}, "sources": ["MinorCoursesforBTech_Students.xlsx"]}
        names = match
    where, args = [f"minor IN ({','.join('?' * len(names))})"], list(names)
    if batch:
        where.append("batch=?"); args.append(batch)
    summ = q(f"SELECT * FROM minor_batches WHERE {' AND '.join(where)} ORDER BY minor, batch", *args)
    if not minor:  # overview only
        return {"result": {"minors": [{"minor": s["minor"], "batch": s["batch"], "status": s["status"], "total_credits": s["total_credits"],
                                       "tba_credits": s["tba_credits"]} for s in summ]},
                "sources": ["MinorCoursesforBTech_Students.xlsx (all sheets)"]}
    rows = q(f"SELECT * FROM minors WHERE {' AND '.join(where)} ORDER BY minor, batch, semester, option_no", *args)
    sources, courses = [], []
    for r in rows[:MAX_ROWS]:
        sources.append(minor_source(r))
        courses.append({"minor": r["minor"], "batch": r["batch"], "semester": r["semester"], "code": r["code"] or f"({r['code_status']})",
                        "title": r["title"], "L-T-P": ltp(r), "credits": r["C"], "prerequisites": r["prereq_text"],
                        "prerequisite_title": r["prereq_title"], "choice": r["n_options"] > 1, "src": len(sources)})
    notes = issues_for(batches=[batch] if batch else (), severities=(),
                       categories=("minor_courses_tba", "minor_not_offered", "prereq_not_in_batch_plan", "prereq_unknown_code"))
    return {"result": {"summary": summ, "courses": courses, "data_notes": [i for i in notes if any(n in i["issue"] for n in names)][:15]},
            "sources": sources}


def list_data_issues(batch: int | None = None, course: str | None = None, category: str | None = None):
    """Known conflicts/inconsistencies in the source documents (policy conflicts, missing prerequisites, basket mismatches...)."""
    codes = resolve_course(course) if course else []
    rows = q("SELECT * FROM issues ORDER BY severity DESC, category")
    out = [{"severity": i["severity"], "category": i["category"], "issue": i["title"], "detail": i["detail"], "sources": json.loads(i["sources"])}
           for i in rows if (not batch or i["batch"] in (None, batch)) and (not codes or i["code"] in codes)
           and (not category or category.lower() in i["category"])]
    sources = []
    for o in out[:40]:
        sources.append("; ".join(o.pop("sources")) or "derived consistency check over the curriculum data")
        o["src"] = len(sources)
    return {"result": {"count": len(out), "issues": out[:40]}, "sources": sources}


# ------------------------------------------------------------------ eligibility
def check_eligibility(course: str, batch: int | None = None, completed: list[str] | None = None, failed: list[str] | None = None,
                      cgpa: float | None = None, target_semester: int | None = None):
    """Deterministic eligibility check: offered to the batch? prerequisites passed? progression CGPA? already passed?
    `course` = the title or code exactly as the student wrote it (resolved here; do not guess codes)."""
    missing, reasons, sources = [], [], []

    def cite(s):
        sources.append(s)
        return len(sources)

    codes = resolve_course(course)
    if not codes:
        return {"result": {"decision": "cannot_determine", "reasons": [f"No course matching '{course}' in the curriculum data."]}, "sources": []}
    code = codes[0]
    if batch is None:
        return {"result": {"decision": "needs_info", "missing_information": ["your batch (admission year, 2022-2026)"],
                           "why": "Semester, prerequisites and credits of this course differ between batches."}, "sources": []}
    rows = q("SELECT * FROM options WHERE batch=? AND (code=? OR title=?)", batch, code, code)
    minor_rows = q("SELECT * FROM minors WHERE batch=? AND code=?", batch, code)
    if not rows and not minor_rows:
        other = q("SELECT DISTINCT batch, semester FROM options WHERE code=?", code)
        return {"result": {"decision": "not_offered", "course": code,
                           "reasons": [f"{code} is not in the batch {batch} semester plan or minor lists."
                                       + (f" It appears for other batches: {', '.join(f'{o['batch']} (S{o['semester']})' for o in other)}." if other else "")]},
                "sources": [f"Semester spread / minor workbook for batch {batch}"]}
    r = rows[0] if rows else minor_rows[0]
    src = cite(cell_source(r) if rows else minor_source(r))
    sem = r["semester"]
    reasons.append(f"{code} {r['title']} is scheduled in semester {sem} of the batch {batch} plan ({_term(batch, sem)}), {r['C']} credits [src {src}].")
    completed = {norm_code(c) for c in completed} if completed is not None else None
    failed = {norm_code(c) for c in failed or []}
    blocked = False
    if completed is not None and code in completed:
        reasons.append(f"You have already passed {code}; re-registration for passed courses is not permitted except under the grade-improvement provisions [src {cite(REREGISTER_RULE)}].")
        return {"result": {"decision": "not_eligible", "course": code, "reasons": reasons}, "sources": sources}
    if code in failed:
        reasons.append(f"You previously failed {code} (F/FA); you must re-register for it when it is next offered (or use the make-up "
                       f"exam for an F) [src {cite(FAIL_RULE)}].")
    groups = json.loads(r["prereq_groups"] or "[]")
    known = {x["code"] for x in q("SELECT code FROM courses")}
    for grp in groups:
        cs = [g for g in grp if re.fullmatch(r"[A-Z]{3,4}\d{3}", g)]
        label = " or ".join(grp)
        unknown = [g for g in cs if g not in known]
        if unknown and len(unknown) == len(cs):
            reasons.append(f"Prerequisite {label} does not exist anywhere in the curriculum data, so it cannot be verified (source inconsistency) [src {src}].")
            blocked = blocked or "unverifiable"
            continue
        if not cs:
            reasons.append(f"Prerequisite '{label}' is not a course code and cannot be checked automatically [src {src}].")
            blocked = blocked or "unverifiable"
            continue
        if completed is None:
            missing.append(f"whether you have passed {label}")
            continue
        if any(g in completed for g in cs):
            reasons.append(f"Prerequisite {label}: passed.")
        elif any(g in failed for g in cs):
            reasons.append(f"Prerequisite {label}: failed. You must clear it (make-up exam / re-registration) before taking {code} [src {cite(FAIL_RULE)}].")
            blocked = True
        else:
            reasons.append(f"Prerequisite {label}: not completed.")
            blocked = True
    if groups:
        reasons.append(f"Prerequisites must be satisfied to register [src {cite(PREREQ_RULE)}].")
    else:
        reasons.append(f"No prerequisites listed ({r['prereq_text'] or 'blank'}) [src {src}].")
    target = target_semester or sem
    if target in (3, 5, 7):
        need = 4.0 if target == 3 else 5.0
        if cgpa is None:
            missing.append(f"your current CGPA (minimum {need:.2f} needed to progress into semester {target})")
        elif cgpa < need:
            reasons.append(f"CGPA {cgpa:.2f} is below the {need:.2f} required to progress into semester {target} [src {cite(PROGRESSION_RULE)}].")
            blocked = True
        else:
            reasons.append(f"CGPA {cgpa:.2f} meets the {need:.2f} progression requirement for semester {target} [src {cite(PROGRESSION_RULE)}].")
    if target_semester and target_semester != sem:
        reasons.append(f"Note: the batch plan schedules {code} in semester {sem}, not semester {target_semester}; courses are normally taken in their planned semester.")
    reasons.append(f"Registration also requires fees/dues cleared and no debarment [src {cite(REGISTRATION_RULE)}].")
    decision = ("not_eligible" if blocked is True else "needs_info" if missing else
                "cannot_determine" if blocked == "unverifiable" else "eligible")
    res = {"decision": decision, "course": code, "reasons": reasons, "data_notes": issues_for([code], [batch])}
    if missing:
        res["missing_information"] = missing
    return {"result": res, "sources": sources}


# ------------------------------------------------------------------ SQL + retrieval
def run_sql(sql: str):
    """Read-only SELECT over the knowledge base for aggregates not covered by other tools."""
    s = sql.strip().rstrip(";")
    if not re.match(r"(?is)^\s*(select|with)\b", s) or re.search(r"(?i)\b(insert|update|delete|drop|alter|attach|pragma|create|replace)\b", s):
        return {"result": {"error": "Only a single read-only SELECT statement is allowed."}, "sources": []}
    try:
        rows = q(f"SELECT * FROM ({s}) LIMIT {MAX_ROWS}")
    except sqlite3.Error as e:
        return {"result": {"error": str(e)}, "sources": []}
    return {"result": {"rows": rows, "row_count": len(rows)}, "sources": [f"SQL over knowledge base (derived from the Excel workbooks): {s[:200]}"]}


def search_documents(query: str, scope: str = "all", k: int = 6):
    """Hybrid (BM25 + dense, RRF) search over Handbook/SOP sections and curriculum cards."""
    from .retrieval import search
    hits = search(query, scope=scope, k=k)
    sources, out = [], []
    for h in hits:
        sources.append({
            "citation": h["citation"],
            "source_file": h.get("meta", {}).get("source_file"),
            "sheet": h.get("meta", {}).get("sheet"),
            "locator": h.get("meta", {}).get("locator"),
            "pdf_page": h.get("meta", {}).get("pdf_pages", [None])[0] if h.get("meta", {}).get("pdf_pages") else None,
            "exact_text": h.get("meta", {}).get("exact_text") or h["text"][:1400],
        })
        out.append({"src": len(sources), "citation": h["citation"], "text": h["text"][:1400]})
    result = {"passages": out}
    if not hits or hits[0]["weak"]:
        result["warning"] = "Weak or no match: the provided documents may not contain this information."
    return {"result": result, "sources": sources}


SCHEMA_HINT = """Tables (SQLite): options(batch, semester, basket_code, basket_name, slot, code, code_status, title, L, T, P, C, prereq_text,
prereq_groups, sheet, cells) — one row per course option in a batch plan; offerings(id, batch, semester, basket_code, C, n_options, ...)
— one row per plan slot (sum C here for credit totals); minors(minor, batch, code, title, L, T, P, C, semester, prereq_text, is_placeholder);
minor_batches(minor, batch, status, total_credits, tba_credits); baskets(batch, basket_code, basket_name, credits_struct,
credits_computed, fixed_min_semspread); terms(batch, semester, term, months, calendar_year, label); courses(code, canonical_title, kind);
issues(severity, category, batch, code, title, detail); structure_courses(batch, section, title, credits, linked_code)."""
