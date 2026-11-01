"""Build the knowledge base from the four source files.

    python -m advisor.ingest.build            # full build (embeddings only if Azure is configured)

Outputs in knowledge_base/: advisor.sqlite, curated/handbook.md, chunks.jsonl, embeddings.npy,
exports/*.json + academic_advisor_normalized.xlsx, sources.json, VALIDATION_REPORT.md.
Exits non-zero if any validation invariant fails.
"""
import difflib
import hashlib
import json
import re
import sqlite3
import sys
from collections import Counter, defaultdict

from .. import config
from . import excel, pdf
from .excel import text

PROGRAMME = "B.Tech (DS)"
BASKET_RULES = [  # order matters
    (r"univ", "B1", "University Core"), (r"foundation", "B2", "Foundation"),
    (r"program\s*core", "B3", "Program Core"), (r"internship|capstone", "B7", "Internship/Capstone Project"),
    (r"sp\s*track|speciali", "B5", "Specialization Tracks"), (r"elective", "B5", "Electives"),
    (r"hon", "B4", "Program Honors"), (r"open|minor", "B6", "Open/Minor"),
]
PLACEHOLDER = re.compile(r"^(TBA|TBD|NEW|DON.?T\s*KNOW)$", re.I)
CODE = re.compile(r"^[A-Z]{3,4}\d{3}$")
CODE_IN_TEXT = re.compile(r"[A-Z]{3,4}\s?\d{3}")
TYPO_FIXES = {"Fiancial": "Financial", "Finacial": "Financial", "Anaylsis": "Analysis", "Visuatlization": "Visualization",
              "Desciplinary": "Disciplinary", "Interdesciplinary": "Interdisciplinary", "Summe Training": "Summer Training",
              "Instituation": "Institution", "Behavourial": "Behavioural", "(R )": "(R)"}
HB_SECTIONS = [(9, 12, "Section I – Preamble, Short Title, Commencement and Application"),
               (13, 20, "Section II – Admission Rules, Fee Policy and Scholarship Policy"),
               (21, 48, "Section III – Academic Regulations"), (49, 54, "Section IV – Code of Conduct"),
               (55, 62, "Section V – Guidelines on Banning Ragging and Anti-Ragging Measures"),
               (63, 68, "Section VI – Guidelines on Prevention of Sexual Harassment"),
               (69, 72, "Section VII – Ombudsperson"), (73, 76, "Section VIII – Dress Code"),
               (77, 80, "Section IX – Malpractice in Examination Hall"),
               (81, 84, "Section X – Knowledge Resource Centre (KRC)"), (85, 90, "Section XI – Sports Policy"),
               (91, 92, "Section XII – Important Contact Numbers and Campus Address")]


# ---------------------------------------------------------------- helpers
def basket_of(label):
    low = (label or "").lower()
    for pat, code, name in BASKET_RULES:
        if re.search(pat, low):
            return code, name
    raise ValueError(f"Unknown basket label: {label!r}")


def norm_code(raw):
    s = re.sub(r"\s+", "", str(raw)).upper() if raw is not None else ""
    return s or None


def fix_typos(s):
    for a, b in TYPO_FIXES.items():
        s = s.replace(a, b)
    return s


def title_key(s):
    return re.sub(r"[^a-z0-9]+", " ", fix_typos(s or "").lower()).strip()


def parse_prereq(raw):
    """-> (text, groups). ',' separates required groups; '/' inside a group means alternatives."""
    t = text(raw)
    if t is None:
        return None, []
    if re.fullmatch(r"(?i)nil|none|-", t):
        return "NIL", []
    groups = []
    for part in t.split(","):
        alts = [a.replace(" ", "") for a in CODE_IN_TEXT.findall(part.upper())]
        if alts:
            groups.append(alts)
        elif part.strip(" ()"):
            groups.append([part.strip(" ()")])  # non-code token: FAMA, TBA, TBD
    return t, groups


def ltp_credits(L, T, P):
    """Handbook clause 3.1: 1 credit per L/T hour, 1 credit per 2 P hours."""
    return L + T + P / 2


def cite_cells(sheet, cells, file="Semester_Spread_Structures_Sept_2026.xlsx"):
    return f"{file} › {sheet} › {cells}"


# ---------------------------------------------------------------- offerings
def split_options(code_raw, title_raw, prereq_raw, L, T, P, C):
    """One grid cell-block can hold alternative courses ('MATH401/COMP401', 'SPT-Course #1-A LTP: 3-0-2/B LTP: 2-0-4')."""
    codes = norm_code(code_raw).split("/") if norm_code(code_raw) else [None]
    title = text(title_raw) or ""
    slot, body = None, title
    m = re.match(r"SPT-Course\s*#\s*(\d+)\s*[-:]?\s*(.*)", title)
    if m:
        slot, body = f"Specialization course #{m.group(1)}", m.group(2).strip()
        if body.startswith("(") and body.endswith(")"):
            body = body[1:-1]
    titles = [x.strip() for x in body.split("/")] if (len(codes) > 1 or slot) else [body]
    parsed = []
    for t in titles:
        lm = re.search(r"LTP:\s*(\d+)\s*-\s*(\d+)\s*-\s*(\d+)", t)
        parsed.append((re.sub(r"\s*LTP:\s*\d+\s*-\s*\d+\s*-\s*\d+", "", t).strip(),
                       tuple(int(x) for x in lm.groups()) if lm else None))
    seen, uniq = set(), []
    for t, ltp in parsed:  # 'Natural Language Processing / Natural Language Processing' -> one option
        if title_key(t) not in seen:
            seen.add(title_key(t))
            uniq.append((t, ltp))
    n = max(len(codes), len(uniq))
    codes += [None] * (n - len(codes))
    uniq += [(None, None)] * (n - len(uniq))
    pre_parts = str(prereq_raw).split("/") if (n > 1 and prereq_raw and len(str(prereq_raw).split("/")) == n) else [prereq_raw] * n
    out = []
    for i, (code, (t, ltp)) in enumerate(zip(codes, uniq), 1):
        if code and PLACEHOLDER.match(code):
            status, code = "placeholder", code.upper()
        elif code:
            status = "ok"
        else:
            status = "slot" if n == 1 else "not_given"
        l, tt, p = ltp if ltp else (L, T, P)
        ptext, groups = parse_prereq(pre_parts[i - 1])
        out.append({"option_no": i, "code": code, "code_status": status, "title": t, "L": l, "T": tt, "P": p, "C": C,
                    "prereq_text": ptext, "prereq_groups": groups, "ltp_from_title": bool(ltp)})
    return slot, out


def build_offerings(units):
    offerings, options = [], []
    for u in units:
        c = u["content"]
        if c["table"] != "offering":
            continue
        bcode, bname = basket_of(c["basket_label"])
        if bcode == "B5" and "elective" in c["basket_label"].lower():
            bname = "Electives"
        title_clean = text(c["title"])
        orphan = title_clean is None and all(c[k] is None for k in "LTPC")  # a bare code, e.g. 2022 AI19 'COMP401'
        slot, opts = (None, []) if orphan else split_options(c["code"], c["title"], c["prereq"], c["L"], c["T"], c["P"], c["C"])
        placeholder = len(opts) == 1 and opts[0]["code_status"] == "slot"  # "Minor/Open (#1 Course)", "Elective 1"
        oid = len(offerings) + 1
        offerings.append({"id": oid, "batch": c["batch"], "semester": c["semester"], "basket_code": bcode, "basket_name": bname,
                          "slot": "orphan code cell" if orphan else slot or (title_clean if placeholder else None),
                          "title_raw": c["title"], "code_raw": c["code"],
                          "prereq_raw": c["prereq"], "L": c["L"], "T": c["T"], "P": c["P"], "C": c["C"],
                          "is_placeholder": int(placeholder), "n_options": len(opts), "sheet": u["source"]["sheet"],
                          "cells": u["source"]["locator"]})
        for o in opts:
            options.append({"offering_id": oid, "batch": c["batch"], "semester": c["semester"], "basket_code": bcode,
                            "basket_name": bname, "slot": offerings[-1]["slot"], **o,
                            "title": o["title"] or title_clean, "sheet": u["source"]["sheet"], "cells": u["source"]["locator"]})
    return offerings, options


# ---------------------------------------------------------------- baskets / structure
def build_baskets(units, options, offerings):
    spread = {}
    for u in units:
        c = u["content"]
        if c["table"] == "basket_spread":
            code, name = basket_of(c["label"])
            spread[(c["batch"], code)] = (c, u["source"])
    struct = {}
    for u in units:
        c = u["content"]
        if c["table"] == "basket_struct" and c["kind"] == "basket":
            struct[(c["batch"], basket_of(c["label"])[0])] = (c, u["source"])
    computed = defaultdict(float)
    for o in offerings:
        computed[(o["batch"], o["basket_code"])] += o["C"] or 0
    rows = []
    for key in sorted(set(spread) | set(struct)):
        sp, st = spread.get(key), struct.get(key)
        name = basket_of((sp or st)[0]["label"])[1]
        if key[1] == "B5" and "elective" in (sp or st)[0]["label"].lower():
            name = "Electives"
        rows.append({"batch": key[0], "basket_code": key[1], "basket_name": name,
                     "label_semspread": sp[0]["label"] if sp else None, "basket_id_cell": sp[0]["basket_id"] if sp else None,
                     "fixed_min_semspread": sp[0]["fixed_min"] if sp else None,
                     "credits_struct": st[0]["credits"] if st else None, "credits_computed": computed.get(key, 0),
                     "semspread_cell": cite_cells(sp[1]["sheet"], f"D{sp[0]['row']}") if sp else None,
                     "struct_cell": cite_cells(st[1]["sheet"], st[1]["locator"]) if st else None})
    return rows


def title_score(a, b):
    if a == b:
        return 1.0
    short, long_ = sorted((a, b), key=len)
    if short and short in long_ and len(short) / len(long_) >= 0.5:
        return 0.9
    return difflib.SequenceMatcher(None, a, b).ratio()


def best_match(key, candidates):
    return max(((title_score(key, tk), code, cr) for tk, code, cr in candidates), default=(0.0, None, None))


def build_structure(units, options, offerings):
    rows = []
    by_batch, anywhere = defaultdict(list), []
    for o in options:
        if o["title"]:
            by_batch[o["batch"]].append((title_key(o["title"]), o["code"], o["C"]))
            if o["code_status"] == "ok":
                anywhere.append((title_key(o["title"]), o["code"], o["C"]))
    for off in offerings:
        t = text(off["title_raw"])
        if t and off["n_options"] > 1:
            codes = "/".join(o["code"] or "?" for o in options if o["offering_id"] == off["id"])
            by_batch[off["batch"]].append((title_key(t), codes, off["C"]))
    for u in units:
        c = u["content"]
        if c["table"] != "structure":
            continue
        sec = c["section"]
        low = sec.lower()
        m = re.search(r"track\s*#\s*(\d+)\s*-\s*(.*?)\s*-\s*courses", sec, re.I)
        section = (f"Specialization Track {m.group(1)} – {m.group(2)}" if m else
                   "Electives" if low == "electives" else "Minor/Open" if "minor" in low else basket_of(sec)[1])
        bcode = "B5" if (m or low == "electives") else "B6" if "minor" in low else basket_of(sec)[0]
        key = title_key(c["title"])
        best, scope = best_match(key, by_batch[c["batch"]]), "same_batch"
        if best[0] < 0.8:  # e.g. 2026 Electives pool vs generic "Elective 1-5" slots: borrow the code other batches use
            best, scope = best_match(key, anywhere), "other_batch"
        matched = best[0] >= 0.8
        rows.append({"batch": c["batch"], "basket_code": bcode, "section": section, "section_raw": sec,
                     "number": (c["number"] or "").strip() or None, "title": c["title"], "credits": c["credits"],
                     "matched": int(matched), "link_scope": scope if matched else None,
                     "linked_code": best[1] if matched else None, "link_score": round(best[0], 2),
                     "linked_credits": best[2] if matched else None,
                     "sheet": u["source"]["sheet"], "cells": u["source"]["locator"]})
    return rows


# ---------------------------------------------------------------- minors
def build_minors(units):
    rows, status = [], {}
    for u in units:
        c = u["content"]
        if c["table"] != "minor":
            continue
        code_raw, title = (text(v).replace("�", "'") if text(v) else None for v in (c["code"], c["title"]))
        key = (c["minor"], c["batch"])
        if code_raw and code_raw.lower() == "no students":
            status[key] = "no_students"
            continue
        status.setdefault(key, "offered")
        placeholder = bool(title and PLACEHOLDER.match(title)) and not code_raw
        codes = norm_code(code_raw).split("/") if code_raw else [None]
        titles = [t.strip() for t in title.split("/")] if (title and len(codes) > 1) else [title]
        for i, (code, t) in enumerate(zip(codes, titles), 1):
            if placeholder:
                cstatus, code, t = "tba_credits", None, f"To be announced ({title})"
            elif code and PLACEHOLDER.match(code):
                cstatus, code = ("new_course_code_pending" if code == "NEW" else "code_unknown"), None
            else:
                cstatus = "ok" if code else "code_unknown"
            ptext, groups = parse_prereq(c["prereq"])
            rows.append({"minor": c["minor"], "batch": c["batch"], "option_no": i, "n_options": len(codes), "code": code,
                         "code_raw": code_raw, "code_status": cstatus, "title": t, "L": c["L"], "T": c["T"], "P": c["P"],
                         "C": c["C"], "semester": c["semester"], "prereq_text": ptext, "prereq_groups": groups,
                         "prereq_title": text(c["prereq_title"]), "is_placeholder": int(placeholder),
                         "sheet": u["source"]["sheet"], "cells": u["source"]["locator"]})
    summary = []
    for (minor, batch), st in sorted(status.items()):
        mine = [r for r in rows if r["minor"] == minor and r["batch"] == batch and r["option_no"] == 1]
        total = sum(r["C"] or 0 for r in mine)
        tba = sum(r["C"] or 0 for r in mine if r["is_placeholder"])
        summary.append({"minor": minor, "batch": batch, "status": st, "total_credits": total, "tba_credits": tba,
                        "confirmed_credits": total - tba, "n_courses": sum(1 for r in mine if not r["is_placeholder"])})
    return rows, summary


# ---------------------------------------------------------------- terms (derived month map)
def build_terms():
    rows = []
    for batch in range(2022, 2027):
        for sem in range(1, 9):
            ay = batch + (sem - 1) // 2
            odd = sem % 2 == 1
            rows.append({"batch": batch, "semester": sem, "term": "Odd Semester" if odd else "Even Semester",
                         "academic_year": f"{ay}-{str(ay + 1)[2:]}", "months": "Jul/Aug–Dec" if odd else "Jan–Apr/May",
                         "calendar_year": ay if odd else ay + 1, "start_month": 7 if odd else 1, "end_month": 12 if odd else 5,
                         "label": f"Semester {sem}: {'Jul/Aug–Dec' if odd else 'Jan–Apr/May'} {ay if odd else ay + 1} (AY {ay}-{str(ay + 1)[2:]})"})
            if not odd and sem < 8:
                rows.append({"batch": batch, "semester": sem + 0.5, "term": "Summer Term", "academic_year": f"{ay}-{str(ay + 1)[2:]}",
                             "months": "Jun–Jul", "calendar_year": ay + 1, "start_month": 6, "end_month": 7,
                             "label": f"Summer Term after Semester {sem}: Jun–Jul {ay + 1} (optional)"})
    return rows


# ---------------------------------------------------------------- issues
def build_issues(units, offerings, options, baskets, structure, minors, minor_summary):
    issues = []

    def add(id_, severity, category, title, detail, sources, batch=None, code=None):
        issues.append({"id": id_, "severity": severity, "category": category, "batch": batch, "code": code,
                       "title": title, "detail": detail, "sources": sources})

    real = [o for o in options if o["code_status"] == "ok"]
    known = {o["code"] for o in real} | {m["code"] for m in minors if m["code"]}

    # cross-batch variants of the same course code
    per_code = defaultdict(dict)
    for o in real:
        per_code[o["code"]][o["batch"]] = o
    for code, bmap in sorted(per_code.items()):
        if len(bmap) < 2:
            continue
        fields = {"semester": lambda o: f"S{o['semester']}", "L-T-P/credits": lambda o: f"{o['L']}-{o['T']}-{o['P']} / {o['C']} cr",
                  "prerequisites": lambda o: re.sub(r"\s+", "", (o["prereq_text"] or "NIL").upper()),
                  "basket": lambda o: o["basket_code"], "title": lambda o: title_key(o["title"])}
        for fname, f in fields.items():
            vals = {b: f(o) for b, o in sorted(bmap.items())}
            if len(set(vals.values())) > 1:
                shown = {b: (bmap[b]["title"] if fname == "title" else v) for b, v in vals.items()}
                add(f"variant::{code}::{fname}", "info", "batch_variant", f"{code} {fname} differs across batches",
                    "; ".join(f"batch {b}: {v}" for b, v in shown.items()),
                    [cite_cells(o["sheet"], o["cells"]) for o in bmap.values()], code=code)

    # prerequisite integrity (programme courses)
    by_batch = defaultdict(dict)
    for o in real:
        by_batch[o["batch"]][o["code"]] = o["semester"]
    for o in real:
        for group in o["prereq_groups"]:
            codes = [g for g in group if CODE.match(g)]
            if not codes:
                add(f"prereq_text::{o['batch']}::{o['code']}::{'/'.join(group)}", "info", "prereq_non_code",
                    f"{o['code']} (batch {o['batch']}) lists a non-code prerequisite '{'/'.join(group)}'",
                    "The prerequisite is not a course code, so it cannot be checked automatically.",
                    [cite_cells(o["sheet"], o["cells"])], o["batch"], o["code"])
                continue
            for g in codes:
                if g not in known:
                    add(f"prereq_unknown::{o['batch']}::{o['code']}::{g}", "warning", "prereq_unknown_code",
                        f"{o['code']} {o['title']} (batch {o['batch']}) requires {g}, which does not exist anywhere in the curriculum data",
                        f"Prerequisite text: '{o['prereq_text']}'. No course with code {g} appears in any semester plan or minor list, so this prerequisite cannot be verified (possible typo in the source).",
                        [cite_cells(o["sheet"], o["cells"])], o["batch"], o["code"])
            in_plan = [by_batch[o["batch"]][g] for g in codes if g in by_batch[o["batch"]]]
            if not in_plan and all(g in known for g in codes):
                add(f"prereq_not_in_plan::{o['batch']}::{o['code']}::{'/'.join(codes)}", "warning", "prereq_not_in_batch_plan",
                    f"{o['code']} {o['title']} (batch {o['batch']}) requires {' or '.join(codes)}, which is not in the batch {o['batch']} semester plan",
                    f"Prerequisite text: '{o['prereq_text']}'. {' / '.join(codes)} exists in other batches or minors but is not scheduled for batch {o['batch']}.",
                    [cite_cells(o["sheet"], o["cells"])], o["batch"], o["code"])
            elif in_plan and min(in_plan) >= o["semester"]:
                add(f"prereq_order::{o['batch']}::{o['code']}::{'/'.join(codes)}", "warning", "prereq_same_or_later_semester",
                    f"{o['code']} {o['title']} (batch {o['batch']}, S{o['semester']}) requires {' or '.join(codes)}, scheduled in S{min(in_plan)} of the same plan",
                    "The prerequisite is scheduled in the same or a later semester than the course itself, so it cannot be completed beforehand as planned.",
                    [cite_cells(o["sheet"], o["cells"])], o["batch"], o["code"])

    # minors: prerequisites
    minor_codes, minor_titles, code_titles = defaultdict(set), defaultdict(set), defaultdict(set)
    for m in minors:
        minor_titles[(m["minor"], m["batch"])].add(title_key(m["title"]))
        if m["code"]:
            minor_codes[(m["minor"], m["batch"])].add(m["code"])
    for o in real + [m for m in minors if m["code"]]:
        code_titles[o["code"]].add(title_key(o["title"]))
    for m in minors:
        for group in m["prereq_groups"]:
            codes = [g for g in group if CODE.match(g)]
            ref = f"MinorCoursesforBTech_Students.xlsx › {m['sheet']} › {m['cells']}"
            for g in codes:
                if g not in known:
                    add(f"prereq_unknown::minor::{m['minor']}::{m['batch']}::{m['code'] or m['title']}::{g}", "warning", "prereq_unknown_code",
                        f"{m['minor']} minor (batch {m['batch']}): {m['code'] or m['title']} requires {g}, which does not exist anywhere in the data",
                        f"Prerequisite text: '{m['prereq_text']}'" + (f" ({m['prereq_title']})" if m["prereq_title"] else "") + ".",
                        [ref], m["batch"], m["code"])
            avail = by_batch[m["batch"]].keys() | minor_codes[(m["minor"], m["batch"])]
            same_title = any(code_titles[g] & minor_titles[(m["minor"], m["batch"])] for g in codes)  # e.g. 2022 'Marketing Management' listed with code "DON'T KNOW"
            if codes and all(g in known for g in codes) and not any(g in avail for g in codes) and not same_title:
                add(f"prereq_not_in_plan::minor::{m['minor']}::{m['batch']}::{m['code'] or m['title']}::{'/'.join(codes)}", "warning",
                    "prereq_not_in_batch_plan",
                    f"{m['minor']} minor (batch {m['batch']}): {m['code'] or m['title']} requires {' or '.join(codes)}, which is not offered to batch {m['batch']}",
                    f"Prerequisite text: '{m['prereq_text']}'. It is neither in the batch {m['batch']} semester plan nor in this minor's course list for that batch.",
                    [ref], m["batch"], m["code"])
        if m["L"] is not None and m["T"] is not None and m["P"] is not None and (m["L"] + m["T"] + m["P"]) and m["C"] is not None \
                and ltp_credits(m["L"], m["T"], m["P"]) != m["C"]:
            add(f"credit_norm::minor::{m['minor']}::{m['batch']}::{m['code']}", "info", "credit_norm_mismatch",
                f"{m['code']} {m['title']} ({m['minor']} minor, batch {m['batch']}): L-T-P {m['L']}-{m['T']}-{m['P']} implies {ltp_credits(m['L'], m['T'], m['P']):g} credits but {m['C']} are listed",
                "Handbook clause 3.1: 1 credit per lecture/tutorial hour, 1 credit per 2 practical hours.",
                [f"MinorCoursesforBTech_Students.xlsx › {m['sheet']} › {m['cells']}"], m["batch"], m["code"])

    # programme: credit norm and LTP taken from title text
    for o in real + [o for o in options if o["code_status"] != "ok" and o["title"]]:
        if o["ltp_from_title"]:
            off = next(x for x in offerings if x["id"] == o["offering_id"])
            if (off["L"], off["T"], off["P"]) != (o["L"], o["T"], o["P"]):
                add(f"ltp_title::{o['batch']}::{o['offering_id']}::{o['option_no']}", "info", "ltp_in_title",
                    f"{o['code'] or o['title']} (batch {o['batch']}, S{o['semester']}): course-specific L-T-P {o['L']}-{o['T']}-{o['P']} is written in the title; the row's L-T-P columns say {off['L']}-{off['T']}-{off['P']}",
                    "The alternative courses in this specialization slot have different L-T-P; the value written next to each course title is used.",
                    [cite_cells(o["sheet"], o["cells"])], o["batch"], o["code"])
        if None not in (o["L"], o["T"], o["P"], o["C"]) and (o["L"] + o["T"] + o["P"]) and ltp_credits(o["L"], o["T"], o["P"]) != o["C"]:
            add(f"credit_norm::{o['batch']}::{o['offering_id']}::{o['option_no']}", "info", "credit_norm_mismatch",
                f"{o['code'] or o['title']} (batch {o['batch']}): L-T-P {o['L']}-{o['T']}-{o['P']} implies {ltp_credits(o['L'], o['T'], o['P']):g} credits but {o['C']} are listed",
                "Handbook clause 3.1: 1 credit per lecture/tutorial hour, 1 credit per 2 practical hours.",
                [cite_cells(o["sheet"], o["cells"])], o["batch"], o["code"])

    # placeholders / missing codes
    for o in options:
        if o["code_status"] in ("placeholder", "not_given"):
            add(f"code::{o['batch']}::{o['offering_id']}::{o['option_no']}", "info", "course_code_missing",
                f"'{o['title']}' (batch {o['batch']}, S{o['semester']}) has no confirmed course code ({o['code'] or 'not given in source'})",
                "The source lists this course without a final course code.", [cite_cells(o["sheet"], o["cells"])], o["batch"])
    for m in minors:
        if m["code_status"] in ("code_unknown", "new_course_code_pending"):
            add(f"code::minor::{m['minor']}::{m['batch']}::{m['cells']}", "info", "course_code_missing",
                f"{m['minor']} minor (batch {m['batch']}): '{m['title']}' has no course code (source says '{m['code_raw']}')",
                "The source lists this minor course without a final course code.",
                [f"MinorCoursesforBTech_Students.xlsx › {m['sheet']} › {m['cells']}"], m["batch"])
    for s in minor_summary:
        if s["tba_credits"]:
            add(f"minor_tba::{s['minor']}::{s['batch']}", "info", "minor_courses_tba",
                f"{s['minor']} minor, batch {s['batch']}: {s['tba_credits']:g} of {s['total_credits']:g} credits are not yet assigned to courses (TBA/TBD)",
                f"Confirmed courses cover {s['confirmed_credits']:g} credits; the remaining {s['tba_credits']:g} credits are listed as TBA/TBD/'Don't know' in the source.",
                ["MinorCoursesforBTech_Students.xlsx"], s["batch"])
        if s["status"] == "no_students":
            add(f"minor_none::{s['minor']}::{s['batch']}", "info", "minor_not_offered",
                f"{s['minor']} minor was not run for batch {s['batch']} ('No Students')", "The source marks this batch as 'No Students'.",
                ["MinorCoursesforBTech_Students.xlsx"], s["batch"])

    # basket credit totals
    for b in baskets:
        vals = {"semester-spread Fixed Min": b["fixed_min_semspread"], "Struct sheet": b["credits_struct"],
                "sum of listed course credits": b["credits_computed"]}
        if len({v for v in vals.values() if v is not None}) > 1:
            add(f"basket::{b['batch']}::{b['basket_code']}", "warning", "basket_total_mismatch",
                f"Batch {b['batch']} {b['basket_name']} ({b['basket_code']}) credits disagree between sources",
                "; ".join(f"{k}: {v:g}" for k, v in vals.items() if v is not None) + ". The course-level sum and the Struct sheet agree; the Fixed Min column appears not to have been updated.",
                [s for s in (b["semspread_cell"], b["struct_cell"]) if s], b["batch"])
        if b["label_semspread"] and not b["basket_id_cell"]:
            add(f"basket_id::{b['batch']}::{b['basket_code']}", "info", "sheet_anomaly",
                f"Batch {b['batch']}: basket id label ({b['basket_code']}) missing in column A for '{b['label_semspread']}'",
                "Basket assignment was taken from the column-B label.", [b["semspread_cell"]], b["batch"])

    for off in offerings:
        if off["n_options"] == 0:
            add(f"orphan::{off['batch']}::{off['cells']}", "info", "sheet_anomaly",
                f"Batch {off['batch']} S{off['semester']} cell {off['cells']}: bare code '{text(off['code_raw'])}' with no title, L-T-P or credits",
                "Treated as an overflow of the paired entry in the row above, not as a separate course.",
                [cite_cells(off["sheet"], off["cells"])], off["batch"], norm_code(off["code_raw"]))

    # sheet anomalies found by the adapter
    for u in units:
        c = u["content"]
        if c["table"] == "anomaly":
            add(f"sheet::{u['source']['sheet']}::{u['source']['locator']}", "info", "sheet_anomaly",
                f"{u['source']['sheet']} {u['source']['locator']}: {c['what']}", f"Cell value: {c['value']}",
                [cite_cells(u["source"]["sheet"], u["source"]["locator"])])

    # Struct sheet vs semester plan
    for s in structure:
        ref = [cite_cells(s["sheet"], s["cells"])]
        if s["title"] and not s["matched"] and not re.search(r"courses of|credits each", s["title"], re.I):
            add(f"struct_unlinked::{s['batch']}::{s['cells']}", "info", "struct_unlinked",
                f"Struct_{s['batch']}: '{s['title']}' ({s['section']}) could not be matched to any course in the semester plans",
                "Title matching found no course with a similar title.", ref, s["batch"])
        elif s["link_scope"] == "other_batch":
            add(f"struct_other_batch::{s['batch']}::{s['cells']}", "info", "struct_code_from_other_batch",
                f"Struct_{s['batch']}: '{s['title']}' ({s['section']}) is not scheduled by name in the batch {s['batch']} semester plan; other batches use code {s['linked_code']}",
                "The code is inferred from the same title in other batches' semester plans and is not confirmed for this batch.", ref, s["batch"], s["linked_code"])
        elif s["matched"] and s["linked_credits"] is not None and s["credits"] is not None and s["credits"] != s["linked_credits"]:
            add(f"struct_credit::{s['batch']}::{s['cells']}", "warning", "struct_credit_mismatch",
                f"Batch {s['batch']}: '{s['title']}' has {s['credits']} credits in the Struct sheet but {s['linked_credits']} in the semester plan ({s['linked_code']})",
                "The two sheets of the same workbook disagree.", ref, s["batch"], s["linked_code"])

    # code prefixes that look like typos of a common prefix (e.g. PSCY vs PSYC)
    prefixes = Counter(re.match(r"[A-Z]+", c).group() for c in known)
    for p, n in prefixes.items():
        twins = [q for q in prefixes if q != p and sorted(q) == sorted(p) and prefixes[q] > n]
        if n == 1 and twins:
            code = next(c for c in known if c.startswith(p))
            add(f"code_typo::{code}", "info", "suspicious_code", f"{code} uses prefix '{p}', which appears nowhere else; '{twins[0]}' is the usual prefix",
                "Possible typo in the source course code.", [], code=code)

    for pc in json.loads((config.CURATED / "policy_conflicts.json").read_text(encoding="utf-8")):
        add(pc["id"], pc["severity"], "policy_conflict", pc["title"], pc["detail"], pc["sources"])
    return issues


# ---------------------------------------------------------------- policy documents -> markdown + chunks
def write_handbook_md(pages):
    out = ["# Student Handbook (August 2026) – Vidyashilp University", "",
           "<!-- generated by advisor/ingest/pdf.py from '4. Student Handbook Aug 2026.pdf' with PyMuPDF; "
           "known extraction defects fixed via reviewed patches (Table 1/2/3, SGPA/CGPA formulas, malpractice table, contacts header). "
           "Printed page = PDF page - 8. Front matter (cover, contents) omitted. -->", ""]
    current = None
    for p in pages:
        c = p["content"]
        n = c["pdf_page"]
        sec = next((s for a, b, s in HB_SECTIONS if a <= n <= b), None)
        if sec is None or len(c["text"]) < 150:  # front matter / section divider pages
            continue
        out.append(f"<!-- pdf={n} printed={c['printed_page']} -->")  # anchor first: the section heading belongs to this page
        if sec != current:
            out += [f"## {sec}", ""]
            current = sec
        out += [c["text"], ""]
    md = "\n".join(out)
    (config.CURATED / "handbook.md").write_text(md, encoding="utf-8")
    return md


CLAUSE = re.compile(r"^[-*#\s]*(\d{1,2}\.\d{1,2}(?:\.\d{1,2})?)\.?\s", re.M)  # clause ids start a line


def chunk_markdown(md, doc, doc_label, target=1100):
    """Heading/page-aware chunks. Each chunk keeps its section, nearest heading, pages and clause ids."""
    chunks, buf, pages = [], [], []
    section = heading = None
    page = (None, None)

    def flush():
        body = "\n".join(buf).strip()
        if len(body) > 40:
            clauses = sorted(set(CLAUSE.findall(body)), key=lambda x: [int(y) for y in x.split(".")])
            printed = sorted({p for _, p in pages if p})
            crumb = " › ".join(x for x in (doc_label, section, heading) if x)
            page_txt = (f"p.{printed[0]}" if len(printed) == 1 else f"pp.{printed[0]}-{printed[-1]}") if printed else ""
            cl = f"§{clauses[0]}" + (f"–{clauses[-1]}" if len(clauses) > 1 else "") if clauses else ""
            chunks.append({"id": f"{doc}::{len(chunks) + 1:03d}", "doc_type": doc,
                           "text": f"[{crumb}]\n{body}",
                           "citation": ", ".join(x for x in (doc_label, section.split(" – ")[-1] if section else None, cl, page_txt) if x),
                           "meta": {"section": section, "heading": heading, "clauses": clauses,
                                    "pdf_pages": sorted({p for p, _ in pages}), "printed_pages": printed,
                                    "source_file": "4. Student Handbook Aug 2026.pdf" if doc == "handbook" else "SOP STUDENT 17082026 - Final.pdf",
                                    "exact_text": body}})
        buf.clear()
        pages.clear()
        if page[0]:
            pages.append(page)

    for line in md.splitlines():
        m = re.match(r"<!-- pdf=(\d+) printed=(\w+) -->", line)
        if m:
            page = (int(m.group(1)), int(m.group(2)) if m.group(2).isdigit() else None)
            pages.append(page)
            continue
        if line.startswith("<!--") or line.startswith("# "):
            continue
        if line.startswith("## "):
            flush()
            pages[:] = [page] if page[0] else []  # page anchors always precede section headings
            section, heading = line[3:].strip(), None
            continue
        h = re.match(r"#{3,4} (.*)", line)
        if h:
            if sum(len(x) for x in buf) > 250:
                flush()
            heading = h.group(1).replace("*", "").strip()
        buf.append(line)
        if sum(len(x) for x in buf) > target and not line.startswith("|"):
            flush()
    flush()
    return chunks


# ---------------------------------------------------------------- structured cards (retrieval docs that always name the batch)
def fmt_ltp(o):
    return f"{o['L']}-{o['T']}-{o['P']}" if None not in (o["L"], o["T"], o["P"]) else "not given"


def build_cards(options, offerings, baskets, minors, minor_summary, terms, issues, courses):
    cards = []
    term = {(t["batch"], t["semester"]): t for t in terms}
    notes = defaultdict(list)
    for i in issues:
        if i["code"] and i["severity"] == "warning":
            notes[(i["batch"], i["code"])].append(i["title"])
    for o in options:
        if o["code_status"] == "slot":
            continue
        t = term[(o["batch"], o["semester"])]
        slot = f", {o['slot']}" if o["slot"] else ""
        pre = o["prereq_text"] or "not given"
        note = (" Data note: " + "; ".join(notes[(o["batch"], o["code"])]) + ".") if notes.get((o["batch"], o["code"])) else ""
        cards.append({"id": f"course::{o['batch']}::S{o['semester']}::{o['code'] or 'nocode'}::{o['offering_id']}.{o['option_no']}",
                      "doc_type": "course",
                      "text": f"{o['title']} ({o['code'] or 'course code not assigned'}) – {PROGRAMME} batch {o['batch']}: semester {o['semester']} "
                              f"({t['months']} {t['calendar_year']}), basket {o['basket_code']} {o['basket_name']}{slot}. "
                              f"L-T-P {fmt_ltp(o)}, {o['C']} credits. Prerequisites: {pre}.{note}",
                      "citation": f"Semester spread {o['batch']} › {o['sheet']} › {o['cells']}",
                      "meta": {"batch": o["batch"], "semester": o["semester"], "code": o["code"]}})
    for (batch, sem), group in sorted(_group(offerings, ("batch", "semester")).items()):
        t = term[(batch, sem)]
        lines = []
        for off in group:
            opts = [o for o in options if o["offering_id"] == off["id"]]
            desc = " OR ".join(f"{o['code'] or '—'} {o['title']}" for o in opts)
            lines.append(f"- {off['basket_code']} {desc} ({fmt_ltp(opts[0]) if len(opts) == 1 else 'see options'}, {off['C']} cr)")
        total = sum(o["C"] or 0 for o in group)
        cards.append({"id": f"plan::{batch}::S{sem}", "doc_type": "plan",
                      "text": f"{PROGRAMME} batch {batch}, semester {sem} ({t['label']}): {len(group)} course slots, total {total:g} credits.\n" + "\n".join(lines),
                      "citation": f"Semester spread {batch} › {group[0]['sheet']} › semester {sem} block",
                      "meta": {"batch": batch, "semester": sem}})
    for batch, group in sorted(_group(baskets, ("batch",)).items()):
        lines = [f"- {b['basket_code']} {b['basket_name']}: {b['credits_struct'] if b['credits_struct'] is not None else b['fixed_min_semspread']:g} credits"
                 + (f" (Fixed Min column says {b['fixed_min_semspread']:g})" if b["fixed_min_semspread"] not in (None, b["credits_struct"]) else "")
                 for b in group]
        cards.append({"id": f"baskets::{batch[0]}", "doc_type": "basket",
                      "text": f"{PROGRAMME} batch {batch[0]} credit structure (180 credits total for the degree):\n" + "\n".join(lines),
                      "citation": f"Struct_{batch[0]} / semester spread {batch[0]} basket columns", "meta": {"batch": batch[0]}})
    for s in minor_summary:
        rows = [m for m in minors if m["minor"] == s["minor"] and m["batch"] == s["batch"]]
        if s["status"] == "no_students":
            body = "Not offered to this batch (source: 'No Students')."
        else:
            body = "\n".join(f"- S{m['semester'] or '?'} {m['code'] or '—'} {m['title']} (L-T-P {fmt_ltp(m)}, {m['C']} cr; prereq {m['prereq_text'] or 'not given'})"
                             for m in rows)
        cards.append({"id": f"minor::{s['minor']}::{s['batch']}", "doc_type": "minor",
                      "text": f"{s['minor']} minor for batch {s['batch']}: total {s['total_credits']:g} credits, {s['tba_credits']:g} credits TBA.\n{body}",
                      "citation": f"MinorCoursesforBTech_Students.xlsx › {rows[0]['sheet'] if rows else s['minor']}", "meta": {"batch": s["batch"], "minor": s["minor"]}})
    for c in courses:
        cards.append({"id": f"courseinfo::{c['code']}", "doc_type": "course_overview",
                      "text": f"{c['canonical_title']} ({c['code']}): {c['summary']}", "citation": "Semester spread / minor workbooks (all batches)",
                      "meta": {"code": c["code"]}})
    for batch in range(2022, 2027):
        lines = [t["label"] for t in terms if t["batch"] == batch]
        cards.append({"id": f"terms::{batch}", "doc_type": "terms",
                      "text": f"Calendar months for {PROGRAMME} batch {batch} (derived: batch year = admission year; Handbook §1.2 odd semester Jul/Aug–Dec, even semester Jan–Apr/May, §1.3 summer Jun–Jul):\n- " + "\n- ".join(lines),
                      "citation": "Student Handbook §1.2–1.3, p.15 + batch year", "meta": {"batch": batch}})
    for i in issues:
        if i["severity"] == "warning":
            cards.append({"id": f"issue::{i['id']}", "doc_type": "issue", "text": f"Data/policy conflict: {i['title']}. {i['detail']}",
                          "citation": "; ".join(i["sources"]) or "derived check", "meta": {"issue_id": i["id"]}})

    # Attach stable provenance to every retrieval card.  This data is used by both
    # citations and the source-navigation endpoint; it is never inferred from the text at answer time.
    for card in cards:
        meta = card.setdefault("meta", {})
        if card["doc_type"] == "terms":
            meta.update(source_file="4. Student Handbook Aug 2026.pdf", pdf_page=15, exact_text=card["text"])
        elif card["doc_type"] == "minor":
            meta.update(source_file="MinorCoursesforBTech_Students.xlsx",
                        sheet=card["citation"].split("›", 1)[1].strip() if "›" in card["citation"] else None,
                        locator=None, exact_text=card["text"])
        elif card["doc_type"] == "issue":
            meta.update(source_file="Semester_Spread_Structures_Sept_2026.xlsx",
                        sheet=None, locator=None, exact_text=card["text"])
        else:
            match = re.search(r"›\s*([^›]+)›\s*(.+)$", card["citation"])
            sheet = match.group(1).strip() if match else None
            locator = match.group(2).strip() if match else None
            meta.update(source_file="Semester_Spread_Structures_Sept_2026.xlsx",
                        sheet=sheet, locator=locator, exact_text=card["text"])
            if card["doc_type"] == "basket":
                batch = meta.get("batch")
                meta.update(source_file="Semester_Spread_Structures_Sept_2026.xlsx",
                            sheet=f"Struct_{batch}" if batch else None, locator=None)
    return cards


def _group(rows, keys):
    out = defaultdict(list)
    for r in rows:
        out[tuple(r[k] for k in keys)].append(r)
    return out


def build_courses(options, minors):
    info = defaultdict(lambda: {"titles": Counter(), "batches": defaultdict(list), "minor": set(), "programme": False})
    for o in options:
        if o["code_status"] == "ok":
            info[o["code"]]["programme"] = True
            info[o["code"]]["titles"][o["title"]] += 1
            info[o["code"]]["batches"][o["batch"]].append(f"S{o['semester']} {o['basket_code']} {fmt_ltp(o)} {o['C']}cr")
    for m in minors:
        if m["code"]:
            info[m["code"]]["titles"][m["title"]] += 1
            info[m["code"]]["minor"].add(m["minor"])
            info[m["code"]]["batches"][m["batch"]].append(f"{m['minor']} minor S{m['semester']} {fmt_ltp(m)} {m['C']}cr")
    rows = []
    for code, d in sorted(info.items()):
        keys = Counter()
        for t, n in d["titles"].items():
            keys[title_key(t)] += n
        best_key = keys.most_common(1)[0][0]
        rep = max((t for t in d["titles"] if title_key(t) == best_key), key=lambda t: d["titles"][t])
        summary = "; ".join(f"batch {b}: {', '.join(v)}" for b, v in sorted(d["batches"].items()))
        rows.append({"code": code, "canonical_title": fix_typos(rep), "titles_seen": json.dumps(sorted(d["titles"])),
                     "batches": json.dumps(sorted(d["batches"])), "kind": ("both" if d["minor"] else "programme") if d["programme"] else "minor",
                     "minors": json.dumps(sorted(d["minor"])), "summary": summary})
    return rows


# ---------------------------------------------------------------- persistence
def write_sqlite(tables):
    config.DB_PATH.unlink(missing_ok=True)
    con = sqlite3.connect(config.DB_PATH)
    for name, rows in tables.items():
        cols = list(rows[0].keys())
        con.execute(f"CREATE TABLE {name} ({', '.join(cols)})")
        con.executemany(f"INSERT INTO {name} VALUES ({', '.join('?' * len(cols))})",
                        [[json.dumps(r[c]) if isinstance(r[c], (list, dict)) else r[c] for c in cols] for r in rows])
    con.executescript("""
        CREATE INDEX ix_opt_code ON options(code); CREATE INDEX ix_opt_bs ON options(batch, semester);
        CREATE VIEW course_offerings AS
          SELECT o.batch, o.semester, o.basket_code, o.basket_name, o.slot, o.code, o.code_status, o.title,
                 o.L, o.T, o.P, o.C, o.prereq_text, t.months, t.calendar_year, t.academic_year, o.sheet, o.cells
          FROM options o JOIN terms t ON t.batch = o.batch AND t.semester = o.semester;
    """)
    con.commit()
    con.close()


def export(tables):
    import pandas as pd
    out = config.KB / "exports"
    out.mkdir(exist_ok=True)
    with pd.ExcelWriter(config.KB / "academic_advisor_normalized.xlsx") as xw:
        for name, rows in tables.items():
            (out / f"{name}.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
            pd.DataFrame(rows).map(lambda v: json.dumps(v) if isinstance(v, (list, dict)) else v).to_excel(xw, sheet_name=name[:31], index=False)


def embed(chunks):
    """Embed chunk texts (Bedrock Titan v2 or Azure); cached per model+text hash so unchanged chunks are never re-embedded."""
    import numpy as np
    from .. import embeddings
    mid = embeddings.model_id()
    if mid is None:
        print("  embeddings skipped (no embedding provider configured) -> retrieval runs BM25-only")
        config.EMB_PATH.unlink(missing_ok=True)
        config.EMB_MODEL_PATH.unlink(missing_ok=True)
        return
    cache = json.loads(config.EMB_CACHE.read_text()) if config.EMB_CACHE.exists() else {}
    keys = [hashlib.sha256((mid + "|" + c["text"]).encode()).hexdigest() for c in chunks]
    todo = sorted({i for i, k in enumerate(keys) if k not in cache})
    if todo:
        vecs = embeddings.embed([chunks[i]["text"] for i in todo])
        for i, v in zip(todo, vecs):
            cache[keys[i]] = [round(float(x), 6) for x in v]
        config.EMB_CACHE.write_text(json.dumps(cache))
    np.save(config.EMB_PATH, np.array([cache[k] for k in keys], dtype="float32"))
    config.EMB_MODEL_PATH.write_text(mid)
    print(f"  embeddings ({mid}): {len(chunks)} chunks ({len(todo)} newly embedded)")


def main():
    from . import validate
    print("Extracting sources ...")
    units = excel.extract(config.SOURCES["semester_spread"]) + excel.extract(config.SOURCES["minors"])
    hb_pages = pdf.extract(config.SOURCES["handbook"])

    offerings, options = build_offerings(units)
    baskets = build_baskets(units, options, offerings)
    structure = build_structure(units, options, offerings)
    minors, minor_summary = build_minors(units)
    terms = build_terms()
    courses = build_courses(options, minors)
    issues = build_issues(units, offerings, options, baskets, structure, minors, minor_summary)
    sources = [{"key": k, "file": p.name, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for k, p in config.SOURCES.items()]
    tables = {"offerings": offerings, "options": options, "baskets": baskets, "structure_courses": structure,
              "minors": minors, "minor_batches": minor_summary, "terms": terms, "courses": courses, "issues": issues, "sources": sources}
    write_sqlite(tables)
    export(tables)
    (config.KB / "sources.json").write_text(json.dumps(sources, indent=1), encoding="utf-8")

    handbook_md = write_handbook_md(hb_pages)
    sop_md = (config.CURATED / "sop.md").read_text(encoding="utf-8")
    chunks = (chunk_markdown(handbook_md, "handbook", "Student Handbook") + chunk_markdown(sop_md, "sop", "SOP")
              + build_cards(options, offerings, baskets, minors, minor_summary, terms, issues, courses))
    with open(config.CHUNKS_PATH, "w", encoding="utf-8") as fh:
        for c in chunks:
            fh.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"  {len(offerings)} offering slots, {len(options)} course options, {len(minors)} minor rows, "
          f"{len(structure)} structure rows, {len(issues)} issues, {len(chunks)} retrieval chunks")
    embed(chunks)

    ok, report = validate.run(units, tables, chunks, handbook_md)
    (config.KB / "VALIDATION_REPORT.md").write_text(report, encoding="utf-8")
    print(report.split("\n\n")[1] if "\n\n" in report else report)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
