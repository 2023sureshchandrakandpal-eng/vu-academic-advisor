"""Thin LangGraph flow:  agent (LLM + tools) <-> tools  ->  verify  -> END.

- The agent decides which deterministic tools / retrieval to call (that is the router).
- The tools node numbers every source as S1, S2, ... across the conversation; the model may only cite those.
- verify is pure code: drops citations that no tool produced and computes UI flags.
"""
import json
import os
import re
import time
from functools import lru_cache
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from . import config, tools
from .prompts import system_prompt

MAX_TOOL_ROUNDS = 4
HISTORY = 16  # messages kept per model call

TOOL_FUNCS = [tools.lookup_courses, tools.course_details, tools.semester_plan, tools.credit_structure, tools.minor_info,
              tools.prerequisite_chain, tools.academic_calendar, tools.check_eligibility, tools.search_documents,
              tools.list_data_issues, tools.run_sql]
LC_TOOLS = [StructuredTool.from_function(f) for f in TOOL_FUNCS]
BY_NAME = {f.__name__: f for f in TOOL_FUNCS}


def merge(a, b):
    return {**(a or {}), **(b or {})}


class State(TypedDict):
    messages: Annotated[list, add_messages]
    evidence: Annotated[dict, merge]
    profile: dict | None
    rounds: int
    flags: dict


@lru_cache(maxsize=2)  # one client per mode: re-creating it costs ~1.5 s and a new TLS handshake per call
def llm(bind_tools=True):
    from langchain_openai import AzureChatOpenAI
    kw = {} if os.getenv("AZURE_OPENAI_TEMPERATURE", "0") == "default" else {"temperature": float(os.getenv("AZURE_OPENAI_TEMPERATURE", "0"))}
    model = AzureChatOpenAI(azure_endpoint=config.AZURE_ENDPOINT, api_key=config.AZURE_API_KEY, api_version=config.AZURE_API_VERSION,
                            azure_deployment=config.CHAT_DEPLOYMENT, timeout=40, max_retries=2, **kw)
    return model.bind_tools(LC_TOOLS) if bind_tools else model


def _history(messages):
    """Recent messages, never starting with an orphan ToolMessage."""
    recent = messages[-HISTORY:]
    while recent and isinstance(recent[0], ToolMessage):
        recent = recent[1:]
    return recent


def _last_question(messages):
    return next((m.content for m in reversed(messages) if isinstance(m, HumanMessage)), "")


def agent(state: State):
    use_tools = state.get("rounds", 0) < MAX_TOOL_ROUNDS
    mentions = tools.course_mentions(_last_question(state["messages"]))
    note = ("\nCOURSES MENTIONED in the student's latest message (resolved from the catalog - use these codes, never others): "
            + "; ".join(f"'{k}' -> {', '.join(v)}" for k, v in mentions.items())) if mentions else ""
    msgs = [SystemMessage(system_prompt(state.get("profile")) + note)] + _history(state["messages"])
    if not use_tools:
        msgs.append(SystemMessage("Tool budget exhausted: answer now from the evidence already gathered, or say what is missing."))
    return {"messages": [llm(use_tools).invoke(msgs)]}


def _globalize(obj, ids):
    """Replace tool-local source numbers (src: 3 / [src 3]) by conversation-wide ids (S7)."""
    if isinstance(obj, dict):
        return {k: (ids.get(v, v) if k == "src" else _globalize(v, ids)) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_globalize(v, ids) for v in obj]
    if isinstance(obj, str):
        return re.sub(r"\[src (\d+)\]", lambda m: f"[{ids.get(int(m.group(1)), '?')}]", obj)
    return obj


def _snippets(result, n):
    """Text shown under a source card: the row / passage that carries that source."""
    found = []

    def walk(o):
        if isinstance(o, dict):
            if o.get("src") == n:
                found.append(o.get("text") or ", ".join(f"{k}: {v}" for k, v in o.items() if k != "src" and v not in (None, "", [])))
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)
        elif isinstance(o, str) and f"[src {n}]" in o:
            found.append(o.replace(f" [src {n}]", ""))
    walk(result)
    return found[0][:600] if found else ""


def run_tools(state: State):
    last = state["messages"][-1]
    evidence, out = dict(state.get("evidence") or {}), []
    profile = state.get("profile") or {}
    question = _last_question(state["messages"])
    mentions = tools.course_mentions(question)
    for call in last.tool_calls:
        name, args = call["name"], dict(call["args"])
        # guard: a course code the student never wrote and the catalog did not resolve is a model guess -> use the resolved course
        guess = re.sub(r"\s+", "", str(args.get("course") or "")).upper()
        allowed = {c for v in mentions.values() for c in v}
        if re.fullmatch(r"[A-Z]{3,4}\d{3}", guess) and guess not in allowed and guess not in question.upper().replace(" ", "") and len(mentions) == 1:
            args["course"] = next(iter(mentions))
        if name == "check_eligibility" and profile:  # the selected profile is the source of truth for the student's record
            args.update(batch=profile.get("batch"), completed=profile.get("completed_courses"),
                        failed=profile.get("failed_courses"), cgpa=profile.get("cgpa"))
        try:
            res = BY_NAME[name](**args)
        except Exception as e:  # bad arguments from the model -> tell it, don't crash the turn
            res = {"result": {"error": f"{type(e).__name__}: {e}"}, "sources": []}
        ids = {}
        for k, cite in enumerate(res.get("sources", []), 1):
            sid = f"S{len(evidence) + 1}"
            ids[k] = sid
            if isinstance(cite, dict):
                citation = cite.get("citation", "")
                source = dict(cite)
            else:
                citation = str(cite)
                source = {}
                lower = citation.lower()
                if "student handbook" in lower:
                    source = {"source_file": "4. Student Handbook Aug 2026.pdf",
                              "pdf_page": int(re.search(r"p\.?\s*(\d+)", citation).group(1)) if re.search(r"p\.?\s*(\d+)", citation) else None,
                              "exact_text": _snippets(res["result"], k)}
                elif "minorcourses" in lower or "minorcourse" in lower:
                    source = {"source_file": "MinorCoursesforBTech_Students.xlsx",
                              "exact_text": _snippets(res["result"], k)}
                elif "sop" in lower:
                    source = {"source_file": "SOP STUDENT 17082026 - Final.pdf",
                              "exact_text": _snippets(res["result"], k)}
                else:
                    source = {"source_file": "Semester_Spread_Structures_Sept_2026.xlsx",
                              "exact_text": _snippets(res["result"], k)}
            evidence[sid] = {
                "citation": citation,
                "tool": name,
                "snippet": _snippets(res["result"], k),
                **source,
            }
        payload = {"result": _globalize(res["result"], ids), "sources": {
            ids[k]: (c.get("citation", "") if isinstance(c, dict) else c)
            for k, c in enumerate(res.get("sources", []), 1)
        }}
        out.append(ToolMessage(json.dumps(payload, ensure_ascii=False, default=str)[:24000], tool_call_id=call["id"], name=name))
    return {"messages": out, "evidence": evidence, "rounds": state.get("rounds", 0) + 1}


def route(state: State):
    return "tools" if getattr(state["messages"][-1], "tool_calls", None) else "verify"


def verify(state: State):
    msg = state["messages"][-1]
    text = msg.content if isinstance(msg.content, str) else str(msg.content)
    text = re.sub(r"(?<=\|)\s*(S\d+)\s*(?=\|)", r" [\1] ", text)  # bare "S3" in a table source column -> [S3]
    text = re.sub(r"\[(S\d+(?:\s*[,;]\s*S\d+)+)\]", lambda m: "".join(f"[{c}]" for c in re.split(r"\s*[,;]\s*", m[1])), text)  # [S1, S5] -> [S1][S5]
    evidence = state.get("evidence") or {}
    cited = list(dict.fromkeys(re.findall(r"\[(S\d+)\]", text)))
    bad = [c for c in cited if c not in evidence]
    for c in bad:
        text = text.replace(f"[{c}]", "")
    text = re.sub(r" +([.,;])", r"\1", text)
    turn_tools = []
    for m in reversed(state["messages"][:-1]):
        if isinstance(m, HumanMessage):
            break
        if isinstance(m, ToolMessage):
            turn_tools.append(m.name)
    turn_ids = [sid for m in state["messages"] if isinstance(m, ToolMessage) and m.name in turn_tools
                for sid in json.loads(m.content).get("sources", {}) if sid in evidence] if turn_tools else []
    good = [c for c in cited if c in evidence]
    flags = {"cited": good or turn_ids[-12:], "sources_not_inline": bool(turn_tools) and not good, "removed_citations": bad,
             "tools": list(reversed(turn_tools)),
             "asked_follow_up": text.rstrip().endswith("?"),
             "conflict_flagged": bool(re.search(r"conflict|disagree|inconsisten|does not exist|not match", text, re.I)),
             "insufficient": bool(re.search(r"(do(es)? not|don't) (contain|have|include|mention|specify)|not (in|covered by) the (provided )?documents", text, re.I)),
             }
    new = AIMessage(content=text, id=msg.id)
    return {"messages": [new], "flags": flags, "rounds": 0}


def build():
    g = StateGraph(State)
    g.add_node("agent", agent)
    g.add_node("tools", run_tools)
    g.add_node("verify", verify)
    g.add_edge(START, "agent")
    g.add_conditional_edges("agent", route, {"tools": "tools", "verify": "verify"})
    g.add_edge("tools", "agent")
    g.add_edge("verify", END)
    return g.compile(checkpointer=MemorySaver())


_GRAPH = None


def ask(message, thread_id="default", profile=None):
    """One chat turn -> {answer, sources (cited only), flags, latency_ms}."""
    global _GRAPH
    _GRAPH = _GRAPH or build()
    t0 = time.perf_counter()
    state = _GRAPH.invoke({"messages": [HumanMessage(message)], "profile": profile, "rounds": 0},
                          {"configurable": {"thread_id": thread_id}, "recursion_limit": 2 * MAX_TOOL_ROUNDS + 6})
    flags = state["flags"]
    ev = state["evidence"] or {}
    return {"answer": state["messages"][-1].content, "sources": [{"id": c, **ev[c]} for c in flags["cited"]],
            "flags": flags, "latency_ms": round((time.perf_counter() - t0) * 1000)}
