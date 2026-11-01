"""FastAPI app: chat + voice transcription + synthetic profiles + static UI.

    uvicorn advisor.server:app --reload        # then open http://127.0.0.1:8000
"""
import json
import time
import uuid
import base64
import html
import re

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import config

app = FastAPI(title="VU AI Academic Advisor")
STATIC = config.ROOT / "advisor" / "static"
PROFILES = {p["id"]: p for p in json.loads((config.ROOT / "eval" / "profiles.json").read_text(encoding="utf-8"))}
MAX_AUDIO_BYTES = 25 * 1024 * 1024  # Azure OpenAI transcription limit


class ChatIn(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    thread_id: str | None = None
    profile_id: str | None = None


@app.on_event("startup")
def warm_up():
    """Load the search index, graph and Azure clients before the first question (avoids a slow first answer)."""
    from .retrieval import embed_query, index
    if index()[2] is not None:
        embed_query("warm up")  # opens the Bedrock connection
    if config.azure_ready(config.CHAT_DEPLOYMENT):
        from . import graph
        graph._GRAPH = graph.build()
        graph.llm(True)
        graph.llm(False).invoke("Reply OK")  # opens the Azure connection (TLS) so the first student question is fast


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/health")
def health():
    return {"chat": config.azure_ready(config.CHAT_DEPLOYMENT), "embeddings": config.EMB_PATH.exists(),
            "transcription": config.azure_ready(config.TRANSCRIBE_DEPLOYMENT), "knowledge_base": config.DB_PATH.exists()}


@app.get("/api/profiles")
def profiles():
    return [{"id": p["id"], "label": p["label"], "batch": p["batch"], "current_semester": p["current_semester"],
             "cgpa": p["cgpa"], "minor": p["minor"], "notes": p["notes"],
             "credits_earned": p.get("credits_earned"), "failed_courses": p.get("failed_courses") or []} for p in PROFILES.values()]


@app.post("/api/chat")
def chat(body: ChatIn):
    if body.profile_id and body.profile_id not in PROFILES:
        raise HTTPException(400, "Unknown profile")
    thread = body.thread_id or uuid.uuid4().hex
    profile = PROFILES.get(body.profile_id)
    if not config.azure_ready(config.CHAT_DEPLOYMENT):
        return {**preview(body.message), "thread_id": thread}
    from .graph import ask
    # A profile switch starts a fresh conversation so one student's record never leaks into another's thread.
    out = ask(body.message, thread_id=f"{thread}:{body.profile_id or 'none'}", profile=profile)
    return {**out, "thread_id": thread}


def preview(message):
    """No LLM configured yet: show the retrieved evidence so the pipeline can still be checked end to end."""
    from .retrieval import search
    t0 = time.perf_counter()
    hits = search(message, k=4)
    sources = []
    for i, h in enumerate(hits, 1):
        meta = h.get("meta", {})
        sources.append({
            "id": f"S{i}", "citation": h["citation"], "tool": "search_documents", "snippet": h["text"][:600],
            "source_file": meta.get("source_file") or ("4. Student Handbook Aug 2026.pdf" if h["doc_type"] == "handbook" else
                                                       "SOP STUDENT 17082026 - Final.pdf" if h["doc_type"] == "sop" else
                                                       "Semester_Spread_Structures_Sept_2026.xlsx"),
            "sheet": meta.get("sheet"), "locator": meta.get("locator"),
            "pdf_page": meta.get("pdf_pages", [None])[0] if meta.get("pdf_pages") else None,
            "exact_text": meta.get("exact_text") or h["text"][:1400],
        })
    answer = ("**Preview mode** – Azure OpenAI chat is not configured yet (fill in `.env`), so no answer is generated. "
              "These are the passages the advisor would ground its answer in:\n\n" + "\n".join(f"- [{s['id']}] {s['citation']}" for s in sources))
    return {"answer": answer, "sources": sources, "latency_ms": round((time.perf_counter() - t0) * 1000),
            "flags": {"tools": ["search_documents"], "cited": [s["id"] for s in sources], "preview": True}}


@app.post("/api/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    if not config.azure_ready(config.TRANSCRIBE_DEPLOYMENT):
        raise HTTPException(503, "Voice input needs AZURE_OPENAI_TRANSCRIBE_DEPLOYMENT in .env")
    data = await audio.read()
    if not data or len(data) > MAX_AUDIO_BYTES:
        raise HTTPException(400, "Audio must be between 1 byte and 25 MB")
    from openai import AzureOpenAI
    client = AzureOpenAI(azure_endpoint=config.AZURE_ENDPOINT, api_key=config.AZURE_API_KEY, api_version=config.TRANSCRIBE_API_VERSION)
    r = client.audio.transcriptions.create(model=config.TRANSCRIBE_DEPLOYMENT, file=(audio.filename or "speech.webm", data, audio.content_type or "audio/webm"),
                                           prompt="Vidyashilp University academic advising. Course codes like DATA301, COMP201, MATH203; CGPA, L-T-P, semester, minor.")
    return {"text": r.text}


def _source_path(filename: str):
    allowed = {p.name for p in config.SOURCES.values()}
    if filename not in allowed:
        raise HTTPException(404, "Unknown source document")
    return next(p for p in config.SOURCES.values() if p.name == filename)


@app.get("/api/source-file/{filename}")
def source_file(filename: str, page: int | None = None, sheet: str | None = None, locator: str | None = None):
    path = _source_path(filename)
    return FileResponse(path, filename=filename)


@app.get("/api/source-view", response_class=HTMLResponse)
def source_view(filename: str, page: int | None = None, sheet: str | None = None, locator: str | None = None, text: str | None = None):
    """Open the original source and present a verified page/cell location with yellow highlighting."""
    path = _source_path(filename)
    original_url = f"/api/source-file/{filename}" + (f"?page={page}" if page else "")
    if path.suffix.lower() == ".pdf":
        import fitz
        doc = fitz.open(path)
        page_number = max(0, min((page or 1) - 1, doc.page_count - 1))
        pdf_page = doc[page_number]
        highlights = []
        needle = re.sub(r"\s+", " ", (text or "").strip())
        for candidate in (needle, needle[:300], " ".join(needle.split()[:12])):
            if candidate:
                rects = pdf_page.search_for(candidate)
                if rects:
                    highlights = rects
                    break
        if not highlights and needle:
            for word in re.findall(r"[A-Za-z]{4,}", needle):
                highlights.extend(pdf_page.search_for(word)[:3])
                if len(highlights) >= 12:
                    break
        pix = pdf_page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5), alpha=False)
        image = base64.b64encode(pix.tobytes("png")).decode("ascii")
        rect_markers = "".join(
            f'<rect x="{r.x0 * 1.5}" y="{r.y0 * 1.5}" width="{(r.x1-r.x0)*1.5}" height="{(r.y1-r.y0)*1.5}" fill="yellow" fill-opacity="0.58"/>'
            for r in highlights
        )
        exact = html.escape(text or "")
        body = f"""
        <h2>{html.escape(filename)} — page {(page_number + 1)}</h2>
        <p><a href="{html.escape(original_url, quote=True)}" target="_blank" rel="noopener">Open the original PDF</a></p>
        <div class="highlight">
          <svg viewBox="0 0 {pix.width} {pix.height}" style="max-width:100%;height:auto;border:1px solid #ccc">
            <image href="data:image/png;base64,{image}" width="{pix.width}" height="{pix.height}"/>
            {rect_markers}
          </svg>
        </div>
        <pre>{exact}</pre>
        <style>body{{font-family:Arial,sans-serif;margin:24px}} h2{{margin-bottom:6px}} .highlight{{position:relative}} pre{{white-space:pre-wrap;background:#fffbea;padding:12px}}</style>
        """
        return HTMLResponse(body)

    sheet_name = html.escape(sheet or "Workbook sheet/row listed below")
    locator_name = html.escape(locator or "Source row or cell is shown in the citation")
    exact = html.escape(text or "")
    return HTMLResponse(f"""
    <h2>{html.escape(filename)}</h2>
    <p><a href="{html.escape(original_url, quote=True)}" target="_blank" rel="noopener">Open the original workbook</a></p>
    <div style="background:#fff2a8;border:1px solid #e0b800;padding:14px;margin:16px 0"><strong>Relevant location:</strong> {sheet_name}<br>{locator_name}</div>
    <pre style="white-space:pre-wrap">{exact}</pre>
    """)


app.mount("/static", StaticFiles(directory=STATIC), name="static")
app.mount("/assets", StaticFiles(directory=config.ROOT / "assets"), name="assets")  # official logo + campus photo
