"""HTTP API and web interface for Ghana Assistant (used locally, on Modal and behind the Hugging Face Space).

  GET  /         the web interface (one question, one answer)
  POST /ask      {"question": "..."} -> {"skill", "answer", "city", "plan" | "sources", "fields"}
  GET  /health   {"ok": true}
  GET  /docs     interactive API docs
"""
from importlib import resources


def create_app(get_assistant):
    """get_assistant: zero-argument callable returning a ready GhanaAssistant (called once, lazily)."""
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import HTMLResponse
    from pydantic import BaseModel, Field

    class AskBody(BaseModel):
        question: str = Field(..., max_length=500, examples=["How do I get from Kejetia Market to Bantama Market?"])

    api = FastAPI(title="Ghana Assistant", description="Landmark-based directions in Accra and Kumasi, and short answers about Ghana.")
    api.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "POST", "OPTIONS"], allow_headers=["*"])
    try:
        page = resources.files("ghana_assistant").joinpath("web/index.html").read_text(encoding="utf-8").replace("__ASSISTANT_URL__", "")
    except (FileNotFoundError, OSError):                     # never take the API down because the page is missing
        page = "<!doctype html><title>Ghana Assistant</title><p>The web page is not packaged here; the API is at <code>POST /ask</code> (see <a href='/docs'>/docs</a>).</p>"
    state = {}

    def ga():
        if "ga" not in state: state["ga"] = get_assistant()
        return state["ga"]

    @api.get("/", response_class=HTMLResponse, include_in_schema=False)
    def index():
        return page

    @api.get("/health")
    def health():
        return {"ok": True}

    @api.post("/ask")
    def ask(body: AskBody):
        q = body.question.strip()
        if not q: return {"error": "Please type a question."}
        out = ga().ask(q)
        return {k: v for k, v in out.items() if k in ("skill", "answer", "city", "plan", "sources", "fields")}

    return api
