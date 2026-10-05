"""Serve Ghana Assistant on Modal as a web endpoint.
  modal deploy deploy/modal_app.py          -> prints the URL of the /ask endpoint
POST {"question": "..."}  ->  {"skill": "navigation"|"knowledge", "answer": "...", ...}
If the data repo is private, create a Modal secret named "huggingface" with HF_TOKEN."""
import modal

image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("torch", extra_index_url="https://download.pytorch.org/whl/cpu")
         .pip_install("transformers>=4.45", "sentencepiece", "spacy>=3.7", "networkx", "numpy", "pandas", "pyarrow", "rapidfuzz",
                      "huggingface_hub", "fastapi[standard]",
                      "https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl")
         .add_local_python_source("ghana_assistant"))
app = modal.App("ghana-assistant", image=image)
cache = modal.Volume.from_name("ghana-assistant-cache", create_if_missing=True)    # keeps model/data downloads between cold starts
import os
SECRETS = [modal.Secret.from_name("huggingface")] if os.environ.get("GA_USE_HF_SECRET") else []   # set GA_USE_HF_SECRET=1 at deploy time if the data repo is private


@app.cls(cpu=4, memory=8192, volumes={"/cache": cache}, secrets=SECRETS,
         scaledown_window=300, timeout=120)
class Assistant:
    @modal.enter()
    def load(self):
        import os
        os.environ.setdefault("HF_HOME", "/cache/hf")
        from ghana_assistant import GhanaAssistant
        self.ga = GhanaAssistant(device="cpu")
        cache.commit()

    @modal.fastapi_endpoint(method="POST")
    def ask(self, body: dict):
        q = (body or {}).get("question", "").strip()
        if not q: return {"error": "empty question"}
        if len(q) > 500: return {"error": "question too long (max 500 characters)"}
        out = self.ga.ask(q)
        return {k: v for k, v in out.items() if k in ("skill", "answer", "city", "plan", "sources", "fields")}
