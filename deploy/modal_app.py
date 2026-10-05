"""Serve Ghana Assistant on Modal (CPU) as a web endpoint.

  modal run deploy/modal_app.py::download     # once: fill the cache volume with the model and data from Hugging Face
  modal deploy deploy/modal_app.py            # prints the URL of the /ask endpoint

<url>/       web interface        POST <url>/ask  {"question": "..."}  ->  {"skill", "answer", ...}        <url>/docs  API docs
CORS is open so the static Hugging Face Space can call it straight from the browser.
If the data repo is private, create a Modal secret named "huggingface" with HF_TOKEN and deploy with GA_USE_HF_SECRET=1."""
import os
import modal

image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("torch", extra_index_url="https://download.pytorch.org/whl/cpu")
         .pip_install("transformers>=4.45", "sentencepiece", "spacy>=3.7", "networkx", "numpy", "pandas", "pyarrow", "rapidfuzz",
                      "huggingface_hub", "fastapi[standard]", "pydantic",
                      "https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl")
         .env({"HF_HOME": "/cache/hf"})
         .add_local_python_source("ghana_assistant")
         .add_local_dir("ghana_assistant/web", "/root/ghana_assistant/web"))     # the web page (add_local_python_source copies .py files only)
app = modal.App("ghana-assistant", image=image)
cache = modal.Volume.from_name("ghana-assistant-cache", create_if_missing=True)      # model + data survive cold starts
SECRETS = [modal.Secret.from_name("huggingface")] if os.environ.get("GA_USE_HF_SECRET") else []
RUNTIME = ["maps/*", "knowledge/*"]


@app.function(volumes={"/cache": cache}, secrets=SECRETS, timeout=3600)
def download():
    from huggingface_hub import snapshot_download
    from ghana_assistant import config
    print("model:", snapshot_download(config.MODEL_REPO))
    print("data :", snapshot_download(config.DATA_REPO, repo_type="dataset", allow_patterns=RUNTIME))
    cache.commit()


@app.cls(cpu=4, memory=8192, volumes={"/cache": cache}, secrets=SECRETS, scaledown_window=300, timeout=120, max_containers=3)
class Assistant:
    @modal.enter()
    def load(self):
        import shutil
        from huggingface_hub import snapshot_download
        from ghana_assistant import GhanaAssistant, config
        src = snapshot_download(config.DATA_REPO, repo_type="dataset", allow_patterns=RUNTIME)   # from the volume cache
        local = "/tmp/ga_data"
        shutil.copytree(src, local, dirs_exist_ok=True)       # local disk: fast memory-mapped reads for retrieval
        self.ga = GhanaAssistant(data_dir=local, device="cpu")

    @modal.asgi_app()
    def web(self):
        from ghana_assistant.server import create_app
        return create_app(lambda: self.ga)              # same API + web interface as `ghana-assistant serve`
