"""Build and publish the static Hugging Face Space: the package's web page pointed at a deployed API.
  python scripts/build_space.py https://<workspace>--ghana-assistant-assistant-web.modal.run [--publish]"""
import sys, os, shutil, tempfile
url = sys.argv[1].rstrip("/"); here = os.path.dirname(os.path.abspath(__file__)); root = os.path.join(here, "..")
out = tempfile.mkdtemp(prefix="ga_space_")
html = open(os.path.join(root, "ghana_assistant", "web", "index.html"), encoding="utf-8").read().replace("__ASSISTANT_URL__", url)
open(os.path.join(out, "index.html"), "w", encoding="utf-8").write(html); shutil.copy(os.path.join(root, "space", "README.md"), out)
print("space files in", out)
if "--publish" in sys.argv:
    from huggingface_hub import HfApi
    api = HfApi(); rid = "ghananlpcommunity/ghana-assistant"
    api.create_repo(rid, repo_type="space", space_sdk="static", exist_ok=True)
    api.upload_folder(folder_path=out, repo_id=rid, repo_type="space", commit_message="Update frontend")
    print("published https://huggingface.co/spaces/" + rid)
