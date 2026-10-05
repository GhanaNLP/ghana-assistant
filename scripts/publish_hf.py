"""Upload runtime artifacts to Hugging Face (organisation: ghananlpcommunity).
  python scripts/publish_hf.py model  runs/t5s_multi                 -> ghananlpcommunity/ghana-assistant-flan-t5-small
  python scripts/publish_hf.py data   out/  [--private]              -> ghananlpcommunity/ghana-assistant-data  (maps/, knowledge/)
  python scripts/publish_hf.py space  space/                         -> ghananlpcommunity/ghana-assistant (Gradio Space)"""
import sys, argparse
from huggingface_hub import HfApi
p = argparse.ArgumentParser(); p.add_argument("what", choices=["model", "data", "space"]); p.add_argument("folder"); p.add_argument("--private", action="store_true")
a = p.parse_args(); api = HfApi()
repo, kind = {"model": ("ghananlpcommunity/ghana-assistant-flan-t5-small", "model"), "data": ("ghananlpcommunity/ghana-assistant-data", "dataset"),
              "space": ("ghananlpcommunity/ghana-assistant", "space")}[a.what]
api.create_repo(repo, repo_type=kind, private=a.private, exist_ok=True, **({"space_sdk": "gradio"} if kind == "space" else {}))
api.upload_folder(folder_path=a.folder, repo_id=repo, repo_type=kind, ignore_patterns=["checkpoint-*", "_work/*", "*.pkl"])
print("published", f"https://huggingface.co/{'spaces/' if kind == 'space' else 'datasets/' if kind == 'dataset' else ''}{repo}")
