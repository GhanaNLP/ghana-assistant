"""ghana-assistant "How do I get from Kejetia Market to KNUST?"  (add --debug to see the plan / sources)"""
import sys, json
from .assistant import GhanaAssistant


def main():
    args = [a for a in sys.argv[1:] if a != "--debug"]; debug = "--debug" in sys.argv
    if not args: print(__doc__); return
    out = GhanaAssistant().ask(" ".join(args))
    print(out["answer"])
    if debug: print(json.dumps({k: v for k, v in out.items() if k != "answer"}, indent=1, ensure_ascii=False))
