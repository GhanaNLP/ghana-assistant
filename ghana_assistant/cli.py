"""Ghana Assistant from the command line.

  ghana-assistant "How do I get from Kejetia Market to Bantama Market?" [--debug]   one question
  ghana-assistant serve [--host 0.0.0.0] [--port 8000]                               API + web interface (http://localhost:8000)
"""
import sys, json, argparse


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "serve":
        p = argparse.ArgumentParser(prog="ghana-assistant serve"); p.add_argument("--host", default="127.0.0.1"); p.add_argument("--port", type=int, default=8000)
        a = p.parse_args(sys.argv[2:])
        import uvicorn
        from .assistant import GhanaAssistant
        from .server import create_app
        ga = GhanaAssistant()                                  # load before serving so the first question is fast
        print(f"Ghana Assistant: web interface http://{a.host}:{a.port}/  |  API POST http://{a.host}:{a.port}/ask  |  docs /docs")
        uvicorn.run(create_app(lambda: ga), host=a.host, port=a.port)
        return
    args = [x for x in sys.argv[1:] if x != "--debug"]; debug = "--debug" in sys.argv
    if not args: print(__doc__); return
    from .assistant import GhanaAssistant
    out = GhanaAssistant().ask(" ".join(args))
    print(out["answer"])
    if debug: print(json.dumps({k: v for k, v in out.items() if k != "answer"}, indent=1, ensure_ascii=False))
