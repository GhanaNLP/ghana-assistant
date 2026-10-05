"""Offline navigation test (no model): python tests/test_navigation.py path/to/kumasi.map.gz"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ghana_assistant.navigation.world import load_map
from ghana_assistant.navigation.resolver import Resolver
from ghana_assistant.navigation.planner import Planner, PlanError
w = load_map(sys.argv[1]); r = Resolver(w); pl = Planner(w, r)
assert r.place("Adum")[0]["cat"] == "area"
assert r.place("Kumasi Zoo")[0] is None                    # unknown places must not be swapped for a lookalike
sc, sk = pl.plan({"task": "route", "start": "Kejetia Market", "end": "Bantama Market"}); assert "Route:" in sk
sc, sk = pl.plan({"task": "avoid", "start": "Adum", "end": "Suame Market", "avoid": "Okomfo Anokye Road"}); assert "Avoid road" in sk
sc, sk = pl.plan({"task": "nearby", "start": "Kejetia Market"}); assert sk.startswith("Task: nearby")
try: pl.plan({"task": "route", "start": "Kumasi Zoo", "end": "Bantama Market"}); raise SystemExit("expected PlanError")
except PlanError: pass
print("navigation tests passed")
