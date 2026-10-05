"""Build city maps (road graph + landmarks) from a Geofabrik Ghana extract (+ optional filtered Foursquare places).
  python scripts/build_maps.py ghana-latest.osm.pbf out/maps [--fsq ghana_fsq.parquet]
Writes out/maps/accra.map.gz and out/maps/kumasi.map.gz. Map data (c) OpenStreetMap contributors, ODbL."""
import os, sys, shutil, subprocess, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ghana_assistant.navigation import world as W
BOXES = {"kumasi": "-1.75,6.58,-1.45,6.82", "accra": "-0.36,5.48,-0.08,5.74"}
p = argparse.ArgumentParser(); p.add_argument("pbf"); p.add_argument("out"); p.add_argument("--fsq"); a = p.parse_args()
os.makedirs(a.out, exist_ok=True); work = os.path.join(a.out, "_work"); os.makedirs(work, exist_ok=True)
if a.fsq: shutil.copy(a.fsq, os.path.join(work, "ghana_fsq.parquet"))
cwd = os.getcwd(); pbf = os.path.abspath(a.pbf); os.chdir(work)
for city, box in BOXES.items():
    subprocess.run(["osmium", "extract", "-b", box, pbf, "-o", f"{city}.osm.pbf", "--overwrite"], check=True)
    w = W.load_world(city); W.save_map(w, os.path.join(cwd, a.out, f"{city}.map.gz"))
    print(city, "landmarks", len(w.poi), "areas", len(w.areas), "graph nodes", w.G.number_of_nodes())
