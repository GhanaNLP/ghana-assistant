"""Scenario -> compact textual 'skeleton' (the reasoning trace). No distances, no coordinates; landmarks, counted turns, sides."""
ORD = {1: "first", 2: "second", 3: "third"}
NEAR = lambda d: "right beside" if d < 60 else "a short walk from" if d < 250 else "a bit further from"


def _lms(st):
    l = st.get("turn_lms") or []
    pos = lambda x: "straight ahead" if x["side"] == "ahead" else f"on your {x['side']}"
    return ("; sees " + ", ".join(f"{x['name']} ({pos(x)})" for x in l)) if l else "; no known landmark here"


def _steps(sc):
    out = []
    for i, st in enumerate(sc.get("steps") or []):
        road = st["road"] + (" (unnamed)" if st.get("generic") else "")
        if i == 0:
            n = st.get("side_roads_passed", 0)
            out.append(f"1. Head {st['heading']} on {road}" + (f", passing about {n} small side roads" if n > 1 else ""))
            continue
        tn = st["turn"]; side = "left" if "left" in tn else "right" if "right" in tn else None
        if side:
            o = st.get("turn_ordinal")
            what = f"TURN {side.upper()} (the {ORD[o]} {side})" if o else f"TURN {side.upper()} (many small roads before it: do not count)"
        else:
            what = "KEEP STRAIGHT"
        skip = st.get("side_roads_passed", 0)
        out.append(f"{i+1}. {what} onto {road}" + (f", skipping {skip} small roads" if skip > 1 else "") + _lms(st))
    return out


def skeleton_text(sc):
    s, e, t = sc["start"], sc.get("end"), sc["task"]
    L = [f"Task: {t}"]
    if s.get("name") is None and s.get("near"):
        L.append(f"Start: a spot {NEAR(s['near']['dist_m'])} {s['near']['name']}, area {s['area']}")
    elif s.get("name") is None:
        L.append(f"Position: area {s['area']}")
    else:
        L.append(f"Start: {s['name']} ({s['type']}), area {s['area']}")
    if e: L.append(f"Destination: {e['name']} ({e['type']}), area {e['area']}")
    if sc.get("avoid"): L.append(f"Avoid {sc['avoid']['kind']}: {sc['avoid']['name']} (route below avoids it)")
    if sc.get("nearby"):
        L.append("Nearby, closest first: " + "; ".join(f"{n['name']} ({n['type']}, {NEAR(n['dist_m'])})" for n in sc["nearby"]))
    if sc.get("question_landmark"):
        L.append(f"Asked about: {sc['question_landmark']} -> on the route: {'yes' if sc['on_route'] else 'no (well off the route)'}")
    if sc.get("steps"):
        L.append("Route:"); L += _steps(sc)
        near = f", beside {sc['dest_near'][0]['name']}" if sc.get("dest_near") else ""
        L.append(f"Arrive: destination on your {sc['dest_side']}{near}" if sc["dest_side"] != "ahead" else f"Arrive: destination straight ahead{near}")
    return "\n".join(L)
