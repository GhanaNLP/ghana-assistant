"""Parsed request -> route on the road graph -> structured scenario -> skeleton text (the model's route plan)."""
import numpy as np
from .world import hav, bearing, compass, rnd_m
from .skeleton import skeleton_text


class PlanError(Exception):
    """raised with a user-facing reason when a request can't be routed"""


def _pt(l):
    return dict(name=l["name"], lat=l["lat"], lon=l["lon"], area=l.get("area") or l["name"], type=l["cat"])


class Planner:
    def __init__(self, world, resolver):
        self.w, self.r = world, resolver

    def _place(self, fields, key, area_key):
        """-> landmark/area dict, or None if not given; raises PlanError naming a place we don't know"""
        for k in (key, area_key):
            if not fields.get(k): continue
            name = fields[k][5:] if fields[k].lower().startswith("near ") else fields[k]
            l, _ = self.r.place(name)
            if l: return l
            raise PlanError(f"I don't know where \"{name}\" is in {self.w.city.title()}. Try a nearby landmark, market, school or junction.")
        return None

    def _route_scenario(self, task, a, b, banned_nodes=(), banned_road=None, extra=None):
        w = self.w
        path = w.route(a["node"], b["node"], banned_nodes=banned_nodes, banned_road=banned_road)
        if not path: raise PlanError("I could not find a road route between those places.")
        st = w.steps(path, exclude={a["name"], b["name"]})
        sc = dict(task=task, city=w.city.title(), start=_pt(a), end=_pt(b), steps=st, dest_side=w.dest_side(path, b),
                  dest_near=w.nearby(b["lat"], b["lon"], r=100, k=1, exclude={a["name"], b["name"]}),
                  bearing=compass(bearing(a["lat"], a["lon"], b["lat"], b["lon"])), route_km=round(w.plen(path) / 1000, 1))
        sc.update(extra or {}); return sc, path

    def plan(self, fields):
        """-> (scenario dict, skeleton text)"""
        task = fields.get("task", "route")
        a = self._place(fields, "start", "start_area")
        if task == "nearby":
            if not a: raise PlanError("Tell me where you are (a landmark or neighbourhood) and I will list what is around you.")
            near = sorted(self.w.nearby(a["lat"], a["lon"], r=800, k=6, exclude={a["name"]}), key=lambda x: x["dist_m"])
            sc = dict(task="nearby", city=self.w.city.title(), start=_pt(a), nearby=near, end=None); return sc, skeleton_text(sc)
        b = self._place(fields, "end", "end_area")
        if not a or not b: raise PlanError("I need both where you are starting from and where you want to go.")
        if task == "avoid" and fields.get("avoid"):
            road, rs = self.r.road(fields["avoid"]); lm, ls = self.r.place(fields["avoid"])
            if road and rs >= ls:
                sc, _ = self._route_scenario("avoid_road", a, b, banned_road=road, extra=dict(avoid=dict(kind="road", name=road)))
            elif lm:
                banned = self.w.nodes_within(lm["lat"], lm["lon"], 150) - {a["node"], b["node"]}
                sc, _ = self._route_scenario("avoid_landmark", a, b, banned_nodes=banned, extra=dict(avoid=dict(kind="landmark", name=lm["name"])))
            else:
                sc, _ = self._route_scenario("route", a, b)
            return sc, skeleton_text(sc)
        if task == "connects" and fields.get("asked"):
            q, _ = self.r.place(fields["asked"])
            sc, path = self._route_scenario("connects", a, b)
            if q:
                xy = np.array([self.w.coord[n] for n in path]); dm = float(np.min(self.w._d(xy, q["lat"], q["lon"])))
                sc.update(variant="on_the_way", question_landmark=q["name"], on_route=dm < 200, **({} if dm < 200 else {"off_route_m": rnd_m(dm)}))
            return sc, skeleton_text(sc)
        sc, _ = self._route_scenario("route", a, b)
        return sc, skeleton_text(sc)
