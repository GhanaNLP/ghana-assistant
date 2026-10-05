"""Match names the model extracted ("Kejetia Market", "Adum", "Ring Road") to real landmarks, areas and roads in a city map.
Rules: exact names win; neighbourhood names win over shops that merely contain them; a fuzzy match must share a
distinctive word (not "market", "school", "kumasi"...), so an unknown place is reported instead of silently swapped."""
import re
from rapidfuzz import fuzz, process

GENERIC = set("""the a an of and at in on near by to from road street avenue highway junction roundabout circle market school
senior high junior primary basic college university hospital clinic hotel church mosque station bus lorry park bank police
office ltd limited company shop store mall centre center international ghana accra kumasi new old main""".split())


def _tok(s): return [t for t in re.findall(r"[a-z0-9]+", s.lower())]
def _distinct(s): return {t for t in _tok(s) if t not in GENERIC and len(t) > 2}


class Resolver:
    def __init__(self, world):
        self.w = world
        self.poi = {l["name"]: l for l in world.poi}
        self.areas = {l["name"]: l for l in world.areas}
        self.exact = {}
        for d in (self.poi, self.areas):                       # areas inserted last so an exact area name wins
            for n, l in d.items(): self.exact[n.lower().strip()] = l
        self.roads = sorted({d["name"] for _, _, d in world.G.edges(data=True) if d["name"]})

    @staticmethod
    def _fuzzy(query, choices, cutoff):
        q = _distinct(query)
        for name, score, _ in process.extract(query, choices, scorer=fuzz.WRatio, limit=10, score_cutoff=cutoff):
            if not q or q & _distinct(name): return name, score     # must share a distinctive word
        return None, 0

    def place(self, name, cutoff=80):
        """-> (landmark or area dict, score 0-100); (None, 0) if unknown"""
        if not name: return None, 0
        key = name.lower().strip()
        if key in self.exact: return self.exact[key], 100
        a, sa = self._fuzzy(name, list(self.areas), cutoff)
        p, sp = self._fuzzy(name, list(self.poi), cutoff)
        if p and _distinct(name) and _distinct(name) <= _distinct(p): return self.poi[p], sp     # every distinctive word matches a landmark
        if a and (sa >= sp or len(_tok(name)) <= 2 and fuzz.ratio(name.lower(), a.lower()) >= 90): return self.areas[a], sa
        if p: return self.poi[p], sp
        return (self.areas[a], sa) if a else (None, 0)

    def road(self, name, cutoff=85):
        return self._fuzzy(name, self.roads, cutoff)

    def score_city(self, names):
        """how well a set of names matches this city (used to pick Accra vs Kumasi)"""
        return sum(self.place(n, cutoff=60)[1] for n in names if n)
