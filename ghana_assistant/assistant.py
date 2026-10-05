"""One entry point: ask(message) -> answer, routing to navigation or knowledge."""
import os
from huggingface_hub import snapshot_download
from . import config
from .model import AssistantModel
from .navigation.world import load_map
from .navigation.resolver import Resolver
from .navigation.planner import Planner, PlanError


class GhanaAssistant:
    def __init__(self, model_repo=config.MODEL_REPO, data_repo=config.DATA_REPO, data_dir=None, device=None, knowledge=True):
        self.model = AssistantModel(model_repo, device)
        self.data_dir = data_dir or snapshot_download(data_repo, repo_type="dataset", token=config.HF_TOKEN,
                                                          allow_patterns=["maps/*", "knowledge/*"])   # skip the training splits
        self.maps = {}
        for c in config.CITIES:
            p = os.path.join(self.data_dir, "maps", f"{c}.map.gz")
            if os.path.exists(p):
                w = load_map(p); r = Resolver(w); self.maps[c] = (w, r, Planner(w, r))
        self.retriever = None
        if knowledge and os.path.exists(os.path.join(self.data_dir, "knowledge", "store.sqlite")):
            from .knowledge.phrases import PhraseStore                # noun-phrase retrieval, memory-mapped (no embeddings, no FAISS)
            self.retriever = PhraseStore(os.path.join(self.data_dir, "knowledge"))

    def _city(self, fields):
        names = [fields.get(k) for k in ("start", "end", "start_area", "end_area", "avoid", "asked")]
        names = [n[5:] if n and n.lower().startswith("near ") else n for n in names]
        return max(self.maps, key=lambda c: self.maps[c][1].score_city(names))

    def directions(self, message):
        fields = self.model.parse(message)
        city = self._city(fields); _, _, planner = self.maps[city]
        try:
            sc, skeleton = planner.plan(fields)
        except PlanError as e:
            return dict(skill="navigation", answer=str(e), fields=fields, city=city)
        return dict(skill="navigation", answer=self.model.directions(message, skeleton), fields=fields, city=city, plan=skeleton)

    def knowledge(self, question):
        if not self.retriever: return dict(skill="knowledge", answer="The knowledge index is not loaded.")
        hits = self.retriever.search(question)
        if not hits: return dict(skill="knowledge", answer="I could not find anything about that in the news and research I know.", sources=[])
        return dict(skill="knowledge", answer=self.model.answer(question, [h["line"] for h in hits]), sources=hits)

    def ask(self, message):
        intent = self.model.intent(message)
        return self.directions(message) if intent.startswith("nav") else self.knowledge(message)
