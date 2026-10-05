# Ghana Assistant

A small, fast assistant for Ghana built around one tiny model (**Flan-T5-small, 77M parameters**) that never has to *remember* facts: the facts come from a map or from retrieved text, and the model only understands the request and writes the reply.

| Skill | Example | Where the facts come from |
|---|---|---|
| **Directions** (Accra, Kumasi) | *How do I get from Kejetia Market to Bantama Market, avoiding Okomfo Anokye Road?* | Shortest path on an OpenStreetMap road graph, turned into a route plan with counted turns and landmarks at each turn |
| **Ghana knowledge** | *Why is bird flu a concern for Ghana's poultry industry?* | Top-8 sentences from Ghanaian news and KNUST research, found with FAISS |

Answers are single-turn: one question, one answer (no conversation memory).

> **Status:** the model is being trained and the Hugging Face artifacts are not published yet, so `GhanaAssistant()` will not download anything until then. The navigation pipeline (maps, name matching, routing, route plans) works today; see `tests/test_navigation.py`.

## How it works

```
message ─► model "intent:" ─┬─ navigation ─► model "parse:"  → {task, start, end, avoid, asked}
                            │                 → match names to landmarks (rapidfuzz) → route on the road graph
                            │                 → route plan ─► model "directions:" ─► spoken directions
                            └─ knowledge  ─► bge-small embedding → FAISS top-8 sentences ─► model "answer:" ─► short answer
```

One model, four prefixes:

| Prefix | Input | Output |
|---|---|---|
| `intent:` | the user's message | `navigation` or `knowledge` |
| `parse:` | a navigation message | `task: avoid \| start: … \| end: … \| avoid: …` |
| `directions:` | message + route plan | spoken, landmark-based directions |
| `answer:` | question + 8 retrieved sentences | a short answer |

Unknown places are reported ("I don't know where X is…") instead of being swapped for a lookalike.

## Use it

```bash
pip install git+https://github.com/GhanaNLP/ghana-assistant
ghana-assistant "How do I get from Kejetia Market to Bantama Market?" --debug
```

```python
from ghana_assistant import GhanaAssistant
ga = GhanaAssistant()                 # downloads the model and data from Hugging Face
print(ga.ask("What is around Asafo Market?")["answer"])
```

Runtime pieces on Hugging Face (`ghananlpcommunity`): the model `ghana-assistant-flan-t5-small` and the data repo `ghana-assistant-data` (`maps/` city graphs, `knowledge/` FAISS index + sentences). Override with `GA_MODEL_REPO` / `GA_DATA_REPO`.

## Deploy

- **Inference on Modal:** `modal deploy deploy/modal_app.py` gives a `POST /ask` endpoint (`{"question": "..."}`). CPU-only; model and data are cached in a Modal volume. If the data repo is private, create a Modal secret `huggingface` with `HF_TOKEN` and deploy with `GA_USE_HF_SECRET=1`.
- **Frontend on Hugging Face Spaces:** `space/` is a Gradio app that is deliberately not a chat: a question box, a "generating" view, then the answer with a collapsible "how this was grounded" section and an "Ask another question" button. Set the Space secret `ASSISTANT_URL` to the Modal endpoint.

## Build the data and model

| Step | Script |
|---|---|
| City maps from OSM (+ optional Foursquare landmarks) | `scripts/build_maps.py ghana-latest.osm.pbf out/maps` |
| Navigation training data (scenarios, Gemini wording, fact-check) | [GhanaOpenAI/ghana-landmark-navigation](https://github.com/GhanaOpenAI/ghana-landmark-navigation), dataset `ghanaopenai/ghana-landmark-navigation` |
| Route plan ("skeleton") format | `training/prep_reasoning.py` |
| GhanaQA corpus, embeddings, retrieval, FAISS | `training/ghanaqa_build.py corpus \| qa \| embed \| retrieve \| index` |
| Train + evaluate the model (all four prefixes) | `training/t5_multi.py train …` / `eval …` |
| Package and publish | `scripts/build_knowledge.py`, `scripts/publish_hf.py model \| data \| space` |

## Data sources and licences
- Road graph and most landmarks: © OpenStreetMap contributors, ODbL. Extra landmarks: Foursquare OS Places, Apache-2.0.
- Knowledge: Ghanaian news (Citi News, MyJoyOnline, GhanaWeb) and KNUST research text; QA pairs from the GhanaQA project (CC-BY-NC-4.0). Check the rights to the source text before redistributing it.
- Code: MIT.

## Limitations
- Only Accra and Kumasi have maps. Directions are only as good as OpenStreetMap; the router ignores turn restrictions.
- Famous places with nicknames need an alias list (e.g. "KNUST").
- Knowledge answers can only be as good as what retrieval finds; the model can still word things loosely.
