# Ghana Assistant

A small, fast assistant for Ghana built around one tiny model (**Flan-T5-small, 77M parameters**) that never has to *remember* facts: the facts come from a map or from retrieved text, and the model only understands the request and writes the reply.

| Skill | Example | Where the facts come from |
|---|---|---|
| **Directions** (Accra, Kumasi) | *How do I get from Kejetia Market to Bantama Market, avoiding Okomfo Anokye Road?* | Shortest path on an OpenStreetMap road graph, turned into a route plan with counted turns and landmarks at each turn |
| **Ghana knowledge** | *Why is bird flu a concern for Ghana's poultry industry?* | Sentences from Ghanaian news and KNUST research that share the question's noun phrases (dated, newest preferred on ties) |

Answers are single-turn: one question, one answer (no conversation memory).

**Try it:** web app on [Hugging Face Spaces](https://huggingface.co/spaces/ghananlpcommunity/ghana-assistant) · API `https://michsethowusuwfp--ghana-assistant-assistant-web.modal.run/ask` · model [ghananlpcommunity/ghana-assistant-flan-t5-small](https://huggingface.co/ghananlpcommunity/ghana-assistant-flan-t5-small) · data [ghananlpcommunity/ghana-assistant-data](https://huggingface.co/datasets/ghananlpcommunity/ghana-assistant-data)

## How it works

```
message ─► model "intent:" ─┬─ navigation ─► model "parse:"  → {task, start, end, avoid, asked}
                            │                 → match names to landmarks (rapidfuzz) → route on the road graph
                            │                 → route plan ─► model "directions:" ─► spoken directions
                            └─ knowledge  ─► spaCy noun phrases → phrase index (memory-mapped) → up to 600 tokens of dated sentences ─► model "answer:" ─► short answer
```

One model, four prefixes:

| Prefix | Input | Output |
|---|---|---|
| `intent:` | the user's message | `navigation` or `knowledge` |
| `parse:` | a navigation message | `task: avoid \| start: … \| end: … \| avoid: …` |
| `directions:` | message + route plan | spoken, landmark-based directions |
| `answer:` | question + retrieved dated sentences (`[2025-10] …`) | a short answer |

Unknown places are reported ("I don't know where X is…") instead of being swapped for a lookalike.

## Use it

Everything (model, city maps, knowledge store) downloads from Hugging Face on first use (about 2 GB). Runs on CPU: about 2-4 s for directions, 1-2 s for knowledge answers.

**1. Web interface + API on your machine**
```bash
pip install "ghana-assistant[serve] @ git+https://github.com/GhanaNLP/ghana-assistant"
ghana-assistant serve --port 8000
```
- Web interface: http://localhost:8000 (a question box, then the answer; not a chat)
- API: `POST /ask` and interactive docs at http://localhost:8000/docs
```bash
curl -X POST http://localhost:8000/ask -H "Content-Type: application/json" \
     -d '{"question": "How do I get from Kejetia Market to Bantama Market?"}'
# {"skill": "navigation", "city": "kumasi", "answer": "Start by heading north-west on ...", "plan": "...", "fields": {...}}
```

**2. Command line**
```bash
ghana-assistant "What is around Asafo Market?" --debug
```

**3. Python**
```python
from ghana_assistant import GhanaAssistant
ga = GhanaAssistant()
print(ga.ask("What does the Ghana Cocoa Board do?")["answer"])
```

Responses: `skill` (`navigation` / `knowledge`), `answer`, and how it was grounded: `plan` (the route plan from the map) or `sources` (the dated sentences used).

## Deploy

The same server (`ghana_assistant/server.py`: web interface at `/`, API at `/ask`, docs at `/docs`) runs locally, on Modal and behind the Space.

- **Modal (CPU):**
  ```bash
  modal run deploy/modal_app.py::download   # once: cache the model and data in a Modal volume
  modal deploy deploy/modal_app.py          # prints https://<workspace>--ghana-assistant-assistant-web.modal.run
  ```
  Cold start is about 30 s, then a few seconds per answer; at most 3 containers. CORS is open so browser pages can call it.
- **Hugging Face Space (static HTML):** `python scripts/build_space.py <modal-url> --publish` puts the package's web page (`ghana_assistant/web/index.html`) on `ghananlpcommunity/ghana-assistant`, pointed at the API.

## Build the data and model

All data-processing, training, evaluation and experiment code is in this repository; see **[docs/PIPELINE.md](docs/PIPELINE.md)** for every script in the order it was used.

| Step | Script |
|---|---|
| City maps from OSM (+ optional Foursquare landmarks) | `scripts/build_maps.py ghana-latest.osm.pbf out/maps` |
| Navigation training data (scenarios, Gemini wording, fact-check) | `data_pipelines/navigation/` (released as dataset `ghanaopenai/ghana-landmark-navigation`) |
| Route plan ("skeleton") format | `training/qwen_and_bakeoffs/prep_reasoning.py` |
| GhanaQA corpus, noun-phrase index, dates, context selection | `data_pipelines/ghanaqa/ghanaqa_build.py corpus \| qa \| nouns \| meta \| select` (embedding + FAISS stages kept for comparison) |
| Train + evaluate the model (all four prefixes) | `training/t5_multi.py train …` / `eval …` |
| Package and publish | `scripts/build_knowledge.py`, `scripts/publish_hf.py model \| data \| space` |

## Data sources and licences
- Road graph and most landmarks: © OpenStreetMap contributors, ODbL. Extra landmarks: Foursquare OS Places, Apache-2.0.
- Knowledge: Ghanaian news (Citi News, MyJoyOnline, GhanaWeb) and KNUST research text; QA pairs from the GhanaQA project (CC-BY-NC-4.0). Check the rights to the source text before redistributing it.
- Code: MIT.

## Limitations
- Only Accra and Kumasi have maps. Directions are only as good as OpenStreetMap; the router ignores turn restrictions.
- Famous places with nicknames need an alias list (e.g. "KNUST").
- Knowledge answers are weak: often not supported by the retrieved sentences and sometimes wrong. Treat them as unverified.
