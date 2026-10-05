# Everything in this repository, in the order it was used

Scripts in `data_pipelines/`, `training/` and `evaluation/` are the exact ones used to build the data and models. They are written to be run from the folder they live in (flat imports such as `from world import ...`), with their input files next to them; paths to the GPU server (`/mnt/volume_d2wey28/...`) appear in a few shell runners.

## 1. Navigation data (`data_pipelines/navigation/`)
| Step | Script | What it does |
|---|---|---|
| Landmarks (extra) | `fsq_ghana.py`, `fsq.py` | Pull Ghana rows from Foursquare OS Places; filter to landmark-like, open, de-duplicated places |
| Map | `world.py` | OSM extract -> drivable road graph + landmarks (OSM + Foursquare, `source` tagged); turn facts: counted turns, side roads, landmarks at each turn |
| Scenarios | `tasks.py`, `build_scenarios.py` | 8 task types sampled from the graph; held-out `test_area` / `test_pair` splits |
| Text | `generate_queries.py` | Gemini writes 5 user requests + answers per scenario from the facts (plain English, no GPS, no distances) |
| Checks | `factcheck.py`, `clean_leaks.py` | Reject answers that contradict the facts; re-queue scenarios with leaked style text |
| Release | `build_hf.py` | Hugging Face dataset `ghanaopenai/ghana-landmark-navigation` |
| Runners | `run_all.sh`, `run_gen.sh`, `gen_*.sh`, `expand.sh` | How the above were run (32,000 scenarios per city) |

## 2. Route plan ("skeleton") and Qwen experiments (`training/qwen_and_bakeoffs/`)
| Script | Purpose |
|---|---|
| `prep_data.py`, `train.py`, `eval.py`, `bakeoff.sh` | First bake-off (memorise-the-streets setup): Qwen3.5 0.8B/2B/4B, Gemma-3-1B, Gemma-4-E4B, MiniCPM5-1B |
| `prep_reasoning.py` | Route plan as the model's input/thinking |
| `train_unsloth.py`, `eval_reason.py`, `small_router.sh`, `chain2.sh` | Qwen3.8-27B and Qwen3.5 0.8B-9B with Unsloth; router mode vs model mode |
| `chain3.sh` | Started (then stopped) full-data Qwen3.5-0.8B run |

## 3. Tiny models (`training/tiny_models/`)
`train_small.py`, `eval_small.py`, `small_bakeoff.sh`: Falcon-H1-Tiny 100M, SmolLM2-135M, Flan-T5-small, Flan-T5-base in router mode. Flan-T5-small was chosen.

## 4. Knowledge data (`data_pipelines/ghanaqa/`, `data_pipelines/ghok/`)
- `ghanaqa_build.py corpus | qa | embed | retrieve | index`: news + KNUST sentences, GhanaQA English QA pairs (batch 1 and batch 2 with source articles), bge-small embeddings (fast parallel path), top-k retrieval, FAISS IVF-PQ index.
- `ghok_build.py`: earlier exploration of the GHOK parallel corpus (not used for training).

## 5. The final model (`training/t5_multi.py`)
One Flan-T5-small, four prefixes (`intent:`, `parse:`, `directions:`, `answer:`); trains on navigation + GhanaQA + intent + parse examples and evaluates all four.

## 6. Translation pilots (`data_pipelines/translation_pilots/`)
Twi/Ewe pilots (not used, English only for now): NLLB-3.3B, Gemini 3.5/3.8 Flash, Google Translate (pilot only); `score_translation.py` scores them via back-translation.

## 7. Evaluation and benchmarks (`evaluation/`)
`score_answers.py` (route-plan coverage, order, extra names, Gemini judge), `factcheck.py`, `gpu_bench.py`, `cpu_bench.py`.

## 8. Experiments (`experiments/`)
Gemini filling in landmarks from its own knowledge, and the Gemini + Google Maps grounding probes.
