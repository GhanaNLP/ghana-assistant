"""Where the runtime pieces live on Hugging Face. Override with environment variables or GhanaAssistant(...) arguments."""
import os

MODEL_REPO = os.environ.get("GA_MODEL_REPO", "ghananlpcommunity/ghana-assistant-flan-t5-small")   # the trained model (all skills)
DATA_REPO = os.environ.get("GA_DATA_REPO", "ghananlpcommunity/ghana-assistant-data")              # city maps, FAISS index, sentence store
CITIES = ("accra", "kumasi")
HF_TOKEN = os.environ.get("HF_TOKEN")                                                        # only needed if the data repo is private
