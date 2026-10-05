"""The one small model, used through four prefixes: intent / parse / directions / answer."""
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


class AssistantModel:
    def __init__(self, repo, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(repo)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(repo).to(self.device).eval()

    def _gen(self, text, max_new_tokens=200, **kw):
        enc = self.tok(text, return_tensors="pt", truncation=True, max_length=768).to(self.device)
        with torch.no_grad(): out = self.model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, **kw)
        return self.tok.decode(out[0], skip_special_tokens=True).strip()

    def intent(self, message):
        """-> "navigation" or "knowledge" """
        return self._gen("intent: " + message, 4).lower()

    def parse(self, message):
        """navigation message -> fields, e.g. {"task": "avoid", "start": "...", "end": "...", "avoid": "..."}"""
        out = self._gen("parse: " + message, 64)
        return dict(x.split(": ", 1) for x in out.split(" | ") if ": " in x)

    def directions(self, message, skeleton):
        return self._gen(f"directions: Request: {message}\nRoute plan:\n{skeleton}")

    def answer(self, question, contexts):
        # beam search + no repeated word pairs + a mild repetition penalty: fixes the repetition seen with greedy decoding on open questions
        # (only here: directions legitimately repeat phrases like "on your left", so they keep plain greedy decoding)
        return self._gen("answer: Question: " + question + "\nContext:\n" + "\n".join(f"{i+1}. {c}" for i, c in enumerate(contexts)),
                         120, num_beams=4, no_repeat_ngram_size=2, repetition_penalty=1.3, early_stopping=True)
