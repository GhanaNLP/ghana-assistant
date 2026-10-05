"""Ghana Assistant - Hugging Face Space frontend. Not a chat: one question in, one answer out, then ask another.
Set the Space secret ASSISTANT_URL to the Modal /ask endpoint."""
import os, requests, gradio as gr

URL = os.environ.get("ASSISTANT_URL", "")
EXAMPLES = ["How do I get from Kejetia Market to Komfo Anokye Teaching Hospital?",
            "I am at Accra Mall, how do I get to Kwame Nkrumah Circle without using the motorway?",
            "What is around Asafo Market?",
            "Why is bird flu a concern for Ghana's poultry industry?",
            "What does the Ghana Cocoa Board do?"]
CSS = """
#wrap {max-width: 760px; margin: 0 auto;}
#answer {font-size: 1.15rem; line-height: 1.6; padding: 18px 20px; border-radius: 12px; border: 1px solid var(--border-color-primary);}
#q-echo {opacity: .75; font-style: italic;}
.badge {display:inline-block; padding: 2px 10px; border-radius: 999px; font-size: .8rem; background: var(--color-accent-soft);}
"""


def ask(q):
    q = (q or "").strip()
    if not q: raise gr.Error("Please type a question first.")
    try:
        r = requests.post(URL, json={"question": q}, timeout=90); r.raise_for_status(); out = r.json()
    except Exception as e:
        raise gr.Error(f"The assistant is not reachable right now ({type(e).__name__}). Please try again.")
    if "error" in out: raise gr.Error(out["error"])
    skill = "Directions" if out.get("skill") == "navigation" else "Ghana knowledge"
    extra = ""
    if out.get("plan"): extra = "**Route plan used** (from the map):\n```\n" + out["plan"] + "\n```"
    if out.get("sources"): extra = "**Sentences used:**\n" + "\n".join(f"- {s['text']}" for s in out["sources"])
    return (gr.update(visible=False), gr.update(visible=True),
            f'<span class="badge">{skill}</span>', f"> {q}", out.get("answer", ""), extra)


def generating():
    return gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)


def reset():
    return gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), ""


with gr.Blocks(css=CSS, title="Ghana Assistant", theme=gr.themes.Soft()) as demo:
    with gr.Column(elem_id="wrap"):
        gr.Markdown("# Ghana Assistant\nAsk for **directions in Accra or Kumasi** (using landmarks), or a **question about Ghana** (news and research). "
                    "Each question is answered on its own; there is no conversation memory.")
        with gr.Column(visible=True) as ask_view:
            q = gr.Textbox(label="Your question", placeholder="e.g. How do I get from Kejetia Market to KNUST?", lines=3, max_lines=6)
            go = gr.Button("Ask", variant="primary")
            gr.Examples(EXAMPLES, inputs=q)
        with gr.Column(visible=False) as wait_view:
            gr.Markdown("### Generating your answer…\nFinding the facts first (a route on the map, or matching news and research), then writing the reply.")
        with gr.Column(visible=False) as answer_view:
            badge = gr.HTML(); echo = gr.Markdown(elem_id="q-echo"); ans = gr.Markdown(elem_id="answer")
            with gr.Accordion("How this answer was grounded", open=False): details = gr.Markdown()
            again = gr.Button("Ask another question", variant="primary")
    go.click(generating, None, [ask_view, wait_view, answer_view]).then(ask, q, [wait_view, answer_view, badge, echo, ans, details])
    q.submit(generating, None, [ask_view, wait_view, answer_view]).then(ask, q, [wait_view, answer_view, badge, echo, ans, details])
    again.click(reset, None, [ask_view, wait_view, answer_view, q])

if __name__ == "__main__":
    demo.launch()
