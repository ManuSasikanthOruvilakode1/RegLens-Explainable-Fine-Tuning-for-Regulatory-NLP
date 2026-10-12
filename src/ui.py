"""Local web demo of the trained models (Gradio).

    uv sync --extra deeplearning --extra demo
    uv run python src/ui.py                 # then open http://127.0.0.1:7860

Requires the trained models in models/ (see scripts/run_transformer_suite.py).
"""

import argparse
import os
import sys

os.environ.setdefault("NLP_TRACKING", "0")
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import gradio as gr

from src.inference import DomainClassifier, EntityTagger, OnnxDomainClassifier, Summariser

CLASSIFY_EXAMPLES = [
    "Can my boss make me work on Anzac Day without extra pay?",
    "I sold my investment flat after two years, how much of the profit is taxed?",
    "Our chairman wants the company to keep borrowing even though we cannot pay suppliers.",
    "Do I need to register for GST if my side business earns sixty thousand a year?",
]
NER_EXAMPLES = [
    "The Fair Work Ombudsman took action under the Fair Work Act 2009 against a franchise operator.",
    "ASIC and APRA released joint guidance on superannuation trustee obligations.",
    "The ATO reminded employers that payroll tax and fringe benefits tax are separate obligations.",
]
SUMMARY_EXAMPLE = (
    "The small business restructuring process allows an eligible company to propose a plan to its creditors while "
    "the directors remain in control of the business. A small business restructuring practitioner helps the company "
    "develop the plan, which creditors vote on within fifteen business days. The company must have liabilities of "
    "less than one million dollars and must have paid employee entitlements that are due."
)
ENTITY_COLOURS = {"GOV_BODY": "#2a78d6", "REGULATORY_ACT": "#eb6834", "FINANCIAL_TERM": "#1baf7a", "ORG": "#6250d6"}

INTRO = """
# Australian Regulatory NLP

Models fine-tuned on Australian tax, workplace-relations and corporate-governance text.
Every number quoted here comes from held-out evaluation; see the project README for confidence intervals.
*Educational demo, not legal or tax advice.*
"""


def build_app(device: str = "auto") -> gr.Blocks:
    import torch
    dev = torch.device("cpu") if device == "cpu" or not torch.cuda.is_available() else torch.device("cuda")
    print(f"Loading models on {dev} ...")
    classifier = DomainClassifier(device=dev)
    tagger = EntityTagger(device=dev)
    summariser = Summariser(device=dev)
    onnx_clf = OnnxDomainClassifier()
    print("Models loaded.")

    def classify(text):
        if not text.strip():
            return {}, []
        return classifier.predict(text), classifier.explain(text)

    def entities(text):
        if not text.strip():
            return []
        out = []
        for chunk, etype in tagger.spans(text):
            if out and chunk not in ".,;:!?)":
                out.append((" ", None))
            out.append((chunk, etype))
        return out

    def summarise(text):
        if not text.strip():
            return "", ""
        out = summariser.summarise(text)
        return out["zero_shot"], out["fine_tuned"]

    def fast(text):
        if not text.strip():
            return {}, ""
        probs, ms = onnx_clf.predict(text)
        return probs, f"**{ms:.1f} ms** on CPU (ONNX Runtime, INT8, single sentence)"

    with gr.Blocks(title="Australian Regulatory NLP") as app:
        gr.Markdown(INTRO)
        with gr.Tab("Classify"):
            gr.Markdown("Which regulatory domain does a question belong to, and which words drove the decision? "
                        "Highlighting uses Integrated Gradients: the deeper the red, the more a word pushed the model towards its prediction; blue words pushed away. "
                        "Model: BERT + LoRA (0.5% of parameters trained).")
            with gr.Row():
                with gr.Column():
                    c_in = gr.Textbox(label="Question or statement", lines=3, placeholder=CLASSIFY_EXAMPLES[0])
                    c_btn = gr.Button("Classify", variant="primary")
                    gr.Examples(CLASSIFY_EXAMPLES, inputs=c_in)
                with gr.Column():
                    c_label = gr.Label(label="Domain", num_top_classes=3)
                    c_expl = gr.HighlightedText(label="Why: word attributions", show_legend=False)
            c_btn.click(classify, c_in, [c_label, c_expl])
            c_in.submit(classify, c_in, [c_label, c_expl])

        with gr.Tab("Entities"):
            gr.Markdown("Named entity recognition for government bodies, legislation, financial terms and organisations. "
                        "Model: RoBERTa + LoRA, strict span-F1 0.71 on held-out sentences.")
            n_in = gr.Textbox(label="Sentence", lines=2, placeholder=NER_EXAMPLES[0])
            n_btn = gr.Button("Find entities", variant="primary")
            n_out = gr.HighlightedText(label="Entities", color_map=ENTITY_COLOURS, show_legend=True)
            gr.Examples(NER_EXAMPLES, inputs=n_in)
            n_btn.click(entities, n_in, n_out)
            n_in.submit(entities, n_in, n_out)

        with gr.Tab("Summarise"):
            gr.Markdown("One-sentence summaries from flan-t5-base, before and after LoRA fine-tuning on 22 examples "
                        "(ROUGE-L 0.23 -> 0.38 on held-out passages).")
            s_in = gr.Textbox(label="Regulatory passage", lines=6, value=SUMMARY_EXAMPLE)
            s_btn = gr.Button("Summarise", variant="primary")
            with gr.Row():
                s_zero = gr.Textbox(label="Zero-shot (base model)", lines=3)
                s_tuned = gr.Textbox(label="Fine-tuned (LoRA)", lines=3)
            s_btn.click(summarise, s_in, [s_zero, s_tuned])

        with gr.Tab("Fast CPU mode"):
            gr.Markdown("The classifier exported to ONNX and quantised to INT8: 4x smaller (105 MB) and about 2.9x faster "
                        "on CPU, agreeing with the full-precision model on 94% of evaluation sentences.")
            f_in = gr.Textbox(label="Question or statement", lines=2, placeholder=CLASSIFY_EXAMPLES[1])
            f_btn = gr.Button("Classify on CPU", variant="primary")
            with gr.Row():
                f_label = gr.Label(label="Domain", num_top_classes=3)
                f_time = gr.Markdown()
            gr.Examples(CLASSIFY_EXAMPLES, inputs=f_in)
            f_btn.click(fast, f_in, [f_label, f_time])
            f_in.submit(fast, f_in, [f_label, f_time])
    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--device", choices=["auto", "cpu"], default="auto")
    args = parser.parse_args()
    build_app(args.device).launch(server_name="127.0.0.1", server_port=args.port)


if __name__ == "__main__":
    main()
