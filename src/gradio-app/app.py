import gradio as gr
import pandas as pd
import tempfile
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from treetse.pipeline import run_pipeline

def handle_file(file):
    if file is None:
        return "No file uploaded."

    with open(file.name, "r", encoding="utf-8") as f:
        content = f.read()

    # Return first 10 lines as a preview
    preview = "\n".join(content.splitlines()[:10])
    return f"File uploaded: {file.name}\n\nPreview:\n{preview}"

def process_grew_query(treebank: str, query: str, node: str, feature: str):
    config = {
        "grew_query": query,
        "grew_variable_for_masking": node,
        "treebank_filepath": treebank
    }
    masked_dataset = run_pipeline(config)

    # Save to a temporary CSV file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
    masked_dataset.to_csv(temp_file.name, index=False)
    return masked_dataset, temp_file.name

def run_query(query, node, feature, treebank_selection, upload_file_selection):
    if treebank_selection == "None":
        return pd.DataFrame(), "your-uploaded-file"
    else:
        df, file_path = process_grew_query(query, node, feature)
        return df, file_path

with gr.Blocks() as demo:
    gr.Image("logo.svg")
    gr.Markdown("# GREW-TSE: A Pipeline for Query-based Targeted Syntactic Evaluation")

    with gr.Row():
        query_input = gr.Textbox(label="GREW Query", lines=5, placeholder="Enter your GREW query here...")
        node_input = gr.Textbox(label="Node", placeholder="The variable in your grew query that you wish to isolate e.g., N")
        feature_input = gr.Textbox(label="Alternative Feature", placeholder="How do you want to alternate the above variable for the minimal pair e.g., Case=Acc")

    with gr.Row():
        # specify one of the available treebanks
        dropdown = gr.Dropdown(
            choices=["None", "spanish-test-sm.conllu", "polish-test-lg.conllu"],  
            label="Select a treebank",
            value="None"  
        )
        
        # alternatively, upload a treebank yourself
        gr.Markdown("## Upload a .conllu File")
        file_input = gr.File(
            label="Upload .conllu file",
            file_types=[".conllu"],
            type="filepath"
        )
        output = ""
        file_input.change(fn=handle_file, inputs=file_input, outputs=output)
        gr.Textbox(value=output)

    run_button = gr.Button("Run Query")

    output_table = gr.Dataframe(label="Output Table")
    download_file = gr.File(label="Download CSV")

    run_button.click(fn=run_query, inputs=[query_input, node_input, feature_input, dropdown], outputs=[output_table, download_file])

demo.launch()