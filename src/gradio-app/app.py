import gradio as gr
import pandas as pd
import tempfile
import ast
import sys
import os
from PIL import Image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from grewtse.pipeline import Grewtse

grewtse = Grewtse()
treebank_path = None

def parse_treebank(path: str, treebank_selection: str) -> pd.DataFrame:
    if treebank_selection == "None":
        successful_treebank_parse = grewtse.parse_treebank(path)
        treebank_path = path
    else:
        successful_treebank_parse = grewtse.parse_treebank(treebank_selection)
        treebank_path = treebank_selection

    print("changing treebank parse success")
    is_treebank_parse_success = True
    return grewtse.get_morphological_features().head()

def to_masked_dataset(query, node) -> pd.DataFrame:
    df = grewtse.generate_masked_dataset(query, node)
    return df

def safe_str_to_dict(s):
    try:
        return ast.literal_eval(s)
    except (ValueError, SyntaxError):
        return None  

def generate_minimal_pairs(query: str, node: str, alt_features: str):
    if not grewtse.is_treebank_loaded():
        raise ValueError("Please parse a treebank first.")

    # mask each sentence
    resulting_dataset = to_masked_dataset(query, node)

    # determine whether an alternative LI should be found
    alt_features_as_dict = safe_str_to_dict(alt_features)
    if alt_features_as_dict is not None:
        resulting_dataset = grewtse.generate_minimal_pairs(alt_features_as_dict, {})
    # resulting_dataset = grewtse.get_masked_dataset()
    print(resulting_dataset)

    # save to a temporary CSV file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
    resulting_dataset.to_csv(temp_file.name, index=False)
    return resulting_dataset, temp_file.name

def evaluate_model(model_repo: str, target_x_label: str, alt_x_label: str, x_axis_label: str, title: str):
    if not grewtse.are_minimal_pairs_generated():
        raise ValueError("Please parse a treebank, mask a dataset and generate minimal pairs first.")

    mp_with_eval_dataset = grewtse.evaluate_bert_mlm(model_repo)
    vis_filename = "vis.png"

    grewtse.visualise_syntactic_performance(vis_filename,
        mp_with_eval_dataset,
        target_x_label,
        alt_x_label,
        x_axis_label,
        "Confidence",
        title)

    # save to a temporary CSV file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
    mp_with_eval_dataset.to_csv(temp_file.name, index=False)
    return mp_with_eval_dataset, temp_file.name, vis_filename

def show_df():
    return gr.update(visible=True)

with gr.Blocks(theme=gr.themes.Ocean()) as demo:
    is_treebank_parse_success = False

    with gr.Row():
        gr.Markdown("# GREW-TSE: A Pipeline for Query-based Targeted Syntactic Evaluation")

    with gr.Row():
        with gr.Column():
            gr.Markdown("""
            #### Load a Treebank
            You can begin by loading up a particular treebank that you'd like to work with.<br>
            You can either select a treebank from the pre-loaded options below, or upload your own.<br>
            """)

        with gr.Column():
            with gr.Tabs():
                with gr.TabItem("Choose Treebank"):
                    treebank_selection = gr.Dropdown(
                        choices=["None", "spanish-test-sm.conllu", "polish-test-lg.conllu"],  
                        label="Select a treebank",
                        value="spanish-test-sm.conllu"  
                    )
            
                with gr.TabItem("Upload Your Own"):
                    gr.Markdown("## Upload a .conllu File")
                    file_input = gr.File(
                        label="Upload .conllu file",
                        file_types=[".conllu"],
                        type="filepath"
                    )
            parse_file_button = gr.Button("Parse Treebank", size='sm', scale=1)

    gr.Markdown("## Isolate A Syntactic Phenomenon")
    morph_table = gr.Dataframe(interactive=False, visible=False)

    parse_file_button.click(
        fn=parse_treebank,
        inputs=[file_input, treebank_selection],
        outputs=[morph_table]
    )
    parse_file_button.click(
        fn=show_df,
        outputs=morph_table
    )

    with gr.Row():
        with gr.Column():
            gr.Markdown("""
                **GREW (Graph Rewriting for Universal Dependencies)** is a query and transformation language used to search within and manipulate dependency treebanks. A GREW query allows linguists and NLP researchers to find specific syntactic patterns in parsed linguistic data (such as Universal Dependencies treebanks).
                Queries are expressed as graph constraints using a concise pattern-matching syntax.

                #### Example
                The following short GREW query will find target any verbs. Try it with one of the sample treebanks above.
                Make sure to include the variable V as the target that we're trying to isolate.

                ```grew
                V [upos=\"VERB\"];
                ```
            """)
        with gr.Column():
            query_input = gr.Textbox(label="GREW Query", lines=5, placeholder="Enter your GREW query here...", value="V [upos=\"VERB\"];")
            node_input = gr.Textbox(label="Node", placeholder="The variable in your GREW query to isolate, e.g., N", value="V")
            feature_input = gr.Textbox(
                label="Enter Alternative Feature Values for Minimal Pair as a Dictionary",
                placeholder='e.g. {"case": "Acc", "number": "Sing"}',
                value="{\"mood\": \"Sub\"}",
                lines=3
            )
            run_button = gr.Button("Run Query", size='sm', scale=3)

    output_table = gr.Dataframe(label="Output Table", visible=False)
    download_file = gr.File(label="Download CSV")
    run_button.click(
        fn=generate_minimal_pairs,
        inputs=[query_input, node_input, feature_input],
        outputs=[output_table, download_file]
    )
    run_button.click(
        fn=show_df,
        outputs=output_table
    )

    with gr.Row():
        with gr.Column():
            gr.Markdown("""
            ## Evaluate A Model
            You can evaluate any BERT for MLM model by providing the name of the model repository.
            """)
        with gr.Column():
            repository_input = gr.Textbox(label="Model Repository", lines=1, placeholder="Enter the model repository here...", value="dccuchile/distilbert-base-spanish-uncased")

    with gr.Row():
        with gr.Column():
            gr.Markdown("""
            ## Choose Visualisation Settings
            The results will be displayed as a visualisation which you can edit using the following settings.
            """)
        with gr.Column():
            target_x_label_textbox = gr.Textbox(label="Original Label Name i.e type of the 'right' token", lines=1, placeholder="Genitive Version")
            alt_x_label_textbox = gr.Textbox(label="Alternative Label Name i.e type of the 'wrong' token", lines=1, placeholder="Accusative Version")
            x_axis_label_textbox = gr.Textbox(label="X Axis Title i.e what features are you comparing?", lines=1, placeholder="Case of Nouns in Transitive Verbs")
            title_textbox = gr.Textbox(label="Visualisation Title", lines=1, placeholder="Syntactic Performance of BERT on English Transitive Noun Case")

            evaluate_button = gr.Button("Evaluate Model", size='sm', scale=3)

    mp_with_eval_output_dataset = gr.Dataframe(label="Output Table", visible=False)
    mp_with_eval_output_download = gr.File(label="Download CSV")
    visualisation_widget = gr.Image(type="pil", label="Loaded Image")

    evaluate_button.click(
        fn=evaluate_model,
        inputs=[repository_input, target_x_label_textbox, alt_x_label_textbox, x_axis_label_textbox, title_textbox],
        outputs=[mp_with_eval_output_dataset, mp_with_eval_output_download, visualisation_widget]
    )
    evaluate_button.click(
        fn=show_df,
        outputs=[mp_with_eval_output_dataset]
    )

if __name__ == "__main__":
    demo.launch(share=True)
