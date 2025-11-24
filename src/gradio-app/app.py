import gradio as gr
import pandas as pd
import tempfile
import ast
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from grewtse.pipeline import GrewTSEPipe
from grewtse.evaluators import GrewTSEvaluator
from grewtse.visualise import GrewTSEVisualiser

grewtse = GrewTSEPipe()


def parse_treebank(path: str, treebank_selection: str) -> pd.DataFrame:
    if treebank_selection == "None":
        parsed_treebank = grewtse.parse_treebank(path)
        # treebank_path = path
    else:
        treebank_selection = f"./datasets/{treebank_selection}"
        parsed_treebank = grewtse.parse_treebank(treebank_selection)
        # treebank_path = treebank_selection

    return grewtse.get_morphological_features().tail()


def to_masked_dataset(query, node) -> pd.DataFrame:
    df = grewtse.generate_masked_dataset(query, node)
    return df


def to_prompt_dataset(query, node) -> pd.DataFrame:
    df = grewtse.generate_prompt_dataset(query, node)
    return df


def safe_str_to_dict(s):
    try:
        return ast.literal_eval(s)
    except (ValueError, SyntaxError):
        return None


def truncate_text(text, max_len=50):
    """
    Truncate a string to max_len characters and append '...' if it was longer.
    """
    if not isinstance(text, str):
        return text  # Keep non-string values unchanged
    return text[:max_len] + "..." if len(text) > max_len else text


def generate_minimal_pairs(query: str, node: str, alt_features: str, task_type: str):
    if not grewtse.is_treebank_parsed():
        raise ValueError("Please parse a treebank first.")

    # determine whether an alternative LI should be found
    alt_features_as_dict = safe_str_to_dict(alt_features)
    if alt_features_as_dict is None:
        raise Exception("Invalid features provided.")

    has_leading_whitespace = False
    is_encoder = False
    masked_or_prompt_dataset = None
    if task_type.lower() == "masked":
        # mask the target word in the sentence
        masked_or_prompt_df = to_masked_dataset(query, node)
        has_leading_whitespace = False
        is_encoder = True
    elif task_type.lower() == "prompt":
        # create prompts from each sentence (i.e. cut them off right at the target word)
        masked_or_prompt_dataset = to_prompt_dataset(query, node)
        has_leading_whitespace = True
    else:
        raise Exception("Invalid task type.")

    full_dataset = grewtse.generate_minimal_pair_dataset(
        alt_features_as_dict,
        {},
        ood_pairs=None,
        has_leading_whitespace=has_leading_whitespace,
    )

    # save to a temporary CSV file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
    full_dataset.to_csv(temp_file.name, index=False)

    if is_encoder:
        dataset_for_vis = full_dataset[
            ["masked_text", "form_grammatical", "form_ungrammatical"]
        ]
        dataset_for_vis["masked_text"] = dataset_for_vis["masked_text"].apply(
            truncate_text
        )
    else:
        dataset_for_vis = full_dataset[
            ["prompt_text", "form_grammatical", "form_ungrammatical"]
        ]
        dataset_for_vis["prompt_text"] = dataset_for_vis["prompt_text"].apply(
            truncate_text
        )

    num_exceptions = grewtse.get_num_exceptions()
    num_targets_parsed = len(masked_or_prompt_df)
    num_success = len(full_dataset)

    exceptions_info = f"{num_targets_parsed+num_exceptions} targets identified and turned into masks/prompts. {num_exceptions} of these could not be used due to treebank structure issues. After searching for minimal pairs, a total of <br>{num_success} minimal-pair syntactic tests</br> were successfully generated."
    gr.Info(exceptions_info, duration=60, title="Grew-TSE Results")

    return dataset_for_vis, temp_file.name


def evaluate_model(model_repo: str, task_type: str):
    if not grewtse.are_minimal_pairs_generated():
        raise ValueError(
            "Please parse a treebank, mask a dataset and generate minimal pairs first."
        )

    g_eval = GrewTSEvaluator()
    g_vis = GrewTSEVisualiser()

    model_type = "encoder" if task_type.lower() == "masked" else "decoder"
    mp_with_eval_dataset = g_eval.evaluate_model(
        grewtse.get_minimal_pair_dataset(), model_repo, model_type
    )
    metrics = g_eval.get_all_metrics()
    metrics = pd.DataFrame(metrics.items(), columns=["Metric", "Value"])
    print("===METRICS===")
    print(metrics)
    print("----")

    # save to a temporary CSV file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
    mp_with_eval_dataset.to_csv(temp_file.name, index=False)
    return metrics, temp_file.name


def show_df():
    return gr.update(visible=True)


with gr.Blocks(theme=gr.themes.Ocean()) as demo:
    is_treebank_parse_success = False

    with gr.Row():
        gr.Markdown(
            "# GREW-TSE: A Pipeline for Query-based Targeted Syntactic Evaluation"
        )

    with gr.Row():
        with gr.Column():
            gr.Markdown(
                """
            #### Load a Treebank
            You can begin by loading up a particular treebank that you'd like to work with.<br>
            You can either select a treebank from the pre-loaded options below, or upload your own.<br>
            """
            )

        with gr.Column():
            with gr.Tabs():
                with gr.TabItem("Choose Treebank"):
                    treebank_selection = gr.Dropdown(
                        choices=[
                            "None",
                            "en/English-EWT-UD-Treebank.conllu",
                            "Polish-Test-Treebank.conllu",
                            "Spanish-Test-SM.conllu",
                        ],
                        label="Select a treebank",
                        value="en/English-EWT-UD-Treebank.conllu",
                    )

                with gr.TabItem("Upload Your Own"):
                    gr.Markdown("## Upload a .conllu File")
                    file_input = gr.File(
                        label="Upload .conllu file",
                        file_types=[".conllu"],
                        type="filepath",
                    )
            parse_file_button = gr.Button("Parse Treebank", size="sm", scale=0)

            morph_table = gr.Dataframe(interactive=False, visible=False)

    parse_file_button.click(
        fn=parse_treebank,
        inputs=[file_input, treebank_selection],
        outputs=[morph_table],
    )
    parse_file_button.click(fn=show_df, outputs=morph_table)

    gr.Markdown("## Isolate A Syntactic Phenomenon")

    with gr.Row():
        with gr.Column():
            gr.Markdown(
                """
                **GREW (Graph Rewriting for Universal Dependencies)** is a query and transformation language used to search within and manipulate dependency treebanks. A GREW query allows linguists and NLP researchers to find specific syntactic patterns in parsed linguistic data (such as Universal Dependencies treebanks).
                Queries are expressed as graph constraints using a concise pattern-matching syntax.

                #### Example
                The following short GREW query will find target any verbs. Try it with one of the sample treebanks above.
                Make sure to include the variable V as the target that we're trying to isolate.

                ```grew
                pattern {
                    V [upos=\"VERB\"];
                }
                ```
            """
            )
        with gr.Column():
            query_input = gr.Textbox(
                label="GREW Query",
                lines=5,
                placeholder="Enter your GREW query here...",
                value="pattern { V [upos=VERB, Number=Sing]; }",
            )
            node_input = gr.Textbox(
                label="Target",
                placeholder="The variable in your GREW query to isolate, e.g., N",
                value="V",
            )
            feature_input = gr.Textbox(
                label="Enter Alternative Feature Values for Minimal Pair as a Dictionary",
                placeholder='e.g. {"case": "Acc", "number": "Sing"}',
                value='{"number": "Plur"}',
                lines=3,
            )
            task_type = gr.Dropdown(
                choices=[
                    "Masked",
                    "Prompt",
                ],
                label="Select whether you want masked- or prompt-based tests.",
                value="Masked",
            )
            run_button = gr.Button("Run Query", size="sm", scale=0)

            output_table = gr.Dataframe(label="Output Table", visible=False)
            download_file = gr.File(label="Download CSV")
    run_button.click(
        fn=generate_minimal_pairs,
        inputs=[query_input, node_input, feature_input, task_type],
        outputs=[output_table, download_file],
    )
    run_button.click(fn=show_df, outputs=output_table)

    with gr.Row():
        with gr.Column():
            gr.Markdown(
                """
            ## Evaluate A Model (BETA Version)
            You can evaluate models trained either for MLM or NTP tasks that are available on the Hugging Face platform.
            """
            )
        with gr.Column():
            repository_input = gr.Textbox(
                label="Model Repository",
                lines=1,
                placeholder="Enter the model repository here...",
                value="google-bert/bert-base-multilingual-cased",
            )

            with gr.Column():
                evaluate_button = gr.Button("Evaluate Model", size="sm", scale=0)

            mp_with_eval_output_dataset = gr.Dataframe(
                label="Output Table", visible=False
            )
            mp_with_eval_output_download = gr.File(label="Download CSV")

            evaluate_button.click(
                fn=evaluate_model,
                inputs=[repository_input, task_type],
                outputs=[
                    gr.DataFrame(),
                    mp_with_eval_output_download,
                ],
            )

if __name__ == "__main__":
    demo.launch(share=True)
