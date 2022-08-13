import os
from pathlib import Path

import pandas as pd
import streamlit as st
import yaml
from datasets import get_dataset_config_names
from dotenv import load_dotenv
from huggingface_hub import list_datasets

from evaluation import filter_evaluated_models
from utils import (
    AUTOTRAIN_TASK_TO_HUB_TASK,
    commit_evaluation_log,
    create_autotrain_project_name,
    format_col_mapping,
    get_compatible_models,
    get_dataset_card_url,
    get_key,
    get_metadata,
    http_get,
    http_post,
)

if Path(".env").is_file():
    load_dotenv(".env")

HF_TOKEN = os.getenv("HF_TOKEN")
AUTOTRAIN_USERNAME = os.getenv("AUTOTRAIN_USERNAME")
AUTOTRAIN_BACKEND_API = os.getenv("AUTOTRAIN_BACKEND_API")
DATASETS_PREVIEW_API = os.getenv("DATASETS_PREVIEW_API")

# Put image tasks on top
TASK_TO_ID = {
    "image_binary_classification": 17,
    "image_multi_class_classification": 18,
    "binary_classification": 1,
    "multi_class_classification": 2,
    "entity_extraction": 4,
    "extractive_question_answering": 5,
    "translation": 6,
    "summarization": 8,
    "zero_shot_classification": 22,
}

TASK_TO_DEFAULT_METRICS = {
    "binary_classification": ["f1", "precision", "recall", "auc", "accuracy"],
    "multi_class_classification": [
        "f1",
        "precision",
        "recall",
        "accuracy",
    ],
    "entity_extraction": ["precision", "recall", "f1", "accuracy"],
    "extractive_question_answering": ["f1", "exact_match"],
    "translation": ["sacrebleu"],
    "summarization": ["rouge1", "rouge2", "rougeL", "rougeLsum"],
    "image_binary_classification": ["f1", "precision", "recall", "auc", "accuracy"],
    "image_multi_class_classification": [
        "f1",
        "precision",
        "recall",
        "accuracy",
    ],
    "zero_shot_classification": ["accuracy", "loss"]
}

AUTOTRAIN_TASK_TO_LANG = {
    "translation": "en2de",
    "image_binary_classification": "unk",
    "image_multi_class_classification": "unk",
}


SUPPORTED_TASKS = list(TASK_TO_ID.keys())
UNSUPPORTED_TASKS = []

# Extracted from utils.get_supported_metrics
# Hardcoded for now due to speed / caching constraints
SUPPORTED_METRICS = [
    "accuracy",
    "bertscore",
    "bleu",
    "cer",
    "chrf",
    "code_eval",
    "comet",
    "competition_math",
    "coval",
    "cuad",
    "exact_match",
    "f1",
    "frugalscore",
    "google_bleu",
    "mae",
    "mahalanobis",
    "matthews_correlation",
    "mean_iou",
    "meteor",
    "mse",
    "pearsonr",
    "perplexity",
    "precision",
    "recall",
    "roc_auc",
    "rouge",
    "sacrebleu",
    "sari",
    "seqeval",
    "spearmanr",
    "squad",
    "squad_v2",
    "ter",
    "trec_eval",
    "wer",
    "wiki_split",
    "xnli",
    "angelina-wang/directional_bias_amplification",
    "jordyvl/ece",
    "lvwerra/ai4code",
    "lvwerra/amex",
    "lvwerra/test",
    "lvwerra/test_metric",
]


#######
# APP #
#######
st.title("Evaluation on the Hub")
st.markdown(
    """
    Welcome to Hugging Face's automatic model evaluator 👋!

    This application allows you to evaluate 🤗 Transformers
    [models](https://huggingface.co/models?library=transformers&sort=downloads)
    across a wide variety of [datasets](https://huggingface.co/datasets) on the
    Hub. Please select the dataset and configuration below. The results of your
    evaluation will be displayed on the [public
    leaderboards](https://huggingface.co/spaces/autoevaluate/leaderboards). For
    more details, check out out our [blog
    post](https://huggingface.co/blog/eval-on-the-hub).
    """
)

all_datasets = [d.id for d in list_datasets()]
query_params = st.experimental_get_query_params()
if "first_query_params" not in st.session_state:
    st.session_state.first_query_params = query_params
first_query_params = st.session_state.first_query_params
default_dataset = all_datasets[0]
if "dataset" in first_query_params:
    if len(first_query_params["dataset"]) > 0 and first_query_params["dataset"][0] in all_datasets:
        default_dataset = first_query_params["dataset"][0]

selected_dataset = st.selectbox(
    "Select a dataset",
    all_datasets,
    index=all_datasets.index(default_dataset),
    help="""Datasets with metadata can be evaluated with 1-click. Configure an evaluation job to add \
        new metadata to a dataset card.""",
)
st.experimental_set_query_params(**{"dataset": [selected_dataset]})

# Check if selected dataset can be streamed
is_valid_dataset = http_get(
    path="/is-valid",
    domain=DATASETS_PREVIEW_API,
    params={"dataset": selected_dataset},
).json()
if is_valid_dataset["valid"] is False:
    st.error(
        """The dataset you selected is not currently supported. Open a \
            [discussion](https://huggingface.co/spaces/autoevaluate/model-evaluator/discussions) for support."""
    )

metadata = get_metadata(selected_dataset, token=HF_TOKEN)
print(f"INFO -- Dataset metadata: {metadata}")
if metadata is None:
    st.warning("No evaluation metadata found. Please configure the evaluation job below.")

with st.expander("Advanced configuration"):
    # Select task
    # Hack to filter for unsupported tasks
    # TODO(lewtun): remove this once we have SQuAD metrics support
    if metadata is not None and metadata[0]["task_id"] in UNSUPPORTED_TASKS:
        metadata = None
    selected_task = st.selectbox(
        "Select a task",
        SUPPORTED_TASKS,
        index=SUPPORTED_TASKS.index(metadata[0]["task_id"]) if metadata is not None else 0,
        help="""Don't see your favourite task here? Open a \
            [discussion](https://huggingface.co/spaces/autoevaluate/model-evaluator/discussions) to request it!""",
    )
    # Select config
    configs = get_dataset_config_names(selected_dataset)
    selected_config = st.selectbox(
        "Select a config",
        configs,
        help="""Some datasets contain several sub-datasets, known as _configurations_. \
            Select one to evaluate your models on. \
            See the [docs](https://huggingface.co/docs/datasets/master/en/load_hub#configurations) for more details.
            """,
    )

    # Select splits
    splits_resp = http_get(
        path="/splits",
        domain=DATASETS_PREVIEW_API,
        params={"dataset": selected_dataset},
    )
    if splits_resp.status_code == 200:
        split_names = []
        all_splits = splits_resp.json()
        for split in all_splits["splits"]:
            if split["config"] == selected_config:
                split_names.append(split["split"])

        if metadata is not None:
            eval_split = metadata[0]["splits"].get("eval_split", None)
        else:
            eval_split = None
        selected_split = st.selectbox(
            "Select a split",
            split_names,
            index=split_names.index(eval_split) if eval_split is not None else 0,
            help="Be wary when evaluating models on the `train` split.",
        )

    # Select columns
    rows_resp = http_get(
        path="/rows",
        domain=DATASETS_PREVIEW_API,
        params={
            "dataset": selected_dataset,
            "config": selected_config,
            "split": selected_split,
        },
    ).json()
    col_names = list(pd.json_normalize(rows_resp["rows"][0]["row"]).columns)

    st.markdown("**Map your dataset columns**")
    st.markdown(
        """The model evaluator uses a standardised set of column names for the input examples and labels. \
        Please define the mapping between your dataset columns (right) and the standardised column names (left)."""
    )
    col1, col2 = st.columns(2)

    # TODO: find a better way to layout these items
    # TODO: need graceful way of handling dataset <--> task mismatch for datasets with metadata
    col_mapping = {}
    if selected_task in ["binary_classification", "multi_class_classification"]:
        with col1:
            st.markdown("`text` column")
            st.text("")
            st.text("")
            st.text("")
            st.text("")
            st.markdown("`target` column")
        with col2:
            text_col = st.selectbox(
                "This column should contain the text to be classified",
                col_names,
                index=col_names.index(get_key(metadata[0]["col_mapping"], "text")) if metadata is not None else 0,
            )
            target_col = st.selectbox(
                "This column should contain the labels associated with the text",
                col_names,
                index=col_names.index(get_key(metadata[0]["col_mapping"], "target")) if metadata is not None else 0,
            )
            col_mapping[text_col] = "text"
            col_mapping[target_col] = "target"

    elif selected_task == "zero_shot_classification":
        with col1:
            st.markdown("`text` column")
            st.text("")
            st.text("")
            st.text("")
            st.text("")
            st.markdown("`classes` column")
            st.text("")
            st.text("")
            st.text("")
            st.text("")
            st.markdown("`target` column")
        with col2:
            text_col = st.selectbox(
                "This column should contain the text to be classified",
                col_names,
                index=col_names.index(get_key(metadata[0]["col_mapping"], "text")) if metadata is not None else 0,
            )
            classes_col = st.selectbox(
                "This column should contain the classes associated with the text",
                col_names,
                index=col_names.index(get_key(metadata[0]["col_mapping"], "classes")) if metadata is not None else 0,
            )
            target_col = st.selectbox(
                "This column should contain the index of the correct class",
                col_names,
                index=col_names.index(get_key(metadata[0]["col_mapping"], "target")) if metadata is not None else 0,
            )
            col_mapping[text_col] = "text"
            col_mapping[classes_col] = "classes"
            col_mapping[target_col] = "target"

    elif selected_task == "entity_extraction":
        with col1:
            st.markdown("`tokens` column")
            st.text("")
            st.text("")
            st.text("")
            st.text("")
            st.markdown("`tags` column")
        with col2:
            tokens_col = st.selectbox(
                "This column should contain the array of tokens to be classified",
                col_names,
                index=col_names.index(get_key(metadata[0]["col_mapping"], "tokens")) if metadata is not None else 0,
            )
            tags_col = st.selectbox(
                "This column should contain the labels associated with each part of the text",
                col_names,
                index=col_names.index(get_key(metadata[0]["col_mapping"], "tags")) if metadata is not None else 0,
            )
            col_mapping[tokens_col] = "tokens"
            col_mapping[tags_col] = "tags"

    elif selected_task == "translation":
        with col1:
            st.markdown("`source` column")
            st.text("")
            st.text("")
            st.text("")
            st.text("")
            st.markdown("`target` column")
        with col2:
            text_col = st.selectbox(
                "This column should contain the text to be translated",
                col_names,
                index=col_names.index(get_key(metadata[0]["col_mapping"], "source")) if metadata is not None else 0,
            )
            target_col = st.selectbox(
                "This column should contain the target translation",
                col_names,
                index=col_names.index(get_key(metadata[0]["col_mapping"], "target")) if metadata is not None else 0,
            )
            col_mapping[text_col] = "source"
            col_mapping[target_col] = "target"

    elif selected_task == "summarization":
        with col1:
            st.markdown("`text` column")
            st.text("")
            st.text("")
            st.text("")
            st.text("")
            st.markdown("`target` column")
        with col2:
            text_col = st.selectbox(
                "This column should contain the text to be summarized",
                col_names,
                index=col_names.index(get_key(metadata[0]["col_mapping"], "text")) if metadata is not None else 0,
            )
            target_col = st.selectbox(
                "This column should contain the target summary",
                col_names,
                index=col_names.index(get_key(metadata[0]["col_mapping"], "target")) if metadata is not None else 0,
            )
            col_mapping[text_col] = "text"
            col_mapping[target_col] = "target"

    elif selected_task == "extractive_question_answering":
        if metadata is not None:
            col_mapping = metadata[0]["col_mapping"]
            # Hub YAML parser converts periods to hyphens, so we remap them here
            col_mapping = format_col_mapping(col_mapping)
        with col1:
            st.markdown("`context` column")
            st.text("")
            st.text("")
            st.text("")
            st.text("")
            st.markdown("`question` column")
            st.text("")
            st.text("")
            st.text("")
            st.text("")
            st.markdown("`answers.text` column")
            st.text("")
            st.text("")
            st.text("")
            st.text("")
            st.markdown("`answers.answer_start` column")
        with col2:
            context_col = st.selectbox(
                "This column should contain the question's context",
                col_names,
                index=col_names.index(get_key(col_mapping, "context")) if metadata is not None else 0,
            )
            question_col = st.selectbox(
                "This column should contain the question to be answered, given the context",
                col_names,
                index=col_names.index(get_key(col_mapping, "question")) if metadata is not None else 0,
            )
            answers_text_col = st.selectbox(
                "This column should contain example answers to the question, extracted from the context",
                col_names,
                index=col_names.index(get_key(col_mapping, "answers.text")) if metadata is not None else 0,
            )
            answers_start_col = st.selectbox(
                "This column should contain the indices in the context of the first character of each `answers.text`",
                col_names,
                index=col_names.index(get_key(col_mapping, "answers.answer_start")) if metadata is not None else 0,
            )
            col_mapping[context_col] = "context"
            col_mapping[question_col] = "question"
            col_mapping[answers_text_col] = "answers.text"
            col_mapping[answers_start_col] = "answers.answer_start"
    elif selected_task in ["image_binary_classification", "image_multi_class_classification"]:
        with col1:
            st.markdown("`image` column")
            st.text("")
            st.text("")
            st.text("")
            st.text("")
            st.markdown("`target` column")
        with col2:
            image_col = st.selectbox(
                "This column should contain the images to be classified",
                col_names,
                index=col_names.index(get_key(metadata[0]["col_mapping"], "image")) if metadata is not None else 0,
            )
            target_col = st.selectbox(
                "This column should contain the labels associated with the images",
                col_names,
                index=col_names.index(get_key(metadata[0]["col_mapping"], "target")) if metadata is not None else 0,
            )
            col_mapping[image_col] = "image"
            col_mapping[target_col] = "target"

    # Select metrics
    st.markdown("**Select metrics**")
    st.markdown("The following metrics will be computed")
    html_string = " ".join(
        [
            '<div style="padding-right:5px;padding-left:5px;padding-top:5px;padding-bottom:5px;float:left">'
            + '<div style="background-color:#D3D3D3;border-radius:5px;display:inline-block;padding-right:5px;'
            + 'padding-left:5px;color:white">'
            + metric
            + "</div></div>"
            for metric in TASK_TO_DEFAULT_METRICS[selected_task]
        ]
    )
    st.markdown(html_string, unsafe_allow_html=True)
    selected_metrics = st.multiselect(
        "(Optional) Select additional metrics",
        sorted(list(set(SUPPORTED_METRICS) - set(TASK_TO_DEFAULT_METRICS[selected_task]))),
        help="""User-selected metrics will be computed with their default arguments. \
            For example, `f1` will report results for binary labels. \
            Check out the [available metrics](https://huggingface.co/metrics) for more details.""",
    )

with st.form(key="form"):
    compatible_models = get_compatible_models(selected_task, [selected_dataset])
    selected_models = st.multiselect(
        "Select the models you wish to evaluate",
        compatible_models,
        help="""Don't see your favourite model in this list? Add the dataset and task it was trained on to the \
            [model card metadata.](https://huggingface.co/docs/hub/models-cards#model-card-metadata)""",
    )
    print("INFO -- Selected models before filter:", selected_models)

    hf_username = st.text_input("Enter your 🤗 Hub username to be notified when the evaluation is finished")

    submit_button = st.form_submit_button("Evaluate models 🚀")

    if submit_button:
        if len(hf_username) == 0:
            st.warning("No 🤗 Hub username provided! Please enter your username and try again.")
        elif len(selected_models) == 0:
            st.warning("⚠️ No models were selected for evaluation! Please select at least one model and try again.")
        elif len(selected_models) > 10:
            st.warning("Only 10 models can be evaluated at once. Please select fewer models and try again.")
        else:
            # Filter out previously evaluated models
            selected_models = filter_evaluated_models(
                selected_models,
                selected_task,
                selected_dataset,
                selected_config,
                selected_split,
                selected_metrics,
            )
            print("INFO -- Selected models after filter:", selected_models)
            if len(selected_models) > 0:
                project_payload = {
                    "username": AUTOTRAIN_USERNAME,
                    "proj_name": create_autotrain_project_name(selected_dataset),
                    "task": TASK_TO_ID[selected_task],
                    "config": {
                        "language": AUTOTRAIN_TASK_TO_LANG[selected_task]
                        if selected_task in AUTOTRAIN_TASK_TO_LANG
                        else "en",
                        "max_models": 5,
                        "instance": {
                            "provider": "aws",
                            "instance_type": "ml.g4dn.4xlarge",
                            "max_runtime_seconds": 172800,
                            "num_instances": 1,
                            "disk_size_gb": 150,
                        },
                        "evaluation": {
                            "metrics": selected_metrics,
                            "models": selected_models,
                            "hf_username": hf_username,
                        },
                    },
                }
                print(f"INFO -- Payload: {project_payload}")
                project_json_resp = http_post(
                    path="/projects/create",
                    payload=project_payload,
                    token=HF_TOKEN,
                    domain=AUTOTRAIN_BACKEND_API,
                ).json()
                print(f"INFO -- Project creation response: {project_json_resp}")

                if project_json_resp["created"]:
                    data_payload = {
                        "split": 4,  # use "auto" split choice in AutoTrain
                        "col_mapping": col_mapping,
                        "load_config": {"max_size_bytes": 0, "shuffle": False},
                    }
                    data_json_resp = http_post(
                        path=f"/projects/{project_json_resp['id']}/data/{selected_dataset}",
                        payload=data_payload,
                        token=HF_TOKEN,
                        domain=AUTOTRAIN_BACKEND_API,
                        params={
                            "type": "dataset",
                            "config_name": selected_config,
                            "split_name": selected_split,
                        },
                    ).json()
                    print(f"INFO -- Dataset creation response: {data_json_resp}")
                    if data_json_resp["download_status"] == 1:
                        train_json_resp = http_get(
                            path=f"/projects/{project_json_resp['id']}/data/start_process",
                            token=HF_TOKEN,
                            domain=AUTOTRAIN_BACKEND_API,
                        ).json()
                        print(f"INFO -- AutoTrain job response: {train_json_resp}")
                        if train_json_resp["success"]:
                            train_eval_index = {
                                "train-eval-index": [
                                    {
                                        "config": selected_config,
                                        "task": AUTOTRAIN_TASK_TO_HUB_TASK[selected_task],
                                        "task_id": selected_task,
                                        "splits": {"eval_split": selected_split},
                                        "col_mapping": col_mapping,
                                    }
                                ]
                            }
                            selected_metadata = yaml.dump(train_eval_index, sort_keys=False)
                            dataset_card_url = get_dataset_card_url(selected_dataset)
                            st.success("✅ Successfully submitted evaluation job!")
                            st.markdown(
                                f"""
                            Evaluation can take up to 1 hour to complete, so grab a ☕️ or 🍵 while you wait:

                            * 🔔 A [Hub pull request](https://huggingface.co/docs/hub/repositories-pull-requests-discussions) with the evaluation results will be opened for each model you selected. Check your email for notifications.
                            * 📊 Click [here](https://hf.co/spaces/autoevaluate/leaderboards?dataset={selected_dataset}) to view the results from your submission once the Hub pull request is merged.
                            * 🥱 Tired of configuring evaluations? Add the following metadata to the [dataset card]({dataset_card_url}) to enable 1-click evaluations:
                            """  # noqa
                            )
                            st.markdown(
                                f"""
                            ```yaml
                            {selected_metadata}
                            """
                            )
                            print("INFO -- Pushing evaluation job logs to the Hub")
                            evaluation_log = {}
                            evaluation_log["payload"] = project_payload
                            evaluation_log["project_creation_response"] = project_json_resp
                            evaluation_log["dataset_creation_response"] = data_json_resp
                            evaluation_log["autotrain_job_response"] = train_json_resp
                            commit_evaluation_log(evaluation_log, hf_access_token=HF_TOKEN)
                        else:
                            st.error("🙈 Oh no, there was an error submitting your evaluation job!")
            else:
                st.warning("⚠️ No models left to evaluate! Please select other models and try again.")
