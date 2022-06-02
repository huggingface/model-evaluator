import inspect
import os
import uuid
from pathlib import Path

import pandas as pd
import streamlit as st
from datasets import get_dataset_config_names
from dotenv import load_dotenv
from evaluate import load
from huggingface_hub import list_datasets, list_metrics
from tqdm import tqdm

from evaluation import filter_evaluated_models
from utils import (
    format_col_mapping,
    get_compatible_models,
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


TASK_TO_ID = {
    "binary_classification": 1,
    "multi_class_classification": 2,
    # "multi_label_classification": 3, # Not fully supported in AutoTrain
    "entity_extraction": 4,
    "extractive_question_answering": 5,
    "translation": 6,
    "summarization": 8,
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
    "extractive_question_answering": [],
    "translation": ["sacrebleu"],
    "summarization": ["rouge1", "rouge2", "rougeL", "rougeLsum"],
}

SUPPORTED_TASKS = list(TASK_TO_ID.keys())


@st.cache
def get_supported_metrics():
    metrics = list_metrics()
    supported_metrics = []
    for metric in tqdm(metrics):
        try:
            metric_func = load(metric)
        except Exception as e:
            print(e)
            print("Skipping the following metric, which cannot load:", metric)
            continue

        argspec = inspect.getfullargspec(metric_func.compute)
        if "references" in argspec.kwonlyargs and "predictions" in argspec.kwonlyargs:
            # We require that "references" and "predictions" are arguments
            # to the metric function. We also require that the other arguments
            # besides "references" and "predictions" have defaults and so do not
            # need to be specified explicitly.
            defaults = True
            for key, value in argspec.kwonlydefaults.items():
                if key not in ("references", "predictions"):
                    if value is None:
                        defaults = False
                        break

            if defaults:
                supported_metrics.append(metric)
    return supported_metrics


supported_metrics = get_supported_metrics()


#######
# APP #
#######
st.title("Evaluation as a Service")
st.markdown(
    """
    Welcome to Hugging Face's Evaluation as a Service! This application allows
    you to evaluate 🤗 Transformers
    [models](https://huggingface.co/models?library=transformers&sort=downloads)
    with a dataset on the Hub. Please select the dataset and configuration
    below. The results of your evaluation will be displayed on the [public
    leaderboard](https://huggingface.co/spaces/autoevaluate/leaderboards).
    """
)

all_datasets = [d.id for d in list_datasets()]
query_params = st.experimental_get_query_params()
default_dataset = all_datasets[0]
if "dataset" in query_params:
    if len(query_params["dataset"]) > 0 and query_params["dataset"][0] in all_datasets:
        default_dataset = query_params["dataset"][0]

selected_dataset = st.selectbox("Select a dataset", all_datasets, index=all_datasets.index(default_dataset))
st.experimental_set_query_params(**{"dataset": [selected_dataset]})


metadata = get_metadata(selected_dataset)
print(metadata)
if metadata is None:
    st.warning("No evaluation metadata found. Please configure the evaluation job below.")

with st.expander("Advanced configuration"):
    # Select task
    selected_task = st.selectbox(
        "Select a task",
        SUPPORTED_TASKS,
        index=SUPPORTED_TASKS.index(metadata[0]["task_id"]) if metadata is not None else 0,
    )
    # Select config
    configs = get_dataset_config_names(selected_dataset)
    selected_config = st.selectbox("Select a config", configs)

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

    st.markdown("**Map your data columns**")
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
                "This column should contain the text you want to classify",
                col_names,
                index=col_names.index(get_key(metadata[0]["col_mapping"], "text")) if metadata is not None else 0,
            )
            target_col = st.selectbox(
                "This column should contain the labels you want to assign to the text",
                col_names,
                index=col_names.index(get_key(metadata[0]["col_mapping"], "target")) if metadata is not None else 0,
            )
            col_mapping[text_col] = "text"
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
                "This column should contain the array of tokens",
                col_names,
                index=col_names.index(get_key(metadata[0]["col_mapping"], "tokens")) if metadata is not None else 0,
            )
            tags_col = st.selectbox(
                "This column should contain the labels to associate to each part of the text",
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
                "This column should contain the text you want to translate",
                col_names,
                index=col_names.index(get_key(metadata[0]["col_mapping"], "source")) if metadata is not None else 0,
            )
            target_col = st.selectbox(
                "This column should contain an example translation of the source text",
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
                "This column should contain the text you want to summarize",
                col_names,
                index=col_names.index(get_key(metadata[0]["col_mapping"], "text")) if metadata is not None else 0,
            )
            target_col = st.selectbox(
                "This column should contain an example summarization of the text",
                col_names,
                index=col_names.index(get_key(metadata[0]["col_mapping"], "target")) if metadata is not None else 0,
            )
            col_mapping[text_col] = "text"
            col_mapping[target_col] = "target"

    elif selected_task == "extractive_question_answering":
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
                "This column should contain the indices in the context of the first character of each answers.text",
                col_names,
                index=col_names.index(get_key(col_mapping, "answers.answer_start")) if metadata is not None else 0,
            )
            col_mapping[context_col] = "context"
            col_mapping[question_col] = "question"
            col_mapping[answers_text_col] = "answers.text"
            col_mapping[answers_start_col] = "answers.answer_start"

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
        list(set(supported_metrics) - set(TASK_TO_DEFAULT_METRICS[selected_task])),
    )
    st.info(
        """"Note: user-selected metrics will be run with their default arguments. \
            Check out the [available metrics](https://huggingface.co/metrics) for more details."""
    )

with st.form(key="form"):

    compatible_models = get_compatible_models(selected_task, selected_dataset)
    selected_models = st.multiselect("Select the models you wish to evaluate", compatible_models)
    print("Selected models:", selected_models)

    if len(selected_models) > 0:
        selected_models = filter_evaluated_models(
            selected_models,
            selected_task,
            selected_dataset,
            selected_config,
            selected_split,
        )
        print("Selected models:", selected_models)

    submit_button = st.form_submit_button("Make submission")

    if submit_button:
        if len(selected_models) > 0:
            project_id = str(uuid.uuid4())[:8]
            payload = {
                "username": AUTOTRAIN_USERNAME,
                "proj_name": f"eval-project-{project_id}",
                "task": TASK_TO_ID[selected_task],
                "config": {
                    "language": "en"
                    if selected_task != "translation"
                    else "en2de",  # Need this dummy pair to enable translation
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
                    },
                },
            }
            print(f"Payload: {payload}")
            project_json_resp = http_post(
                path="/projects/create",
                payload=payload,
                token=HF_TOKEN,
                domain=AUTOTRAIN_BACKEND_API,
            ).json()
            print(project_json_resp)

            if project_json_resp["created"]:
                payload = {
                    "split": 4,  # use "auto" split choice in AutoTrain
                    "col_mapping": col_mapping,
                    "load_config": {"max_size_bytes": 0, "shuffle": False},
                }
                data_json_resp = http_post(
                    path=f"/projects/{project_json_resp['id']}/data/{selected_dataset}",
                    payload=payload,
                    token=HF_TOKEN,
                    domain=AUTOTRAIN_BACKEND_API,
                    params={
                        "type": "dataset",
                        "config_name": selected_config,
                        "split_name": selected_split,
                    },
                ).json()
                print(data_json_resp)
                if data_json_resp["download_status"] == 1:
                    train_json_resp = http_get(
                        path=f"/projects/{project_json_resp['id']}/data/start_process",
                        token=HF_TOKEN,
                        domain=AUTOTRAIN_BACKEND_API,
                    ).json()
                    print(train_json_resp)
                    if train_json_resp["success"]:
                        st.success(f"✅ Successfully submitted evaluation job with project ID {project_id}")
                        st.markdown(
                            f"""
                        Evaluation takes appoximately 1 hour to complete, so grab a ☕ or 🍵 while you wait:

                        📊 Click [here](https://hf.co/spaces/autoevaluate/leaderboards?dataset={selected_dataset}) \
                            to view the results from your submission
                        """
                        )
                    else:
                        st.error("🙈 Oh no, there was an error submitting your evaluation job!")
        else:
            st.warning("⚠️ No models were selected for evaluation!")
