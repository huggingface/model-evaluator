import os
from pathlib import Path

import streamlit as st
from datasets import get_dataset_config_names
from dotenv import load_dotenv
from huggingface_hub import list_datasets

from utils import get_compatible_models, get_metadata, http_get, http_post

if Path(".env").is_file():
    load_dotenv(".env")

HF_TOKEN = os.getenv("HF_TOKEN")
AUTOTRAIN_USERNAME = os.getenv("AUTOTRAIN_USERNAME")
AUTOTRAIN_BACKEND_API = os.getenv("AUTOTRAIN_BACKEND_API")
DATASETS_PREVIEW_API = os.getenv("DATASETS_PREVIEW_API")


TASK_TO_ID = {
    "binary_classification": 1,
    "multi_class_classification": 2,
    "multi_label_classification": 3,
    "entity_extraction": 4,
    "extractive_question_answering": 5,
    "translation": 6,
    "summarization": 8,
    "single_column_regression": 10,
}

AUTOTRAIN_TASK_TO_HUB_TASK = {
    "binary_classification": "text-classification",
    "multi_class_classification": "text-classification",
    "multi_label_classification": "text-classification",
    "entity_extraction": "token-classification",
    "extractive_question_answering": "question-answering",
    "translation": "translation",
    "summarization": "summarization",
    "single_column_regression": 10,
}

# TODO: remove this hardcorded logic and accept any dataset on the Hub
# DATASETS_TO_EVALUATE = ["emotion", "conll2003", "imdb", "squad", "xsum", "ncbi_disease", "go_emotions"]

###########
### APP ###
###########
st.title("Evaluation as a Service")
st.markdown(
    """
    Welcome to Hugging Face's Evaluation as a Service! This application allows
    you to evaluate any 🤗 Transformers model with a dataset on the Hub. Please
    select the dataset and configuration below. The results of your evaluation
    will be displayed on the public leaderboard
    [here](https://huggingface.co/spaces/huggingface/leaderboards).
    """
)

all_datasets = [d.id for d in list_datasets()]
selected_dataset = st.selectbox("Select a dataset", all_datasets)
print(f"Dataset name: {selected_dataset}")

# TODO: remove this step once we select real datasets
# Strip out original dataset name
# original_dataset_name = dataset_name.split("/")[-1].split("__")[-1]

# In general this will be a list of multiple configs => need to generalise logic here
metadata = get_metadata(selected_dataset)
print(metadata)
if metadata is None:
    st.warning("No evaluation metadata found. Please configure the evaluation job below.")

with st.expander("Advanced configuration"):
    ## Select task
    selected_task = st.selectbox("Select a task", list(AUTOTRAIN_TASK_TO_HUB_TASK.values()))
    ### Select config
    configs = get_dataset_config_names(selected_dataset)
    selected_config = st.selectbox("Select a config", configs)

    ## Select splits
    splits_resp = http_get(path="/splits", domain=DATASETS_PREVIEW_API, params={"dataset": selected_dataset})
    if splits_resp.status_code == 200:
        split_names = []
        all_splits = splits_resp.json()
        print(all_splits)
        for split in all_splits["splits"]:
            print(selected_config)
            if split["config"] == selected_config:
                split_names.append(split["split"])

        selected_split = st.selectbox("Select a split", split_names)  # , index=split_names.index(eval_split))

    ## Show columns
    rows_resp = http_get(
        path="/rows",
        domain="https://datasets-preview.huggingface.tech",
        params={"dataset": selected_dataset, "config": selected_config, "split": selected_split},
    ).json()
    columns = rows_resp["columns"]
    col_names = []
    for c in columns:
        col_names.append(c["column"]["name"])
    # splits = metadata[0]["splits"]
    # split_names = list(splits.values())
    # eval_split = splits.get("eval_split", split_names[0])

    # selected_split = st.selectbox("Select a split", split_names, index=split_names.index(eval_split))

    # TODO: add a function to handle the mapping task <--> column mapping
    # col_mapping = metadata[0]["col_mapping"]
    # col_names = list(col_mapping.keys())

    # TODO: figure out how to get all dataset column names (i.e. features) without download dataset itself
    st.markdown("**Map your data columns**")
    col1, col2 = st.columns(2)

    # TODO: find a better way to layout these items
    # TODO: propagate this information to payload
    # TODO: make it task specific
    with col1:
        st.markdown("`text` column")
        st.text("")
        st.text("")
        st.text("")
        st.text("")
        st.markdown("`target` column")
    with col2:
        st.selectbox("This column should contain the text you want to classify", col_names, index=0)
        st.selectbox("This column should contain the labels you want to assign to the text", col_names, index=1)

with st.form(key="form"):

    compatible_models = get_compatible_models(selected_task, selected_dataset)

    selected_models = st.multiselect(
        "Select the models you wish to evaluate", compatible_models
    )  # , compatible_models[0])
    print(selected_models)

    submit_button = st.form_submit_button("Make submission")

    # if submit_button:
    #     for model in selected_models:
    #         payload = {
    #             "username": AUTOTRAIN_USERNAME,
    #             "task": TASK_TO_ID[metadata[0]["task_id"]],
    #             "model": model,
    #             "col_mapping": metadata[0]["col_mapping"],
    #             "split": selected_split,
    #             "dataset": original_dataset_name,
    #             "config": selected_config,
    #         }
    # json_resp = http_post(
    #     path="/evaluate/create", payload=payload, token=HF_TOKEN, domain=AUTOTRAIN_BACKEND_API
    # ).json()
    # if json_resp["status"] == 1:
    #     st.success(f"✅ Successfully submitted model {model} for evaluation with job ID {json_resp['id']}")
    #     st.markdown(
    #         f"""
    #     Evaluation takes appoximately 1 hour to complete, so grab a ☕ or 🍵 while you wait:

    #     * 📊 Click [here](https://huggingface.co/spaces/huggingface/leaderboards) to view the results from your submission
    #     """
    #     )
    # else:
    #     st.error("🙈 Oh noes, there was an error submitting your submission!")

    # st.write("Creating project!")
    # payload = {
    #     "username": AUTOTRAIN_USERNAME,
    #     "proj_name": "my-eval-project-1",
    #     "task": TASK_TO_ID[metadata[0]["task_id"]],
    #     "config": {
    #         "language": "en",
    #         "max_models": 5,
    #         "instance": {
    #             "provider": "aws",
    #             "instance_type": "ml.g4dn.4xlarge",
    #             "max_runtime_seconds": 172800,
    #             "num_instances": 1,
    #             "disk_size_gb": 150,
    #         },
    #     },
    # }
    # json_resp = http_post(
    #     path="/projects/create", payload=payload, token=HF_TOKEN, domain=AUTOTRAIN_BACKEND_API
    # ).json()
    # # print(json_resp)

    # # st.write("Uploading data")
    # payload = {
    #     "split": 4,
    #     "col_mapping": metadata[0]["col_mapping"],
    #     "load_config": {"max_size_bytes": 0, "shuffle": False},
    # }
    # json_resp = http_post(
    #     path="/projects/522/data/emotion",
    #     payload=payload,
    #     token=HF_TOKEN,
    #     domain=AUTOTRAIN_BACKEND_API,
    #     params={"type": "dataset", "config_name": "default", "split_name": "train"},
    # ).json()
    # print(json_resp)

    # st.write("Training")
    # json_resp = http_get(
    #     path="/projects/522/data/start_process", token=HF_TOKEN, domain=AUTOTRAIN_BACKEND_API
    # ).json()
    # print(json_resp)
