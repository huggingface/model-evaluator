import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from utils import get_compatible_models, get_metadata, http_post

if Path(".env").is_file():
    load_dotenv(".env")

HF_TOKEN = os.getenv("HF_TOKEN")
AUTOTRAIN_USERNAME = os.getenv("AUTOTRAIN_USERNAME")
AUTOTRAIN_BACKEND_API = os.getenv("AUTOTRAIN_BACKEND_API")


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

# TODO: remove this hardcorded logic and accept any dataset on the Hub
DATASETS_TO_EVALUATE = ["emotion", "conll2003", "imdb", "squad", "xsum", "ncbi_disease", "go_emotions"]

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

dataset_name = st.selectbox("Select a dataset", [f"lewtun/autoevaluate__{dset}" for dset in DATASETS_TO_EVALUATE])

with st.form(key="form"):

    # TODO: remove this step once we select real datasets
    # Strip out original dataset name
    original_dataset_name = dataset_name.split("/")[-1].split("__")[-1]

    # In general this will be a list of multiple configs => need to generalise logic here
    metadata = get_metadata(dataset_name)

    dataset_config = st.selectbox("Select a config", [metadata[0]["config"]])

    splits = metadata[0]["splits"]
    split_names = list(splits.values())
    eval_split = splits.get("eval_split", split_names[0])

    selected_split = st.selectbox("Select a split", split_names, index=split_names.index(eval_split))

    # TODO: add a function to handle the mapping task <--> column mapping
    col_mapping = metadata[0]["col_mapping"]
    col_names = list(col_mapping.keys())

    # TODO: figure out how to get all dataset column names (i.e. features) without download dataset itself
    st.markdown("**Map your data columns**")
    col1, col2 = st.columns(2)

    # TODO: find a better way to layout these items
    # TODO: propagate this information to payload
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

    compatible_models = get_compatible_models(metadata[0]["task"], original_dataset_name)

    selected_models = st.multiselect("Select the models you wish to evaluate", compatible_models, compatible_models[0])

    submit_button = st.form_submit_button("Make submission")

    if submit_button:
        for model in selected_models:
            payload = {
                "username": AUTOTRAIN_USERNAME,
                "task": TASK_TO_ID[metadata[0]["task_id"]],
                "model": model,
                "col_mapping": metadata[0]["col_mapping"],
                "split": selected_split,
                "dataset": original_dataset_name,
                "config": dataset_config,
            }
            json_resp = http_post(
                path="/evaluate/create", payload=payload, token=HF_TOKEN, domain=AUTOTRAIN_BACKEND_API
            ).json()
            if json_resp["status"] == 1:
                st.success(f"✅ Successfully submitted model {model} for evaluation with job ID {json_resp['id']}")
                st.markdown(
                    f"""
                Evaluation takes appoximately 1 hour to complete, so grab a ☕ or 🍵 while you wait:

                * 📊 Click [here](https://huggingface.co/spaces/huggingface/leaderboards) to view the results from your submission
                * 💾 Click [here](https://huggingface.co/datasets/autoevaluate/eval-staging-{json_resp['id']}) to view the stored predictions on the Hugging Face Hub
                """
                )
            else:
                st.error("🙈 Oh noes, there was an error submitting your submission!")
