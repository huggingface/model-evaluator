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


with st.form(key="form"):

    dataset_name = st.selectbox("Select a dataset to evaluate on", ["lewtun/autoevaluate__emotion"])

    # TODO: remove this step once we select real datasets
    # Strip out original dataset name
    original_dataset_name = dataset_name.split("/")[-1].split("__")[-1]

    # In general this will be a list of multiple configs => need to generalise logic here
    metadata = get_metadata(dataset_name)

    dataset_config = st.selectbox("Select the subset to evaluate on", [metadata[0]["config"]])

    splits = metadata[0]["splits"]
    split_names = list(splits.values())
    eval_split = splits.get("eval_split", split_names[0])

    selected_split = st.selectbox("Select the split to evaluate on", split_names, index=split_names.index(eval_split))

    compatible_models = get_compatible_models(metadata[0]["task"], original_dataset_name)

    selected_models = st.multiselect("Select the models you wish to evaluate", compatible_models, compatible_models[0])

    submit_button = st.form_submit_button("Make Submission")

    if submit_button:
        for model in selected_models:
            payload = {
                "username": AUTOTRAIN_USERNAME,
                "task": 1,
                "model": model,
                "col_mapping": {"sentence": "text", "label": "target"},
                "split": selected_split,
                "dataset": original_dataset_name,
                "config": dataset_config,
            }
            json_resp = http_post(
                path="/evaluate/create", payload=payload, token=HF_TOKEN, domain=AUTOTRAIN_BACKEND_API
            ).json()

            st.success(f"✅ Successfully submitted model {model} for evaluation with job ID {json_resp['id']}")
