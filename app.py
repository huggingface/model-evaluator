import streamlit as st
from huggingface_hub import DatasetFilter, HfApi, ModelFilter

api = HfApi()


def get_metadata(dataset_name):
    filt = DatasetFilter(dataset_name=dataset_name)
    data = api.list_datasets(filter=filt, full=True)
    return data[0].cardData["train-eval-index"]


def get_compatible_models(task, dataset_name):
    filt = ModelFilter(task=task, trained_dataset=dataset_name)
    compatible_models = api.list_models(filter=filt)
    return [model.modelId for model in compatible_models]


with st.form(key="form"):

    dataset_name = st.selectbox("Select a dataset to evaluate on", ["lewtun/autoevaluate_emotion"])

    metadata = get_metadata(dataset_name)
    # st.write(metadata)

    dataset_config = st.selectbox("Select the subset to evaluate on", [metadata[0]["config"]])

    splits = metadata[0]["splits"]

    # st.write(splits)

    evaluation_split = st.selectbox("Select the split to evaluate on", [v for d in splits for k, v in d.items()])

    compatible_models = get_compatible_models(metadata[0]["task"], dataset_name.split("/")[-1].split("_")[-1])

    options = st.multiselect("Select the models you wish to evaluate", compatible_models, compatible_models[0])

    submit_button = st.form_submit_button("Make Submission")

    if submit_button:
        st.success(f"✅ Evaluation was successfully submitted for evaluation with job ID 42")
