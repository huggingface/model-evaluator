import inspect
import uuid
from typing import Dict, List, Union

import jsonlines
import requests
import streamlit as st
from evaluate import load
from huggingface_hub import HfApi, ModelFilter, Repository, dataset_info, list_metrics, create_repo
from tqdm import tqdm

AUTOTRAIN_TASK_TO_HUB_TASK = {
    "binary_classification": "text-classification",
    "multi_class_classification": "text-classification",
    "natural_language_inference": "text-classification",
    "entity_extraction": "token-classification",
    "extractive_question_answering": "question-answering",
    "translation": "translation",
    "summarization": "summarization",
    "image_binary_classification": "image-classification",
    "image_multi_class_classification": "image-classification",
    "text_zero_shot_classification": "text-generation",
}


HUB_TASK_TO_AUTOTRAIN_TASK = {v: k for k, v in AUTOTRAIN_TASK_TO_HUB_TASK.items()}
LOGS_REPO = "evaluation-job-logs"


def get_auth_headers(token: str, prefix: str = "Bearer"):
    return {"Authorization": f"{prefix} {token}"}


def http_post(path: str, token: str, payload=None, domain: str = None, params=None) -> requests.Response:
    """HTTP POST request to the AutoNLP API, raises UnreachableAPIError if the API cannot be reached"""
    try:
        response = requests.post(
            url=domain + path,
            json=payload,
            headers=get_auth_headers(token=token),
            allow_redirects=True,
            params=params,
        )
    except requests.exceptions.ConnectionError:
        print("❌ Failed to reach AutoNLP API, check your internet connection")
    response.raise_for_status()
    return response


def http_get(path: str, domain: str, token: str = None, params: dict = None) -> requests.Response:
    """HTTP POST request to `path`, raises UnreachableAPIError if the API cannot be reached"""
    try:
        response = requests.get(
            url=domain + path,
            headers=get_auth_headers(token=token),
            allow_redirects=True,
            params=params,
        )
    except requests.exceptions.ConnectionError:
        print(f"❌ Failed to reach {path}, check your internet connection")
    response.raise_for_status()
    return response


def get_metadata(dataset_name: str, token: str) -> Union[Dict, None]:
    data = dataset_info(dataset_name, token=token)
    if data.cardData is not None and "train-eval-index" in data.cardData.keys():
        return data.cardData["train-eval-index"]
    else:
        return None


def get_compatible_models(task: str, dataset_ids: List[str]) -> List[str]:
    """
    Returns all model IDs that are compatible with the given task and dataset names.

    Args:
        task (`str`): The task to search for.
        dataset_names (`List[str]`): A list of dataset names to search for.

    Returns:
        A list of model IDs, sorted alphabetically.
    """
    compatible_models = []
    # Allow any summarization model to be used for summarization tasks
    # and allow any text-generation model to be used for text_zero_shot_classification
    if task in ("summarization", "text_zero_shot_classification"):
        model_filter = ModelFilter(
            task=AUTOTRAIN_TASK_TO_HUB_TASK[task],
            library=["transformers", "pytorch"],
        )
        compatible_models.extend(HfApi().list_models(filter=model_filter))
    # Include models trained on SQuAD datasets, since these can be evaluated on
    # other SQuAD-like datasets
    if task == "extractive_question_answering":
        dataset_ids.extend(["squad", "squad_v2"])

    # TODO: relax filter on PyTorch models if TensorFlow supported in AutoTrain
    for dataset_id in dataset_ids:
        model_filter = ModelFilter(
            task=AUTOTRAIN_TASK_TO_HUB_TASK[task],
            trained_dataset=dataset_id,
            library=["transformers", "pytorch"],
        )
        compatible_models.extend(HfApi().list_models(filter=model_filter))
    return sorted(set([model.modelId for model in compatible_models]))


def get_key(col_mapping, val):
    for key, value in col_mapping.items():
        if val == value:
            return key

    return "key doesn't exist"


def format_col_mapping(col_mapping: dict) -> dict:
    for k, v in col_mapping["answers"].items():
        col_mapping[f"answers.{k}"] = f"answers.{v}"
    del col_mapping["answers"]
    return col_mapping


def commit_evaluation_log(evaluation_log, hf_access_token=None):
    create_repo(
        repo_id=f"autoevaluate/{LOGS_REPO}",
        repo_type="dataset",
        exists_ok=True,
        private=True,
        token=hf_access_token,
    )
    logs_repo_url = f"https://huggingface.co/datasets/autoevaluate/{LOGS_REPO}"
    logs_repo = Repository(
        local_dir=LOGS_REPO,
        clone_from=logs_repo_url,
        repo_type="dataset",
        use_auth_token=hf_access_token,
    )
    logs_repo.git_pull()
    with jsonlines.open(f"{LOGS_REPO}/logs.jsonl") as r:
        lines = []
        for obj in r:
            lines.append(obj)

    lines.append(evaluation_log)
    with jsonlines.open(f"{LOGS_REPO}/logs.jsonl", mode="w") as writer:
        for job in lines:
            writer.write(job)
    logs_repo.push_to_hub(
        commit_message=f"Evaluation submitted with project name {evaluation_log['payload']['proj_name']}"
    )
    print("INFO -- Pushed evaluation logs to the Hub")


@st.experimental_memo
def get_supported_metrics():
    """Helper function to get all metrics compatible with evaluation service.

    Requires all metric dependencies installed in the same environment, so wait until
    https://github.com/huggingface/evaluate/issues/138 is resolved before using this.
    """
    metrics = [metric.id for metric in list_metrics()]
    supported_metrics = []
    for metric in tqdm(metrics):
        # TODO: this currently requires all metric dependencies to be installed
        # in the same environment. Refactor to avoid needing to actually load
        # the metric.
        try:
            print(f"INFO -- Attempting to load metric: {metric}")
            metric_func = load(metric)
        except Exception as e:
            print(e)
            print("WARNING -- Skipping the following metric, which cannot load:", metric)
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


def get_dataset_card_url(dataset_id: str) -> str:
    """Gets the URL to edit the dataset card for the given dataset ID."""
    if "/" in dataset_id:
        return f"https://huggingface.co/datasets/{dataset_id}/edit/main/README.md"
    else:
        return f"https://github.com/huggingface/datasets/edit/master/datasets/{dataset_id}/README.md"


def create_autotrain_project_name(dataset_id: str, dataset_config: str) -> str:
    """Creates an AutoTrain project name for the given dataset ID."""
    # Project names cannot have "/", so we need to format community datasets accordingly
    dataset_id_formatted = dataset_id.replace("/", "__")
    dataset_config_formatted = dataset_config.replace("--", "__")
    # Project names need to be unique, so we append a random string to guarantee this while adhering to naming rules
    basename = f"eval-{dataset_id_formatted}-{dataset_config_formatted}"
    basename = basename[:60] if len(basename) > 60 else basename  # Hub naming limitation
    return f"{basename}-{str(uuid.uuid4())[:6]}"


def get_config_metadata(config: str, metadata: List[Dict] = None) -> Union[Dict, None]:
    """Gets the dataset card metadata for the given config."""
    if metadata is None:
        return None
    config_metadata = [m for m in metadata if m["config"] == config]
    if len(config_metadata) >= 1:
        return config_metadata[0]
    else:
        return None
