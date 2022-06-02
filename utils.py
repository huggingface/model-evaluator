from typing import Dict, Union

import requests
from huggingface_hub import HfApi, ModelFilter

AUTOTRAIN_TASK_TO_HUB_TASK = {
    "binary_classification": "text-classification",
    "multi_class_classification": "text-classification",
    # "multi_label_classification": "text-classification", # Not fully supported in AutoTrain
    "entity_extraction": "token-classification",
    "extractive_question_answering": "question-answering",
    "translation": "translation",
    "summarization": "summarization",
    # "single_column_regression": 10,
}

HUB_TASK_TO_AUTOTRAIN_TASK = {v: k for k, v in AUTOTRAIN_TASK_TO_HUB_TASK.items()}

api = HfApi()


def get_auth_headers(token: str, prefix: str = "autonlp"):
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


def get_metadata(dataset_name: str) -> Union[Dict, None]:
    data = requests.get(f"https://huggingface.co/api/datasets/{dataset_name}").json()
    if data["cardData"] is not None and "train-eval-index" in data["cardData"].keys():
        return data["cardData"]["train-eval-index"]
    else:
        return None


def get_compatible_models(task, dataset_name):
    # TODO: relax filter on PyTorch models once supported in AutoTrain
    filt = ModelFilter(
        task=AUTOTRAIN_TASK_TO_HUB_TASK[task],
        trained_dataset=dataset_name,
        library=["transformers", "pytorch"],
    )
    compatible_models = api.list_models(filter=filt)
    return sorted([model.modelId for model in compatible_models])


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
