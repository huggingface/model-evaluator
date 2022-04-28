from typing import Dict, Union

import requests
from huggingface_hub import DatasetFilter, HfApi, ModelFilter

api = HfApi()


def get_auth_headers(token: str, prefix: str = "autonlp"):
    return {"Authorization": f"{prefix} {token}"}


def http_post(path: str, token: str, payload=None, domain: str = None, params=None) -> requests.Response:
    """HTTP POST request to the AutoNLP API, raises UnreachableAPIError if the API cannot be reached"""
    try:
        response = requests.post(
            url=domain + path, json=payload, headers=get_auth_headers(token=token), allow_redirects=True, params=params
        )
    except requests.exceptions.ConnectionError:
        print("❌ Failed to reach AutoNLP API, check your internet connection")
    response.raise_for_status()
    return response


def http_get(path: str, domain: str, token: str = None, params: dict = None) -> requests.Response:
    """HTTP POST request to the AutoNLP API, raises UnreachableAPIError if the API cannot be reached"""
    try:
        response = requests.get(
            url=domain + path, headers=get_auth_headers(token=token), allow_redirects=True, params=params
        )
    except requests.exceptions.ConnectionError:
        print("❌ Failed to reach AutoNLP API, check your internet connection")
    response.raise_for_status()
    return response


def get_metadata(dataset_name: str) -> Union[Dict, None]:
    filt = DatasetFilter(dataset_name=dataset_name)
    data = api.list_datasets(filter=filt, full=True)
    if data[0].cardData is not None and "train-eval-index" in data[0].cardData.keys():
        return data[0].cardData["train-eval-index"]
    else:
        return None


def get_compatible_models(task, dataset_name):
    filt = ModelFilter(task=task, trained_dataset=dataset_name, library="transformers")
    compatible_models = api.list_models(filter=filt)
    return [model.modelId for model in compatible_models]
