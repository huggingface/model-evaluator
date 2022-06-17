from dataclasses import dataclass

import streamlit as st
from huggingface_hub import DatasetFilter, HfApi
from huggingface_hub.hf_api import DatasetInfo


@dataclass(frozen=True, eq=True)
class EvaluationInfo:
    task: str
    model: str
    dataset_name: str
    dataset_config: str
    dataset_split: str


def compute_evaluation_id(dataset_info: DatasetInfo) -> int:
    if dataset_info.cardData is not None:
        metadata = dataset_info.cardData["eval_info"]
        metadata.pop("col_mapping", None)
        evaluation_info = EvaluationInfo(**metadata)
        return hash(evaluation_info)
    else:
        return None


def get_evaluation_ids():
    filt = DatasetFilter(author="autoevaluate")
    evaluation_datasets = HfApi().list_datasets(filter=filt, full=True)
    return [compute_evaluation_id(dset) for dset in evaluation_datasets]


def filter_evaluated_models(models, task, dataset_name, dataset_config, dataset_split):
    evaluation_ids = get_evaluation_ids()

    for idx, model in enumerate(models):
        evaluation_info = EvaluationInfo(
            task=task,
            model=model,
            dataset_name=dataset_name,
            dataset_config=dataset_config,
            dataset_split=dataset_split,
        )
        candidate_id = hash(evaluation_info)
        # if candidate_id in evaluation_ids:
        #     st.info(f"Model `{model}` has already been evaluated on this configuration. Skipping evaluation...")
        #     models.pop(idx)

    return models
