import copy
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
    metrics: set


def create_evaluation_info(dataset_info: DatasetInfo) -> int:
    if dataset_info.cardData is not None:
        metadata = dataset_info.cardData["eval_info"]
        metadata.pop("col_mapping", None)
        # TODO(lewtun): populate dataset cards with metric info
        if "metrics" not in metadata:
            metadata["metrics"] = frozenset()
        else:
            metadata["metrics"] = frozenset(metadata["metrics"])
        return EvaluationInfo(**metadata)


def get_evaluation_infos():
    evaluation_datasets = []
    filt = DatasetFilter(author="autoevaluate")
    autoevaluate_datasets = HfApi().list_datasets(filter=filt, full=True)
    for dset in autoevaluate_datasets:
        try:
            evaluation_datasets.append(create_evaluation_info(dset))
        except Exception as e:
            print(f"Error processing dataset {dset}: {e}")
    return evaluation_datasets


def filter_evaluated_models(models, task, dataset_name, dataset_config, dataset_split, metrics):
    evaluation_infos = get_evaluation_infos()
    models_to_filter = copy.copy(models)

    for model in models_to_filter:
        evaluation_info = EvaluationInfo(
            task=task,
            model=model,
            dataset_name=dataset_name,
            dataset_config=dataset_config,
            dataset_split=dataset_split,
            metrics=frozenset(metrics),
        )
        if evaluation_info in evaluation_infos:
            st.info(
                f"Model [`{model}`](https://huggingface.co/{model}) has already been evaluated on this configuration. \
                    This model will be excluded from the evaluation job..."
            )
            models.remove(model)

    return models
