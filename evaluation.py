from dataclasses import dataclass

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
    metadata = dataset_info.cardData["eval_info"]
    metadata.pop("col_mapping", None)
    evaluation_info = EvaluationInfo(**metadata)
    return hash(evaluation_info)


def get_evaluation_ids():
    filt = DatasetFilter(author="autoevaluate")
    evaluation_datasets = HfApi().list_datasets(filter=filt, full=True)
    return [compute_evaluation_id(dset) for dset in evaluation_datasets]
