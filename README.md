---
title: Model Evaluator
emoji: 📊
colorFrom: red
colorTo: red
sdk: streamlit
sdk_version: 1.10.0
app_file: app.py
---

# Model Evaluator

> Submit evaluation jobs to AutoTrain from the Hugging Face Hub

## Supported tasks

The table below shows which tasks are currently supported for evaluation in the AutoTrain backend:

| Task                               | Supported |
|:-----------------------------------|:---------:|
| `binary_classification`            |     ✅     |
| `multi_class_classification`       |     ✅     |
| `multi_label_classification`       |     ❌     |
| `entity_extraction`                |     ✅     |
| `extractive_question_answering`    |     ✅     |
| `translation`                      |     ✅     |
| `summarization`                    |     ✅     |
| `image_binary_classification`      |     ✅     |
| `image_multi_class_classification` |     ✅     |
| `text_zero_shot_evaluation`        |     ✅     |


## Installation

To run the application locally, first clone this repository and install the dependencies as follows:

```
pip install -r requirements.txt
```

Next, copy the example file of environment variables:

```
cp .env.template .env
```

and set the `HF_TOKEN` variable with a valid API token from the [`autoevaluator`](https://huggingface.co/autoevaluator) bot user. Finally, spin up the application by running:

```
streamlit run app.py
```

## Usage

Evaluation on the Hub involves two main steps:

1. Submitting an evaluation job via the UI. This creates an AutoTrain project with `N` models for evaluation. At this stage, the dataset is also processed and prepared for evaluation.
2. Triggering the evaluation itself once the dataset is processed.

From the user perspective, only step (1) is needed since step (2) is handled by a cron job on GitHub Actions that executes the `run_evaluation_jobs.py` script every 15 minutes.

See below for details on manually triggering evaluation jobs.

### Triggering an evaluation

To evaluate the models in an AutoTrain project, run:

```
python run_evaluation_jobs.py
```

This will download the [`autoevaluate/evaluation-job-logs`](https://huggingface.co/datasets/autoevaluate/evaluation-job-logs) dataset from the Hub and check which evaluation projects are ready for evaluation (i.e. those whose dataset has been processed).

## AutoTrain configuration details

Models are evaluated by the [`autoevaluator`](https://huggingface.co/autoevaluator) bot user in AutoTrain, with the payload sent to the `AUTOTRAIN_BACKEND_API` environment variable. Evaluation projects are created and run on either the `prod` or `staging` environments. You can view the status of projects in the AutoTrain UI by navigating to one of the links below (ask internally for access to the staging UI):

| AutoTrain environment |                                                AutoTrain UI URL                                                |           `AUTOTRAIN_BACKEND_API`            |
|:---------------------:|:--------------------------------------------------------------------------------------------------------------:|:--------------------------------------------:|
|        `prod`         |         [`https://ui.autotrain.huggingface.co/projects`](https://ui.autotrain.huggingface.co/projects)         |     https://api.autotrain.huggingface.co     |
|       `staging`       | [`https://ui-staging.autotrain.huggingface.co/projects`](https://ui-staging.autotrain.huggingface.co/projects) | https://api-staging.autotrain.huggingface.co |


The current configuration for evaluation jobs running on [Spaces](https://huggingface.co/spaces/autoevaluate/model-evaluator) is:

```
AUTOTRAIN_BACKEND_API=https://api.autotrain.huggingface.co
```

To evaluate models with a _local_ instance of AutoTrain, change the environment to:

```
AUTOTRAIN_BACKEND_API=http://localhost:8000
```

### Migrating from staging to production (and vice versa)

In general, evaluation jobs should run in AutoTrain's `prod` environment, which is defined by the following environment variable:

```
AUTOTRAIN_BACKEND_API=https://api.autotrain.huggingface.co
```

However, there are times when it is necessary to run evaluation jobs in AutoTrain's `staging` environment (e.g. because a new evaluation pipeline is being deployed). In these cases the corresponding environement variable is:

```
AUTOTRAIN_BACKEND_API=https://api-staging.autotrain.huggingface.co
```

To migrate between these two environments, update the `AUTOTRAIN_BACKEND_API` in two places:

* In the [repo secrets](https://huggingface.co/spaces/autoevaluate/model-evaluator/settings) associated with the `model-evaluator` Space. This will ensure evaluation projects are created in the desired environment.
* In the [GitHub Actions secrets](https://github.com/huggingface/model-evaluator/settings/secrets/actions) associated with this repo. This will ensure that the correct evaluation jobs are approved and launched via the `run_evaluation_jobs.py` script.
