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

## Installation

To run the application locally, first clone this repository and install the dependencies as follows:

```
pip install -r requirements.txt
```

Next, copy the example file of environment variables:

```
cp .env.template .env
```

and set the `HF_TOKEN` variable with a valid API token from the `autoevaluator` user. Finally, spin up the application by running:

```
streamlit run app.py
```

## AutoTrain configuration details

Models are evaluated by AutoTrain, with the payload sent to the `AUTOTRAIN_BACKEND_API` environment variable. The current configuration for evaluation jobs running on Spaces is:

```
AUTOTRAIN_BACKEND_API=https://api-staging.autotrain.huggingface.co
```

To evaluate models with a _local_ instance of AutoTrain, change the environment to:

```
AUTOTRAIN_BACKEND_API=http://localhost:8000
```