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
| `extractive_question_answering`    |     ❌     |
| `translation`                      |     ✅     |
| `summarization`                    |     ✅     |
| `image_binary_classification`      |     ✅     |
| `image_multi_class_classification` |     ✅     |

## Installation

To run the application, first clone this repository and install the dependencies as follows:

```
pip install -r requirements.txt
```

Then spin up the application by running:

```
streamlit run app.py
```