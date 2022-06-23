---
title: Model Evaluator
emoji: 📊
colorFrom: red
colorTo: red
sdk: streamlit
sdk_version: 1.10.0
app_file: app.py
---

# AutoEvaluate

> Submit evaluation jobs to AutoTrain from the Hugging Face Hub

## Supported tasks

The table below shows which tasks are currently supported for evaluation in the AutoTrain backend:

| Task                            | Supported | Sample prediction repository                                                        |
|:--------------------------------|:---------:|:------------------------------------------------------------------------------------|
| `binary_classification`         |     ✅     | [`eval-staging-835`](https://huggingface.co/datasets/autoevaluate/eval-staging-835) |
| `multi_class_classification`    |     ✅     | [`eval-staging-822`](https://huggingface.co/datasets/autoevaluate/eval-staging-822) |
| `multi_label_classification`    |     ❌     |                                                                                     |
| `entity_extraction`             |     ✅     | [`eval-staging-838`](https://huggingface.co/datasets/autoevaluate/eval-staging-838) |
| `extractive_question_answering` |     ✅     |                                                                                     |
| `translation`                   |     ✅     |                                                                                     |
| `summarization`                 |     ✅     |                                                                                     |