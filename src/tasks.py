import numpy as np
import os

# Each entry: (hf_dataset_name, hf_config, split, text_fn, label_fn)
# text_fn tells us how to make one string from a row
# label_fn tells us how to extract the integer label

task_configs = {
    "sst2": {
        "dataset": "glue", "config": "sst2", "split": "validation",
        "text_fn": lambda row: row["sentence"],
        "label_fn": lambda row: int(row["label"])
    },
    "mnli": {
        "dataset": "glue", "config": "mnli", "split": "validation_matched",
        "text_fn": lambda row: f"{row['premise'][:300]} [SEP] {row['hypothesis']}",
        "label_fn": lambda row: int(row["label"])
    },
    "qnli": {
        "dataset": "glue", "config": "qnli", "split": "validation",
        "text_fn": lambda row: f"{row['question']} [SEP] {row['sentence']}",
        "label_fn": lambda row: int(row["label"])
    },
    "qqp": {
        "dataset": "glue", "config": "qqp", "split": "validation",
        "text_fn": lambda row: f"{row['question1']} [SEP] {row['question2']}",
        "label_fn": lambda row: int(row["label"])
    },
    "rte": {
        "dataset": "glue", "config": "rte", "split": "validation",
        "text_fn": lambda row: f"{row['sentence1']} [SEP] {row['sentence2']}",
        "label_fn": lambda row: int(row["label"])
    },
    "mrpc": {
        "dataset": "glue", "config": "mrpc", "split": "validation",
        "text_fn": lambda row: f"{row['sentence1']} [SEP] {row['sentence2']}",
        "label_fn": lambda row: int(row["label"])
    },
    "cola": {
        "dataset": "glue", "config": "cola", "split": "validation",
        "text_fn": lambda row: row["sentence"],
        "label_fn": lambda row: int(row["label"])
    },
    "stsb": {
        "dataset": "glue", "config": "stsb", "split": "validation",
        "text_fn": lambda row: f"{row['sentence1']} [SEP] {row['sentence2']}",
        "label_fn": lambda row: int(round(row["label"]))
    },
    "wnli": {
        "dataset": "glue", "config": "wnli", "split": "validation",
        "text_fn": lambda row: f"{row['sentence1']} [SEP] {row['sentence2']}",
        "label_fn": lambda row: int(row["label"])
    },
    "cb": {
        "dataset": "super_glue", "config": "cb", "split": "validation",
        "text_fn": lambda row: f"{row['premise'][:300]} [SEP] {row['hypothesis']}",
        "label_fn": lambda row: int(row["label"])
    },
    "boolq": {
        "dataset": "super_glue", "config": "boolq", "split": "validation",
        "text_fn": lambda row: f"{row['question']} [SEP] {row['passage'][:300]}",
        "label_fn": lambda row: int(row["label"])
    },
    "wic": {
        "dataset": "super_glue", "config": "wic", "split": "validation",
        "text_fn": lambda row: f"{row['word']} [SEP] {row['sentence1']} [SEP] {row['sentence2']}",
        "label_fn": lambda row: int(row["label"])
    },
    "multirc": {
        "dataset": "super_glue", "config": "multirc", "split": "validation",
        "text_fn": lambda row: f"{row['question']} [SEP] {row['answer']} [SEP] {row['paragraph'][:200]}",
        "label_fn": lambda row: int(row["label"])
    },
}