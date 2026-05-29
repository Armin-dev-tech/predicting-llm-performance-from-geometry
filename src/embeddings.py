# src/embeddings.py

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel


def load_roberta(model_name="roberta-base"):
    """
    Load pretrained RoBERTa tokenizer and model.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModel.from_pretrained(model_name)
    model.eval()

    return tokenizer, model


def get_embeddings(
    texts,
    tokenizer,
    model,
    batch_size=32,
    max_length=128
):
    """
    Convert a list of texts into RoBERTa CLS embeddings.
    """

    all_embeddings = []

    for i in range(0, len(texts), batch_size):

        batch = texts[i:i + batch_size]

        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )

        with torch.no_grad():
            output = model(**encoded)

        # CLS token embedding
        embeddings = output.last_hidden_state[:, 0, :].numpy()

        all_embeddings.append(embeddings)

        if i % (batch_size * 10) == 0:
            print(f"  batch {i // batch_size + 1}...", end="\r")

    return np.vstack(all_embeddings)


def load_and_embed_task(
    task_name,
    cfg,
    tokenizer,
    model,
    max_examples=2000,
    batch_size=32
):
    """
    Load a dataset task and compute embeddings + labels.
    """

    ds = load_dataset(
        cfg["dataset"],
        cfg["config"],
        split=cfg["split"]
    )

    texts = [cfg["text_fn"](row) for row in ds]

    labels = np.array([
        cfg["label_fn"](row)
        for row in ds
    ])

    # Optional subsampling
    if len(texts) > max_examples:

        idx = np.random.choice(
            len(texts),
            size=max_examples,
            replace=False
        )

        texts = [texts[i] for i in idx]
        labels = labels[idx]

        print(f"   Subsampled to {max_examples} examples")

    print(f"   {len(texts)} examples, encoding now...")

    embeddings = get_embeddings(
        texts,
        tokenizer,
        model,
        batch_size=batch_size
    )

    return embeddings, labels