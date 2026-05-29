# Predicting Zero-Shot Success from Representation Geometry

Research project investigating whether geometric properties of pretrained RoBERTa-base embeddings can predict zero-shot task difficulty across GLUE and SuperGLUE benchmarks.

**Key finding:** Intrinsic dimensionality of pretrained embeddings shows moderate, statistically significant correlation with downstream task accuracy (r = 0.57, p = 0.044, n = 13 tasks). Four other geometric metrics show weak, non-significant correlations, suggesting that most pretrained geometric properties are poor proxies for post-finetuning task performance.

A poster summarizing this work is available at `poster.pdf`.

---

## Motivation

Evaluating large language models on new tasks requires expensive brute-force benchmarking. Zero-shot performance is highly unpredictable, with no reliable way to anticipate success or failure on superficially similar tasks. This project investigates whether geometric properties of a model's pretrained representation space can serve as lightweight proxies for task difficulty, enabling prediction of downstream performance without full task-specific training.

---

## Methods

- **Model:** RoBERTa-base; embeddings extracted from the CLS token of the final hidden layer (768 dimensions)
- **Tasks:** 13 benchmarks from GLUE and SuperGLUE, spanning entailment, sentiment, paraphrase detection, reading comprehension, and word sense disambiguation
- **Metrics:** Five geometric properties computed on frozen pretrained embeddings per task: linear probe loss, mean cosine similarity, intrinsic dimensionality, Fisher separability, and CKA
- **Ground truth:** Published RoBERTa-base fine-tuned accuracy from Liu et al. (2019) (GLUE) and the SuperGLUE leaderboard (SuperGLUE tasks), used as a proxy for task difficulty
- **Analysis:** Pearson correlation between each metric and task accuracy across all 13 tasks

> **Note on zero-shot vs. fine-tuned accuracy:** The original hypothesis concerns zero-shot performance evaluated directly from pretrained representations without task-specific adaptation. In practice, standardized zero-shot accuracy numbers for RoBERTa-base are not available across all 13 tasks, so published fine-tuned accuracy is used as a proxy for task difficulty. This is a known limitation: fine-tuned accuracy reflects post-adaptation performance, while the geometry metrics are computed on pretrained embeddings before any fine-tuning occurs. The gap between these two settings is part of what the results reflect, and closing it by evaluating against true zero-shot performance is the most direct future direction for this work.

---

## Repository Structure

```
reprgeom/
├── notebooks/
│   └── main.ipynb          # Full pipeline: embedding → metrics → figures
├── src/
│   ├── tasks.py            # Task configs for all 13 GLUE/SuperGLUE datasets
│   ├── embeddings.py       # RoBERTa loading and CLS embedding extraction
│   └── metrics.py          # Five geometry metric functions
├── data/
│   └── embeddings_roberta/ # Cached .npy embeddings (not tracked, see below)
├── outputs/
│   ├── metrics.csv         # Computed geometry metrics + accuracy scores
│   └── figures/            # Generated plots
├── poster.pdf              # Poster presented at undergrad research symposium
└── requirements.txt
```

---

## Reproducing Results

**Install dependencies:**
```bash
pip install -r requirements.txt
```

**Run the notebook:**
```bash
cd notebooks
jupyter notebook main.ipynb
```

Run all cells top to bottom. Embeddings are generated and cached automatically on first run.

> **Note:** Embedding files (`data/embeddings_roberta/`) are not tracked by git due to file size. They will be regenerated automatically when the notebook is run.

---

## Results

| Metric | Pearson r | Significant? |
|--------|-----------|--------------|
| Intrinsic Dimensionality | 0.57 | ✅ p = 0.044 |
| Fisher Separability | < 0.20 | ❌ |
| Mean Cosine Similarity | < 0.20 | ❌ |
| CKA | < 0.20 | ❌ |
| Linear Probe Loss | < 0.20 | ❌ |

---

## References

- Aghajanyan et al. (2021). Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning. [arXiv:2012.13255](https://arxiv.org/abs/2012.13255)
- Kornblith et al. (2019). Similarity of Neural Network Representations Revisited. [arXiv:1905.00414](https://arxiv.org/abs/1905.00414)
- Liu et al. (2019). RoBERTa: A Robustly Optimized BERT Pretraining Approach. [arXiv:1907.11692](https://arxiv.org/abs/1907.11692)
- Merchant et al. (2020). What Happens to BERT Embeddings During Fine-tuning? [arXiv:2004.14448](https://arxiv.org/abs/2004.14448)
- Wang et al. (2018). GLUE: A Multi-Task Benchmark and Analysis Platform. [arXiv:1804.07461](https://arxiv.org/abs/1804.07461)
- Wang et al. (2019). SuperGLUE: A Stickier Benchmark for General-Purpose Language Understanding. [arXiv:1905.00537](https://arxiv.org/abs/1905.00537)

---

## Acknowledgements

This project was completed as part of an undergraduate research symposium. Part of the code and analysis were developed with assistance from Claude (Anthropic) and ChatGPT (OpenAI).