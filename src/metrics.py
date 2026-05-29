from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd

def linear_probe_loss(embeddings, labels):
    """Train logistic regression on embeddings, return cross-val error."""
    clf = LogisticRegression(max_iter=1000, random_state=42)
    # cross_val_score returns accuracy; we want loss = 1 - accuracy
    scores = cross_val_score(clf, embeddings, labels, cv=5, scoring='accuracy')
    return 1.0 - scores.mean()

def mean_cosine_similarity(embeddings):
    """Average pairwise cosine similarity across a sample of examples."""
    # Subsample to 500 to keep it fast
    idx = np.random.choice(len(embeddings), size=min(500, len(embeddings)), replace=False)
    sample = embeddings[idx]
    normed = normalize(sample)
    sim_matrix = normed @ normed.T
    # Take upper triangle only (exclude diagonal = self-similarity)
    upper = sim_matrix[np.triu_indices(len(normed), k=1)]
    return upper.mean()

def intrinsic_dimensionality(embeddings, threshold=0.90):
    """Number of PCA dims needed to explain `threshold` variance."""
    pca = PCA().fit(embeddings)
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    n_dims = np.searchsorted(cumvar, threshold) + 1
    return int(n_dims)

def fisher_separability(embeddings, labels):
    """Ratio of between-class to within-class variance (Fisher criterion)."""
    classes = np.unique(labels)
    overall_mean = embeddings.mean(axis=0)
    
    between = np.zeros(embeddings.shape[1])
    within = np.zeros(embeddings.shape[1])
    
    for c in classes:
        mask = labels == c
        class_emb = embeddings[mask]
        class_mean = class_emb.mean(axis=0)
        n = mask.sum()
        between += n * (class_mean - overall_mean) ** 2
        within += ((class_emb - class_mean) ** 2).sum(axis=0)
    
    # Scalar: mean ratio across all dimensions
    ratio = (between / (within + 1e-8)).mean()
    return float(ratio)

def cka(embeddings, labels, subsample=500):
    """
    Centered Kernel Alignment between embedding Gram matrix
    and label Gram matrix. Higher = embeddings align with label structure.
    """
    idx = np.random.choice(len(embeddings), size=min(subsample, len(embeddings)), replace=False)
    X = embeddings[idx].astype(np.float32)
    Y = labels[idx].reshape(-1, 1).astype(np.float32)
    
    def center(K):
        n = K.shape[0]
        H = np.eye(n) - np.ones((n, n)) / n
        return H @ K @ H
    
    K = X @ X.T
    L = Y @ Y.T
    
    Kc = center(K)
    Lc = center(L)
    
    num = np.sum(Kc * Lc)
    denom = np.sqrt(np.sum(Kc * Kc) * np.sum(Lc * Lc))
    return float(num / (denom + 1e-8))