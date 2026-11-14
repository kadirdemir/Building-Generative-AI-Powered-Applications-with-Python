"""
Embedding and Similarity Search
================================
This module demonstrates how to generate dense vector embeddings from text
and use cosine similarity to find the most semantically similar passages
from a small corpus.

Key concepts covered:
- Loading a pre-trained sentence embedding model
- Encoding sentences into fixed-length vectors
- Computing pairwise cosine similarities
- Ranking results and returning the top-k matches

Requirements:
    pip install sentence-transformers torch
"""

from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# ---------------------------------------------------------------------------
# Sample corpus — a small collection of technical sentences
# ---------------------------------------------------------------------------
CORPUS: list[str] = [
    "Gradient descent is an iterative optimisation algorithm that minimises a loss function.",
    "Backpropagation computes gradients by applying the chain rule through a computation graph.",
    "A convolutional layer applies learned filters to local patches of an input tensor.",
    "Recurrent networks process sequential data by maintaining a hidden state across timesteps.",
    "Attention mechanisms allow a model to focus on relevant parts of an input sequence.",
    "Transfer learning reuses weights from a model trained on a large dataset for a new task.",
    "Dropout randomly zeroes out neurons during training to reduce overfitting.",
    "Batch normalisation standardises layer inputs to accelerate training and improve stability.",
    "An encoder maps variable-length inputs to a fixed-size latent representation.",
    "A decoder transforms a latent representation back into a desired output format.",
    "Early stopping halts training when validation loss stops improving.",
    "Data augmentation artificially expands a training set by applying label-preserving transforms.",
    "K-fold cross-validation partitions data into k subsets for more reliable evaluation.",
    "Hyperparameter tuning searches for the configuration that maximises model performance.",
    "Regularisation techniques penalise model complexity to improve generalisation.",
]


def encode_texts(
    model: SentenceTransformer,
    texts: list[str],
    batch_size: int = 32,
) -> np.ndarray:
    """Return a 2-D array of embeddings, one row per text.

    Args:
        model: A loaded SentenceTransformer model.
        texts: The list of strings to encode.
        batch_size: Number of texts to encode in one forward pass.

    Returns:
        NumPy array of shape (len(texts), embedding_dim).
    """
    embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=False)
    return np.array(embeddings)


def find_top_k(
    query_embedding: np.ndarray,
    corpus_embeddings: np.ndarray,
    corpus: list[str],
    k: int = 3,
) -> list[tuple[float, str]]:
    """Return the top-k corpus sentences most similar to the query.

    Args:
        query_embedding: Shape (1, dim) or (dim,) embedding of the query.
        corpus_embeddings: Shape (n, dim) matrix of corpus embeddings.
        corpus: The original corpus strings, aligned with corpus_embeddings.
        k: Number of results to return.

    Returns:
        A list of (score, sentence) pairs sorted from highest to lowest score.
    """
    if query_embedding.ndim == 1:
        query_embedding = query_embedding.reshape(1, -1)

    scores: np.ndarray = cosine_similarity(query_embedding, corpus_embeddings)[0]
    top_indices = np.argsort(scores)[::-1][:k]
    return [(float(scores[i]), corpus[i]) for i in top_indices]


def run_demo(model_name: str = "all-MiniLM-L6-v2") -> None:
    """Encode the corpus and answer several example queries."""

    print(f"Loading embedding model '{model_name}' …\n")
    model = SentenceTransformer(model_name)

    # Pre-compute corpus embeddings once
    corpus_embeddings = encode_texts(model, CORPUS)
    print(f"Encoded {len(CORPUS)} corpus sentences "
          f"(dim={corpus_embeddings.shape[1]}).\n")

    queries = [
        "How do neural networks learn from data?",
        "What techniques prevent a model from memorising the training set?",
        "Explain how sequence models handle temporal dependencies.",
    ]

    for query in queries:
        print(f"Query: {query!r}")
        query_embedding = encode_texts(model, [query])
        results = find_top_k(query_embedding, corpus_embeddings, CORPUS, k=3)
        for rank, (score, sentence) in enumerate(results, start=1):
            print(f"  {rank}. [{score:.4f}] {sentence}")
        print()


def main() -> None:
    """Entry point."""
    run_demo()


if __name__ == "__main__":
    main()
