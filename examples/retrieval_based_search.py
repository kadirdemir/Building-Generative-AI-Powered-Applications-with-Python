"""
Retrieval-Based Search with Dense Passage Retrieval

This module demonstrates how to build a retrieval-based search system using
sentence-transformers to encode documents and queries into dense vector
representations. The system finds the most relevant passages from a corpus
based on semantic similarity rather than exact keyword matching.

Key components:
- Document encoding: transform text passages into fixed-size vectors
- Query encoding: encode search queries into the same vector space
- Similarity computation: rank documents by cosine similarity to the query
- Top-k retrieval: return the most relevant results

Dependencies:
    pip install sentence-transformers torch numpy
"""

import numpy as np
from typing import List, Tuple


def encode_texts(texts: List[str], model) -> np.ndarray:
    """
    Encode a list of text strings into dense vector representations.

    Args:
        texts: List of text strings to encode.
        model: A sentence-transformers SentenceTransformer model instance.

    Returns:
        A 2D numpy array where each row is the embedding for one text.
    """
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    return embeddings


def cosine_similarity_matrix(query_vecs: np.ndarray, doc_vecs: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity between each query vector and all document vectors.

    Args:
        query_vecs: Array of shape (num_queries, dim).
        doc_vecs: Array of shape (num_docs, dim).

    Returns:
        Similarity matrix of shape (num_queries, num_docs).
    """
    # Normalise rows to unit length
    query_norms = np.linalg.norm(query_vecs, axis=1, keepdims=True)
    doc_norms = np.linalg.norm(doc_vecs, axis=1, keepdims=True)

    query_unit = query_vecs / np.clip(query_norms, 1e-10, None)
    doc_unit = doc_vecs / np.clip(doc_norms, 1e-10, None)

    return query_unit @ doc_unit.T  # (num_queries, num_docs)


def retrieve_top_k(
    query: str,
    corpus: List[str],
    query_embedding: np.ndarray,
    corpus_embeddings: np.ndarray,
    k: int = 3,
) -> List[Tuple[float, str]]:
    """
    Retrieve the top-k most relevant passages for a single query.

    Args:
        query: The search query string (used only for display).
        corpus: List of document/passage strings.
        query_embedding: Pre-computed embedding for the query (shape: (dim,)).
        corpus_embeddings: Pre-computed embeddings for all corpus passages.
        k: Number of top results to return.

    Returns:
        List of (score, passage) tuples sorted by descending similarity.
    """
    scores = cosine_similarity_matrix(
        query_embedding.reshape(1, -1), corpus_embeddings
    )[0]  # shape: (num_docs,)

    top_indices = np.argsort(scores)[::-1][:k]
    results = [(float(scores[i]), corpus[i]) for i in top_indices]
    return results


def build_index(corpus: List[str], model) -> np.ndarray:
    """
    Pre-encode all corpus documents to build a searchable index.

    Args:
        corpus: List of text passages to index.
        model: Sentence-transformers model for encoding.

    Returns:
        Numpy array of embeddings with shape (len(corpus), embedding_dim).
    """
    print(f"Building index for {len(corpus)} passages …")
    embeddings = encode_texts(corpus, model)
    print(f"Index shape: {embeddings.shape}")
    return embeddings


def main():
    from sentence_transformers import SentenceTransformer

    # --- Sample corpus ---
    corpus = [
        "Python is a high-level, interpreted programming language known for its clear syntax.",
        "Machine learning involves training statistical models on data to make predictions.",
        "Neural networks are composed of layers of interconnected nodes that process information.",
        "NumPy provides efficient multi-dimensional array operations for numerical computing in Python.",
        "Transfer learning reuses a pre-trained model as the starting point for a new task.",
        "Gradient descent is an optimisation algorithm that iteratively minimises a loss function.",
        "Tokenisation splits raw text into smaller units such as words or subwords before processing.",
        "Pandas offers data structures and analysis tools built on top of NumPy arrays.",
        "Backpropagation computes gradients of the loss with respect to model parameters.",
        "Cosine similarity measures the angle between two vectors as a proxy for semantic closeness.",
    ]

    queries = [
        "How does optimisation work in deep learning?",
        "What tools are available for working with arrays in Python?",
        "How is text prepared for processing in NLP?",
    ]

    # Load a lightweight bi-encoder model
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    print(f"Loading model: {model_name}")
    model = SentenceTransformer(model_name)

    # Build corpus index once
    corpus_embeddings = build_index(corpus, model)

    # Encode all queries in a single batch
    print(f"\nEncoding {len(queries)} queries …")
    query_embeddings = encode_texts(queries, model)

    # Retrieve and display results
    top_k = 3
    for query, q_emb in zip(queries, query_embeddings):
        print(f"\n{'='*60}")
        print(f"Query : {query}")
        print(f"Top-{top_k} results:")
        results = retrieve_top_k(query, corpus, q_emb, corpus_embeddings, k=top_k)
        for rank, (score, passage) in enumerate(results, start=1):
            print(f"  {rank}. [{score:.4f}] {passage}")

    print("\nRetrieval complete.")


if __name__ == "__main__":
    main()
