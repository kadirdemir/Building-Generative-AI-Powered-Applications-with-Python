"""
Semantic Textual Similarity
============================
Measures how similar two sentences are in meaning using sentence-transformers.
Cosine similarity scores range from -1 (opposite meaning) to 1 (identical meaning).

Dependencies:
    pip install sentence-transformers

Usage:
    python semantic_textual_similarity.py
"""

from sentence_transformers import SentenceTransformer, util


def compute_similarity(model: SentenceTransformer, sentence1: str, sentence2: str) -> float:
    """Encode two sentences and return their cosine similarity score."""
    embeddings = model.encode([sentence1, sentence2], convert_to_tensor=True)
    score = util.cos_sim(embeddings[0], embeddings[1])
    return float(score)


def rank_candidates(
    model: SentenceTransformer, query: str, candidates: list[str]
) -> list[tuple[float, str]]:
    """Rank a list of candidate sentences by similarity to the query."""
    query_embedding = model.encode(query, convert_to_tensor=True)
    candidate_embeddings = model.encode(candidates, convert_to_tensor=True)
    scores = util.cos_sim(query_embedding, candidate_embeddings)[0]
    ranked = sorted(
        zip(scores.tolist(), candidates), key=lambda x: x[0], reverse=True
    )
    return ranked


def main() -> None:
    print("Loading sentence encoder model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # --- Pairwise similarity examples ---
    pairs = [
        (
            "The cat sat on the mat.",
            "A cat is resting on a rug.",
        ),
        (
            "The cat sat on the mat.",
            "Dogs love playing fetch in the park.",
        ),
        (
            "She enjoys reading novels in her spare time.",
            "In her free time she likes to read books.",
        ),
        (
            "The stock market closed higher today.",
            "Equity markets ended the session with gains.",
        ),
    ]

    print("\n=== Pairwise Similarity Scores ===")
    print(f"{'Score':>6}  Sentence pair")
    print("-" * 70)
    for s1, s2 in pairs:
        score = compute_similarity(model, s1, s2)
        preview1 = s1[:45] + ("…" if len(s1) > 45 else "")
        preview2 = s2[:45] + ("…" if len(s2) > 45 else "")
        print(f"{score:>6.3f}  '{preview1}'")
        print(f"       vs '{preview2}'")
        print()

    # --- Candidate ranking example ---
    query = "How do I cook pasta?"
    candidates = [
        "Boil water, add salt, then cook the pasta until al dente.",
        "Pasta should be served with a good tomato sauce.",
        "The best way to bake bread is at 200 °C for 30 minutes.",
        "Italian cuisine uses many different types of noodles.",
        "To prepare spaghetti, boil it in salted water for about 9 minutes.",
    ]

    print("=== Candidate Ranking ===")
    print(f"Query: '{query}'\n")
    ranked = rank_candidates(model, query, candidates)
    for rank, (score, sentence) in enumerate(ranked, start=1):
        print(f"  {rank}. [{score:.3f}] {sentence}")

    print("\nDone.")


if __name__ == "__main__":
    main()
