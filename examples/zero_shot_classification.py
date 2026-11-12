"""
Zero-Shot Classification Example
=================================
This module demonstrates zero-shot text classification using the
Hugging Face `transformers` library and a cross-encoder (NLI) pipeline.

Zero-shot classification allows a model to categorise text into labels
it has never been explicitly trained on by framing each label as a
natural-language hypothesis and running natural language inference (NLI).

Dependencies
------------
    pip install transformers torch

Usage
-----
    python zero_shot_classification.py
"""

from transformers import pipeline


def classify_zero_shot(
    text: str,
    candidate_labels: list[str],
    multi_label: bool = False,
    model_name: str = "facebook/bart-large-mnli",
) -> dict:
    """Classify *text* into one of the *candidate_labels* without prior training.

    Parameters
    ----------
    text:
        The input sentence or paragraph to classify.
    candidate_labels:
        A list of category names to consider.
    multi_label:
        When True the model treats every label independently, allowing
        more than one label to score high.  When False (default) the
        softmax is computed over all labels so their probabilities sum
        to 1.
    model_name:
        HuggingFace model identifier.  ``facebook/bart-large-mnli`` is a
        reliable default; smaller alternatives include
        ``cross-encoder/nli-MiniLM2-L6-H768``.

    Returns
    -------
    dict
        ``{"labels": [...], "scores": [...], "sequence": "..."}``
        Labels and scores are sorted from highest to lowest confidence.
    """
    classifier = pipeline(
        task="zero-shot-classification",
        model=model_name,
    )
    result = classifier(text, candidate_labels, multi_label=multi_label)
    return result


def print_results(result: dict) -> None:
    """Print classification results in a readable format."""
    print(f"\nInput text:\n  {result['sequence']}\n")
    print(f"{'Label':<30} {'Score':>8}")
    print("-" * 40)
    for label, score in zip(result["labels"], result["scores"]):
        bar = "█" * int(score * 30)
        print(f"{label:<30} {score:>7.1%}  {bar}")
    print()


def main() -> None:
    """Run several zero-shot classification demonstrations."""

    # ------------------------------------------------------------------ #
    # Example 1 – News topic classification                                #
    # ------------------------------------------------------------------ #
    news_headline = (
        "Scientists discover a new species of deep-sea fish near the "
        "hydrothermal vents of the Pacific Ocean."
    )
    news_labels = ["science", "politics", "sports", "entertainment", "finance"]

    print("=" * 50)
    print("Example 1 — News topic classification")
    print("=" * 50)
    result1 = classify_zero_shot(news_headline, news_labels)
    print_results(result1)

    # ------------------------------------------------------------------ #
    # Example 2 – Customer-support intent detection                        #
    # ------------------------------------------------------------------ #
    customer_query = "My order hasn't arrived yet and it has been two weeks!"
    intent_labels = [
        "shipping complaint",
        "product inquiry",
        "returns & refunds",
        "account issue",
        "general question",
    ]

    print("=" * 50)
    print("Example 2 — Customer-support intent detection")
    print("=" * 50)
    result2 = classify_zero_shot(customer_query, intent_labels)
    print_results(result2)

    # ------------------------------------------------------------------ #
    # Example 3 – Multi-label emotion tagging                              #
    # ------------------------------------------------------------------ #
    social_post = (
        "I can't believe they cancelled the show — I'm both heartbroken "
        "and absolutely furious about the decision."
    )
    emotion_labels = ["joy", "sadness", "anger", "surprise", "fear", "disgust"]

    print("=" * 50)
    print("Example 3 — Multi-label emotion tagging")
    print("=" * 50)
    result3 = classify_zero_shot(social_post, emotion_labels, multi_label=True)
    print_results(result3)

    # ------------------------------------------------------------------ #
    # Example 4 – Domain classification for academic abstracts             #
    # ------------------------------------------------------------------ #
    abstract = (
        "We present a convolutional neural network architecture that achieves "
        "state-of-the-art accuracy on the CIFAR-10 benchmark while reducing "
        "inference latency by 40 % through structured pruning."
    )
    domain_labels = [
        "computer vision",
        "natural language processing",
        "robotics",
        "bioinformatics",
        "quantum computing",
    ]

    print("=" * 50)
    print("Example 4 — Academic abstract domain classification")
    print("=" * 50)
    result4 = classify_zero_shot(abstract, domain_labels)
    print_results(result4)


if __name__ == "__main__":
    main()
