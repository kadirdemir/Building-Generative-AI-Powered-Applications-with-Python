"""
Sentiment Analysis with Transformers
=====================================

This module demonstrates sentiment analysis using the Hugging Face `transformers`
library. It classifies text as POSITIVE or NEGATIVE and returns a confidence score.

The pipeline uses a pre-trained model (distilbert-base-uncased-finetuned-sst-2-english)
fine-tuned on the Stanford Sentiment Treebank (SST-2) dataset.

Requirements:
    pip install transformers torch

Usage:
    python sentiment_analysis.py
"""

from transformers import pipeline


def build_sentiment_pipeline(model_name: str = "distilbert-base-uncased-finetuned-sst-2-english"):
    """
    Load and return a sentiment analysis pipeline.

    Args:
        model_name: Hugging Face model identifier. Defaults to DistilBERT SST-2.

    Returns:
        A callable pipeline that accepts text and returns sentiment labels + scores.
    """
    return pipeline("sentiment-analysis", model=model_name)


def analyze_texts(classifier, texts: list[str]) -> list[dict]:
    """
    Run sentiment analysis on a list of texts.

    Args:
        classifier: A Hugging Face sentiment-analysis pipeline.
        texts: List of strings to classify.

    Returns:
        List of dicts, each containing 'text', 'label', and 'score'.
    """
    results = classifier(texts)
    enriched = []
    for text, result in zip(texts, results):
        enriched.append({
            "text": text,
            "label": result["label"],
            "score": round(result["score"], 4),
        })
    return enriched


def print_results(results: list[dict]) -> None:
    """Pretty-print sentiment analysis results."""
    print(f"\n{'Text':<55} {'Label':<10} {'Confidence':>10}")
    print("-" * 78)
    for item in results:
        truncated = item["text"][:52] + "..." if len(item["text"]) > 52 else item["text"]
        print(f"{truncated:<55} {item['label']:<10} {item['score']:>10.4f}")


def main():
    sample_texts = [
        "I absolutely loved this product — it exceeded every expectation!",
        "The service was terrible and the staff was rude.",
        "It is an okay experience, nothing spectacular.",
        "Best purchase I have made in years. Highly recommend!",
        "Complete waste of money. Would not buy again.",
        "Delivery was fast and the packaging was intact.",
        "The instructions were confusing and the parts were missing.",
        "A decent option for the price point.",
    ]

    print("Loading sentiment analysis pipeline...")
    classifier = build_sentiment_pipeline()

    print("Analysing sample texts...")
    results = analyze_texts(classifier, sample_texts)
    print_results(results)

    positive_count = sum(1 for r in results if r["label"] == "POSITIVE")
    negative_count = len(results) - positive_count
    print(f"\nSummary: {positive_count} positive, {negative_count} negative out of {len(results)} texts.")


if __name__ == "__main__":
    main()
