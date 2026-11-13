"""
Named Entity Recognition (NER) with Hugging Face Transformers

This module demonstrates how to identify and classify named entities in text,
such as persons, organizations, locations, dates, and other proper nouns.
Named entity recognition is a fundamental natural language processing task used
in information extraction, document analysis, and text understanding pipelines.

Dependencies:
    pip install transformers torch
"""

from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification


def load_ner_pipeline(model_name: str = "dslim/bert-base-NER") -> pipeline:
    """
    Load a token classification pipeline for named entity recognition.

    Args:
        model_name: Hugging Face model identifier for NER task.

    Returns:
        A Hugging Face pipeline configured for token classification.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(model_name)
    ner_pipeline = pipeline(
        "ner",
        model=model,
        tokenizer=tokenizer,
        aggregation_strategy="simple"
    )
    return ner_pipeline


def extract_entities(ner_pipeline: pipeline, text: str) -> list[dict]:
    """
    Extract named entities from the given text.

    Args:
        ner_pipeline: A loaded NER pipeline.
        text: Input text to analyze.

    Returns:
        A list of dictionaries, each containing entity group, word, score,
        start and end character positions.
    """
    results = ner_pipeline(text)
    return results


def format_entities(entities: list[dict]) -> str:
    """
    Format extracted entities into a human-readable string.

    Args:
        entities: List of entity dictionaries from the NER pipeline.

    Returns:
        A formatted string displaying each entity and its type.
    """
    if not entities:
        return "No entities found."

    lines = []
    for entity in entities:
        label = entity.get("entity_group", entity.get("entity", "UNKNOWN"))
        word = entity["word"]
        score = entity["score"]
        lines.append(f"  [{label}] '{word}' (confidence: {score:.2%})")
    return "\n".join(lines)


def analyze_text(ner_pipeline: pipeline, text: str) -> None:
    """
    Analyze a text sample and print its named entities.

    Args:
        ner_pipeline: A loaded NER pipeline.
        text: Input text to analyze.
    """
    print(f"\nText:\n  {text}")
    entities = extract_entities(ner_pipeline, text)
    print("Entities found:")
    print(format_entities(entities))
    print("-" * 60)


def main():
    """
    Demonstrate named entity recognition on several sample sentences.
    """
    print("Named Entity Recognition Demo")
    print("=" * 60)

    print("\nLoading NER model (dslim/bert-base-NER)...")
    ner = load_ner_pipeline()
    print("Model loaded successfully.\n")

    sample_texts = [
        "Apple Inc. was founded by Steve Jobs and Steve Wozniak in Cupertino, California in 1976.",
        "The World Health Organization announced new guidelines in Geneva last Tuesday.",
        "Elon Musk, CEO of Tesla, spoke at the Berlin conference about electric vehicles.",
        "Amazon Web Services opened a new data center in São Paulo, Brazil.",
        "The Paris Agreement was signed by 196 countries at the United Nations Climate Change Conference.",
    ]

    for text in sample_texts:
        analyze_text(ner, text)

    # Demonstrate entity type distribution
    all_entities = []
    for text in sample_texts:
        all_entities.extend(extract_entities(ner, text))

    entity_types: dict[str, int] = {}
    for entity in all_entities:
        label = entity.get("entity_group", entity.get("entity", "UNKNOWN"))
        entity_types[label] = entity_types.get(label, 0) + 1

    print("\nEntity Type Summary:")
    for entity_type, count in sorted(entity_types.items(), key=lambda x: x[1], reverse=True):
        print(f"  {entity_type}: {count}")


if __name__ == "__main__":
    main()
