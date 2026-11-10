"""
Text Summarization with Transformer-based Models
=================================================

This module demonstrates extractive and abstractive text summarization
using the Hugging Face `transformers` library with a sequence-to-sequence
model (facebook/bart-large-cnn).

Key concepts covered:
- Loading a pre-trained summarization pipeline
- Adjusting summary length via min_length / max_length parameters
- Batch summarization of multiple documents
- Comparing short vs. detailed summary outputs

Dependencies:
    pip install transformers torch
"""

from transformers import pipeline


SAMPLE_TEXTS = [
    """
    The Amazon rainforest, often referred to as the "lungs of the Earth," covers more than
    5.5 million square kilometres across nine countries in South America. It produces roughly
    20 percent of the world's oxygen and houses an estimated 10 percent of all species on the
    planet, many of which have yet to be identified by scientists. Deforestation driven by
    agriculture, logging, and infrastructure development has destroyed nearly 17 percent of
    the forest over the past five decades. Conservation organisations and governments are
    working together on monitoring programmes that use satellite imagery to detect illegal
    clearing in near real time, enabling faster enforcement responses than traditional
    ground-based patrols.
    """,
    """
    Quantum computing leverages the principles of superposition and entanglement to process
    information in ways that classical computers cannot. While a traditional bit can represent
    either 0 or 1, a quantum bit (qubit) can represent both simultaneously, enabling quantum
    processors to explore vast solution spaces in parallel. Current quantum hardware suffers
    from high error rates caused by environmental interference — a challenge researchers call
    decoherence. Error-correction schemes that require thousands of physical qubits to protect
    a single logical qubit are an active area of research. Despite these hurdles, early
    demonstrations have shown that specific optimisation and simulation tasks can be solved
    exponentially faster on quantum devices than on the best available classical supercomputers.
    """,
]


def load_summarizer(model_name: str = "facebook/bart-large-cnn") -> pipeline:
    """Load and return a summarization pipeline.

    Args:
        model_name: Hugging Face model identifier for the summarizer.

    Returns:
        A Hugging Face summarization pipeline instance.
    """
    print(f"Loading summarization model: {model_name}")
    summarizer = pipeline("summarization", model=model_name)
    print("Model loaded successfully.\n")
    return summarizer


def summarize_text(
    summarizer: pipeline,
    text: str,
    max_length: int = 130,
    min_length: int = 30,
    do_sample: bool = False,
) -> str:
    """Generate a summary for the given text.

    Args:
        summarizer:  A pre-loaded summarization pipeline.
        text:        The source document to summarize.
        max_length:  Maximum number of tokens in the summary.
        min_length:  Minimum number of tokens in the summary.
        do_sample:   Whether to use sampling (True) or greedy decoding (False).

    Returns:
        The generated summary string.
    """
    result = summarizer(
        text.strip(),
        max_length=max_length,
        min_length=min_length,
        do_sample=do_sample,
    )
    return result[0]["summary_text"]


def batch_summarize(
    summarizer: pipeline,
    texts: list[str],
    max_length: int = 100,
    min_length: int = 25,
) -> list[str]:
    """Summarize a list of documents in one batched call.

    Args:
        summarizer: A pre-loaded summarization pipeline.
        texts:      List of source documents.
        max_length: Maximum summary token length.
        min_length: Minimum summary token length.

    Returns:
        List of summary strings, one per input document.
    """
    cleaned = [t.strip() for t in texts]
    results = summarizer(cleaned, max_length=max_length, min_length=min_length, do_sample=False)
    return [r["summary_text"] for r in results]


def main() -> None:
    """Run the text summarization demonstration."""
    summarizer = load_summarizer()

    # --- Single document, two length variants ---
    text = SAMPLE_TEXTS[0]
    print("=" * 60)
    print("SOURCE TEXT (truncated):")
    print(text[:300].strip(), "...")

    short_summary = summarize_text(summarizer, text, max_length=60, min_length=20)
    detailed_summary = summarize_text(summarizer, text, max_length=130, min_length=50)

    print("\nSHORT SUMMARY:")
    print(short_summary)
    print("\nDETAILED SUMMARY:")
    print(detailed_summary)

    # --- Batch summarization ---
    print("\n" + "=" * 60)
    print("BATCH SUMMARIZATION (2 documents)\n")
    summaries = batch_summarize(summarizer, SAMPLE_TEXTS)
    for idx, (original, summary) in enumerate(zip(SAMPLE_TEXTS, summaries), start=1):
        print(f"Document {idx} summary:")
        print(summary)
        print()


if __name__ == "__main__":
    main()
