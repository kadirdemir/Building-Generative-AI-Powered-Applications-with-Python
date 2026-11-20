"""
Text Translation using the Transformers Library

This module demonstrates how to translate text between languages using
pre-trained sequence-to-sequence models from the Helsinki-NLP collection.
It covers single-sentence translation, batch processing, and round-trip
verification (translate to target language and back to check fidelity).

Dependencies:
    pip install transformers torch sentencepiece
"""

from transformers import pipeline, MarianMTModel, MarianTokenizer


def load_translation_pipeline(src_lang: str, tgt_lang: str):
    """
    Load a MarianMT translation pipeline for the given language pair.

    Args:
        src_lang: Source language code (e.g. 'en', 'de', 'fr').
        tgt_lang: Target language code.

    Returns:
        A Hugging Face translation pipeline.
    """
    model_name = f"Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}"
    translator = pipeline("translation", model=model_name)
    return translator


def translate_text(translator, text: str, max_length: int = 512) -> str:
    """
    Translate a single string.

    Args:
        translator: A loaded translation pipeline.
        text: The source text to translate.
        max_length: Maximum token length for the output.

    Returns:
        Translated string.
    """
    result = translator(text, max_length=max_length)
    return result[0]["translation_text"]


def translate_batch(translator, texts: list[str], max_length: int = 512) -> list[str]:
    """
    Translate a list of strings in a single forward pass.

    Args:
        translator: A loaded translation pipeline.
        texts: List of source strings.
        max_length: Maximum token length per output.

    Returns:
        List of translated strings, in the same order as the input.
    """
    results = translator(texts, max_length=max_length)
    return [r["translation_text"] for r in results]


def round_trip_translation(text: str, src_lang: str, pivot_lang: str) -> dict:
    """
    Translate text to a pivot language and then back to the source language.

    Round-trip translation is a quick sanity check: if the meaning survives
    both hops, the model is likely capturing the semantics well.

    Args:
        text: Original text in `src_lang`.
        src_lang: Source (and final) language code.
        pivot_lang: Intermediate language code.

    Returns:
        Dictionary with keys 'original', 'translated', and 'back_translated'.
    """
    forward = load_translation_pipeline(src_lang, pivot_lang)
    backward = load_translation_pipeline(pivot_lang, src_lang)

    translated = translate_text(forward, text)
    back_translated = translate_text(backward, translated)

    return {
        "original": text,
        "translated": translated,
        "back_translated": back_translated,
    }


def demonstrate_marian_tokenizer(src_lang: str, tgt_lang: str, text: str) -> None:
    """
    Show how the MarianMT tokenizer encodes source text before translation.

    Args:
        src_lang: Source language code.
        tgt_lang: Target language code.
        text: Example sentence to tokenize.
    """
    model_name = f"Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}"
    tokenizer = MarianTokenizer.from_pretrained(model_name)

    tokens = tokenizer(text, return_tensors="pt")
    decoded_ids = tokenizer.convert_ids_to_tokens(tokens["input_ids"][0])

    print(f"\nTokenizer demo for '{model_name}':")
    print(f"  Input text : {text}")
    print(f"  Token IDs  : {tokens['input_ids'][0].tolist()}")
    print(f"  Tokens     : {decoded_ids}")


def main():
    # ------------------------------------------------------------------ #
    # 1.  Single-sentence translation: English → German
    # ------------------------------------------------------------------ #
    print("=== Single-sentence translation (EN → DE) ===")
    en_de = load_translation_pipeline("en", "de")
    sentence = "The quick brown fox jumps over the lazy dog."
    german = translate_text(en_de, sentence)
    print(f"Source   : {sentence}")
    print(f"German   : {german}")

    # ------------------------------------------------------------------ #
    # 2.  Batch translation: English → French
    # ------------------------------------------------------------------ #
    print("\n=== Batch translation (EN → FR) ===")
    en_fr = load_translation_pipeline("en", "fr")
    sentences = [
        "Machine learning models learn patterns from data.",
        "Natural language processing enables computers to understand text.",
        "Transfer learning reduces the need for large labelled datasets.",
    ]
    french_sentences = translate_batch(en_fr, sentences)
    for src, tgt in zip(sentences, french_sentences):
        print(f"  EN: {src}")
        print(f"  FR: {tgt}\n")

    # ------------------------------------------------------------------ #
    # 3.  Round-trip translation: English → Spanish → English
    # ------------------------------------------------------------------ #
    print("=== Round-trip translation (EN → ES → EN) ===")
    original = "Knowledge is the foundation of innovation."
    result = round_trip_translation(original, src_lang="en", pivot_lang="es")
    print(f"  Original       : {result['original']}")
    print(f"  → Spanish      : {result['translated']}")
    print(f"  → Back to EN   : {result['back_translated']}")

    # ------------------------------------------------------------------ #
    # 4.  Tokenizer internals
    # ------------------------------------------------------------------ #
    demonstrate_marian_tokenizer("en", "de", "Hello, world!")

    # ------------------------------------------------------------------ #
    # 5.  Direct model usage (no pipeline wrapper)
    # ------------------------------------------------------------------ #
    print("\n=== Direct model usage (EN → RO) ===")
    model_name = "Helsinki-NLP/opus-mt-en-ro"
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)

    source = "Data is the new oil of the digital economy."
    encoded = tokenizer([source], return_tensors="pt", padding=True)
    translated_ids = model.generate(**encoded)
    romanian = tokenizer.decode(translated_ids[0], skip_special_tokens=True)
    print(f"  EN : {source}")
    print(f"  RO : {romanian}")


if __name__ == "__main__":
    main()
