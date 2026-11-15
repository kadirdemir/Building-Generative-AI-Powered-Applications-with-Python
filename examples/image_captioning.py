"""
Image Captioning with Vision-Language Models

This module demonstrates automatic caption generation for images using
the Salesforce BLIP (Bootstrapped Language-Image Pre-training) model
via the Hugging Face transformers library.

Two captioning modes are shown:
  - Unconditional: the model freely describes the image.
  - Conditional:   the model continues from a user-supplied text prefix.

Requirements:
    pip install transformers torch Pillow requests

Usage:
    python image_captioning.py
"""

import requests
from io import BytesIO

from PIL import Image
from transformers import BlipForConditionalGeneration, BlipProcessor


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------

def load_model(model_name: str = "Salesforce/blip-image-captioning-base"):
    """Load the BLIP processor and captioning model from the Hub."""
    processor = BlipProcessor.from_pretrained(model_name)
    model = BlipForConditionalGeneration.from_pretrained(model_name)
    return processor, model


def fetch_image_from_url(url: str) -> Image.Image:
    """Download an image from *url* and return a PIL Image in RGB mode."""
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    return Image.open(BytesIO(response.content)).convert("RGB")


# ---------------------------------------------------------------------------
# Captioning functions
# ---------------------------------------------------------------------------

def caption_unconditional(
    image: Image.Image,
    processor: BlipProcessor,
    model: BlipForConditionalGeneration,
    max_new_tokens: int = 50,
) -> str:
    """Generate a free-form caption without any text prompt."""
    inputs = processor(image, return_tensors="pt")
    output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    return processor.decode(output_ids[0], skip_special_tokens=True)


def caption_conditional(
    image: Image.Image,
    text_prefix: str,
    processor: BlipProcessor,
    model: BlipForConditionalGeneration,
    max_new_tokens: int = 50,
) -> str:
    """Generate a caption that starts with *text_prefix*."""
    inputs = processor(image, text_prefix, return_tensors="pt")
    output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    return processor.decode(output_ids[0], skip_special_tokens=True)


def caption_batch(
    images: list[Image.Image],
    processor: BlipProcessor,
    model: BlipForConditionalGeneration,
    max_new_tokens: int = 50,
) -> list[str]:
    """Caption a list of images in a single forward pass."""
    inputs = processor(images, return_tensors="pt", padding=True)
    output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    return [
        processor.decode(ids, skip_special_tokens=True) for ids in output_ids
    ]


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

SAMPLE_URLS = [
    "https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/PNG_transparency_demonstration_1.png/280px-PNG_transparency_demonstration_1.png",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/320px-Cat03.jpg",
]

CONDITIONAL_PREFIXES = [
    "a photograph of",
    "this image shows",
]


def main():
    print("Loading BLIP image-captioning model …")
    processor, model = load_model()
    print("Model ready.\n")

    # --- Single-image unconditional ---
    print("=== Unconditional captioning ===")
    for url in SAMPLE_URLS:
        try:
            img = fetch_image_from_url(url)
            caption = caption_unconditional(img, processor, model)
            print(f"URL : {url}")
            print(f"Caption: {caption}\n")
        except Exception as exc:
            print(f"Could not process {url}: {exc}\n")

    # --- Single-image conditional ---
    print("=== Conditional captioning ===")
    try:
        img = fetch_image_from_url(SAMPLE_URLS[1])
        for prefix in CONDITIONAL_PREFIXES:
            caption = caption_conditional(img, prefix, processor, model)
            print(f"Prefix : '{prefix}'")
            print(f"Caption: {caption}\n")
    except Exception as exc:
        print(f"Could not run conditional captioning: {exc}\n")

    # --- Batch captioning ---
    print("=== Batch captioning ===")
    images = []
    loaded_urls = []
    for url in SAMPLE_URLS:
        try:
            images.append(fetch_image_from_url(url))
            loaded_urls.append(url)
        except Exception as exc:
            print(f"Skipping {url}: {exc}")

    if images:
        captions = caption_batch(images, processor, model)
        for url, cap in zip(loaded_urls, captions):
            print(f"URL    : {url}")
            print(f"Caption: {cap}\n")


if __name__ == "__main__":
    main()
