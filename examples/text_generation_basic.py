"""
Basic Text Generation Example
Demonstrates simple text generation using the Hugging Face transformers library.
"""

from transformers import pipeline

def generate_text(prompt: str, max_length: int = 100) -> str:
    """Generate text from a prompt using GPT-2."""
    generator = pipeline("text-generation", model="gpt2")
    result = generator(prompt, max_length=max_length, num_return_sequences=1)
    return result[0]["generated_text"]


def main():
    prompts = [
        "Artificial intelligence is transforming",
        "The future of generative AI will",
        "Python is the best language for machine learning because",
    ]

    for prompt in prompts:
        print(f"\nPrompt: {prompt}")
        print(f"Generated: {generate_text(prompt)}")
        print("-" * 60)


if __name__ == "__main__":
    main()
