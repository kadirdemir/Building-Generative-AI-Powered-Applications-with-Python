"""
Question Answering with Extractive Reading Comprehension

This module demonstrates extractive question answering using the
transformers library. Given a context passage and a question, the
model identifies the span within the context that best answers
the question.

Extractive QA works by finding the start and end positions of
the answer within the provided context, without generating new text.
"""

from transformers import pipeline


def build_qa_pipeline(model_name: str = "distilbert-base-cased-distilled-squad"):
    """
    Initialize a question-answering pipeline.

    Args:
        model_name: Hugging Face model identifier for the QA task.

    Returns:
        A callable pipeline object ready for inference.
    """
    return pipeline("question-answering", model=model_name)


def answer_question(
    qa_pipeline,
    question: str,
    context: str,
) -> dict:
    """
    Extract an answer from the context for the given question.

    Args:
        qa_pipeline: Initialized question-answering pipeline.
        question: The question to answer.
        context: The passage that contains the answer.

    Returns:
        A dictionary with keys:
            - answer (str): The extracted answer span.
            - score (float): Confidence score between 0 and 1.
            - start (int): Character start index in context.
            - end (int): Character end index in context.
    """
    result = qa_pipeline(question=question, context=context)
    return result


def batch_qa(qa_pipeline, pairs: list[dict]) -> list[dict]:
    """
    Answer multiple question-context pairs in one call.

    Args:
        qa_pipeline: Initialized question-answering pipeline.
        pairs: List of dicts, each with 'question' and 'context' keys.

    Returns:
        List of result dicts corresponding to each input pair.
    """
    return [
        answer_question(qa_pipeline, p["question"], p["context"])
        for p in pairs
    ]


def main():
    print("Loading question-answering pipeline...\n")
    qa = build_qa_pipeline()

    # Example passages and questions
    pairs = [
        {
            "context": (
                "Python is a high-level, general-purpose programming language. "
                "Its design philosophy emphasizes code readability through the use "
                "of significant indentation. Python is dynamically typed and "
                "garbage-collected. It supports multiple programming paradigms, "
                "including structured, object-oriented, and functional programming. "
                "Guido van Rossum began working on Python in the late 1980s as a "
                "successor to the ABC programming language."
            ),
            "question": "Who created Python?",
        },
        {
            "context": (
                "The transformer architecture was introduced in the 2017 paper "
                "'Attention Is All You Need' by Vaswani et al. It relies entirely "
                "on self-attention mechanisms to draw global dependencies between "
                "input and output, dispensing with recurrence and convolutions entirely. "
                "Transformers have become the dominant architecture for natural "
                "language processing tasks."
            ),
            "question": "In which year was the transformer architecture introduced?",
        },
        {
            "context": (
                "Transfer learning is a technique where a model trained on one task "
                "is reused as the starting point for a model on a second task. "
                "It is popular in deep learning because it can allow you to train "
                "deep neural networks with comparatively little data. "
                "Fine-tuning is a common form of transfer learning where only "
                "the last few layers of a pre-trained model are retrained."
            ),
            "question": "What is fine-tuning in the context of transfer learning?",
        },
    ]

    results = batch_qa(qa, pairs)

    for i, (pair, result) in enumerate(zip(pairs, results), start=1):
        print(f"--- Example {i} ---")
        print(f"Question : {pair['question']}")
        print(f"Answer   : {result['answer']}")
        print(f"Score    : {result['score']:.4f}")
        print()


if __name__ == "__main__":
    main()
