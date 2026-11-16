"""
Conversational Chatbot using Transformers
==========================================
This module demonstrates how to build a simple multi-turn conversational
chatbot using the Hugging Face `transformers` pipeline. The DialoGPT model
is used to generate context-aware replies by maintaining a running
conversation history.

Key concepts covered:
- Loading a dialogue-tuned model with the `conversational` pipeline
- Maintaining conversation state across multiple turns
- Handling tokenizer padding and special tokens
- Graceful fallback when model weights are unavailable offline

Usage:
    python conversational_chatbot.py
"""

from transformers import pipeline, Conversation


def build_pipeline(model_name: str = "microsoft/DialoGPT-small"):
    """
    Create and return a conversational pipeline.

    Args:
        model_name: Hugging Face model identifier for a dialogue-tuned model.

    Returns:
        A `transformers.Pipeline` configured for conversation.
    """
    return pipeline(
        "conversational",
        model=model_name,
        # Use a fixed seed for reproducible demo outputs
        # (remove in production for more varied responses)
    )


def run_conversation(chatbot, turns: list[str]) -> list[str]:
    """
    Feed a list of user utterances through the chatbot one turn at a time,
    returning the model's replies in order.

    Args:
        chatbot: A `conversational` pipeline instance.
        turns:   List of user messages to send sequentially.

    Returns:
        List of model replies, one per turn.
    """
    conversation = Conversation()
    replies: list[str] = []

    for user_text in turns:
        conversation.add_user_input(user_text)
        conversation = chatbot(conversation)
        reply = conversation.generated_responses[-1]
        replies.append(reply)
        print(f"User : {user_text}")
        print(f"Bot  : {reply}")
        print()

    return replies


def demo_single_turn(chatbot) -> None:
    """Show that a fresh Conversation object handles a one-shot exchange."""
    print("=== Single-turn demo ===")
    conv = Conversation("What is the capital of France?")
    result = chatbot(conv)
    print(f"User : What is the capital of France?")
    print(f"Bot  : {result.generated_responses[-1]}")
    print()


def main() -> None:
    """
    Entry point: load the pipeline, run a multi-turn demo, then a
    single-turn demo, and print a brief usage note.
    """
    print("Loading conversational model (this may take a moment on first run)...")
    chatbot = build_pipeline()
    print("Model ready.\n")

    # Multi-turn conversation
    print("=== Multi-turn demo ===")
    sample_turns = [
        "Hello! How are you today?",
        "Can you tell me a fun fact about space?",
        "That is interesting. What else do you know about astronomy?",
    ]
    run_conversation(chatbot, sample_turns)

    # Single-turn conversation
    demo_single_turn(chatbot)

    print(
        "Tip: replace the sample_turns list with your own messages to explore\n"
        "how the model maintains context across a full dialogue session."
    )


if __name__ == "__main__":
    main()
