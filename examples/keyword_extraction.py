"""
Keyword Extraction from Text Documents

This module demonstrates how to extract meaningful keywords and keyphrases
from text using TF-IDF (Term Frequency-Inverse Document Frequency) and
YAKE (Yet Another Keyword Extractor) approaches.

TF-IDF scores words based on their frequency within a document relative
to their frequency across a corpus, identifying terms that are both
frequent and distinctive.

YAKE is a statistical, unsupervised, and domain-independent keyword
extraction method that works at the sentence and document level.

Requirements:
    pip install scikit-learn yake
"""

import re
import math
from collections import Counter
from typing import List, Tuple


def preprocess_text(text: str) -> List[str]:
    """
    Clean and tokenize text into individual words.

    Args:
        text: Raw input text string.

    Returns:
        List of lowercase tokens with punctuation removed.
    """
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    tokens = text.split()
    return tokens


def get_stopwords() -> set:
    """
    Return a set of common English stopwords to filter out.

    Returns:
        Set of stopword strings.
    """
    return {
        'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to',
        'for', 'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were',
        'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
        'will', 'would', 'could', 'should', 'may', 'might', 'shall', 'can',
        'that', 'this', 'these', 'those', 'it', 'its', 'as', 'up', 'out',
        'about', 'into', 'through', 'during', 'before', 'after', 'above',
        'below', 'between', 'each', 'more', 'most', 'other', 'some', 'such',
        'than', 'too', 'very', 's', 't', 'just', 'not', 'also', 'so',
        'then', 'there', 'when', 'where', 'which', 'who', 'how', 'what',
        'all', 'both', 'any', 'their', 'they', 'them', 'we', 'our', 'us',
        'you', 'your', 'he', 'she', 'his', 'her', 'i', 'my', 'me', 'if',
        'no', 'while', 'now', 'only', 'even', 'much', 'many', 'over'
    }


def compute_tfidf(
    documents: List[str],
    top_n: int = 10,
    min_word_length: int = 3
) -> List[List[Tuple[str, float]]]:
    """
    Compute TF-IDF scores for each document and return top keywords.

    TF (Term Frequency) = count of term in document / total terms in document
    IDF (Inverse Document Frequency) = log(total documents / documents containing term)

    Args:
        documents: List of text documents.
        top_n: Number of top keywords to return per document.
        min_word_length: Minimum number of characters for a word to be considered.

    Returns:
        List of (keyword, score) tuples for each document.
    """
    stopwords = get_stopwords()

    # Tokenize all documents
    tokenized_docs = []
    for doc in documents:
        tokens = preprocess_text(doc)
        filtered = [
            t for t in tokens
            if t not in stopwords and len(t) >= min_word_length
        ]
        tokenized_docs.append(filtered)

    # Build vocabulary and document frequencies
    num_docs = len(documents)
    doc_freq = Counter()
    for tokens in tokenized_docs:
        unique_tokens = set(tokens)
        doc_freq.update(unique_tokens)

    # Compute TF-IDF for each document
    results = []
    for tokens in tokenized_docs:
        if not tokens:
            results.append([])
            continue

        term_count = Counter(tokens)
        total_terms = len(tokens)

        tfidf_scores = {}
        for term, count in term_count.items():
            tf = count / total_terms
            # Smoothed IDF to avoid division by zero
            idf = math.log((num_docs + 1) / (doc_freq[term] + 1)) + 1.0
            tfidf_scores[term] = tf * idf

        sorted_keywords = sorted(
            tfidf_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        results.append(sorted_keywords[:top_n])

    return results


def extract_keyphrases(
    text: str,
    phrase_length: int = 2,
    top_n: int = 10
) -> List[Tuple[str, int]]:
    """
    Extract multi-word keyphrases using n-gram frequency analysis.

    Counts adjacent word pairs (bigrams) or triplets (trigrams) to find
    frequently occurring phrases that may be more meaningful than
    individual keywords.

    Args:
        text: Input text document.
        phrase_length: Number of words per phrase (2 for bigrams, 3 for trigrams).
        top_n: Number of top phrases to return.

    Returns:
        List of (phrase, count) tuples sorted by frequency.
    """
    stopwords = get_stopwords()
    tokens = preprocess_text(text)
    filtered_tokens = [
        t for t in tokens
        if t not in stopwords and len(t) >= 3
    ]

    # Generate n-grams
    phrases = []
    for i in range(len(filtered_tokens) - phrase_length + 1):
        phrase = ' '.join(filtered_tokens[i:i + phrase_length])
        phrases.append(phrase)

    phrase_counts = Counter(phrases)
    # Only keep phrases that appear more than once
    meaningful = [(p, c) for p, c in phrase_counts.most_common(top_n * 2) if c > 1]
    return meaningful[:top_n]


def summarize_keywords(keywords: List[Tuple[str, float]], doc_index: int) -> None:
    """
    Print a formatted summary of extracted keywords for a document.

    Args:
        keywords: List of (keyword, score) tuples.
        doc_index: Index number of the document (for display).
    """
    print(f"\nDocument {doc_index + 1} — Top Keywords:")
    print("-" * 40)
    if not keywords:
        print("  No keywords found.")
        return
    for rank, (word, score) in enumerate(keywords, start=1):
        print(f"  {rank:2d}. {word:<20s} (score: {score:.4f})")


def main() -> None:
    """
    Demonstrate keyword extraction on a small corpus of sample documents.

    Runs both TF-IDF keyword scoring and n-gram keyphrase extraction,
    then prints the results for each sample document.
    """
    sample_documents = [
        """
        Machine learning is a subset of computer science that focuses on building
        systems that learn from data. Supervised learning algorithms are trained on
        labeled datasets to predict outputs. Common algorithms include decision trees,
        random forests, support vector machines, and gradient boosting methods.
        Feature engineering plays a critical role in improving model performance.
        Cross-validation helps evaluate generalization ability and prevent overfitting.
        """,
        """
        Climate change refers to long-term shifts in global temperatures and weather
        patterns. Greenhouse gases, primarily carbon dioxide and methane, trap heat
        in the atmosphere. Rising sea levels, more frequent extreme weather events,
        and biodiversity loss are among the major consequences of global warming.
        Renewable energy sources like solar and wind power reduce fossil fuel dependence
        and contribute to sustainable development goals.
        """,
        """
        Python is a versatile programming language known for its simple and readable
        syntax. It supports multiple programming paradigms including object-oriented,
        functional, and procedural programming. Popular Python libraries include NumPy
        for numerical computing, Pandas for data manipulation, and Matplotlib for
        data visualization. Python is widely used in web development, scientific
        computing, automation, and data analysis workflows.
        """
    ]

    print("=" * 60)
    print("TF-IDF KEYWORD EXTRACTION")
    print("=" * 60)

    keyword_results = compute_tfidf(sample_documents, top_n=8)
    for idx, keywords in enumerate(keyword_results):
        summarize_keywords(keywords, idx)

    print("\n" + "=" * 60)
    print("KEYPHRASE EXTRACTION (Bigrams)")
    print("=" * 60)

    for idx, doc in enumerate(sample_documents):
        keyphrases = extract_keyphrases(doc, phrase_length=2, top_n=5)
        print(f"\nDocument {idx + 1} — Top Bigram Keyphrases:")
        print("-" * 40)
        if keyphrases:
            for phrase, count in keyphrases:
                print(f"  '{phrase}'  (appears {count}x)")
        else:
            print("  No repeated bigrams found.")

    print("\n" + "=" * 60)
    print("CUSTOM TEXT INPUT EXAMPLE")
    print("=" * 60)

    custom_text = """
    Neural networks are computational models inspired by the structure of the human brain.
    Deep learning uses multiple layers of neural networks to learn hierarchical representations
    of data. Convolutional neural networks excel at image recognition tasks, while recurrent
    neural networks are well-suited for sequential data such as time series and text.
    Training deep neural networks requires large datasets and significant computational resources.
    Backpropagation and gradient descent are fundamental optimization techniques used during
    the neural network training process.
    """

    print("\nCustom document keyword extraction:")
    custom_results = compute_tfidf([custom_text], top_n=10)
    summarize_keywords(custom_results[0], 0)

    print("\nCustom document bigram keyphrases:")
    custom_phrases = extract_keyphrases(custom_text, phrase_length=2, top_n=5)
    print("-" * 40)
    for phrase, count in custom_phrases:
        print(f"  '{phrase}'  (appears {count}x)")

    print("\nDone.")


if __name__ == '__main__':
    main()
