import nltk
import statistics

# Ensure the tokenizer model is available
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt')
    nltk.download('punkt_tab')

def gopher_quality_filter(text: str) -> bool:
    """
    Applies the Gopher quality filters to a text document.

    Returns True if the document passes all filters, False otherwise.

    Filters:
    1. Word count between 50 and 100,000.
    2. Mean word length between 3 and 10 characters.
    3. Less than 30% of lines end with an ellipsis.
    4. At least 80% of words contain at least one alphabetic character.
    """
    # Tokenize text into words
    words = nltk.word_tokenize(text)
    num_words = len(words)

    # 1. Check Word Count
    # "Contain less than 50 or more than 100,000 words."
    if num_words < 50 or num_words > 100000:
        return False

    # 2. Check Mean Word Length
    # "Have a mean word length outside the range of 3 to 10 characters."
    word_lengths = [len(w) for w in words]
    if num_words > 0:
        mean_word_len = statistics.mean(word_lengths)
        if mean_word_len < 3 or mean_word_len > 10:
            return False

    # 3. Check Ellipsis Lines
    # "Have more than 30% of lines ending with an ellipsis ('...')."
    lines = text.splitlines()
    if len(lines) > 0:
        # We strip whitespace from the right to catch "..." followed by a space
        ellipsis_lines = sum(1 for line in lines if line.rstrip().endswith("..."))
        if (ellipsis_lines / len(lines)) > 0.3:
            return False

    # 4. Check Alphabetic Words
    # "Contain less than 80% of words with at least one alphabetic character."
    # (This filters out documents that are mostly numbers or symbols/code)
    alpha_words = sum(1 for w in words if any(c.isalpha() for c in w))
    if num_words > 0:
        if (alpha_words / num_words) < 0.8:
            return False

    return True