import numpy
from statistics import mean
from typing import List

def check_model_drift(current_answer: str, previous_answer_lengths: List[int], window_size: int = 20, threshold: int = 50) -> bool:
    """
    Checks for model drift based on significant changes in answer length.
    Compares the current answer length to the average length of the last 'window_size' answers.
    """
    if not previous_answer_lengths:
        return False

    last_n_lengths = previous_answer_lengths[-window_size:]
    avg_previous_length = mean(last_n_lengths) if last_n_lengths else 0

    if abs(len(current_answer) - avg_previous_length) > threshold:
        return True
    return False


def check_token_drift(current_tokens: int, previous_tokens: List[int], window_size: int = 20, threshold: int = 20) -> bool:
    """
    Checks for token drift based on significant changes in token count.
    Compares the current token count to the average token count of the last 'window_size' answers.
    """
    if not previous_tokens:
        return False

    last_n_tokens = previous_tokens[-window_size:]
    avg_previous_tokens = mean(last_n_tokens) if last_n_tokens else 0

    if abs(current_tokens - avg_previous_tokens) > threshold:
        return True
    return False

def validate_context(context):
    return context if context.strip() not in {"", "seed document"} else "No relevant context available"



def summarize_text(summarizer, text: str, max_length: int = 150) -> str:
    """Summarize text using DistilGPT-2 with prompt engineering"""
    prompt = f"Summarize this in one sentence:\n{text}\nSummary:"
    
    summary = summarizer(
        prompt,
        max_new_tokens=max_length,
        temperature=0.3,  # Lower = more deterministic
        no_repeat_ngram_size=2,
        do_sample=False  # Disable randomness for consistency
    )
    print("summary: ", summary)
    summary = summary #[0]['generated_text']
    
    # Extract just the summary part
    return summary.split("Summary:")[-1].strip()[:50]