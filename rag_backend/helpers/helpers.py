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