# -*- coding: utf-8 -*-
import numpy as np


def binarize_top_percent(score, top_percent=0.15):
    """
    Selects the top_percent frames of the video to be included in the summary, by binarizing the top percentage of scores into a binary indicator array.
    :param numpy.ndarray score: 1D array of scores for frames.
    :param float top_percent: Percentage of frames to be included in the summary. Defaults to 0.15.
    :return numpy.ndarray: Binary array with values 1.0 for the top_percent highest scores and 0.0 otherwise.
    """
    n = score.size
    k = int((top_percent * n))
    sorted_descending = np.sort(score)[::-1]
    threshold_val = sorted_descending[k]
    # Create binary array (values ≥ threshold become 1)
    binary_score = (score >= threshold_val).astype(np.float32)
    return binary_score


def evaluate_summary(score, gtscore, selection_upper_limit=0.25):
    """
    Compute the F-score between a predicted and ground truth video summary.
    :param torch.Tensor | numpy.ndarray score: Predicted importance scores for each frame.
    :param array-like gtscore: Ground truth binary summary for each frame (0 or 1).
    :param float selection_upper_limit: Upper limit of accepted selected frames in the summary (in case there are equal scores in frames)
    :return float: f_score (%) evaluating overlap between predicted and ground truth summary.
    """

    predicted_summary = binarize_top_percent(score.numpy(), top_percent=0.15)
    limit = np.mean(predicted_summary)
    problematic_selection = False
    if limit > selection_upper_limit:  # Summary should be ~15%, but this is not always possible. If however the selection of frames leads to a summary with more than 25%, ignore this summary.
        problematic_selection = True
        print(limit)
    max_len = max(len(predicted_summary), len(gtscore))
    S = np.zeros(max_len, dtype=int)
    G = np.zeros(max_len, dtype=int)
    S[:len(predicted_summary)] = predicted_summary
    G[:len(gtscore)] = gtscore
    overlapped = S & G
    # Compute precision, recall, f-score
    precision = sum(overlapped) / sum(S + 1e-8)
    recall = sum(overlapped) / sum(G + 1e-8)
    if precision + recall == 0 or problematic_selection:
        f_score = 0
    else:
        f_score = (2 * precision * recall * 100) / (precision + recall)
    return f_score
