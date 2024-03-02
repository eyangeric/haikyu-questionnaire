from random import choice
import numpy as np

def assign_score(score_range: tuple) -> float:
    scores = np.arange(score_range[0], score_range[1], 0.1)
    random_score = choice(scores)
    return random_score

def impute_scores(score_range: tuple,
                  iterations: int) -> list:
    scores = [assign_score(score_range) for i in range(iterations)]
    return scores
