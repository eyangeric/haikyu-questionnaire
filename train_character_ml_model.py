from random import choice
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import joblib


def assign_score(score_range: tuple) -> float:
    scores = np.arange(score_range[0], score_range[1], 0.1)
    random_score = choice(scores)
    return random_score

def impute_scores(score_range: tuple,
                  iterations: int) -> list:
    scores = [assign_score(score_range) for i in range(iterations)]
    return scores

def score_questions(questions: dict) -> pd.DataFrame:
    df = pd.DataFrame()
    for key, value in questions.items():
        df[key] = impute_scores(value)
    return df

def score_characters(characters: list) -> pd.DataFrame:
    dfs = []
    for character in characters:
        name = character['name']
        character_df = character['questions']
        character_df['character'] = name
        dfs.append(character_df)
    df = pd.concat(dfs)
    return df

def fit_character_model(df: pd.DataFrame, outcome: str, features: list):
    label_encoder = LabelEncoder()
    clf = RandomForestClassifier()
    df[f'{outcome}_num'] = label_encoder.fit_transform(outcome)
    clf.fit(df[features], df[f'{outcome}_num'])
    return clf





