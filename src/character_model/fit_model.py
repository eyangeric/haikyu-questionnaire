from random import choice
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import pickle
from google.cloud import storage
from io import BytesIO
import sys
sys.path.append('../../')
from config import ML_MODEL_GCS_BUCKET, ML_MODEL_FILE_NAME
from character_config import character_score_ranges


def assign_score(score_range: tuple) -> float:
    scores = np.arange(score_range[0], score_range[1], 0.1)
    random_score = choice(scores)
    return random_score


def impute_scores(score_range: tuple,
                  iterations: 100) -> list:
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
        questions = character['questions']
        character_df = score_questions(questions)
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


def export_model(model: RandomForestClassifier) -> None:
    cs_client = storage.Client()
    bucket = cs_client.bucket(ML_MODEL_GCS_BUCKET)
    blob = bucket.blob(ML_MODEL_FILE_NAME)
    model_bytes = BytesIO()
    pickle.dump(model, model_bytes)
    model_bytes.seek(0)
    blob.upload_from_file(model_bytes, content_type='application/octet-stream')





