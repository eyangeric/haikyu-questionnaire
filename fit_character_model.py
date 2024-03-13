from random import choice
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import pickle
from google.cloud import storage
from io import BytesIO
from config import ML_MODEL_GCS_BUCKET, ML_MODEL_FILE_NAME


character_score_ranges = [
    {
        "name": "Daichi Sawamura",
        "questions": {
            1: (5, 6),
            2: (7, 9),
            3: (10, 10),
            4: (9, 10),
            5: (5, 7),
            6: (5, 7),
            7: (5, 7),
            8: (6, 7),
            9: (6, 6),
            10: (7, 8),
        },
    },
    {
        "name": "Koshi Sugawara",
        "questions": {
            1: (5, 6),
            2: (8, 9),
            3: (10, 10),
            4: (10, 10),
            5: (5, 7),
            6: (5, 7),
            7: (5, 6),
            8: (6, 8),
            9: (5, 6),
            10: (6, 8),
        },
    },
    {
        "name": "Asahi Azumane",
        "questions": {
            1: (6, 8),
            2: (5, 7),
            3: (3, 5),
            4: (4, 7),
            5: (6, 8),
            6: (5, 7),
            7: (5, 5),
            8: (3, 7),
            9: (2, 5),
            10: (5, 7),
        },
    },
    {
        "name": "Yu Nishinoya",
        "questions": {
            1: (8, 10),
            2: (1, 3),
            3: (4, 6),
            4: (1, 4),
            5: (10, 10),
            6: (10, 10),
            7: (5, 7),
            8: (9, 10),
            9: (7, 10),
            10: (8, 10),
        },
    },
    {
        "name": "Ryunosuke Tanaka",
        "questions": {
            1: (1, 3),
            2: (2, 4),
            3: (1, 3),
            4: (1, 4),
            5: (1, 2),
            6: (7, 10),
            7: (5, 5),
            8: (9, 10),
            9: (8, 10),
            10: (8, 10),
        },
    },
    {
        "name": "Tobio Kageyama",
        "questions": {
            1: (1, 1),
            2: (7, 10),
            3: (3, 8),
            4: (1, 2),
            5: (5, 5),
            6: (10, 10),
            7: (10, 10),
            8: (10, 10),
            9: (10, 10),
            10: (10, 10),
        },
    },
    {
        "name": "Shoyo Hinata",
        "questions": {
            1: (1, 1),
            2: (1, 4),
            3: (1, 4),
            4: (6, 7),
            5: (10, 10),
            6: (10, 10),
            7: (10, 10),
            8: (10, 10),
            9: (10, 10),
            10: (10, 10),
        },
    },
    {
        "name": "Kei Tsukishima",
        "questions": {
            1: (5, 7),
            2: (10, 10),
            3: (10, 10),
            4: (5, 10),
            5: (5, 5),
            6: (5, 7),
            7: (5, 5),
            8: (4, 6),
            9: (1, 2),
            10: (5, 7),
        },
    },
    {
        "name": "Tadashi Yamaguchi",
        "questions": {
            1: (4, 7),
            2: (7, 9),
            3: (7, 9),
            4: (9, 10),
            5: (5, 5),
            6: (4, 6),
            7: (5, 5),
            8: (4, 7),
            9: (1, 2),
            10: (5, 7),
        },
    },
    {
        "name": "Tetsuro Kuroo",
        "questions": {
            1: (1, 1),
            2: (8, 10),
            3: (8, 10),
            4: (6, 10),
            5: (5, 5),
            6: (7, 9),
            7: (6, 8),
            8: (7, 9),
            9: (3, 7),
            10: (6, 9),
        },
    },
    {
        "name": "Kenma Kozume",
        "questions": {
            1: (10, 10),
            2: (10, 10),
            3: (10, 10),
            4: (10, 10),
            5: (1, 5),
            6: (6, 8),
            7: (6, 8),
            8: (4, 6),
            9: (1, 1),
            10: (6, 10),
        },
    },
    {
        "name": "Toru Oikawa",
        "questions": {
            1: (1, 1),
            2: (7, 10),
            3: (2, 4),
            4: (1, 3),
            5: (8, 10),
            6: (10, 10),
            7: (10, 10),
            8: (10, 10),
            9: (10, 10),
            10: (10, 10),
        },
    },
    {
        "name": "Wakatoshi Ushijima",
        "questions": {
            1: (1, 1),
            2: (3, 8),
            3: (6, 9),
            4: (10, 10),
            5: (3, 5),
            6: (4, 6),
            7: (10, 10),
            8: (10, 10),
            9: (10, 10),
            10: (10, 10),
        },
    },
]


def assign_score(score_range: tuple) -> float:
    if score_range[0] == score_range[1]:
        random_score = score_range[0]
    else:
        scores = np.arange(score_range[0], score_range[1], 0.1)
        random_score = choice(scores)
    return random_score


def impute_scores(score_range: tuple,
                  iterations: int) -> list:
    scores = [assign_score(score_range) for i in range(iterations)]
    return scores


def score_questions(questions: dict,
                    iterations: int) -> pd.DataFrame:
    df = pd.DataFrame()
    for key, value in questions.items():
        df[key] = impute_scores(value, iterations)
    return df


def score_characters(characters: list,
                     iterations: int) -> pd.DataFrame:
    dfs = []
    for character in characters:
        name = character['name']
        questions = character['questions']
        character_df = score_questions(questions, iterations)
        character_df['character'] = name
        dfs.append(character_df)
    df = pd.concat(dfs)
    return df


def fit_character_model(df: pd.DataFrame, outcome: str, features: list) -> RandomForestClassifier:
    label_encoder = LabelEncoder()
    clf = RandomForestClassifier()
    df[f'{outcome}_num'] = label_encoder.fit_transform(df[f'{outcome}'])
    clf.fit(df[features], df[f'{outcome}_num'])
    return clf


def export_model(model: RandomForestClassifier,
                 bucket_name: str,
                 file_name: str) -> None:
    cs_client = storage.Client()
    bucket = cs_client.bucket(bucket_name)
    blob = bucket.blob(file_name)
    model_bytes = BytesIO()
    pickle.dump(model, model_bytes)
    model_bytes.seek(0)
    blob.upload_from_file(model_bytes, content_type='application/octet-stream')


df = score_characters(character_score_ranges, 10000)
model = fit_character_model(df, 'character', list(range(1, 11)))
export_model(model, ML_MODEL_GCS_BUCKET, ML_MODEL_FILE_NAME)
