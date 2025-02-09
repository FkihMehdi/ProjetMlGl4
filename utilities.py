import re
import pandas as pd
import ast
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler


# Load transformation mappings
with open('mapGenreAndActorToScore.pkl', 'rb') as f:
    mapGenreAndActorToScore = pickle.load(f)

with open('mapDirectorToScore.pkl', 'rb') as f:
    mapDirectorToScore = pickle.load(f)

with open('mapCreatorToScore.pkl', 'rb') as f:
    mapCreatorToScore = pickle.load(f)

with open('mapkeywordToScore.pkl', 'rb') as f:
    mapkeywordToScore = pickle.load(f)

# Constants
RATING_VALUE_MEAN = 6.5
MEDIAN_BUDGET = 15.0
MEAN_MINUTES = 48.1
MEDIAN_TODAY_AGO_PUBLISHED = 6354.0
min_age_by_default = 14
rating_to_age = {
    'nan': None, 'Not Rated': None, 'TV-14': 14, 'TV-PG': 10, 'PG-13': 13,
    'R': 17, 'X': 18, 'PG': 10, 'TV-Y': 0, 'TV-MA': 17, 'TV-G': 0, 'TV-Y7': 7,
    'Unrated': None, 'Approved': 0, 'E': 0, 'K-A': 0, 'M': 18, 'TV-Y7-FV': 7,
    'E10+': 10, 'T': 10, 'G': 0, '18+': 18, '16+': 16, '6+': 6, '12+': 12,
    '13+': 13, 'NC-17': 17, 'EC': 3, 'GP': 0
}

def normalise_date(date):
    if pd.isna(date):
        return None
    date = re.sub(r'(\d{2})[-/](\d{2})[-/](\d{4})', r'\3-\2-\1', date)
    date = re.sub(r'(\d{4})[-/](\d{2})[-/](\d{2})', r'\1-\2-\3', date)
    return date

def compute_actor_genre_score(actors, genres):
    actor_genre_pairs = [(actor, genre) for actor in actors for genre in genres]
    scores = [mapGenreAndActorToScore.get(pair, RATING_VALUE_MEAN) for pair in actor_genre_pairs]
    return sum(scores) / len(scores) if scores else RATING_VALUE_MEAN

def compute_director_score(directors):
    scores = [mapDirectorToScore.get(d, RATING_VALUE_MEAN) for d in directors]
    return sum(scores) / len(scores) if scores else RATING_VALUE_MEAN

def compute_creator_score(creators):
    scores = [mapCreatorToScore.get(c, RATING_VALUE_MEAN) for c in creators]
    return sum(scores) / len(scores) if scores else RATING_VALUE_MEAN

def compute_keyword_score(keywords):
    scores = [mapkeywordToScore.get(k, RATING_VALUE_MEAN) for k in keywords]
    return sum(scores) / len(scores) if scores else RATING_VALUE_MEAN

def todayAgoPublished(date):
    try:
        date = pd.to_datetime(date)
        return (pd.to_datetime("2025-01-01") - date).days
    except:
        return MEDIAN_TODAY_AGO_PUBLISHED

def convert_to_minutes(duration):
    if pd.isna(duration) or duration == 'nan':
        return None
    hours = minutes = 0
    hour_match = re.search(r'PT(\d+)H', duration)
    minute_match = re.search(r'(\d+)M', duration)
    if hour_match:
        hours = int(hour_match.group(1))
    if minute_match:
        minutes = int(minute_match.group(1))
    return hours * 60 + minutes if hours or minutes else MEAN_MINUTES

def convert_rating_to_age(rating):
    return rating_to_age.get(rating, min_age_by_default)

def parse_list(value):
    try:
        return [item['name'] for item in ast.literal_eval(value)] if value else []
    except:
        return []

def convert(movie_data):

    return {
        "ratingCount": movie_data.get("ratingCount", 0),
        "budget": movie_data.get("budget", MEDIAN_BUDGET),
        "actorsGenresScore": compute_actor_genre_score(
            parse_list(movie_data.get("actors")),
            ast.literal_eval(movie_data.get("genre", "[]"))
        ),
        "directorsScore": compute_director_score(parse_list(movie_data.get("directors"))),
        "creatorsScore": compute_creator_score(parse_list(movie_data.get("creators"))),
        "keywordsScore": compute_keyword_score(movie_data.get("keywords", "").split(",")),
        "todayAgoPublished": todayAgoPublished(movie_data.get("datePublished")),
        "minutes": convert_to_minutes(movie_data.get("Minutes")),
        "minAgeToWatch": convert_rating_to_age(movie_data.get("contentRating"))
    }

