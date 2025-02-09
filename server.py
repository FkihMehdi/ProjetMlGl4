from fastapi import FastAPI
import pandas as pd
import pickle
from pydantic import BaseModel
from sklearn.preprocessing import StandardScaler

from model import load_model
from utilities import convert

# Load scaler and model
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

model = load_model()

app = FastAPI()


class MovieData(BaseModel):
    imdbId: str
    name: str
    genre: str
    datePublished: str
    contentRating: str
    keywords: str
    ratingCount: int
    budget: float
    Minutes: str
    actors: str
    directors: str
    creators: str


@app.post("/predict")
def predict_rating(movie: MovieData):
    movie_dict = movie.model_dump()  # Updated from .dict() to .model_dump()
    transformed_data = convert(movie_dict)

    feature_order = ['ratingCount', 'budget', 'actorsGenresScore',
                     'directorsScore', 'creatorsScore', 'keywordsScore',
                     'todayAgoPublished', 'minutes', 'minAgeToWatch']

    data_df = pd.DataFrame([transformed_data], columns=feature_order)
    scaled_data = scaler.transform(data_df)
    prediction = model.predict(scaled_data)

    return {"imdbId": movie.imdbId, "predictedRating": float(prediction[0])}  # Convert numpy float to Python float

@app.post("/predict/list")
def predict_rating_list(movies: list[MovieData]):
    predictions = []
    feature_order = ['ratingCount', 'budget', 'actorsGenresScore',
                     'directorsScore', 'creatorsScore', 'keywordsScore',
                     'todayAgoPublished', 'minutes', 'minAgeToWatch']
    for movie in movies:
        movie_dict = movie.model_dump()
        transformed_data = convert(movie_dict)
        data_df = pd.DataFrame([transformed_data], columns=feature_order)
        scaled_data = scaler.transform(data_df)
        prediction = model.predict(scaled_data)
        predictions.append({"imdbId": movie.imdbId, "predictedRating": float(prediction[0])})

    return predictions



if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
