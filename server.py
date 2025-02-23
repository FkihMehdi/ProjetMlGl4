from fastapi import FastAPI
import pandas as pd
import pickle
from pydantic import BaseModel
from sklearn.preprocessing import StandardScaler
from typing import Optional
from fastapi.middleware.cors import CORSMiddleware

from starlette.middleware.cors import CORSMiddleware

from model import load_model
from utilities import convert

# Load scaler and model
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

model = load_model()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (replace "*" with your frontend URL in production)
    allow_methods=["POST", "OPTIONS"],  # Allow POST and OPTIONS methods
    allow_headers=["*"],  # Allow all headers
)

class MovieData(BaseModel):
    name: str
    genre: Optional[list[str]] = None
    datePublished: Optional[str] = None
    contentRating: Optional[str] = None
    keywords: Optional[list[str]] = None
    ratingCount: Optional[int] = None
    budget: Optional[float] = None
    duration: Optional[int] = None
    actors: Optional[list[str]] = None
    directors: Optional[list[str]] = None
    creators: Optional[list[str]] = None


@app.post("/predict")
def predict_rating(movie: MovieData):
    # Transform data to match expected format
    print(movie)
    movie_dict = movie.model_dump()
    movie_dict["genre"] = str(movie.genre) if movie.genre else "[]"
    movie_dict["keywords"] = ",".join(movie.keywords) if movie.keywords else "[]"
    movie_dict["Minutes"] = f"PT{movie.duration}M" if movie.duration else "PT0M"
    movie_dict["actors"] = str([{"name": actor} for actor in movie.actors]) if movie.actors else "[]"
    movie_dict["directors"] = str([{"name": director} for director in movie.directors]) if movie.directors else "[]"
    movie_dict["creators"] = str([{"name": creator} for creator in movie.creators]) if movie.creators else "[]"

    transformed_data = convert(movie_dict)

    feature_order = ['ratingCount', 'budget', 'actorsGenresScore',
                     'directorsScore', 'creatorsScore', 'keywordsScore',
                     'todayAgoPublished', 'minutes', 'minAgeToWatch']

    data_df = pd.DataFrame([transformed_data], columns=feature_order)
    scaled_data = scaler.transform(data_df)
    prediction = model.predict(scaled_data)

    return {"name": movie.name, "predictedRating": float(prediction[0])}


@app.post("/predict/list")
def predict_rating_list(movies: list[MovieData]):
    predictions = []
    feature_order = ['ratingCount', 'budget', 'actorsGenresScore',
                     'directorsScore', 'creatorsScore', 'keywordsScore',
                     'todayAgoPublished', 'minutes', 'minAgeToWatch']

    for movie in movies:
        movie_dict = movie.model_dump()
        movie_dict["genre"] = str(movie.genre) if movie.genre else "[]"
        movie_dict["keywords"] = ",".join(movie.keywords) if movie.keywords else "[]"
        movie_dict["Minutes"] = f"PT{movie.duration}M" if movie.duration else "PT0M"
        movie_dict["actors"] = str([{"name": actor} for actor in movie.actors]) if movie.actors else "[]"
        movie_dict["directors"] = str([{"name": director} for director in movie.directors]) if movie.directors else "[]"
        movie_dict["creators"] = str([{"name": creator} for creator in movie.creators]) if movie.creators else "[]"

        transformed_data = convert(movie_dict)
        data_df = pd.DataFrame([transformed_data], columns=feature_order)
        scaled_data = scaler.transform(data_df)
        prediction = model.predict(scaled_data)

        predictions.append({"name": movie.name, "predictedRating": float(prediction[0])})

    return predictions


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
