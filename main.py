import joblib
from fastapi import FastAPI
import pandas as pd
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

COLUMNS = ['latitude', 'longitude', 'price', 'minimum_nights', 'number_of_reviews', 'reviews_per_month', 'calculated_host_listings_count', 'availability_365', 'neighbourhood_group', 'neighbourhood']

model = joblib.load('Model_Pipeline.pkl') # Load the model pipeline from the pickle file

# Pydantic Model = the input validation
class Features(BaseModel):
    latitude: float = Field(..., ge= -90, le=90, description = "Latitude coordinate")
    longitude: float = Field(..., ge= -180, le=180, description = "Longitude coordinate")
    price: float = Field(..., ge=0, description = "Price per night, must be positive")
    minimum_nights: int = Field(..., ge=1, le=365, description = "Minimum nights required for booking")
    number_of_reviews: int = Field(..., ge=0, description = "Total number of reviews")
    reviews_per_month: float = Field(..., ge=0, description = "Average number of reviews per month")
    calculated_host_listings_count: int = Field(..., ge=0, description = "Number of listings the host has")
    availability_365: int = Field(..., ge=0, le=365, description = "Number of days the listing is available in a year")
    neighbourhood_group: str = Field(...,min_length=1, description = "The borough of the listing")
    neighbourhood: str = Field(...,min_length=1, description = "Specific neighborhood name")


@app.get('/')
def greet():
    return "Hello Guys"

@app.post('/predict')
def predict(features: Features):
    row = pd.DataFrame([features.dict()], columns=COLUMNS)
    prediction = model.predict(row)
    probality = model.predict_proba(row)

    return {
        "Predicted_room_type": prediction[0],
        "Probability": probality.tolist()[0]
    }