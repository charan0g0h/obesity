from fastapi import FastAPI
import numpy as np
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import Main
import Maps

app = FastAPI()

origins = [
    "http://localhost:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ObesityFeatures(BaseModel):
    Gender: int
    Age: float
    Height: float
    Weight: float
    family_history_with_overweight: int
    FAVC: int
    FCVC: float
    NCP: float
    CAEC: int
    SMOKE: int
    CH2O: float
    SCC: int
    FAF: float
    TUE: float
    CALC: int
    MTRANS: int

@app.post("/predict")
def predict(features: ObesityFeatures):
    x = np.array([[
        features.Gender,
        features.Age,
        features.Height,
        features.Weight,
        features.family_history_with_overweight,
        features.FAVC,
        features.FCVC,
        features.NCP,
        features.CAEC,
        features.SMOKE,
        features.CH2O,
        features.SCC,
        features.FAF,
        features.TUE,
        features.CALC,
        features.MTRANS
    ]], dtype=np.float32)
    
    return Main.prediction(features = x)

@app.post("/suggest")
def recommend(features: ObesityFeatures):
    suggestions = Main.suggestion(features=features)
    return {
        "suggestions": suggestions
    }
