# POKER-HAND
from fastapi import FastAPI
import joblib
from pydantic import BaseModel

# Modelo cargado
model = joblib.load("best_model.pkl")

# Definir esquema de la mano
class Hand(BaseModel):
    S1: int
    C1: int
    S2: int
    C2: int
    S3: int
    C3: int
    S4: int
    C4: int
    S5: int
    C5: int

app = FastAPI()

@app.post("/predict")
def predict(hand: Hand):
    data = [[
        hand.S1, hand.C1,
        hand.S2, hand.C2,
        hand.S3, hand.C3,
        hand.S4, hand.C4,
        hand.S5, hand.C5
    ]]
    pred = model.predict(data)
    return {"prediction": pred[0]}
