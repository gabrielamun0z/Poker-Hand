# POKER-HAND
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import uvicorn
import os

# Cargar modelo
model = joblib.load("best_model.pkl")

# Inicializar FastAPI
app = FastAPI(title="Poker Hand Prediction API")

# Definir esquema de entrada
class PokerHand(BaseModel):
    cards: list[str]  # Ej: ["2C", "10H", "KD", "AS", "7D"]

# Función para transformar cartas a features numéricas
def cards_to_features(cards):
    # Palo: C, D, H, S → convertir a números
    suit_map = {"C": 0, "D": 1, "H": 2, "S": 3}
    # Valores de cartas
    value_map = {
        "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7,
        "8": 8, "9": 9, "10": 10,
        "J": 11, "Q": 12, "K": 13, "A": 14
    }

    features = []
    for card in cards:
        value, suit = card[:-1], card[-1]  # Ej: "10H" → ("10", "H")
        features.append(suit_map[suit])
        features.append(value_map[value])

    return [features]

@app.post("/predict")
def predict(hand: PokerHand):
    features = cards_to_features(hand.cards)
    prediction = model.predict(features)[0]

    return {"hand": hand.cards, "prediction": int(prediction)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))

