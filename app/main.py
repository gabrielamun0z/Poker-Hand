from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import joblib
import os

# Cargar modelo
model_path = os.path.join("app", "model", "best_model.pkl")
model = joblib.load(model_path)

app = FastAPI()

# Servir frontend (index.html y estáticos)
app.mount("/static", StaticFiles(directory="app/static"), name="static")

@app.get("/")
def read_root():
    return FileResponse("app/index.html")

# Definir el input
class HandInput(BaseModel):
    cards: list[str]  # ej: ["AH", "10D", "3C", "7S", "7H"]

@app.post("/predict")
def predict_hand(data: HandInput):
    # Transforma cartas en features
    X = transform_cards_to_features(data.cards)

    prediction = model.predict([X])[0]
    return {"prediction": str(prediction)}

def transform_cards_to_features(cards):
    """
    Transforma las cartas (ej: ["AH","10D","7S","7H","3C"])
    en el vector de características que tu modelo espera.
    """
    # 👉 aquí debes poner tu lógica real de preprocesamiento
    # de momento, lo dejamos como dummy:
    return [len(cards)]
