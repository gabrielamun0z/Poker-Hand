from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import joblib
import os

# ==== Cargar modelo ====
model_path = os.path.join("app", "model", "best_model.pkl")
model = joblib.load(model_path)

app = FastAPI(title="AsistentePoker")

# Servir frontend (index.html + static)
app.mount("/static", StaticFiles(directory="app/static"), name="static")

@app.get("/")
def root():
    return FileResponse("app/index.html")

# ==== Definir inputs ====
class HandInput(BaseModel):
    cards: list[str]  # Ejemplo: ["AH", "10D", "3C", "7S", "7H"]

# ==== Primera ronda ====
@app.post("/first_round")
def first_round(data: HandInput):
    """
    Recibe 5 cartas iniciales.
    El modelo recomienda si hay que descartar y cuáles.
    Máx. 3 descartes.
    """
    # Aquí debes implementar tu lógica real de descarte
    discard = suggest_discards(data.cards)

    return {
        "initial_hand": data.cards,
        "to_discard": discard,
        "message": "Descarta como máximo 3 cartas sugeridas."
    }

# ==== Segunda ronda ====
@app.post("/second_round")
def second_round(data: HandInput):
    """
    Recibe las 5 cartas finales (tras el descarte y reemplazo).
    Devuelve la jugada según el modelo.
    """
    X = transform_cards_to_features(data.cards)
    prediction = model.predict([X])[0]

    return {
        "final_hand": data.cards,
        "prediction": str(prediction)
    }

# ==== Reiniciar partida ====
@app.post("/reset")
def reset_game():
    return {"message": "Nueva partida iniciada"}

# ==== Helpers ====
def suggest_discards(cards):
    """
    Lógica de descarte:
    de momento, dummy -> descartar las 3 primeras si existen.
    Sustituye por tu lógica real.
    """
    return cards[:3] if len(cards) >= 3 else []

def transform_cards_to_features(cards):
    """
    Convierte la mano en vector de características para el modelo.
    Aquí debes poner tu lógica de preprocesamiento real.
    Ahora dummy: vector con longitud de la mano.
    """
    return [len(cards)]
