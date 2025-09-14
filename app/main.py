from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import os
import joblib

# === Inicialización ===
app = FastAPI()

BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, "model")

# Cargar modelos
win_model = joblib.load(os.path.join(MODEL_DIR, "win_model.pkl"))
policy = joblib.load(os.path.join(MODEL_DIR, "montecarlo_policy.pkl"))
hand_model = joblib.load(os.path.join(MODEL_DIR, "best_model.pkl"))

# Configuración de plantillas y estáticos
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

# === Variables globales (estado de partida) ===
current_hand = []
discarded_cards = []
final_hand = []

# === Rutas ===
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/first_round")
async def first_round(request: Request):
    global current_hand, discarded_cards
    data = await request.json()
    current_hand = data["cards"]

    # 1. Calcular probabilidad de victoria
    # (Aquí adaptas a cómo tu win_model espera recibir los datos)
    win_prob = float(win_model.predict_proba([current_hand])[0][1])  

    # 2. Obtener sugerencia de descartes usando la policy
    # (Ejemplo simple: la policy devuelve lista de cartas a descartar)
    to_discard = policy.get("suggest_discards", lambda x: [])(current_hand)
    discarded_cards = to_discard

    return JSONResponse({
        "win_prob": round(win_prob, 3),
        "to_discard": discarded_cards
    })

@app.post("/second_round")
async def second_round(request: Request):
    global final_hand
    data = await request.json()
    final_hand = data["cards"]

    # Predecir jugada final con best_model
    prediction = hand_model.predict([final_hand])[0]

    return JSONResponse({
        "prediction": str(prediction)
    })

@app.post("/verify")
async def verify(request: Request):
    data = await request.json()
    hand = data["cards"]

    prediction = hand_model.predict([hand])[0]

    return JSONResponse({
        "prediction": str(prediction)
    })

@app.post("/reset")
async def reset():
    global current_hand, discarded_cards, final_hand
    current_hand = []
    discarded_cards = []
    final_hand = []
    return JSONResponse({"status": "reset done"})
