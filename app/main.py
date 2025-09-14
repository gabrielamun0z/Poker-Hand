from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import joblib
import os

# ==== Configuración ====
app = FastAPI()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")

# Cargar modelos (ya están en el entorno aunque no en GitHub)
win_model = joblib.load(os.path.join(MODEL_DIR, "win_model.pkl"))
montecarlo_policy = joblib.load(os.path.join(MODEL_DIR, "montercarlo_policy.pkl"))

# Servir archivos estáticos
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")

templates = Jinja2Templates(directory=BASE_DIR)

# ==== Variables de sesión (simples en memoria) ====
session = {
    "initial_hand": [],
    "to_discard": [],
    "final_hand": []
}

# ==== Página principal ====
@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# ==== Primera ronda ====
@app.post("/first_round")
async def first_round(request: Request):
    data = await request.json()
    cards = data["cards"]

    session["initial_hand"] = cards

    # Probabilidad con el modelo de ML
    # Aquí suponemos que el modelo acepta la mano como lista de strings
    win_prob = win_model.predict_proba([cards])[0][1]

    # Política de descarte con Monte Carlo
    to_discard = montecarlo_policy.predict([cards])[0]
    if isinstance(to_discard, str):
        to_discard = [to_discard]

    session["to_discard"] = to_discard

    return JSONResponse({
        "win_prob": round(float(win_prob), 3),
        "to_discard": to_discard
    })

# ==== Verificar jugada sin descartar ====
@app.post("/verify")
async def verify(request: Request):
    data = await request.json()
    cards = data["cards"]

    prediction = win_model.predict([cards])[0]

    return JSONResponse({
        "prediction": str(prediction)
    })

# ==== Segunda ronda (descarte) ====
@app.post("/second_round")
async def second_round(request: Request):
    data = await request.json()
    cards = data["cards"]

    session["final_hand"] = cards
    prediction = win_model.predict([cards])[0]

    return JSONResponse({
        "prediction": str(prediction)
    })

# ==== Reinicio ====
@app.post("/reset")
async def reset():
    session["initial_hand"] = []
    session["to_discard"] = []
    session["final_hand"] = []
    return JSONResponse({"status": "reset done"})
