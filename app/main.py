# app/main.py
import os
import logging
import itertools
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from collections import Counter
import numpy as np
import random

# ========================
# Configuración y rutas
# ========================
BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, "model")

WIN_MODEL_PATH = os.path.join(MODEL_DIR, "win_model.pkl")
POLICY_MODEL_PATH = os.path.join(MODEL_DIR, "montecarlo_policy.pkl")
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pkl")  # opcional

# ========================
# Cargar modelos
# ========================
win_model = joblib.load(WIN_MODEL_PATH)

policy_bundle = joblib.load(POLICY_MODEL_PATH)
policy_model, policy_features = policy_bundle["model"], policy_bundle["feature_names"]

# best_model opcional
best_model = None
try:
    if os.path.exists(BEST_MODEL_PATH):
        best_model = joblib.load(BEST_MODEL_PATH)
except Exception as _e:
    logging.warning("No se pudo cargar best_model.pkl: %s", _e)
    best_model = None

# ========================
# Utilidades de cartas
# ========================
RANK_ORDER = "23456789TJQKA"
RANK_TO_INT = {r: i for i, r in enumerate(RANK_ORDER, start=2)}
INT_TO_RANK = {v: k for k, v in RANK_TO_INT.items()}
SUITS = ["H", "D", "C", "S"]

def parse_card(txt: str):
    """Convierte 'AH' o '10D' en (14, 'H') o (10, 'D')"""
    t = txt.strip().upper().replace(",", "")
    if len(t) < 2:
        raise ValueError(f"Carta inválida: '{txt}'")
    rank, suit = t[0], t[1]
    if rank == "1" and len(t) >= 3 and t[1] == "0":  # 10 -> T
        rank, suit = "T", t[2]
    if rank not in RANK_ORDER or suit not in SUITS:
        raise ValueError("Formato inválido. Usa ej: AH, 7D, TC")
    return (RANK_TO_INT[rank], suit)

def format_card(c):
    return f"{INT_TO_RANK[c[0]]}{c[1]}"

# ========================
# Features para win_model
# ========================
def build_features_for_win_model(cards: List[str]):
    """Construye el DataFrame esperado por win_model"""
    hand = [parse_card(c) for c in cards]
    ranks = [r for r, _ in hand]
    suits = [s for _, s in hand]

    # DataFrame mínimo
    df = pd.DataFrame({
        "C1": [ranks[0]], "S1": [suits[0]],
        "C2": [ranks[1]], "S2": [suits[1]],
        "C3": [ranks[2]], "S3": [suits[2]],
        "C4": [ranks[3]], "S4": [suits[3]],
        "C5": [ranks[4]], "S5": [suits[4]],
    })

    # 🔑 Añadir las columnas extra que el modelo espera
    for col in [
        'longest_sequence','num_suits_distinct','CLASS','has_four_kind',
        'has_three_kind','num_pairs','has_royal_structure','JUGADA_txt',
        'Max_Same_Suits','JUGADA','max_equal_ranks','JUGADA_class','max_equal_suits'
    ]:
        df[col] = 0

    # Fuerza tipos esperados
    for c in ["C1","C2","C3","C4","C5"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype(int)
    for s in ["S1","S2","S3","S4","S5"]:
        if s in df.columns:
            df[s] = df[s].astype(str)

    return df

# ========================
# Features para policy_model
# ========================
def hand_ranks_suits(hand):
    ranks = sorted([r for r, _ in hand], reverse=True)
    suits = [s for _, s in hand]
    return ranks, suits

def is_straight(ranks_sorted_desc):
    r = sorted(set(ranks_sorted_desc), reverse=True)
    if len(r) < 5:
        return False, None
    for i in range(len(r) - 4):
        w = r[i:i+5]
        if w[0] - w[4] == 4:
            return True, w[0]
    if set([14, 5, 4, 3, 2]).issubset(set(r)):
        return True, 5
    return False, None

def hand_rank(hand):
    ranks, suits = hand_ranks_suits(hand)
    cnt = Counter(ranks)
    byc = sorted(cnt.items(), key=lambda x: (x[1], x[0]), reverse=True)
    is_flush = len(set(suits)) == 1
    is_str, top = is_straight(ranks)
    if is_flush and is_str:
        return (9, (top,))
    if byc[0][1] == 4:
        four = byc[0][0]
        kicker = max([r for r in ranks if r != four])
        return (8, (four, kicker))
    if byc[0][1] == 3 and byc[1][1] == 2:
        return (7, (byc[0][0], byc[1][0]))
    if is_flush:
        return (6, tuple(sorted(ranks, reverse=True)))
    if is_str:
        return (5, (top,))
    if byc[0][1] == 3:
        triple = byc[0][0]
        kick = sorted([r for r in ranks if r != triple], reverse=True)
        return (4, (triple, *kick))
    if byc[0][1] == 2 and byc[1][1] == 2:
        hp = max([x[0] for x in byc if x[1] == 2])
        lp = min([x[0] for x in byc if x[1] == 2])
        kicker = max([r for r in ranks if r not in (hp, lp)])
        return (3, (hp, lp, kicker))
    if byc[0][1] == 2:
        pair = byc[0][0]
        kick = sorted([r for r in ranks if r != pair], reverse=True)
        return (2, (pair, *kick))
    return (1, tuple(sorted(ranks, reverse=True)))

def hand_to_features(hand, mask):
    ranks, suits = hand_ranks_suits(hand)
    rank_oh = np.zeros((5, 13), dtype=int)
    suit_oh = np.zeros((5, 4), dtype=int)
    for i, (r, s) in enumerate(hand):
        rank_oh[i, r-2] = 1
        suit_oh[i, SUITS.index(s)] = 1
    mask_arr = np.array(mask, dtype=int)
    cnt = Counter(ranks)
    num_pairs = sum(1 for v in cnt.values() if v == 2)
    num_trips = sum(1 for v in cnt.values() if v == 3)
    is_flush = int(len(set(suits)) == 1)
    is_str, _ = is_straight(ranks)
    cat, _ = hand_rank(hand)
    f = {
        **{f"r{i}_{RANK_ORDER[j]}": int(rank_oh[i, j]) for i in range(5) for j in range(13)},
        **{f"s{i}_{s}": int(suit_oh[i, si]) for i, s in enumerate(SUITS) for si in range(4) if s == SUITS[si]},
        **{f"mask{i}": int(mask_arr[i]) for i in range(5)},
        "num_pairs": num_pairs,
        "num_trips": num_trips,
        "is_flush": is_flush,
        "is_straight": int(is_str),
        "category": cat
    }
    return f

def discard_masks_allowed(hand):
    ranks, _ = hand_ranks_suits(hand)
    cnt = Counter(ranks)
    involved = {r for r, c in cnt.items() if c >= 2}
    allowed = [i for i, (r, _) in enumerate(hand) if r not in involved]
    masks = {tuple([False]*5)}  # no descartar
    for k in [1, 2, 3]:
        for comb in itertools.combinations(allowed, k):
            m = [False]*5
            for i in comb:
                m[i] = True
            masks.add(tuple(m))
    return sorted(list(masks))

def evaluar_opciones_con_policy(hand, masks, policy_model, feature_names):
    rows = [hand_to_features(hand, m) for m in masks]
    X = pd.DataFrame(rows)
    for c in feature_names:
        if c not in X.columns:
            X[c] = 0
    X = X[feature_names]
    return policy_model.predict(X)

def recomendar_descartes(hand):
    masks = discard_masks_allowed(hand)
    preds = evaluar_opciones_con_policy(hand, masks, policy_model, policy_features)
    resultados = [(m, float(p), "ML") for m, p in zip(masks, preds)]
    resultados.sort(key=lambda x: x[1], reverse=True)
    return resultados

# ========================
# Clasificación de jugada final (best_model opcional)
# ========================
def build_features_for_best_model(best_model, hand_tuples):
    """
    DataFrame con EXACTAMENTE las columnas que espera best_model si tiene feature_names_in_.
    hand_tuples: [(rank_int, suit_char), ...]
    """
    if best_model is None or not hasattr(best_model, "feature_names_in_"):
        return None
    cols = list(best_model.feature_names_in_)
    base = {c: 0 for c in cols}
    for i, (r, s) in enumerate(hand_tuples):
        rname = INT_TO_RANK[r]
        kr = f"r{i}_{rname}"
        ks = f"s{i}_{s}"
        if kr in base: base[kr] = 1
        if ks in base: base[ks] = 1
    X = pd.DataFrame([[base[c] for c in cols]], columns=cols)
    return X

RANK_NAMES = {
    1: "Carta alta",
    2: "Pareja",
    3: "Doble pareja",
    4: "Trío",
    5: "Escalera",
    6: "Color",
    7: "Full",
    8: "Póker",
    9: "Escalera de color",
}

def classify_final_hand(cards_str_list):
    parsed = [parse_card(c) for c in cards_str_list]
    # 1) Intentar best_model
    if best_model is not None:
        try:
            Xb = build_features_for_best_model(best_model, parsed)
            if Xb is not None:
                if hasattr(best_model, "predict_proba"):
                    probs = best_model.predict_proba(Xb)[0]
                    classes = getattr(best_model, "classes_", None)
                    if classes is not None and len(classes) == len(probs):
                        return str(classes[int(np.argmax(probs))])
                # fallback sin predict_proba
                return str(best_model.predict(Xb)[0])
        except Exception as e:
            logging.warning("best_model no usable, usando evaluador interno: %s", e)
    # 2) Evaluador interno
    cat, _ = hand_rank(parsed)
    return RANK_NAMES.get(cat, f"Categoría {cat}")

# ========================
# API
# ========================
app = FastAPI()

class Hand(BaseModel):
    cards: List[str]

# --- Predictor tolerante ---
def _predict_win_prob_tolerant(model, X: pd.DataFrame) -> float:
    """
    Intenta model.predict_proba(X). Si falla por strings en S1..S5,
    convierte palos a números y reintenta.
    """
    try:
        return float(model.predict_proba(X)[0][1])
    except ValueError as e:
        msg = str(e)
        if "could not convert string to float" in msg or "dtype='numeric'" in msg:
            suit_map = {"H": 0, "D": 1, "C": 2, "S": 3}
            X2 = X.copy()
            for c in ["S1","S2","S3","S4","S5"]:
                if c in X2.columns:
                    X2[c] = X2[c].map(suit_map).astype(float)
            return float(model.predict_proba(X2)[0][1])
        raise

@app.post("/first_round")
async def first_round(hand: Hand):
    try:
        # --- Probabilidad de victoria ---
        X = build_features_for_win_model(hand.cards)
        win_prob = _predict_win_prob_tolerant(win_model, X)

        # --- Recomendación de descartes ---
        parsed = [parse_card(c) for c in hand.cards]
        recomendaciones = recomendar_descartes(parsed)
        best_mask, best_policy_prob, fuente = recomendaciones[0]

        to_discard = [i for i, b in enumerate(best_mask) if b]
        discardCards = [format_card(parsed[i]) for i in to_discard]

        return {
            "win_prob": win_prob,
            "win_prob_pct": round(win_prob * 100, 2),
            "to_discard": to_discard,
            "discardCards": discardCards,
            "policy_prob": float(best_policy_prob),
            "fuente": fuente
        }
    except Exception as e:
        logging.exception("first_round failed")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/verify")
async def verify(hand: Hand):
    try:
        prediction = classify_final_hand(hand.cards)
        return {"prediction": prediction}
    except Exception as e:
        logging.exception("verify failed")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/second_round")
async def second_round(hand: Hand):
    """
    Recibe la mano FINAL (tras robar). Devuelve la jugada final.
    """
    try:
        prediction = classify_final_hand(hand.cards)
        return {"prediction": prediction}
    except Exception as e:
        logging.exception("second_round failed")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/reset")
async def reset():
    return {"ok": True}

# ------------------------
# Servir frontend
# ------------------------
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

INDEX_PATH = os.path.join(BASE_DIR, "index.html")
STATIC_DIR = os.path.join(BASE_DIR, "static")

if os.path.isdir(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/")
def serve_frontend():
    return FileResponse(INDEX_PATH)
