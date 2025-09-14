# app/main.py
import os
import joblib
import pandas as pd
from fastapi import FastAPI
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

# ========================
# Cargar modelos
# ========================
win_model = joblib.load(WIN_MODEL_PATH)
policy_bundle = joblib.load(POLICY_MODEL_PATH)
policy_model, policy_features = policy_bundle["model"], policy_bundle["feature_names"]

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
    """Construye el DataFrame esperado por win_model (mismas columnas que el CSV)"""
    hand = [parse_card(c) for c in cards]
    ranks = [r for r, _ in hand]
    suits = [s for _, s in hand]
    df = pd.DataFrame({
        "c1_rank": [ranks[0]], "c1_suit": [suits[0]],
        "c2_rank": [ranks[1]], "c2_suit": [suits[1]],
        "c3_rank": [ranks[2]], "c3_suit": [suits[2]],
        "c4_rank": [ranks[3]], "c4_suit": [suits[3]],
        "c5_rank": [ranks[4]], "c5_suit": [suits[4]],
    })
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
    byc = sorted(cnt.items(), key=lambda x:(x[1], x[0]), reverse=True)
    is_flush = len(set(suits)) == 1
    is_str, top = is_straight(ranks)
    if is_flush and is_str: return (9, (top,))
    if byc[0][1] == 4:
        four = byc[0][0]
        kicker = max([r for r in ranks if r != four])
        return (8, (four, kicker))
    if byc[0][1] == 3 and byc[1][1] == 2:
        return (7, (byc[0][0], byc[1][0]))
    if is_flush: return (6, tuple(sorted(ranks, reverse=True)))
    if is_str:   return (5, (top,))
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
        "num_pairs": num_pairs, "num_trips": num_trips,
        "is_flush": is_flush, "is_straight": int(is_str), "category": cat
    }
    return f

def discard_masks_allowed(hand):
    ranks, _ = hand_ranks_suits(hand)
    cnt = Counter(ranks)
    involved = {r for r, c in cnt.items() if c >= 2}
    allowed = [i for i, (r, _) in enumerate(hand) if r not in involved]
    masks = {tuple([False]*5)}
    for k in [1,2,3]:
        for comb in itertools.combinations(allowed, k):
            m = [False]*5
            for i in comb: m[i] = True
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
# API
# ========================
app = FastAPI()

class Hand(BaseModel):
    cards: List[str]

@app.post("/first_round")
async def first_round(hand: Hand):
    # win_model
    X = build_features_for_win_model(hand.cards)
    win_prob = float(win_model.predict_proba(X)[0][1])

    # policy_model
    recomendaciones = recomendar_descartes([parse_card(c) for c in hand.cards])
    best_mask, best_prob, fuente = recomendaciones[0]
    to_discard = [i for i, b in enumerate(best_mask) if b]

    return {
        "win_prob": win_prob,
        "to_discard": to_discard,
        "policy_prob": best_prob,
        "fuente": fuente
    }


# ------------------------
# Servir frontend
# ------------------------
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os

BASE_DIR = os.path.dirname(__file__)
INDEX_PATH = os.path.join(BASE_DIR, "index.html")

STATIC_DIR = os.path.join(BASE_DIR, "static")
if os.path.isdir(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/")
def serve_frontend():
    return FileResponse(INDEX_PATH)

