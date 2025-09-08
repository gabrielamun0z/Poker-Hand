import os

# Carpeta donde están las cartas
cards_dir = "cards"

# Mapas de valores y palos
value_map = {
    "ace": "A",
    "2": "2",
    "3": "3",
    "4": "4",
    "5": "5",
    "6": "6",
    "7": "7",
    "8": "8",
    "9": "9",
    "10": "10",
    "jack": "J",
    "queen": "Q",
    "king": "K"
}
suit_map = {
    "clubs": "C",
    "diamonds": "D",
    "hearts": "H",
    "spades": "S"
}

for file in os.listdir(cards_dir):
    if not file.endswith(".png"):
        continue

    old_path = os.path.join(cards_dir, file)

    # ignorar comodines y backs
    if "joker" in file or "back" in file:
        os.remove(old_path)
        continue

    name = file.replace(".png", "")
    parts = name.split("_of_")  # ej: "ace_of_clubs"
    value, suit = parts[0], parts[1]

    new_name = f"{value_map[value]}{suit_map[suit]}.png"
    new_path = os.path.join(cards_dir, new_name)

    os.rename(old_path, new_path)
    print(f"{file} -> {new_name}")
