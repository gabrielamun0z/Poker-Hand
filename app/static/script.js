const selectedCards = [];
const cardsContainer = document.getElementById("cards");
const predictBtn = document.getElementById("predictBtn");
const result = document.getElementById("result");

// Listado de todas las cartas (en tu carpeta /static/cards/)
const allCards = [
  "2C","2D","2H","2S","3C","3D","3H","3S","4C","4D","4H","4S",
  "5C","5D","5H","5S","6C","6D","6H","6S","7C","7D","7H","7S",
  "8C","8D","8H","8S","9C","9D","9H","9S","10C","10D","10H","10S",
  "JC","JD","JH","JS","QC","QD","QH","QS","KC","KD","KH","KS",
  "AC","AD","AH","AS"
];

// Pintar imágenes
allCards.forEach(card => {
  const img = document.createElement("img");
  img.src = `/static/cards/${card}.png`;
  img.alt = card;
  img.className = "card";
  img.onclick = () => toggleCard(card, img);
  cardsContainer.appendChild(img);
});

function toggleCard(card, img) {
  if (selectedCards.includes(card)) {
    // quitar selección
    selectedCards.splice(selectedCards.indexOf(card), 1);
    img.classList.remove("selected");
  } else {
    if (selectedCards.length < 5) {
      selectedCards.push(card);
      img.classList.add("selected");
    }
  }
  predictBtn.disabled = selectedCards.length !== 5;
}

predictBtn.onclick = async () => {
  const response = await fetch("/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ cards: selectedCards })
  });
  const data = await response.json();
  result.textContent = "Jugada: " + data.prediction;
};
