let selectedCards = [];
let discardedCards = [];
let finalHand = [];

const cardsContainer = document.getElementById("cards");
const firstRoundBtn = document.getElementById("firstRoundBtn");
const secondRoundBtn = document.getElementById("secondRoundBtn");
const resetBtn = document.getElementById("resetBtn");
const result = document.getElementById("result");
const instructions = document.getElementById("instructions");
const discardedRow = document.getElementById("discardedRow");

// Listado de todas las cartas (en /static/cards/)
const allCards = [
  "2C","2D","2H","2S","3C","3D","3H","3S","4C","4D","4H","4S",
  "5C","5D","5H","5S","6C","6D","6H","6S","7C","7D","7H","7S",
  "8C","8D","8H","8S","9C","9D","9H","9S","10C","10D","10H","10S",
  "JC","JD","JH","JS","QC","QD","QH","QS","KC","KD","KH","KS",
  "AC","AD","AH","AS"
];

// Pintar imágenes
function renderCards() {
  cardsContainer.innerHTML = "";
  allCards.forEach(card => {
    const img = document.createElement("img");
    img.src = `/static/cards/${card}.png`;
    img.alt = card;
    img.className = "card";
    img.onclick = () => toggleCard(card, img);
    cardsContainer.appendChild(img);
  });
}

function toggleCard(card, img) {
  if (selectedCards.includes(card)) {
    selectedCards.splice(selectedCards.indexOf(card), 1);
    img.classList.remove("selected");
  } else {
    if (selectedCards.length < 5) {
      selectedCards.push(card);
      img.classList.add("selected");
    }
  }

  if (discardedCards.length === 0) {
    firstRoundBtn.disabled = selectedCards.length !== 5;
  } else {
    const totalCards = finalHand.length + selectedCards.length;
    secondRoundBtn.disabled = totalCards !== 5;
  }
}

// === Primera ronda ===
firstRoundBtn.onclick = async () => {
  const response = await fetch("/first_round", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ cards: selectedCards })
  });
  const data = await response.json();

  discardedCards = data.to_discard;

  // Mostrar imágenes en lugar de texto
  discardedRow.innerHTML = "<p>El modelo sugiere descartar:</p>";
  discardedCards.forEach(card => {
    const img = document.createElement("img");
    img.src = `/static/cards/${card}.png`;
    img.alt = card;
    img.className = "card discarded";
    discardedRow.appendChild(img);
  });

  // Ahora el jugador debe seleccionar 5 nuevas cartas, no importa qué
  instructions.textContent = "Selecciona tus 5 cartas finales.";

  result.textContent = "";
  selectedCards = [];
  renderCards();

  firstRoundBtn.disabled = true;
  secondRoundBtn.disabled = true;
};

// === Segunda ronda ===
secondRoundBtn.onclick = async () => {
  finalHand = [...selectedCards]; // Ahora son simplemente las 5 que el jugador elige

  const response = await fetch("/second_round", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ cards: finalHand })
  });
  const data = await response.json();

  result.textContent = "Jugada final: " + data.prediction;
  instructions.textContent = "Partida terminada. Pulsa reiniciar para jugar otra vez.";

  firstRoundBtn.disabled = true;
  secondRoundBtn.disabled = true;
};

// === toggleCard ===
function toggleCard(card, img) {
  if (selectedCards.includes(card)) {
    selectedCards.splice(selectedCards.indexOf(card), 1);
    img.classList.remove("selected");
  } else {
    if (selectedCards.length < 5) {
      selectedCards.push(card);
      img.classList.add("selected");
    }
  }

  if (discardedCards.length === 0) {
    firstRoundBtn.disabled = selectedCards.length !== 5;
  } else {
    secondRoundBtn.disabled = selectedCards.length !== 5; // Aquí también ahora son 5 exactas
  }
}


// === Reinicio ===
resetBtn.onclick = async () => {
  await fetch("/reset", { method: "POST" });

  selectedCards = [];
  discardedCards = [];
  finalHand = [];
  result.textContent = "";
  instructions.textContent = "Selecciona 5 cartas iniciales:";
  discardedRow.innerHTML = "";

  renderCards();

  firstRoundBtn.disabled = true;
  secondRoundBtn.disabled = true;
};

// Render inicial
renderCards();
