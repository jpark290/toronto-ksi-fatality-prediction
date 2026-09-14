const form = document.getElementById("risk-form");
const btn = document.getElementById("assess-btn");
const predictedLine = document.getElementById("predicted-line");
const boundaryMarker = document.getElementById("boundary-marker");
const separationLabel = document.getElementById("separation-label");
const rawScoreLine = document.getElementById("raw-score-line");
const statusDot = document.getElementById("status-dot");
const errorBox = document.getElementById("error-box");
const resultDetail = document.getElementById("result-detail");
const resultNote = document.getElementById("result-note");
const factorChips = document.getElementById("factor-chips");
const modelLine = document.getElementById("model-line");
const rawToggle = document.getElementById("raw-toggle");
const rawJson = document.getElementById("raw-json");

const BINARY_IDS = [
  "SPEEDING", "AG_DRIV", "REDLIGHT", "ALCOHOL", "DISABILITY",
  "PEDESTRIAN", "CYCLIST", "AUTOMOBILE", "MOTORCYCLE", "TRUCK",
  "TRSN_CITY_VEH", "PASSENGER",
];

const FACTOR_LABELS = {
  SPEEDING: "Speeding", AG_DRIV: "Aggressive driving", REDLIGHT: "Red light",
  ALCOHOL: "Alcohol", DISABILITY: "Disability", PEDESTRIAN: "Pedestrian",
  CYCLIST: "Cyclist", AUTOMOBILE: "Automobile", MOTORCYCLE: "Motorcycle",
  TRUCK: "Truck", TRSN_CITY_VEH: "Transit/city vehicle", PASSENGER: "Passenger",
};

function buildPayload() {
  const features = {
    LATITUDE: parseFloat(document.getElementById("LATITUDE").value),
    LONGITUDE: parseFloat(document.getElementById("LONGITUDE").value),
    YEAR: parseInt(document.getElementById("YEAR").value, 10),
    MONTH: parseInt(document.getElementById("MONTH").value, 10),
    DAY_OF_WEEK: parseInt(document.getElementById("DAY_OF_WEEK").value, 10),
    HOUR: parseInt(document.getElementById("HOUR").value, 10),
    DISTRICT: document.getElementById("DISTRICT").value,
    DIVISION: document.getElementById("DIVISION").value,
    ROAD_CLASS: document.getElementById("ROAD_CLASS").value,
    ACCLOC: document.getElementById("ACCLOC").value,
    TRAFFCTL: document.getElementById("TRAFFCTL").value,
    LIGHT: document.getElementById("LIGHT").value,
    VISIBILITY: document.getElementById("VISIBILITY").value,
    RDSFCOND: document.getElementById("RDSFCOND").value,
    INITDIR: document.getElementById("INITDIR").value,
  };
  BINARY_IDS.forEach((id) => {
    features[id] = document.getElementById(id).checked ? "Yes" : "No";
  });
  // Their /predict expects {"features": {...}}, not a flat body.
  return { features };
}

// Bands derived from the actual SVM decision_function distribution on the
// notebook's test set (n=1373): 25th pct = -1.05, 75th pct = 0.48, 20th pct
// magnitude ~1.27. |score| < 0.5 covers the typical near-boundary case;
// |score| >= 1.5 covers the outer ~20% of scores (real separation).
const SEPARATION_BANDS = [
  { max: 0.5, label: "Slight" },
  { max: 1.5, label: "Moderate" },
  { max: Infinity, label: "Strong" },
];

// Track visually clamps to +/-3 -- comfortably covers the typical spread
// (test-set scores mostly fall in -1.05 to +0.48) without one rare outlier
// (max observed was 9.9) squashing everything else toward the center.
const TRACK_CLAMP = 3;

function bandFor(score) {
  const magnitude = Math.abs(score);
  return SEPARATION_BANDS.find((b) => magnitude < b.max).label;
}

function positionOnTrack(score) {
  const clamped = Math.max(-TRACK_CLAMP, Math.min(TRACK_CLAMP, score));
  return ((clamped + TRACK_CLAMP) / (TRACK_CLAMP * 2)) * 100;
}

rawToggle.addEventListener("click", () => {
  const isHidden = rawJson.style.display === "none";
  rawJson.style.display = isHidden ? "block" : "none";
  rawToggle.textContent = isHidden ? "Hide raw response \u25BE" : "View raw response \u25B6";
});

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  errorBox.classList.remove("show");
  btn.disabled = true;
  btn.textContent = "Assessing…";
  statusDot.classList.remove("live");

  try {
    const res = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(buildPayload()),
    });

    const data = await res.json();
    if (!res.ok) throw new Error(data.error || `Request failed (${res.status})`);

    const score = data.risk_score;
    const isFatal = data.predicted_fatal === 1;
    const side = score >= 0 ? "Fatal" : "Non-Fatal";
    const band = bandFor(score);
    const markerColor = isFatal ? "#e2504a" : "#4fb583";

    predictedLine.className = `predicted-line ${isFatal ? "fatal" : "non-fatal"}`;
    predictedLine.textContent = `Predicted: ${data.predicted_label}`;

    boundaryMarker.style.left = `${positionOnTrack(score)}%`;
    boundaryMarker.style.background = markerColor;

    separationLabel.textContent = `${band} separation toward ${side}`;
    rawScoreLine.textContent = `SVM decision score: ${score.toFixed(3)} \u00b7 not a probability`;

    statusDot.classList.add("live");

    resultDetail.style.display = "block";
    resultNote.textContent =
      `This score reflects distance from the model's decision boundary, ` +
      `not a calibrated chance of fatality.`;

    const activeFactors = Object.keys(FACTOR_LABELS).filter(
      (id) => document.getElementById(id).checked
    );
    factorChips.innerHTML = activeFactors.length
      ? activeFactors.map((id) => `<span class="factor-chip">${FACTOR_LABELS[id]}</span>`).join("")
      : `<span class="factor-chip">No contributing factors or road users flagged</span>`;

    rawJson.textContent = JSON.stringify(data, null, 2);
    rawJson.style.display = "none";
    rawToggle.textContent = "View raw response \u25B6";

    modelLine.textContent = `served ${new Date().toLocaleTimeString()}`;
  } catch (err) {
    errorBox.textContent = `Couldn't get a prediction: ${err.message}`;
    errorBox.classList.add("show");
    statusDot.classList.remove("live");
  } finally {
    btn.disabled = false;
    btn.innerHTML = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M13 2 3 14h7l-1 8 10-12h-7l1-8z"/></svg> Assess Risk`;
  }
});
