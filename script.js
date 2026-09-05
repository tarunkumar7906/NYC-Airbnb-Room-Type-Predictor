// ==========================================================================
// StayType AI — NYC Airbnb Room Type Predictor
// ==========================================================================

// ---- Config -----------------------------------------------------------
const DEFAULT_API_BASE = "http://127.0.0.1:8000";
const API_BASE_KEY = "staytype_api_base";

function getApiBase() {
  return localStorage.getItem(API_BASE_KEY) || DEFAULT_API_BASE;
}
function setApiBase(url) {
  localStorage.setItem(API_BASE_KEY, url.replace(/\/+$/, ""));
}

// ---- Neighbourhood data (borough -> neighbourhoods), matches the model's
// trained OneHotEncoder categories exactly ------------------------------
const NEIGHBOURHOODS_BY_BOROUGH = {"Bronx": ["Allerton", "Baychester", "Belmont", "Bronxdale", "Castle Hill", "City Island", "Claremont Village", "Clason Point", "Co-op City", "Concourse", "Concourse Village", "East Morrisania", "Eastchester", "Edenwald", "Fieldston", "Fordham", "Highbridge", "Hunts Point", "Kingsbridge", "Longwood", "Melrose", "Morris Heights", "Morris Park", "Morrisania", "Mott Haven", "Mount Eden", "Mount Hope", "North Riverdale", "Norwood", "Olinville", "Parkchester", "Pelham Bay", "Pelham Gardens", "Port Morris", "Riverdale", "Schuylerville", "Soundview", "Spuyten Duyvil", "Throgs Neck", "Tremont", "Unionport", "University Heights", "Van Nest", "Wakefield", "Washington Heights", "West Farms", "Westchester Square", "Williamsbridge", "Woodlawn"], "Brooklyn": ["Bath Beach", "Bay Ridge", "Bedford-Stuyvesant", "Bensonhurst", "Bergen Beach", "Boerum Hill", "Borough Park", "Brighton Beach", "Brooklyn Heights", "Brownsville", "Bushwick", "Canarsie", "Carroll Gardens", "Clinton Hill", "Cobble Hill", "Columbia St", "Coney Island", "Crown Heights", "Cypress Hills", "DUMBO", "Downtown Brooklyn", "Dyker Heights", "East Flatbush", "East New York", "Flatbush", "Flatlands", "Fort Greene", "Fort Hamilton", "Gowanus", "Gravesend", "Greenpoint", "Kensington", "Manhattan Beach", "Midwood", "Mill Basin", "Navy Yard", "Park Slope", "Prospect Heights", "Prospect-Lefferts Gardens", "Red Hook", "Sea Gate", "Sheepshead Bay", "South Slope", "Stuyvesant Town", "Sunset Park", "Vinegar Hill", "Williamsburg", "Windsor Terrace"], "Manhattan": ["Battery Park City", "Chelsea", "Chinatown", "Civic Center", "East Harlem", "East Village", "Financial District", "Flatiron District", "Gramercy", "Greenwich Village", "Harlem", "Hell's Kitchen", "Inwood", "Kips Bay", "Little Italy", "Lower East Side", "Marble Hill", "Midtown", "Morningside Heights", "Murray Hill", "NoHo", "Nolita", "Roosevelt Island", "SoHo", "Theater District", "Tribeca", "Two Bridges", "Upper East Side", "Upper West Side", "West Village"], "Queens": ["Arverne", "Astoria", "Bay Terrace", "Bayside", "Bayswater", "Belle Harbor", "Bellerose", "Breezy Point", "Briarwood", "Cambria Heights", "College Point", "Corona", "Ditmars Steinway", "Douglaston", "East Elmhurst", "Edgemere", "Elmhurst", "Far Rockaway", "Flushing", "Forest Hills", "Fresh Meadows", "Glendale", "Hollis", "Holliswood", "Howard Beach", "Jackson Heights", "Jamaica", "Jamaica Estates", "Jamaica Hills", "Kew Gardens", "Kew Gardens Hills", "Laurelton", "Little Neck", "Long Island City", "Maspeth", "Middle Village", "Neponsit", "Ozone Park", "Queens Village", "Rego Park", "Richmond Hill", "Ridgewood", "Rockaway Beach", "Rosedale", "South Ozone Park", "Springfield Gardens", "St. Albans", "Sunnyside", "Whitestone", "Woodhaven", "Woodside"], "Staten Island": ["Arden Heights", "Arrochar", "Bay Terrace, Staten Island", "Bull's Head", "Castleton Corners", "Clifton", "Concord", "Dongan Hills", "Eltingville", "Emerson Hill", "Graniteville", "Grant City", "Great Kills", "Grymes Hill", "Howland Hook", "Huguenot", "Mariners Harbor", "Midland Beach", "New Brighton", "New Dorp", "New Dorp Beach", "New Springville", "Oakwood", "Port Richmond", "Prince's Bay", "Randall Manor", "Rosebank", "Rossville", "Shore Acres", "Silver Lake", "South Beach", "St. George", "Stapleton", "Todt Hill", "Tompkinsville", "Tottenville", "West Brighton", "Westerleigh", "Willowbrook"]};

// Approx borough centroids, used only for the "sample location" helper.
const BOROUGH_CENTROIDS = {
  "Manhattan": { lat: 40.7831, lng: -73.9712 },
  "Brooklyn": { lat: 40.6782, lng: -73.9442 },
  "Queens": { lat: 40.7282, lng: -73.7949 },
  "Bronx": { lat: 40.8448, lng: -73.8648 },
  "Staten Island": { lat: 40.5795, lng: -74.1502 },
};

const ROOM_TYPE_META = {
  "Entire home/apt": { icon: "🏠", short: "Entire Home / Apt" },
  "Private room": { icon: "🛏️", short: "Private Room" },
  "Shared room": { icon: "🛋️", short: "Shared Room" },
};

// ---- DOM refs -----------------------------------------------------------
const form = document.getElementById("predictForm");
const boroughSelect = document.getElementById("neighbourhood_group");
const neighbourhoodSelect = document.getElementById("neighbourhood");
const submitBtn = document.getElementById("submitBtn");
const resetBtn = document.getElementById("resetBtn");
const useSampleBtn = document.getElementById("useMyLocation");

const resultEmpty = document.getElementById("resultEmpty");
const resultContent = document.getElementById("resultContent");
const resultError = document.getElementById("resultError");
const resultErrorMsg = document.getElementById("resultErrorMsg");
const retryBtn = document.getElementById("retryBtn");
const predictAgainBtn = document.getElementById("predictAgainBtn");

const badgeIcon = document.getElementById("badgeIcon");
const badgeValue = document.getElementById("badgeValue");
const probBars = document.getElementById("probBars");
const sumLocation = document.getElementById("sumLocation");
const sumPrice = document.getElementById("sumPrice");
const sumReviews = document.getElementById("sumReviews");
const sumAvailability = document.getElementById("sumAvailability");

const apiDot = document.getElementById("apiDot");
const apiStatusText = document.getElementById("apiStatusText");
const apiBaseInput = document.getElementById("apiBaseInput");
const saveApiBase = document.getElementById("saveApiBase");

const toastEl = document.getElementById("toast");

let lastPayload = null;

// ---- Populate borough dropdown ------------------------------------------
function populateBoroughs() {
  Object.keys(NEIGHBOURHOODS_BY_BOROUGH).forEach((borough) => {
    const opt = document.createElement("option");
    opt.value = borough;
    opt.textContent = borough;
    boroughSelect.appendChild(opt);
  });
}

function populateNeighbourhoods(borough) {
  neighbourhoodSelect.innerHTML = "";
  const placeholder = document.createElement("option");
  placeholder.value = "";
  placeholder.disabled = true;
  placeholder.selected = true;

  if (!borough || !NEIGHBOURHOODS_BY_BOROUGH[borough]) {
    placeholder.textContent = "Select a borough first…";
    neighbourhoodSelect.appendChild(placeholder);
    neighbourhoodSelect.disabled = true;
    return;
  }

  placeholder.textContent = "Select a neighbourhood…";
  neighbourhoodSelect.appendChild(placeholder);

  NEIGHBOURHOODS_BY_BOROUGH[borough].forEach((n) => {
    const opt = document.createElement("option");
    opt.value = n;
    opt.textContent = n;
    neighbourhoodSelect.appendChild(opt);
  });
  neighbourhoodSelect.disabled = false;
}

boroughSelect.addEventListener("change", () => {
  populateNeighbourhoods(boroughSelect.value);
  clearFieldError("neighbourhood_group");
});

// ---- Sample location helper ---------------------------------------------
useSampleBtn.addEventListener("click", () => {
  const boroughs = Object.keys(BOROUGH_CENTROIDS);
  const borough = boroughs[Math.floor(Math.random() * boroughs.length)];
  const centroid = BOROUGH_CENTROIDS[borough];

  const jitter = () => (Math.random() - 0.5) * 0.04;
  document.getElementById("latitude").value = (centroid.lat + jitter()).toFixed(5);
  document.getElementById("longitude").value = (centroid.lng + jitter()).toFixed(5);

  boroughSelect.value = borough;
  populateNeighbourhoods(borough);
  const options = NEIGHBOURHOODS_BY_BOROUGH[borough];
  neighbourhoodSelect.value = options[Math.floor(Math.random() * options.length)];

  document.getElementById("price").value = Math.floor(50 + Math.random() * 250);
  document.getElementById("minimum_nights").value = [1, 2, 3, 5, 7][Math.floor(Math.random() * 5)];
  document.getElementById("number_of_reviews").value = Math.floor(Math.random() * 120);
  document.getElementById("reviews_per_month").value = (Math.random() * 4).toFixed(2);
  document.getElementById("calculated_host_listings_count").value = Math.floor(1 + Math.random() * 5);
  document.getElementById("availability_365").value = Math.floor(Math.random() * 365);

  showToast("Sample listing filled in — feel free to tweak it.", "success");
  clearAllErrors();
});

// ---- Validation -----------------------------------------------------------
const FIELD_RULES = {
  latitude: { min: -90, max: 90, type: "float" },
  longitude: { min: -180, max: 180, type: "float" },
  price: { min: 0, type: "float" },
  minimum_nights: { min: 1, max: 365, type: "int" },
  number_of_reviews: { min: 0, type: "int" },
  reviews_per_month: { min: 0, type: "float" },
  calculated_host_listings_count: { min: 0, type: "int" },
  availability_365: { min: 0, max: 365, type: "int" },
};

function setFieldError(name, message) {
  const input = document.getElementById(name);
  const errorEl = document.querySelector(`.error[data-for="${name}"]`);
  if (input) input.classList.add("invalid");
  if (errorEl) {
    errorEl.textContent = message;
    errorEl.classList.add("show");
  }
}
function clearFieldError(name) {
  const input = document.getElementById(name);
  const errorEl = document.querySelector(`.error[data-for="${name}"]`);
  if (input) input.classList.remove("invalid");
  if (errorEl) {
    errorEl.textContent = "";
    errorEl.classList.remove("show");
  }
}
function clearAllErrors() {
  Object.keys(FIELD_RULES).forEach(clearFieldError);
  clearFieldError("neighbourhood_group");
  clearFieldError("neighbourhood");
}

function validateForm(data) {
  let valid = true;
  clearAllErrors();

  for (const [name, rule] of Object.entries(FIELD_RULES)) {
    const raw = data[name];
    if (raw === "" || raw === null || raw === undefined || Number.isNaN(raw)) {
      setFieldError(name, "This field is required.");
      valid = false;
      continue;
    }
    if (rule.min !== undefined && raw < rule.min) {
      setFieldError(name, `Must be at least ${rule.min}.`);
      valid = false;
    }
    if (rule.max !== undefined && raw > rule.max) {
      setFieldError(name, `Must be at most ${rule.max}.`);
      valid = false;
    }
  }

  if (!data.neighbourhood_group) {
    setFieldError("neighbourhood_group", "Please select a borough.");
    valid = false;
  }
  if (!data.neighbourhood) {
    setFieldError("neighbourhood", "Please select a neighbourhood.");
    valid = false;
  }

  return valid;
}

// clear a field's error as soon as the user edits it
Object.keys(FIELD_RULES).forEach((name) => {
  const el = document.getElementById(name);
  if (el) el.addEventListener("input", () => clearFieldError(name));
});
neighbourhoodSelect.addEventListener("change", () => clearFieldError("neighbourhood"));

// ---- Form submit -----------------------------------------------------------
form.addEventListener("submit", async (e) => {
  e.preventDefault();

  const formData = new FormData(form);
  const payload = {
    latitude: parseFloat(formData.get("latitude")),
    longitude: parseFloat(formData.get("longitude")),
    price: parseFloat(formData.get("price")),
    minimum_nights: parseInt(formData.get("minimum_nights"), 10),
    number_of_reviews: parseInt(formData.get("number_of_reviews"), 10),
    reviews_per_month: parseFloat(formData.get("reviews_per_month")),
    calculated_host_listings_count: parseInt(formData.get("calculated_host_listings_count"), 10),
    availability_365: parseInt(formData.get("availability_365"), 10),
    neighbourhood_group: formData.get("neighbourhood_group") || "",
    neighbourhood: formData.get("neighbourhood") || "",
  };

  if (!validateForm(payload)) {
    showToast("Please fix the highlighted fields.", "error");
    return;
  }

  lastPayload = payload;
  await runPrediction(payload);
});

resetBtn.addEventListener("click", () => {
  form.reset();
  populateNeighbourhoods("");
  clearAllErrors();
  showResultState("empty");
});

retryBtn.addEventListener("click", () => {
  if (lastPayload) runPrediction(lastPayload);
});
predictAgainBtn.addEventListener("click", () => {
  showResultState("empty");
  document.querySelector(".form-panel").scrollIntoView({ behavior: "smooth", block: "start" });
});

// ---- API call -----------------------------------------------------------
async function runPrediction(payload) {
  setLoading(true);
  try {
    const res = await fetch(`${getApiBase()}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    if (!res.ok) {
      let detail = `Request failed with status ${res.status}.`;
      try {
        const errBody = await res.json();
        if (errBody?.detail) {
          detail = typeof errBody.detail === "string" ? errBody.detail : JSON.stringify(errBody.detail);
        }
      } catch (_) {}
      throw new Error(detail);
    }

    const data = await res.json();
    renderResult(data, payload);
    showToast("Prediction complete.", "success");
  } catch (err) {
    console.error(err);
    resultErrorMsg.textContent =
      err.message?.includes("Failed to fetch")
        ? `Couldn't reach the API at ${getApiBase()}. Make sure the FastAPI server is running and CORS is enabled.`
        : err.message || "Unexpected error while predicting.";
    showResultState("error");
    showToast("Prediction failed.", "error");
  } finally {
    setLoading(false);
  }
}

function setLoading(isLoading) {
  submitBtn.disabled = isLoading;
  submitBtn.querySelector(".btn-label").hidden = isLoading;
  submitBtn.querySelector(".btn-loader").hidden = !isLoading;
}

// ---- Render result --------------------------------------------------------
function renderResult(data, payload) {
  const predicted = data.Predicted_room_type;
  const probabilities = data.Probability;

  // Try to align probabilities with known class order; fall back gracefully.
  const classOrder = ["Entire home/apt", "Private room", "Shared room"];
  let pairs;
  if (Array.isArray(probabilities) && probabilities.length === classOrder.length) {
    pairs = classOrder.map((label, i) => ({ label, value: probabilities[i] }));
  } else if (Array.isArray(probabilities)) {
    pairs = probabilities.map((v, i) => ({ label: `Class ${i}`, value: v }));
  } else {
    pairs = [{ label: predicted, value: 1 }];
  }
  pairs.sort((a, b) => b.value - a.value);

  const meta = ROOM_TYPE_META[predicted] || { icon: "🏷️", short: predicted };
  badgeIcon.textContent = meta.icon;
  badgeValue.textContent = meta.short;

  probBars.innerHTML = "";
  pairs.forEach((p, idx) => {
    const row = document.createElement("div");
    row.className = "prob-row";
    const pct = (p.value * 100).toFixed(1);
    row.innerHTML = `
      <div class="prob-row-head">
        <span class="p-name">${ROOM_TYPE_META[p.label]?.short || p.label}</span>
        <span class="p-val">${pct}%</span>
      </div>
      <div class="prob-track">
        <div class="prob-fill ${idx === 0 ? "top" : ""}" style="width:0%"></div>
      </div>
    `;
    probBars.appendChild(row);
    requestAnimationFrame(() => {
      setTimeout(() => {
        row.querySelector(".prob-fill").style.width = `${pct}%`;
      }, 60 + idx * 90);
    });
  });

  sumLocation.textContent = `${payload.neighbourhood}, ${payload.neighbourhood_group}`;
  sumPrice.textContent = `$${payload.price.toLocaleString()}`;
  sumReviews.textContent = `${payload.number_of_reviews} total · ${payload.reviews_per_month}/mo`;
  sumAvailability.textContent = `${payload.availability_365} / 365 days`;

  showResultState("content");
}

function showResultState(state) {
  resultEmpty.hidden = state !== "empty";
  resultContent.hidden = state !== "content";
  resultError.hidden = state !== "error";
}

// ---- API health check -----------------------------------------------------
async function checkApiStatus() {
  apiDot.className = "dot checking";
  apiStatusText.textContent = "Checking API…";
  try {
    const res = await fetch(`${getApiBase()}/`, { method: "GET" });
    if (res.ok) {
      apiDot.className = "dot online";
      apiStatusText.textContent = "API online";
    } else {
      throw new Error("bad status");
    }
  } catch (_) {
    apiDot.className = "dot offline";
    apiStatusText.textContent = "API offline";
  }
}

// ---- API base config UI -----------------------------------------------------
apiBaseInput.value = getApiBase();
saveApiBase.addEventListener("click", () => {
  const val = apiBaseInput.value.trim();
  if (!val) return;
  setApiBase(val);
  showToast("API URL saved.", "success");
  checkApiStatus();
});
apiBaseInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter") saveApiBase.click();
});

// ---- Toast -----------------------------------------------------------
let toastTimer = null;
function showToast(message, type = "") {
  clearTimeout(toastTimer);
  toastEl.textContent = message;
  toastEl.className = `toast show ${type}`;
  toastTimer = setTimeout(() => {
    toastEl.classList.remove("show");
  }, 3200);
}

// ---- Init -----------------------------------------------------------
populateBoroughs();
populateNeighbourhoods("");
checkApiStatus();
setInterval(checkApiStatus, 20000);
