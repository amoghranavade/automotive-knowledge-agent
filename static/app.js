const form = document.querySelector("#search-form");
const queryInput = document.querySelector("#query");
const voiceButton = document.querySelector("#voice-input");
const voiceStatus = document.querySelector("#voice-status");
const submitButton = document.querySelector("#submit");
const modelToggle = document.querySelector("#use-model");
const liveNhtsaToggle = document.querySelector("#live-nhtsa");
const modelStatus = document.querySelector("#model-status");
const dataNotice = document.querySelector("#data-notice");
const configuration = document.querySelector("#configuration");
const statusBox = document.querySelector("#status");
const summary = document.querySelector("#summary");
const results = document.querySelector("#results");
const template = document.querySelector("#candidate-template");

const money = new Intl.NumberFormat("en-US", {
  style: "currency",
  currency: "USD",
  maximumFractionDigits: 0,
});

const number = new Intl.NumberFormat("en-US");

loadModelStatus();
loadDataStatus();
setupVoiceInput();

document.querySelectorAll("[data-query]").forEach((button) => {
  button.addEventListener("click", () => {
    queryInput.value = button.dataset.query;
    queryInput.focus();
  });
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  setLoading(true);
  clearOutput();
  const controller = new AbortController();
  const timeout = window.setTimeout(() => controller.abort(), 120000);

  try {
    const response = await fetch("/api/recommend", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      signal: controller.signal,
      body: JSON.stringify({
        query: queryInput.value.trim(),
        use_model: modelToggle.checked,
        live_nhtsa: liveNhtsaToggle.checked,
        configuration: configuration.value,
      }),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "The request failed.");
    render(data);
  } catch (error) {
    statusBox.textContent = error.name === "AbortError"
      ? "The request took too long. Check the model server and try again."
      : error.message;
    statusBox.classList.add("error");
  } finally {
    window.clearTimeout(timeout);
    setLoading(false);
  }
});

function setLoading(loading) {
  submitButton.disabled = loading;
  submitButton.textContent = loading ? "Finding vehicles…" : "Find vehicles";
  if (loading) statusBox.textContent = "Checking available vehicles and evidence…";
}

function setupVoiceInput() {
  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SpeechRecognition) {
    voiceButton.disabled = true;
    voiceButton.title = "Speech input is available in Chrome and Edge.";
    voiceButton.querySelector("span").textContent = "Speech unavailable";
    return;
  }

  const recognition = new SpeechRecognition();
  recognition.lang = navigator.language || "en-US";
  recognition.continuous = false;
  recognition.interimResults = true;
  let finalTranscript = "";

  voiceButton.addEventListener("click", () => {
    if (voiceButton.getAttribute("aria-pressed") === "true") {
      recognition.stop();
      return;
    }
    finalTranscript = "";
    queryInput.value = "";
    voiceStatus.hidden = false;
    voiceStatus.textContent = "Starting microphone…";
    recognition.start();
  });

  recognition.onstart = () => {
    voiceButton.setAttribute("aria-pressed", "true");
    voiceButton.classList.add("listening");
    voiceButton.querySelector("span").textContent = "Stop listening";
    voiceStatus.textContent = "Listening… Speak your vehicle request.";
  };

  recognition.onresult = (event) => {
    let interimTranscript = "";
    for (let index = event.resultIndex; index < event.results.length; index += 1) {
      const transcript = event.results[index][0].transcript;
      if (event.results[index].isFinal) finalTranscript += `${transcript} `;
      else interimTranscript += transcript;
    }
    queryInput.value = `${finalTranscript}${interimTranscript}`.trim();
  };

  recognition.onerror = (event) => {
    const messages = {
      "not-allowed": "Microphone permission was denied. Allow microphone access in Chrome and try again.",
      "no-speech": "No speech was detected. Try again and speak after Listening appears.",
      "audio-capture": "No microphone was found.",
      network: "Chrome speech recognition could not reach its speech service.",
    };
    voiceStatus.hidden = false;
    voiceStatus.textContent = messages[event.error] || `Speech recognition stopped: ${event.error}.`;
  };

  recognition.onend = () => {
    voiceButton.setAttribute("aria-pressed", "false");
    voiceButton.classList.remove("listening");
    voiceButton.querySelector("span").textContent = "Speak request";
    if (queryInput.value.trim() && !voiceStatus.textContent.includes("denied")) {
      voiceStatus.textContent = "Speech added. Review the request, then select Find vehicles.";
    }
  };
}

function clearOutput() {
  statusBox.classList.remove("error");
  summary.hidden = true;
  summary.replaceChildren();
  results.replaceChildren();
}

function render(data) {
  const candidates = data.candidates || data.recommendations || [];
  const questions = data.clarification_questions || [];
  renderSummary(data.parsed || data.constraints || {});

  if (!candidates.length) {
    statusBox.textContent = questions.length
      ? questions.join(" ")
      : data.message || "No vehicles match the current request.";
    return;
  }

  statusBox.textContent = data.match_mode === "closest_alternatives"
    ? `No exact matches · showing ${candidates.length} closest alternatives`
    : `${candidates.length} ranked vehicle${candidates.length === 1 ? "" : "s"}`;
  if (questions.length) {
    const clarification = document.createElement("span");
    clarification.className = "mode-notice";
    clarification.textContent = `Optional follow-up: ${questions.join(" ")}`;
    statusBox.append(document.createElement("br"), clarification);
  }
  candidates.forEach((candidate, index) => renderCandidate(candidate, index));
}

async function loadModelStatus() {
  try {
    const response = await fetch("/api/model-status");
    const data = await response.json();
    if (!response.ok) throw new Error();
    modelStatus.textContent = data.available ? "Qwen connected" : "Standard mode";
    modelStatus.classList.toggle("ready", Boolean(data.available));
  } catch {
    modelStatus.textContent = "Standard mode";
  }
}

function renderSummary(parsed) {
  const entries = Object.entries(parsed).filter(([, value]) => {
    return value !== null && value !== "" && (!Array.isArray(value) || value.length);
  });
  if (!entries.length) return;

  const list = document.createElement("dl");
  entries.forEach(([key, value]) => {
    const row = document.createElement("div");
    const term = document.createElement("dt");
    const description = document.createElement("dd");
    term.textContent = label(key);
    description.textContent = valueText(value);
    row.append(term, description);
    list.append(row);
  });
  summary.append(list);
  summary.hidden = false;
}

function renderCandidate(candidate, index) {
  const node = template.content.cloneNode(true);
  const vehicle = candidate.vehicle || candidate;
  const listing = candidate.listing || vehicle.listing || {};
  const safety = candidate.safety || vehicle.safety || {};
  const evidence = candidate.evidence || [];
  const evidenceById = new Map(evidence.map((item) => [item.id, item]));

  node.querySelector(".rank").textContent = candidate.rank || index + 1;
  node.querySelector(".vehicle-name").textContent = vehicleName(vehicle);
  node.querySelector(".score").textContent = formatScore(candidate.score);

  const facts = node.querySelector(".facts");
  addFact(facts, "Price", listing.price ?? vehicle.price, money);
  addFact(facts, "Mileage", listing.mileage ?? vehicle.mileage, number, " mi");
  addFact(facts, "Body", vehicle.body_style);
  addFact(facts, "Drivetrain", vehicle.drivetrain);
  addFact(facts, "Location", candidate.location);
  addFact(facts, "Source", candidate.listing_source);
  addFact(facts, "Record", dataModeLabel(candidate.listing_data_mode));
  addFact(facts, "Evidence", candidate.tier === "complete" ? "Complete" : "Needs review");

  renderAspectBars(node.querySelector(".aspect-bars"), candidate.score_components || {});

  const sections = node.querySelector(".sections");
  addSection(sections, "Why it fits", candidate.pros || candidate.reasons || candidate.highlights || [], evidenceById);
  addSection(sections, "Unmet constraints", candidate.constraint_violations || [], evidenceById);
  addSection(sections, "Evidence", candidate.review_evidence || evidence, evidenceById);
  addSection(sections, "Recall cautions", candidate.cautions || safety.cautions || safety.recalls || [], evidenceById);
  addSection(sections, "Graph paths", candidate.graph_paths || [], evidenceById);
  addSection(sections, "Missing evidence", candidate.missing_evidence || [], evidenceById);

  const source = node.querySelector(".source-link");
  const listingEvidence = (candidate.evidence || []).find((item) => item.type === "listing");
  const url = listing.url || listing.source_url || candidate.source_url || listingEvidence?.url;
  if (url) {
    source.href = url;
    source.textContent = candidate.listing_data_mode === "live"
      ? "View live listing"
      : "View source snapshot";
  } else {
    source.remove();
  }

  results.append(node);
}

function renderAspectBars(container, components) {
  const labels = {
    safety: "Safety",
    reliability: "Reliability",
    maintenance: "Maintenance",
    fuel_economy: "Fuel economy",
    cargo: "Cargo",
    snow: "Snow",
  };
  Object.entries(labels).forEach(([key, name]) => {
    const value = components[key];
    if (value === null || value === undefined) return;
    const row = document.createElement("div");
    row.className = "aspect-row";
    const labelNode = document.createElement("span");
    labelNode.textContent = name;
    const meter = document.createElement("span");
    meter.className = "aspect-meter";
    const fill = document.createElement("span");
    fill.style.width = `${Math.max(0, Math.min(1, Number(value))) * 100}%`;
    meter.append(fill);
    row.append(labelNode, meter);
    container.append(row);
  });
}

function dataModeLabel(value) {
  if (value === "live") return "Live";
  if (value === "live_capture") return "Live capture";
  if (value === "frozen_snapshot") return "Frozen snapshot";
  return value || "Unknown";
}

async function loadDataStatus() {
  try {
    const response = await fetch("/api/data-status");
    const data = await response.json();
    if (!response.ok || !data.ready) throw new Error(data.error || "Data unavailable");
    const counts = data.counts;
    dataNotice.textContent = `${counts.vehicles} vehicle variants · ${number.format(counts.listings)} listings · ${number.format(counts.reviews)} reviews · ${counts.recalls} recalls`;
  } catch (error) {
    dataNotice.textContent = error.message;
    dataNotice.classList.add("error");
  }
}

function addFact(container, name, value, formatter, suffix = "") {
  if (value === undefined || value === null || value === "") return;
  const item = document.createElement("span");
  item.className = "fact";
  const formatted = formatter ? formatter.format(Number(value)) : String(value);
  item.textContent = `${name}: ${formatted}${suffix}`;
  container.append(item);
}

function addSection(container, title, content, evidenceById = new Map()) {
  const items = Array.isArray(content) ? content : content ? [content] : [];
  if (!items.length) return;

  const section = document.createElement("section");
  section.className = "section";
  const heading = document.createElement("h3");
  heading.textContent = title;
  section.append(heading);

  const list = document.createElement("ul");
  items.slice(0, 4).forEach((item) => {
    const entry = document.createElement("li");
    entry.append(document.createTextNode(evidenceText(item)));
    if (item && typeof item === "object" && item.url) {
      const link = document.createElement("a");
      link.href = item.url;
      link.target = "_blank";
      link.rel = "noreferrer";
      link.textContent = "Source";
      entry.append(" ", link);
    }
    const citations = item && typeof item === "object" && Array.isArray(item.evidence_ids)
      ? [...new Set(item.evidence_ids)]
        .map((id) => evidenceById.get(id))
        .filter(Boolean)
      : [];
    citations.forEach((evidence, index) => {
      const link = document.createElement("a");
      link.className = "citation-link";
      link.href = evidence.url;
      link.target = "_blank";
      link.rel = "noreferrer";
      link.textContent = citations.length === 1 ? "Cited source" : `Cited source ${index + 1}`;
      link.title = `${evidence.source}: ${evidence.text}`;
      entry.append(" ", link);
    });
    list.append(entry);
  });
  section.append(list);
  container.append(section);
}

function evidenceText(item) {
  if (typeof item === "string") return item;
  const text = item.text || item.quote || item.description || item.title || JSON.stringify(item);
  const source = item.source ? ` — ${item.source}` : "";
  const indirect = item.indirect ? " (indirect)" : "";
  return `${text}${source}${indirect}`;
}

function vehicleName(vehicle) {
  const values = [vehicle.year, vehicle.make, vehicle.model, vehicle.trim].filter(Boolean);
  return values.join(" ") || vehicle.name || "Vehicle";
}

function formatScore(value) {
  if (value === undefined || value === null) return "";
  const score = Number(value);
  return score <= 1 ? `${Math.round(score * 100)}% match` : `${score.toFixed(1)} score`;
}

function valueText(value) {
  if (Array.isArray(value)) return value.join(", ");
  if (typeof value === "object") {
    return Object.entries(value)
      .map(([key, item]) => `${label(key)} ${item}`)
      .join(", ");
  }
  return String(value);
}

function label(key) {
  return key.replaceAll("_", " ").replace(/\b\w/g, (character) => character.toUpperCase());
}
