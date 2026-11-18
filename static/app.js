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
const notices = document.querySelector("#notices");
const refineControl = document.querySelector("#refine-control");
const refineToggle = document.querySelector("#refine");
const followUpQuestion = document.querySelector("#follow-up-question");

let previousQuery = "";
let awaitingAnswer = false;

const money = new Intl.NumberFormat("en-US", {
  style: "currency",
  currency: "USD",
  maximumFractionDigits: 0,
});
const number = new Intl.NumberFormat("en-US");

const RATINGS = [
  ["safety", "Safety"],
  ["reliability", "Reliability"],
  ["maintenance", "Upkeep cost"],
  ["fuel_economy", "Fuel economy"],
  ["cargo", "Cargo space"],
  ["snow", "Winter traction"],
];

const PRIORITY_NAMES = {
  price: "Low price",
  mileage: "Low mileage",
  safety: "Safety",
  reliability: "Reliability",
  maintenance: "Upkeep cost",
  fuel_economy: "Fuel economy",
  cargo: "Cargo space",
  snow: "Winter traction",
};

loadModelStatus();
loadDataStatus();
setupVoiceInput();

document.querySelectorAll("[data-query]").forEach((button) => {
  button.addEventListener("click", () => {
    queryInput.value = button.dataset.query;
    awaitingAnswer = false;
    refineControl.hidden = true;
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
        query: buildQuery(),
        use_model: modelToggle.checked,
        live_nhtsa: liveNhtsaToggle.checked,
        configuration: configuration.value,
      }),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "The search failed. Try again.");
    render(data);
  } catch (error) {
    statusBox.textContent = error.name === "AbortError"
      ? "The search took too long. Turn off the language model in Settings and try again."
      : error.message;
    statusBox.classList.add("error");
  } finally {
    window.clearTimeout(timeout);
    setLoading(false);
  }
});

function buildQuery() {
  const text = queryInput.value.trim();
  if (awaitingAnswer && refineToggle.checked && previousQuery) {
    return `${previousQuery} ${text}`.slice(0, 1000);
  }
  return text;
}

function setLoading(loading) {
  submitButton.disabled = loading;
  submitButton.textContent = loading ? "Searching…" : "Find cars";
  if (loading) statusBox.textContent = "Checking listings, reviews, and safety records…";
}

function clearOutput() {
  statusBox.classList.remove("error");
  notices.hidden = true;
  notices.textContent = "";
  summary.hidden = true;
  summary.replaceChildren();
  results.replaceChildren();
}

/* ---------- Page-level rendering ---------- */

function render(data) {
  const candidates = data.candidates || [];
  const questions = data.clarification_questions || [];
  renderSummary(data.parsed || {});

  previousQuery = data.query || "";
  awaitingAnswer = questions.length > 0;
  refineControl.hidden = !awaitingAnswer;
  if (awaitingAnswer) {
    followUpQuestion.textContent = questions.join(" ");
    queryInput.value = "";
    queryInput.placeholder = "Type your answer, e.g. under $15,000";
  }

  const noticeText = data.model_notice;
  if (noticeText) {
    notices.textContent = noticeText;
    notices.hidden = false;
  }

  if (!candidates.length) {
    statusBox.textContent = data.message || "No cars match yet. Try a higher budget or fewer limits.";
    return;
  }
  statusBox.textContent = data.match_mode === "closest_alternatives"
    ? "Nothing meets every limit. Here are the closest options."
    : `Top ${candidates.length} ${candidates.length === 1 ? "match" : "matches"}`;
  candidates.forEach((candidate, index) => renderCandidate(candidate, index, data));
}

function renderSummary(parsed) {
  const chips = [];
  if (parsed.max_price) chips.push(`Under ${money.format(parsed.max_price)}`);
  if (parsed.body_style) chips.push(parsed.body_style === "suv" ? "SUV" : capitalize(parsed.body_style));
  if (parsed.drivetrain) chips.push(parsed.drivetrain);
  if (parsed.max_mileage) chips.push(`Under ${number.format(parsed.max_mileage)} mi`);
  if (parsed.min_year) chips.push(`${parsed.min_year} or newer`);
  const priorities = Object.entries(parsed.preferences || {})
    .sort((a, b) => b[1] - a[1])
    .slice(0, 3)
    .map(([key]) => PRIORITY_NAMES[key] || capitalize(key));

  if (!chips.length && !priorities.length) return;
  const limits = chipGroup("You asked for", chips, "chip");
  const ranked = chipGroup("Ranked by", priorities, "chip chip-soft");
  summary.append(...[limits, ranked].filter(Boolean));
  summary.hidden = false;
}

function chipGroup(title, values, className) {
  if (!values.length) return null;
  const group = document.createElement("div");
  group.className = "chip-group";
  const label = document.createElement("span");
  label.className = "chip-label";
  label.textContent = title;
  group.append(label);
  values.forEach((value) => {
    const chip = document.createElement("span");
    chip.className = className;
    chip.textContent = value;
    group.append(chip);
  });
  return group;
}

/* ---------- Card rendering ---------- */

function renderCandidate(candidate, index, data) {
  const node = template.content.cloneNode(true);
  const card = node.querySelector(".candidate");
  const evidence = candidate.evidence || [];
  const evidenceById = new Map(evidence.map((item) => [item.id, item]));
  const unmet = candidate.constraint_violations || [];

  node.querySelector(".rank").textContent = candidate.rank || index + 1;
  node.querySelector(".vehicle-name").textContent = vehicleName(candidate);
  node.querySelector(".facts").append(...factSpans(candidate));

  renderGauge(node.querySelector(".gauge"), candidate.score, unmet.length > 0);

  if (unmet.length) {
    const line = node.querySelector(".unmet");
    line.textContent = unmet[0];
    line.hidden = false;
  }

  renderBadges(node.querySelector(".badges"), candidate);
  renderRatings(node.querySelector(".ratings"), candidate);
  renderHighlights(node.querySelector(".highlights"), candidate, evidenceById);
  renderDetails(node, candidate, data, evidenceById);

  const source = node.querySelector(".source-link");
  const listingEvidence = evidence.find((item) => item.type === "listing");
  const url = listingEvidence?.url || candidate.source_url;
  if (url) {
    source.href = url;
    source.textContent = candidate.listing_data_mode === "live" ? "View live listing" : "View listing";
  } else {
    source.remove();
  }

  const liveButton = node.querySelector(".live-check");
  const liveBox = node.querySelector(".live-results");
  liveButton.addEventListener("click", () => liveCheck(candidate.id, liveButton, liveBox));

  card.dataset.rank = String(index + 1);
  results.append(node);
}

function factSpans(candidate) {
  const facts = [];
  if (candidate.price != null) facts.push(["price", money.format(candidate.price)]);
  if (candidate.mileage != null) facts.push(["", `${number.format(candidate.mileage)} mi`]);
  if (candidate.drivetrain) facts.push(["", candidate.drivetrain]);
  if (candidate.body_style) facts.push(["", candidate.body_style === "suv" ? "SUV" : capitalize(candidate.body_style)]);
  return facts.map(([className, text]) => {
    const span = document.createElement("span");
    if (className) span.className = className;
    span.textContent = text;
    return span;
  });
}

function renderGauge(gauge, score, missesLimits) {
  const value = Math.round(Math.max(0, Math.min(1, Number(score) || 0)) * 100);
  const level = missesLimits ? "poor" : value >= 65 ? "good" : value >= 45 ? "fair" : "poor";
  const label = missesLimits
    ? "Misses a limit"
    : { good: "Strong match", fair: "Fair match", poor: "Weak match" }[level];
  gauge.classList.add(`is-${level}`);
  gauge.querySelector(".gauge-fill").style.strokeDasharray = `${value} 100`;
  gauge.querySelector(".gauge-value").textContent = value;
  gauge.querySelector(".gauge-label").textContent = label;
  gauge.setAttribute("aria-label", `Overall match ${value} out of 100: ${label}`);
}

function renderBadges(list, candidate) {
  const badges = [];
  if (candidate.safety_rating) {
    badges.push(["good", `${candidate.safety_rating}★ crash rating`]);
  } else {
    badges.push(["none", "No crash rating"]);
  }
  badges.push(
    candidate.recall_count
      ? ["fair", `${candidate.recall_count} recall${candidate.recall_count === 1 ? "" : "s"}`]
      : ["good", "No recalls on file"],
  );
  badges.push(
    candidate.listing_data_mode === "live" ? ["live", "Live listing"] : ["none", "Saved listing"],
  );
  if (candidate.tier !== "complete") badges.push(["none", "Limited data"]);

  badges.forEach(([level, text]) => {
    const item = document.createElement("li");
    item.className = `badge is-${level}`;
    item.textContent = text;
    list.append(item);
  });
}

function renderRatings(list, candidate) {
  const components = candidate.score_components || {};
  RATINGS.forEach(([key, name]) => {
    const raw = components[key];
    const item = document.createElement("li");
    item.className = "rating";

    const label = document.createElement("span");
    label.className = "rating-name";
    label.textContent = name;

    const meter = document.createElement("span");
    meter.className = "segments";
    const outOfFive = raw == null ? null : Math.max(0, Math.min(5, Number(raw) * 5));
    const filled = outOfFive == null ? 0 : Math.round(outOfFive);
    const level = outOfFive == null ? "none" : outOfFive >= 3.5 ? "good" : outOfFive >= 2.5 ? "fair" : "poor";
    meter.classList.add(`is-${level}`);
    for (let index = 0; index < 5; index += 1) {
      const segment = document.createElement("span");
      if (index < filled) segment.className = "on";
      meter.append(segment);
    }

    const value = document.createElement("span");
    value.className = "rating-value";
    value.textContent = outOfFive == null ? "No data" : outOfFive.toFixed(1);

    item.setAttribute(
      "aria-label",
      outOfFive == null ? `${name}: no data` : `${name}: ${outOfFive.toFixed(1)} out of 5`,
    );
    item.append(label, meter, value);
    list.append(item);
  });
}

function renderHighlights(list, candidate, evidenceById) {
  const bundles = candidate.aspect_bundles || [];
  const pro = bundles.find((bundle) => bundle.positive)?.positive
    || (candidate.pros || []).find((claim) => !/NHTSA/i.test(claim.text));
  let con = bundles.find((bundle) => bundle.negative)?.negative
    || (candidate.cautions || []).find((claim) => !/fitment|indirect|recall/i.test(claim.text));
  if (con && pro && con.text === pro.text) con = null;

  [["good", "✓", pro], ["fair", "!", con]].forEach(([level, icon, claim]) => {
    if (!claim?.text) return;
    const item = document.createElement("li");
    item.className = `highlight is-${level}`;
    const mark = document.createElement("span");
    mark.className = "highlight-icon";
    mark.setAttribute("aria-hidden", "true");
    mark.textContent = icon;
    const text = document.createElement("span");
    text.className = "highlight-text";
    text.textContent = shorten(claim.text, 110);
    text.title = claim.text;
    const source = (claim.evidence_ids || []).map((id) => evidenceById.get(id)).find((entry) => entry?.url);
    if (source) text.append(" ", sourceLink(source));
    item.append(mark, text);
    list.append(item);
  });
}

function renderDetails(node, candidate, data, evidenceById) {
  const stages = data.model_stages || {};
  const backend = (candidate.graph_paths || [])[0]?.backend;
  const rows = [
    ["Method", `Configuration ${data.configuration || "C"}`],
    ["Request parsing", stages.parsing === "qwen" ? "Qwen" : "Rules"],
    ["Review aspects", { qwen: "Qwen", vader: "VADER", stored_lexicon: "Lexicon" }[stages.review_extraction] || "Lexicon"],
    ["Evidence search", readableRetrieval(candidate.retrieval_mode)],
    ["Knowledge graph", backend ? (backend === "neo4j" ? "Neo4j" : "In-memory graph") : "Not used"],
    ["Summaries", stages.summaries === "qwen" ? "Qwen" : "Templates"],
  ];
  const method = node.querySelector(".method");
  rows.forEach(([term, value]) => {
    const dt = document.createElement("dt");
    dt.textContent = term;
    const dd = document.createElement("dd");
    dd.textContent = value;
    method.append(dt, dd);
  });

  const sections = node.querySelector(".sections");
  const bundles = [];
  (candidate.aspect_bundles || []).forEach((bundle) => {
    const name = PRIORITY_NAMES[bundle.aspect] || capitalize(bundle.aspect);
    if (bundle.positive) bundles.push({ text: `${name}: “${bundle.positive.text}”`, evidence_ids: bundle.positive.evidence_ids, level: "good" });
    if (bundle.negative) bundles.push({ text: `${name}: “${bundle.negative.text}”`, evidence_ids: bundle.negative.evidence_ids, level: "fair" });
  });
  addSection(sections, "What owners say", bundles, evidenceById);
  addSection(
    sections,
    "Recalls",
    (candidate.evidence || []).filter((item) => item.type === "recall"),
    evidenceById,
  );
  addSection(sections, "Graph connections", candidate.graph_paths || [], evidenceById);
  addSection(
    sections,
    "Most relevant passages",
    (candidate.retrieved_passages || []).slice(0, 3).map((item) => ({
      text: item.text,
      evidence_ids: [item.evidence_id],
    })),
    evidenceById,
  );
  addSection(
    sections,
    "Missing data",
    (candidate.missing_evidence || []).map((key) => PRIORITY_NAMES[key] || capitalize(key)),
    evidenceById,
  );
}

function addSection(container, title, items, evidenceById) {
  if (!items.length) return;
  const section = document.createElement("section");
  section.className = "detail-section";
  const heading = document.createElement("h3");
  heading.textContent = title;
  const list = document.createElement("ul");
  items.slice(0, 4).forEach((item) => {
    const entry = document.createElement("li");
    if (item.level) entry.className = `is-${item.level}`;
    const text = typeof item === "string" ? item : item.text || "";
    entry.append(document.createTextNode(shorten(text, 180)));
    const source = typeof item === "object" && item.url
      ? item
      : (item.evidence_ids || []).map((id) => evidenceById.get(id)).find((evidence) => evidence?.url);
    if (source) entry.append(" ", sourceLink(source));
    list.append(entry);
  });
  section.append(heading, list);
  container.append(section);
}

function sourceLink(evidence) {
  const link = document.createElement("a");
  link.className = "source-chip";
  link.href = evidence.url;
  link.target = "_blank";
  link.rel = "noreferrer";
  link.textContent = "Source";
  link.title = evidence.source || "Source";
  return link;
}

/* ---------- Live check ---------- */

async function liveCheck(vehicleId, button, box) {
  button.disabled = true;
  button.textContent = "Checking…";
  box.hidden = false;
  box.replaceChildren();
  try {
    const response = await fetch("/api/live-check", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ vehicle_id: vehicleId }),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "Live check failed.");
    const heading = document.createElement("p");
    heading.className = "live-heading";
    heading.textContent = data.message
      || `${data.listings.length} live listing${data.listings.length === 1 ? "" : "s"} from ${data.sources.join(", ")}, checked ${new Date(data.checked_at).toLocaleTimeString([], { hour: "numeric", minute: "2-digit" })}`;
    box.append(heading);
    const list = document.createElement("ul");
    data.listings.forEach((item) => {
      const entry = document.createElement("li");
      const price = item.price != null ? money.format(item.price) : "No price";
      const miles = item.mileage != null ? `, ${number.format(item.mileage)} mi` : "";
      const link = document.createElement("a");
      link.href = item.url;
      link.target = "_blank";
      link.rel = "noreferrer";
      link.textContent = `${price}${miles}`;
      entry.append(link, ` on ${item.source}`);
      list.append(entry);
    });
    if (data.listings.length) box.append(list);
    (data.errors || []).forEach((message) => {
      const error = document.createElement("p");
      error.className = "live-error";
      error.textContent = message;
      box.append(error);
    });
  } catch (error) {
    box.textContent = error.message;
  } finally {
    button.disabled = false;
    button.textContent = "Check live listings";
  }
}

/* ---------- Status bars ---------- */

async function loadModelStatus() {
  try {
    const response = await fetch("/api/model-status");
    const data = await response.json();
    if (!response.ok) throw new Error();
    modelStatus.textContent = data.available ? "Qwen connected" : "Qwen offline";
    modelStatus.classList.toggle("ready", Boolean(data.available));
  } catch {
    modelStatus.textContent = "Qwen offline";
  }
}

async function loadDataStatus() {
  try {
    const response = await fetch("/api/data-status");
    const data = await response.json();
    if (!response.ok || !data.ready) throw new Error(data.error || "Data unavailable");
    const counts = data.counts;
    const live = Object.entries(data.connectors || {})
      .filter(([, on]) => on)
      .map(([name]) => ({ nhtsa: "NHTSA", "auto.dev": "Auto.dev", serpapi: "SerpApi", ebay: "eBay", bestbuy: "Best Buy", marketcheck: "MarketCheck" }[name] || name));
    dataNotice.textContent = `Searching ${counts.vehicles} models, ${number.format(counts.listings)} listings, and ${number.format(counts.reviews)} owner reviews${live.length ? `. Live: ${live.join(", ")}` : ""}.`;
  } catch (error) {
    dataNotice.textContent = error.message;
    dataNotice.classList.add("error");
  }
}

/* ---------- Voice input ---------- */

function setupVoiceInput() {
  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SpeechRecognition) {
    voiceButton.disabled = true;
    voiceButton.title = "Voice input works in Chrome and Edge.";
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
    voiceStatus.textContent = "Listening…";
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
      "not-allowed": "Microphone access is blocked. Allow it in Chrome's site settings.",
      "no-speech": "No speech detected. Try again.",
      "audio-capture": "No microphone found.",
      network: "Chrome's speech service is unreachable.",
    };
    voiceStatus.hidden = false;
    voiceStatus.textContent = messages[event.error] || `Voice input stopped: ${event.error}.`;
  };

  recognition.onend = () => {
    voiceButton.setAttribute("aria-pressed", "false");
    voiceButton.classList.remove("listening");
    if (queryInput.value.trim() && !voiceStatus.textContent.includes("blocked")) {
      voiceStatus.textContent = "Check the text, then select Find cars.";
    }
  };
}

/* ---------- Helpers ---------- */

function readableRetrieval(mode) {
  if (!mode) return "Not used";
  if (mode === "bm25") return "Keyword search (BM25)";
  if (mode.startsWith("ontology_graph")) {
    return mode.includes("MiniLM") ? "Graph paths + MiniLM" : "Graph paths + TF-IDF";
  }
  if (mode.includes("MiniLM")) return "Vector search (MiniLM)";
  return "Keyword search (TF-IDF)";
}

function vehicleName(candidate) {
  return [candidate.year, candidate.make, candidate.model, candidate.trim].filter(Boolean).join(" ") || "Vehicle";
}

function shorten(text, limit) {
  const clean = String(text || "").replace(/\s+/g, " ").trim();
  return clean.length > limit ? `${clean.slice(0, limit - 1).trimEnd()}…` : clean;
}

function capitalize(value) {
  const text = String(value || "").replaceAll("_", " ");
  return text.charAt(0).toUpperCase() + text.slice(1);
}
