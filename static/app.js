const form = document.querySelector("#search-form");
const queryInput = document.querySelector("#query");
const voiceButton = document.querySelector("#voice-input");
const voiceStatus = document.querySelector("#voice-status");
const submitButton = document.querySelector("#submit");
const resetButton = document.querySelector("#reset-button");
const modelToggle = document.querySelector("#use-model");
const liveNhtsaToggle = document.querySelector("#live-nhtsa");
const modelStatus = document.querySelector("#model-status");
const dataNotice = document.querySelector("#data-notice");
const graphStatus = document.querySelector("#graph-status");
const configuration = document.querySelector("#configuration");
const statusBox = document.querySelector("#status");
const clarificationPanel = document.querySelector("#clarification-panel");
const summary = document.querySelector("#summary");
const results = document.querySelector("#results");
const template = document.querySelector("#candidate-template");

const money = new Intl.NumberFormat("en-US", {
  style: "currency",
  currency: "USD",
  maximumFractionDigits: 0,
});

const moneyExact = new Intl.NumberFormat("en-US", {
  style: "currency",
  currency: "USD",
  minimumFractionDigits: 2,
  maximumFractionDigits: 2,
});

const number = new Intl.NumberFormat("en-US");

const ASPECT_LABELS = [
  ["safety", "Safety"],
  ["reliability", "Reliability"],
  ["maintenance", "Maintenance"],
  ["fuel_economy", "Fuel economy"],
  ["cargo", "Cargo"],
  ["snow", "Snow"],
];

const RETAILER_NAMES = {
  ebay: "eBay Motors",
  amazon: "Amazon Automotive",
  walmart: "Walmart Auto & Tires",
  bestbuy: "Best Buy Car Electronics",
};

loadModelStatus();
loadDataStatus();
setupVoiceInput();

document.querySelectorAll("[data-query]").forEach((button) => {
  button.addEventListener("click", () => {
    queryInput.value = button.dataset.query || "";
    queryInput.focus();
  });
});

resetButton.addEventListener("click", () => {
  queryInput.value = "";
  voiceStatus.hidden = true;
  voiceStatus.textContent = "";
  statusBox.textContent = "";
  clearOutput();
  queryInput.focus();
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  await runSearch();
});

async function runSearch() {
  const query = queryInput.value.trim();
  if (!query) {
    statusBox.textContent = "Enter a vehicle request.";
    statusBox.classList.add("error");
    queryInput.focus();
    return;
  }

  setLoading(true);
  clearOutput();
  const controller = new AbortController();
  const timeout = window.setTimeout(() => controller.abort(), 180000);

  try {
    const response = await fetch("/api/recommend", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      signal: controller.signal,
      body: JSON.stringify({
        query,
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
}

function setLoading(loading) {
  submitButton.disabled = loading;
  resetButton.disabled = loading;
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
  clarificationPanel.hidden = true;
  clarificationPanel.replaceChildren();
  summary.hidden = true;
  summary.replaceChildren();
  results.replaceChildren();
}

function render(data) {
  const candidates = data.candidates || data.recommendations || [];
  const questions = (data.clarification_questions || []).slice(0, 2);
  const parsed = data.parsed || data.constraints || {};

  renderClarificationPanel(questions, parsed, candidates.length > 0);
  renderSummary(data, parsed);

  if (!candidates.length) {
    statusBox.textContent = data.message || "No vehicles match the current request.";
    return;
  }

  statusBox.textContent = data.match_mode === "closest_alternatives"
    ? `No exact matches · showing ${candidates.length} closest alternative${candidates.length === 1 ? "" : "s"}`
    : `Showing ${candidates.length} ranked vehicle${candidates.length === 1 ? "" : "s"}`;

  candidates.forEach((candidate, index) => {
    renderCandidate(candidate, index, data);
  });
}

function renderClarificationPanel(questions, parsed, hasCandidates) {
  clarificationPanel.replaceChildren();
  if (!questions.length) {
    clarificationPanel.hidden = true;
    return;
  }

  const body = document.createElement("div");
  body.className = "clarification-body";

  const heading = document.createElement("p");
  heading.className = "clarification-header";
  heading.textContent = hasCandidates
    ? "Clarify your request to refine these matches"
    : "Clarify your request";
  body.append(heading);

  const list = document.createElement("div");
  list.className = "clarification-list";

  questions.forEach((questionText) => {
    const item = document.createElement("div");
    item.className = "clarification-item";

    const q = document.createElement("p");
    q.className = "clarification-question";
    q.textContent = questionText;
    item.append(q);

    const chips = quickOptionsForQuestion(questionText, parsed);
    if (chips.length) {
      const chipRow = document.createElement("div");
      chipRow.className = "quick-chips";
      chips.forEach(({ labelText, appendText }) => {
        const chip = document.createElement("button");
        chip.type = "button";
        chip.className = "quick-chip";
        chip.textContent = labelText;
        chip.addEventListener("click", () => appendToQuery(appendText));
        chipRow.append(chip);
      });
      item.append(chipRow);
    }

    list.append(item);
  });

  const replyForm = document.createElement("form");
  replyForm.className = "clarification-reply-form";

  const replyInput = document.createElement("input");
  replyInput.type = "text";
  replyInput.className = "clarification-reply-input";
  replyInput.placeholder = "Or type a quick answer (e.g. under $16,000 or SUV)…";
  replyInput.setAttribute("aria-label", "Clarification reply");

  const replyButton = document.createElement("button");
  replyButton.type = "submit";
  replyButton.className = "clarification-submit";
  replyButton.textContent = "Update search";

  replyForm.addEventListener("submit", (event) => {
    event.preventDefault();
    const answer = replyInput.value.trim();
    if (!answer) {
      replyInput.focus();
      return;
    }
    appendToQuery(answer);
  });

  replyForm.append(replyInput, replyButton);
  body.append(list, replyForm);

  if (hasCandidates) {
    const drawer = document.createElement("details");
    drawer.className = "clarification-drawer";
    drawer.open = true;
    const summaryEl = document.createElement("summary");
    summaryEl.textContent = `Clarify your request (${questions.length} optional question${questions.length === 1 ? "" : "s"})`;
    drawer.append(summaryEl, body);
    clarificationPanel.append(drawer);
  } else {
    const card = document.createElement("div");
    card.className = "clarification-card";
    card.append(body);
    clarificationPanel.append(card);
  }

  clarificationPanel.hidden = false;
}

function quickOptionsForQuestion(questionText, parsed) {
  const q = questionText.toLowerCase();
  if (/budget|price|cost|\$/.test(q) || (parsed.max_price == null && /maximum|spend/.test(q))) {
    return [
      { labelText: "$10,000", appendText: "under $10,000" },
      { labelText: "$15,000", appendText: "under $15,000" },
      { labelText: "$18,000", appendText: "under $18,000" },
      { labelText: "$25,000", appendText: "under $25,000" },
    ];
  }
  if (/body|suv|sedan|truck|style/.test(q)) {
    return [
      { labelText: "SUV", appendText: "SUV" },
      { labelText: "Sedan", appendText: "sedan" },
      { labelText: "Truck", appendText: "truck" },
    ];
  }
  if (/mileage|miles/.test(q)) {
    return [
      { labelText: "Under 80k mi", appendText: "under 80,000 miles" },
      { labelText: "Under 100k mi", appendText: "under 100,000 miles" },
      { labelText: "Under 120k mi", appendText: "under 120,000 miles" },
    ];
  }
  if (/drivetrain|awd|4wd|fwd|rwd|snow/.test(q)) {
    return [
      { labelText: "AWD", appendText: "AWD" },
      { labelText: "4WD", appendText: "4WD" },
      { labelText: "FWD", appendText: "FWD" },
    ];
  }
  if (/year|newer|old/.test(q)) {
    return [
      { labelText: "2010+", appendText: "2010 or newer" },
      { labelText: "2012+", appendText: "2012 or newer" },
      { labelText: "2015+", appendText: "2015 or newer" },
    ];
  }
  return [];
}

function appendToQuery(addition) {
  const clean = addition.trim();
  if (!clean) return;
  const current = queryInput.value.trim().replace(/[.\s]+$/, "");
  queryInput.value = current ? `${current}, ${clean}` : clean;
  runSearch();
}

function renderSummary(data, parsed) {
  summary.replaceChildren();

  const row = document.createElement("div");
  row.className = "summary-row";

  const slotDefs = [
    ["Max price", parsed.max_price != null ? money.format(Number(parsed.max_price)) : null],
    ["Max mileage", parsed.max_mileage != null ? `${number.format(Number(parsed.max_mileage))} mi` : null],
    ["Min year", parsed.min_year != null ? `${parsed.min_year}+` : null],
    ["Make", parsed.make || null],
    ["Body", parsed.body_style ? formatBodyStyle(parsed.body_style) : null],
    ["Drivetrain", parsed.drivetrain || null],
  ];

  slotDefs.forEach(([name, val]) => {
    if (!val) return;
    const pill = document.createElement("span");
    pill.className = "slot-pill";
    const strong = document.createElement("strong");
    strong.textContent = `${name}:`;
    pill.append(strong, document.createTextNode(` ${val}`));
    row.append(pill);
  });

  if (data.feasible_count !== undefined && data.feasible_count !== null) {
    const cohort = document.createElement("span");
    cohort.className = "cohort-pill";
    const comparable = data.comparable_cohort_count ?? 0;
    const retention = data.cohort_retention != null
      ? ` (${Math.round(Number(data.cohort_retention) * 100)}%)`
      : "";
    cohort.textContent = `Comparable cohort: ${comparable} / ${data.feasible_count} feasible${retention}`;
    row.append(cohort);
  }

  const stagePill = document.createElement("span");
  stagePill.className = "stage-pill";
  const stages = data.model_stages || {};
  const stageParts = [
    data.configuration ? `Config ${data.configuration}` : null,
    stages.parsing ? `Parse: ${stages.parsing}` : null,
    stages.review_extraction ? `Aspects: ${label(stages.review_extraction)}` : null,
    stages.summaries ? `Summaries: ${stages.summaries}` : null,
  ].filter(Boolean);
  stagePill.textContent = stageParts.join(" · ");
  if (stagePill.textContent) row.append(stagePill);

  if (row.childNodes.length) {
    summary.append(row);
  }

  if (parsed.preferences && typeof parsed.preferences === "object") {
    const topPrefs = Object.entries(parsed.preferences)
      .filter(([, weight]) => Number(weight) > 0)
      .sort((a, b) => Number(b[1]) - Number(a[1]))
      .map(([key, weight]) => `${label(key)} ${Math.round(Number(weight) * 100)}%`);
    if (topPrefs.length) {
      const prefLine = document.createElement("p");
      prefLine.className = "summary-meta";
      prefLine.textContent = `Ranking weights: ${topPrefs.join(" · ")}`;
      summary.append(prefLine);
    }
  }

  if (data.model_notice) {
    const notice = document.createElement("p");
    notice.className = "summary-meta";
    notice.textContent = data.model_notice;
    summary.append(notice);
  }

  if (summary.childNodes.length) {
    summary.hidden = false;
  }
}

function renderCandidate(candidate, index, data) {
  const node = template.content.cloneNode(true);
  const vehicle = candidate.vehicle || candidate;
  const listing = candidate.listing || vehicle.listing || {};
  const safety = candidate.safety || vehicle.safety || {};
  const evidence = candidate.evidence || [];
  const evidenceById = new Map(evidence.map((item) => [item.id, item]));

  node.querySelector(".rank").textContent = candidate.rank || index + 1;
  node.querySelector(".vehicle-name").textContent = vehicleName(vehicle);

  const sublineParts = [
    vehicle.trim ? `Trim: ${vehicle.trim}` : null,
    candidate.retrieval_mode ? `Retrieval: ${candidate.retrieval_mode}` : null,
  ].filter(Boolean);
  const subline = node.querySelector(".vehicle-subline");
  if (sublineParts.length) {
    subline.textContent = sublineParts.join(" · ");
  } else {
    subline.remove();
  }

  const tierPill = node.querySelector(".tier-pill");
  const isClosest = data.match_mode === "closest_alternatives"
    || (Array.isArray(candidate.constraint_violations) && candidate.constraint_violations.length > 0);
  if (isClosest) {
    tierPill.textContent = "Closest alternative";
    tierPill.classList.add("tier-pill--closest");
  } else if (candidate.tier === "complete") {
    tierPill.textContent = "Complete evidence";
    tierPill.classList.add("tier-pill--complete");
  } else {
    tierPill.textContent = "Needs additional evidence";
    tierPill.classList.add("tier-pill--review");
  }

  const priceValue = listing.price ?? candidate.price ?? vehicle.price;
  const mileageValue = listing.mileage ?? candidate.mileage ?? vehicle.mileage;
  node.querySelector(".asking-price").textContent = priceValue != null
    ? money.format(Number(priceValue))
    : "Price unavailable";
  node.querySelector(".mileage-readout").textContent = mileageValue != null
    ? `${number.format(Number(mileageValue))} mi`
    : "";
  node.querySelector(".score-badge").textContent = formatScore(candidate.score);

  const specPills = node.querySelector(".spec-pills");
  if (vehicle.body_style) addSpecPill(specPills, formatBodyStyle(vehicle.body_style));
  if (vehicle.drivetrain) addSpecPill(specPills, vehicle.drivetrain);
  const stars = candidate.safety_rating ?? safety.ncap_stars;
  addSpecPill(specPills, stars != null ? `★ ${stars}/5 NHTSA` : "NHTSA unrated");
  const recallCount = candidate.recall_count ?? (Array.isArray(safety.recalls) ? safety.recalls.length : null);
  if (recallCount != null) {
    addSpecPill(specPills, `${recallCount} recall${recallCount === 1 ? "" : "s"}`);
  }
  if (candidate.location) addSpecPill(specPills, candidate.location);
  const modeText = dataModeLabel(candidate.listing_data_mode);
  const sourceLabel = candidate.listing_source ? `${candidate.listing_source} · ${modeText}` : modeText;
  const sourceClass = candidate.listing_data_mode === "live" ? "spec-pill--live" : "spec-pill--snapshot";
  addSpecPill(specPills, sourceLabel, sourceClass);

  const sourceLink = node.querySelector(".source-link");
  const listingEvidence = evidence.find((item) => item.type === "listing");
  const rawUrl = listing.url || listing.source_url || candidate.source_url || listingEvidence?.url;
  const verifiedUrl = safeUrl(rawUrl);
  if (verifiedUrl) {
    sourceLink.href = verifiedUrl;
    sourceLink.textContent = candidate.listing_data_mode === "live"
      ? "View live listing ↗"
      : "View listing ↗";
  } else {
    sourceLink.remove();
  }

  renderAspectBars(
    node.querySelector(".aspect-bars"),
    candidate.score_components || {},
    candidate.aspect_summary || null,
    stars
  );

  renderProsColumn(node.querySelector(".pros-list"), candidate, evidenceById);
  renderCautionsColumn(node.querySelector(".cautions-list"), candidate, evidenceById);
  renderEvidenceDrawer(node.querySelector(".evidence-drawer"), candidate, data, evidenceById);

  results.append(node);
}

function addSpecPill(container, text, extraClass = "") {
  if (!text) return;
  const pill = document.createElement("span");
  pill.className = extraClass ? `spec-pill ${extraClass}` : "spec-pill";
  pill.textContent = text;
  container.append(pill);
}

function renderAspectBars(container, components, aspectSummary, safetyStars) {
  ASPECT_LABELS.forEach(([key, name]) => {
    const value = components[key];
    const summaryItem = aspectSummary ? aspectSummary[key] : null;
    const row = document.createElement("div");
    row.className = "aspect-row";

    const labelNode = document.createElement("span");
    labelNode.className = "aspect-label";
    labelNode.textContent = name;

    const meter = document.createElement("span");
    meter.className = "aspect-meter";

    const badge = document.createElement("span");
    badge.className = "aspect-badge";

    if (value === null || value === undefined) {
      meter.classList.add("is-missing");
      badge.classList.add("is-missing");
      badge.textContent = "No data";
    } else {
      const clamped = Math.max(0, Math.min(1, Number(value)));
      const fill = document.createElement("span");
      fill.style.width = `${Math.round(clamped * 100)}%`;
      if (clamped < 0.4) fill.classList.add("is-low");
      else if (clamped < 0.65) fill.classList.add("is-mid");
      meter.append(fill);

      if (summaryItem && summaryItem.total > 0) {
        const mixNote = summaryItem.mixed > 0 ? ` · ${summaryItem.mixed}~` : "";
        badge.textContent = `${summaryItem.positive}+ · ${summaryItem.negative}−${mixNote}`;
        row.title = `${summaryItem.positive} positive, ${summaryItem.negative} negative, ${summaryItem.neutral} neutral, ${summaryItem.mixed} mixed (${summaryItem.total} spans)`;
      } else if (key === "safety" && safetyStars != null) {
        badge.textContent = `${safetyStars}/5`;
      } else {
        badge.textContent = `${Math.round(clamped * 100)}%`;
      }
    }

    row.append(labelNode, meter, badge);
    container.append(row);
  });
}

function renderProsColumn(list, candidate, evidenceById) {
  const items = Array.isArray(candidate.pros) ? candidate.pros : [];
  if (!items.length) {
    const empty = document.createElement("li");
    empty.className = "missing-note";
    empty.textContent = "Selected based on listing price, mileage, and available attributes.";
    list.append(empty);
    return;
  }

  items.forEach((item) => {
    list.append(buildClaimItem(item, evidenceById));
  });
}

function renderCautionsColumn(list, candidate, evidenceById) {
  const violations = Array.isArray(candidate.constraint_violations) ? candidate.constraint_violations : [];
  violations.forEach((violationText) => {
    const li = document.createElement("li");
    const tag = document.createElement("span");
    tag.className = "unmet-tag";
    tag.textContent = "Unmet constraint";
    li.append(tag, document.createTextNode(String(violationText)));
    list.append(li);
  });

  const cautions = Array.isArray(candidate.cautions) ? candidate.cautions : [];
  cautions.forEach((item) => {
    list.append(buildClaimItem(item, evidenceById));
  });

  const missing = Array.isArray(candidate.missing_evidence) ? candidate.missing_evidence : [];
  if (missing.length) {
    const li = document.createElement("li");
    li.className = "missing-note";
    li.textContent = `Missing review evidence: ${missing.map(label).join(", ")}.`;
    list.append(li);
  }

  if (!list.childNodes.length) {
    const li = document.createElement("li");
    li.className = "missing-note";
    li.textContent = "No cautions recorded; verify VIN history prior to purchase.";
    list.append(li);
  }
}

function buildClaimItem(item, evidenceById) {
  const li = document.createElement("li");
  if (typeof item === "string") {
    li.textContent = item;
    return li;
  }

  const text = item.text || item.quote || item.description || item.title || "";
  li.append(document.createTextNode(text));

  if (item.url) {
    const href = safeUrl(item.url);
    if (href) {
      li.append(buildCitationLink(href, item.source || "Source", item.text || ""));
    }
  }

  const citationIds = Array.isArray(item.evidence_ids) ? [...new Set(item.evidence_ids)] : [];
  const seenHrefs = new Set();
  citationIds.forEach((id) => {
    const ev = evidenceById.get(id);
    if (!ev) return;
    const href = safeUrl(ev.url);
    if (!href || seenHrefs.has(href)) return;
    seenHrefs.add(href);
    const pillLabel = ev.source || "Source";
    const tooltip = `${ev.source}${ev.observed_at ? ` (${ev.observed_at})` : ""}: ${ev.text || ""}`;
    li.append(buildCitationLink(href, pillLabel, tooltip));
  });

  return li;
}

function buildCitationLink(href, sourceLabel, tooltip) {
  const link = document.createElement("a");
  link.className = "citation-pill";
  link.href = href;
  link.target = "_blank";
  link.rel = "noreferrer";
  link.textContent = `${sourceLabel} ↗`;
  if (tooltip) link.title = tooltip;
  return link;
}

function renderEvidenceDrawer(drawer, candidate, data, evidenceById) {
  const evidence = candidate.evidence || [];
  const passages = candidate.retrieved_passages || [];
  const reviewsFromEvidence = evidence.filter((item) => item.type === "review");
  const seenReviewIds = new Set();
  const reviewEntries = [];

  passages.forEach((passage) => {
    const ev = evidenceById.get(passage.evidence_id);
    seenReviewIds.add(passage.evidence_id);
    reviewEntries.push({
      id: passage.evidence_id,
      type: passage.type || ev?.type || "evidence",
      source: passage.source || ev?.source || "Source",
      text: passage.text,
      score: passage.score,
      url: ev?.url,
      observed_at: ev?.observed_at,
      indirect: Boolean(ev?.indirect),
      data_mode: ev?.data_mode,
    });
  });

  reviewsFromEvidence.forEach((ev) => {
    if (seenReviewIds.has(ev.id)) return;
    reviewEntries.push({
      id: ev.id,
      type: "review",
      source: ev.source,
      text: ev.text,
      score: null,
      url: ev.url,
      observed_at: ev.observed_at,
      indirect: Boolean(ev.indirect),
      data_mode: ev.data_mode,
    });
  });

  const parts = evidence.filter((item) => item.type === "aftermarket_product");
  const recalls = evidence.filter((item) => item.type === "recall");
  const graphPaths = Array.isArray(candidate.graph_paths) ? candidate.graph_paths : [];
  const navPlan = data.navigation_plan && typeof data.navigation_plan === "object"
    ? Object.entries(data.navigation_plan)
    : [];

  const totalCount = reviewEntries.length + parts.length + recalls.length + graphPaths.length + navPlan.length;
  drawer.querySelector(".evidence-drawer-summary").textContent =
    `Supporting evidence, aftermarket parts & GraphRAG paths (${totalCount})`;

  const tabsContainer = drawer.querySelector(".drawer-tabs");
  const panelsContainer = drawer.querySelector(".drawer-panels");

  const tabs = [
    {
      id: "reviews",
      title: `Reviews & Passages (${reviewEntries.length})`,
      count: reviewEntries.length,
      render: () => renderReviewsPanel(reviewEntries),
    },
    {
      id: "parts",
      title: `Aftermarket Parts (${parts.length})`,
      count: parts.length,
      render: () => renderPartsPanel(parts),
    },
    {
      id: "recalls",
      title: `NHTSA Recalls (${recalls.length})`,
      count: recalls.length,
      render: () => renderRecallsPanel(recalls),
    },
    {
      id: "paths",
      title: `GraphRAG Paths (${graphPaths.length})`,
      count: graphPaths.length,
      render: () => renderGraphPathsPanel(graphPaths, data.configuration, candidate.retrieval_mode, evidenceById),
    },
    {
      id: "retailers",
      title: `Cross-Retailer Links (${navPlan.length})`,
      count: navPlan.length,
      render: () => renderNavigationPlanPanel(navPlan),
    },
  ];

  let activeIndex = tabs.findIndex((tab) => tab.count > 0);
  if (activeIndex < 0) activeIndex = 0;

  const panelNodes = [];
  const tabButtons = [];

  tabs.forEach((tab, idx) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "drawer-tab";
    button.setAttribute("role", "tab");
    button.setAttribute("aria-selected", idx === activeIndex ? "true" : "false");
    button.textContent = tab.title;

    const panel = document.createElement("div");
    panel.className = "drawer-panel";
    panel.setAttribute("role", "tabpanel");
    panel.hidden = idx !== activeIndex;
    panel.append(tab.render());

    button.addEventListener("click", () => {
      tabButtons.forEach((btn, bIdx) => {
        btn.setAttribute("aria-selected", bIdx === idx ? "true" : "false");
      });
      panelNodes.forEach((p, pIdx) => {
        p.hidden = pIdx !== idx;
      });
    });

    tabButtons.push(button);
    panelNodes.push(panel);
    tabsContainer.append(button);
    panelsContainer.append(panel);
  });
}

function renderReviewsPanel(entries) {
  if (!entries.length) {
    return emptyNote("No owner reviews or passages retrieved for this variant.");
  }
  const list = document.createElement("ul");
  list.className = "drawer-list";

  entries.forEach((entry) => {
    const li = document.createElement("li");
    li.className = "drawer-item";

    const meta = document.createElement("div");
    meta.className = "drawer-item-meta";
    meta.append(miniPill(entry.source));
    meta.append(miniPill(label(entry.type)));
    if (entry.indirect) {
      meta.append(miniPill("Indirect sibling-year", "mini-pill--warn"));
    } else if (entry.type === "review") {
      meta.append(miniPill("Exact year", "mini-pill--verified"));
    }
    if (entry.score != null) {
      meta.append(miniPill(`Score ${Number(entry.score).toFixed(3)}`));
    }
    if (entry.observed_at) {
      meta.append(document.createTextNode(entry.observed_at));
    }
    const href = safeUrl(entry.url);
    if (href) {
      meta.append(buildCitationLink(href, "Open", entry.source));
    }

    const body = document.createElement("div");
    body.textContent = entry.text;

    li.append(meta, body);
    list.append(li);
  });
  return list;
}

function renderPartsPanel(parts) {
  if (!parts.length) {
    return emptyNote("No aftermarket parts linked in the current catalog.");
  }
  const list = document.createElement("ul");
  list.className = "drawer-list";

  parts.forEach((part) => {
    const li = document.createElement("li");
    li.className = "drawer-item";

    const meta = document.createElement("div");
    meta.className = "drawer-item-meta";
    meta.append(miniPill(part.source));

    const verified = Boolean(part.facts?.verified_fitment);
    meta.append(
      miniPill(
        verified ? "Verified fitment" : "Fitment unverified",
        verified ? "mini-pill--verified" : "mini-pill--warn"
      )
    );

    const rawPrice = part.facts?.price;
    if (rawPrice !== undefined && rawPrice !== null && !Number.isNaN(Number(rawPrice))) {
      meta.append(miniPill(moneyExact.format(Number(rawPrice))));
    }
    meta.append(miniPill(dataModeLabel(part.data_mode)));

    const href = safeUrl(part.url);
    if (href) {
      meta.append(buildCitationLink(href, "Open offer", part.source));
    }

    const body = document.createElement("div");
    body.textContent = part.text;

    li.append(meta, body);
    list.append(li);
  });
  return list;
}

function renderRecallsPanel(recalls) {
  if (!recalls.length) {
    return emptyNote("No recalls recorded in the current NHTSA snapshot/response.");
  }
  const list = document.createElement("ul");
  list.className = "drawer-list";

  recalls.forEach((recall) => {
    const li = document.createElement("li");
    li.className = "drawer-item";

    const meta = document.createElement("div");
    meta.className = "drawer-item-meta";
    if (recall.facts?.campaign_number) {
      meta.append(miniPill(`Campaign ${recall.facts.campaign_number}`, "mini-pill--alert"));
    }
    if (recall.facts?.component) {
      meta.append(miniPill(recall.facts.component));
    }
    if (recall.observed_at) {
      meta.append(document.createTextNode(recall.observed_at));
    }
    const href = safeUrl(recall.url);
    if (href) {
      meta.append(buildCitationLink(href, "NHTSA", "Open NHTSA record"));
    }

    const body = document.createElement("div");
    body.textContent = recall.text;

    li.append(meta, body);
    list.append(li);
  });
  return list;
}

function renderGraphPathsPanel(paths, configName, retrievalMode, evidenceById) {
  if (!paths.length) {
    return emptyNote(
      `GraphRAG paths are not used in Configuration ${configName || ""} (${retrievalMode || "flat retrieval"}).`
    );
  }
  const list = document.createElement("ul");
  list.className = "drawer-list";

  paths.forEach((path) => {
    const li = document.createElement("li");
    li.className = "drawer-item";

    const meta = document.createElement("div");
    meta.className = "drawer-item-meta";
    meta.append(miniPill(path.label || label(path.kind || "path")));
    if (path.component) {
      meta.append(miniPill(`Component: ${label(path.component)}`));
    }
    if (path.polarity) {
      const polarityClass = path.polarity === "positive"
        ? "mini-pill--verified"
        : path.polarity === "negative"
          ? "mini-pill--alert"
          : "mini-pill--warn";
      meta.append(miniPill(label(path.polarity), polarityClass));
    }
    if (path.indirect) {
      meta.append(miniPill("Indirect lineage", "mini-pill--warn"));
    }
    if (path.backend) {
      meta.append(miniPill(formatBackendLabel(path.backend)));
    }

    const relations = Array.isArray(path.relations) ? path.relations : [];
    const chain = document.createElement("div");
    chain.className = "relation-chain";
    relations.forEach((rel, idx) => {
      if (idx > 0) {
        const arrow = document.createElement("span");
        arrow.className = "rel-arrow";
        arrow.textContent = "→";
        chain.append(arrow);
      }
      const pill = document.createElement("span");
      pill.className = "rel-pill";
      pill.textContent = rel;
      chain.append(pill);
    });

    const body = document.createElement("div");
    const pathText = path.text || "";
    body.textContent = path.span && !pathText.includes(path.span)
      ? `${pathText} — “${path.span}”`
      : pathText;

    const citations = Array.isArray(path.evidence_ids) ? [...new Set(path.evidence_ids)] : [];
    citations.forEach((id) => {
      const ev = evidenceById.get(id);
      if (!ev) return;
      const href = safeUrl(ev.url);
      if (href) {
        body.append(buildCitationLink(href, ev.source || "Source", ev.text || ""));
      }
    });

    li.append(meta);
    if (relations.length) li.append(chain);
    li.append(body);
    list.append(li);
  });
  return list;
}

function renderNavigationPlanPanel(navPlanEntries) {
  if (!navPlanEntries.length) {
    return emptyNote("Cross-retailer search links are unavailable for this response.");
  }
  const grid = document.createElement("div");
  grid.className = "retailer-grid";

  navPlanEntries.forEach(([key, plan]) => {
    if (!plan || typeof plan !== "object") return;
    const card = document.createElement("div");
    card.className = "retailer-card";

    const head = document.createElement("div");
    head.className = "retailer-card-header";
    const title = document.createElement("strong");
    title.textContent = RETAILER_NAMES[key] || label(key);
    head.append(title);

    const href = safeUrl(plan.url);
    if (href) {
      const link = document.createElement("a");
      link.className = "retailer-link";
      link.href = href;
      link.target = "_blank";
      link.rel = "noreferrer";
      link.textContent = "Open search ↗";
      head.append(link);
    }

    card.append(head);

    if (plan.target) {
      const target = document.createElement("div");
      target.className = "drawer-item-meta";
      target.textContent = plan.target;
      card.append(target);
    }

    if (plan.query) {
      const q = document.createElement("div");
      q.textContent = `Query: "${plan.query}"`;
      card.append(q);
    }

    const filters = plan.filters && typeof plan.filters === "object"
      ? Object.entries(plan.filters).filter(([, val]) => val !== null && val !== "")
      : [];
    if (filters.length || plan.selector_version) {
      const meta = document.createElement("div");
      meta.className = "drawer-item-meta";
      filters.forEach(([fKey, fVal]) => {
        const display = Array.isArray(fVal)
          ? fVal.join(", ")
          : (typeof fVal === "object" ? JSON.stringify(fVal) : String(fVal));
        meta.append(miniPill(`${label(fKey)}: ${display}`));
      });
      if (plan.selector_version) {
        meta.append(miniPill(plan.selector_version));
      }
      card.append(meta);
    }

    grid.append(card);
  });
  return grid;
}

function miniPill(text, extraClass = "") {
  const pill = document.createElement("span");
  pill.className = extraClass ? `mini-pill ${extraClass}` : "mini-pill";
  pill.textContent = text;
  return pill;
}

function emptyNote(text) {
  const p = document.createElement("p");
  p.className = "empty-drawer-note";
  p.textContent = text;
  return p;
}

async function loadModelStatus() {
  try {
    const response = await fetch("/api/model-status");
    const data = await response.json();
    if (!response.ok) throw new Error();
    const isAvailable = Boolean(data.available);
    modelStatus.textContent = isAvailable ? "Qwen connected" : "Rule-based fallback";
    modelStatus.classList.toggle("is-ready", isAvailable);
    modelStatus.classList.toggle("is-fallback", !isAvailable);
    if (data.model) {
      modelStatus.title = isAvailable
        ? `Model: ${data.model}`
        : `${data.model} unavailable (${data.error || "using rule-based fallback"})`;
    }
  } catch {
    modelStatus.textContent = "Rule-based fallback";
    modelStatus.classList.add("is-fallback");
  }
}

async function loadDataStatus() {
  try {
    const response = await fetch("/api/data-status");
    const data = await response.json();
    if (!response.ok || !data.ready) throw new Error(data.error || "Data unavailable");
    const counts = data.counts || {};
    const liveTotal = (data.live_records?.listings || 0) + (data.live_records?.products || 0);
    const liveSuffix = liveTotal > 0 ? ` · ${liveTotal} live` : "";
    dataNotice.textContent = `${counts.vehicles} variants · ${number.format(counts.listings)} listings · ${number.format(counts.reviews)} reviews · ${counts.recalls} recalls${liveSuffix}`;

    const backend = (data.graph_backend || "memory").toLowerCase();
    graphStatus.textContent = formatBackendLabel(backend);
    graphStatus.classList.toggle("is-ready", backend === "neo4j");
    if (data.retrieval_mode) {
      graphStatus.title = `Retrieval mode: ${data.retrieval_mode}`;
    }
  } catch (error) {
    dataNotice.textContent = error.message;
    dataNotice.classList.add("is-error");
  }
}

function formatBackendLabel(backend) {
  if (backend === "neo4j") return "Neo4j Graph";
  if (backend === "memory_fallback") return "Memory Fallback";
  return "Memory Graph";
}

function dataModeLabel(value) {
  if (value === "live") return "Live";
  if (value === "live_capture") return "Live capture";
  if (value === "frozen_snapshot") return "Frozen snapshot";
  return value || "Snapshot";
}

function formatBodyStyle(value) {
  const lower = String(value).toLowerCase();
  if (lower === "suv") return "SUV";
  return label(lower);
}

function vehicleName(vehicle) {
  const values = [vehicle.year, vehicle.make, vehicle.model].filter(Boolean);
  return values.join(" ") || vehicle.name || "Vehicle";
}

function formatScore(value) {
  if (value === undefined || value === null) return "";
  const score = Number(value);
  return score <= 1 ? `${Math.round(score * 100)}% match` : `${score.toFixed(1)} score`;
}

function label(key) {
  return String(key)
    .replaceAll("_", " ")
    .replace(/\b\w/g, (character) => character.toUpperCase());
}

function safeUrl(raw) {
  if (!raw || typeof raw !== "string") return null;
  try {
    const parsed = new URL(raw, window.location.origin);
    if (parsed.protocol === "http:" || parsed.protocol === "https:") {
      return parsed.href;
    }
  } catch {
    return null;
  }
  return null;
}
