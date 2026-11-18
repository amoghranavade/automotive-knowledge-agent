from __future__ import annotations

import os
import re
from typing import Any

try:
    import streamlit as st
except ImportError:  # pragma: no cover - optional UI extra
    st = None  # type: ignore[assignment]

from clickless import config  # noqa: F401
from clickless.data import CatalogError, load_catalog
from clickless.llm import DEFAULT_QWEN_MODEL, QwenClient, QwenError
from clickless.service import recommend
from clickless.verifier import VerificationError

CONFIGURATION_LABELS = {
    "A": "A · Classical BM25 + VADER",
    "B": "B · Flat MiniLM Vector RAG",
    "C": "C · Full Ontology GraphRAG",
    "D": "D · GraphRAG (Raw Reviews)",
    "E": "E · GraphRAG (No Sentiment Rank)",
}

STARTER_PROMPTS = [
    (
        "Snow-ready SUV",
        "Find a reliable SUV under $18,000 with good safety ratings for snow driving and affordable brake maintenance.",
    ),
    (
        "Commuter Sedan",
        "Find a fuel-efficient sedan under $15,000 with under 90k miles and strong reliability.",
    ),
    (
        "Towing Truck",
        "Find a 4WD truck under $25,000 with strong reliability for hauling and winter driving.",
    ),
    (
        "Clarification Demo",
        "Find a reliable AWD SUV with good cargo space and low brake maintenance.",
    ),
]


def merge_follow_up(base_query: str, follow_up: str) -> str:
    """Append a clarification answer to an existing vehicle request cleanly."""
    clean_follow_up = follow_up.strip()
    if not clean_follow_up:
        return base_query.strip()
    clean_base = re.sub(r"[.\s]+$", "", base_query.strip())
    if not clean_base:
        return clean_follow_up
    return f"{clean_base}, {clean_follow_up}"


def is_follow_up_reply(text: str, pending_questions: list[str]) -> bool:
    """Heuristically detect whether user input answers an active clarification."""
    if not pending_questions:
        return False
    stripped = text.strip()
    lower = stripped.lower()
    if re.match(r"^(find|show|search|recommend|looking for|i need|i want)\b", lower):
        return False
    return len(stripped.split()) <= 6


def clarification_quick_options(questions: list[str]) -> list[tuple[str, str]]:
    """Build quick-reply (label, query_addition) options for <=2 clarification questions."""
    options: list[tuple[str, str]] = []
    joined = " ".join(questions).lower()
    if any(token in joined for token in ("budget", "price", "cost", "$")):
        options.extend(
            [
                ("$10,000", "under $10,000"),
                ("$15,000", "under $15,000"),
                ("$18,000", "under $18,000"),
                ("$25,000", "under $25,000"),
            ]
        )
    if any(token in joined for token in ("body", "suv", "sedan", "truck")):
        options.extend(
            [
                ("SUV", "SUV"),
                ("Sedan", "sedan"),
                ("Truck", "truck"),
            ]
        )
    if "mileage" in joined or "miles" in joined:
        options.extend(
            [
                ("Under 80k mi", "under 80,000 miles"),
                ("Under 100k mi", "under 100,000 miles"),
            ]
        )
    if any(token in joined for token in ("drivetrain", "awd", "4wd", "fwd")):
        options.extend([("AWD", "AWD"), ("4WD", "4WD"), ("FWD", "FWD")])
    return options[:6]


def vehicle_title(candidate: dict[str, Any]) -> str:
    """Format a readable vehicle heading from candidate fields."""
    parts = [candidate.get("year"), candidate.get("make"), candidate.get("model")]
    text = " ".join(str(part) for part in parts if part)
    return text or str(candidate.get("name") or "Vehicle")


def format_currency(value: Any) -> str:
    if value is None:
        return "N/A"
    return f"${float(value):,.0f}"


def format_mileage(value: Any) -> str:
    if value is None:
        return "N/A"
    return f"{int(float(value)):,} mi"


def _tier_label(candidate: dict[str, Any], match_mode: str) -> str:
    if match_mode == "closest_alternatives" or candidate.get("constraint_violations"):
        return "Closest alternative"
    if candidate.get("tier") == "complete":
        return "Complete evidence"
    return "Needs additional evidence"


def _parsed_summary_text(result: dict[str, Any]) -> str:
    parsed = result.get("parsed") or {}
    parts: list[str] = []
    if parsed.get("max_price") is not None:
        parts.append(f"Max price: {format_currency(parsed['max_price'])}")
    if parsed.get("max_mileage") is not None:
        parts.append(f"Max mileage: {format_mileage(parsed['max_mileage'])}")
    if parsed.get("min_year") is not None:
        parts.append(f"Min year: {parsed['min_year']}+")
    if parsed.get("make"):
        parts.append(f"Make: {parsed['make']}")
    if parsed.get("body_style"):
        parts.append(f"Body: {str(parsed['body_style']).upper()}")
    if parsed.get("drivetrain"):
        parts.append(f"Drivetrain: {parsed['drivetrain']}")

    feasible = result.get("feasible_count")
    comparable = result.get("comparable_cohort_count")
    if feasible is not None:
        cohort_text = f"Comparable cohort: {comparable or 0}/{feasible} feasible"
        if result.get("cohort_retention") is not None:
            cohort_text += f" ({round(float(result['cohort_retention']) * 100)}%)"
        parts.append(cohort_text)
    parts.append(f"Config {result.get('configuration', 'C')}")
    return " · ".join(parts)


def _render_claims(
    claims: list[Any],
    evidence_by_id: dict[str, dict[str, Any]],
) -> None:
    for item in claims:
        if isinstance(item, str):
            st.markdown(f"- {item}")
            continue
        text = str(item.get("text") or "")
        links: list[str] = []
        seen_urls: set[str] = set()
        for ev_id in dict.fromkeys(item.get("evidence_ids") or []):
            ev = evidence_by_id.get(str(ev_id))
            if ev and ev.get("url") and ev["url"] not in seen_urls:
                seen_urls.add(ev["url"])
                links.append(f"[{ev.get('source', 'Source')}]({ev['url']})")
        suffix = f" ({', '.join(links)})" if links else ""
        st.markdown(f"- {text}{suffix}")


def _render_candidate_card(
    candidate: dict[str, Any],
    rank: int,
    result: dict[str, Any],
) -> None:
    evidence = candidate.get("evidence") or []
    evidence_by_id = {str(item["id"]): item for item in evidence if "id" in item}
    match_mode = str(result.get("match_mode") or "exact")

    with st.container(border=True):
        header_left, header_right = st.columns([3, 2])
        with header_left:
            st.subheader(f"{rank}. {vehicle_title(candidate)}")
            tier = _tier_label(candidate, match_mode)
            stars = candidate.get("safety_rating")
            star_text = f"NHTSA {stars}/5" if stars is not None else "NHTSA unrated"
            recalls = candidate.get("recall_count", 0)
            mode_label = (
                "Live"
                if candidate.get("listing_data_mode") == "live"
                else "Frozen snapshot"
            )
            st.caption(
                f"{tier} · {str(candidate.get('body_style', '')).upper()} · "
                f"{candidate.get('drivetrain', '')} · {star_text} · "
                f"{recalls} recall(s) · {candidate.get('listing_source', 'Catalog')} ({mode_label})"
            )

        with header_right:
            score_val = candidate.get("score")
            score_str = f"{round(float(score_val) * 100)}% match" if score_val is not None else "N/A"
            st.metric(
                label=score_str,
                value=format_currency(candidate.get("price")),
                delta=format_mileage(candidate.get("mileage")),
                delta_color="off",
            )
            listing_ev = next((item for item in evidence if item.get("type") == "listing"), None)
            if listing_ev and listing_ev.get("url"):
                st.link_button("View listing ↗", listing_ev["url"])

        # Aspect breakdown
        components = candidate.get("score_components") or {}
        aspect_summary = candidate.get("aspect_summary") or {}
        aspect_cols = st.columns(3)
        aspect_keys = [
            ("safety", "Safety"),
            ("reliability", "Reliability"),
            ("maintenance", "Maintenance"),
            ("fuel_economy", "Fuel economy"),
            ("cargo", "Cargo"),
            ("snow", "Snow"),
        ]
        for idx, (key, label_text) in enumerate(aspect_keys):
            val = components.get(key)
            col = aspect_cols[idx % 3]
            with col:
                if val is None:
                    st.caption(f"{label_text}: No data")
                    st.progress(0.0)
                else:
                    clamped = max(0.0, min(1.0, float(val)))
                    summary_item = aspect_summary.get(key)
                    if summary_item and summary_item.get("total"):
                        badge = f"{summary_item['positive']}+ / {summary_item['negative']}-"
                    else:
                        badge = f"{round(clamped * 100)}%"
                    st.caption(f"{label_text}: {badge}")
                    st.progress(clamped)

        pros_col, cautions_col = st.columns(2)
        with pros_col:
            st.markdown("**Why it fits**")
            pros = candidate.get("pros") or []
            if pros:
                _render_claims(pros, evidence_by_id)
            else:
                st.caption("Selected based on listing price, mileage, and vehicle attributes.")

        with cautions_col:
            st.markdown("**Trade-offs & Recall Cautions**")
            for violation in candidate.get("constraint_violations") or []:
                st.markdown(f"- **Unmet constraint:** {violation}")
            cautions = candidate.get("cautions") or []
            if cautions:
                _render_claims(cautions, evidence_by_id)
            missing = candidate.get("missing_evidence") or []
            if missing:
                st.caption(f"Missing review evidence: {', '.join(missing)}")

        passages = candidate.get("retrieved_passages") or []
        passage_ids = {str(item.get("evidence_id")) for item in passages if item.get("evidence_id")}
        extra_reviews = [
            item
            for item in evidence
            if item.get("type") == "review" and str(item.get("id")) not in passage_ids
        ]
        review_count = len(passages) + len(extra_reviews)
        parts = [item for item in evidence if item.get("type") == "aftermarket_product"]
        recalls_list = [item for item in evidence if item.get("type") == "recall"]
        graph_paths = candidate.get("graph_paths") or []
        nav_plan = result.get("navigation_plan") or {}
        total_items = review_count + len(parts) + len(recalls_list) + len(graph_paths) + len(nav_plan)

        with st.expander(f"Supporting evidence, aftermarket parts & GraphRAG paths ({total_items})"):
            tab_reviews, tab_parts, tab_recalls, tab_paths, tab_links = st.tabs(
                [
                    f"Reviews & Passages ({review_count})",
                    f"Aftermarket Parts ({len(parts)})",
                    f"NHTSA Recalls ({len(recalls_list)})",
                    f"GraphRAG Paths ({len(graph_paths)})",
                    f"Cross-Retailer Links ({len(nav_plan)})",
                ]
            )
            with tab_reviews:
                if not review_count:
                    st.caption("No reviews or passages retrieved.")
                for item in passages:
                    ev = evidence_by_id.get(str(item.get("evidence_id") or ""))
                    indirect_tag = " *(Indirect sibling-year)*" if ev and ev.get("indirect") else ""
                    st.markdown(
                        f"- **[{item.get('source', 'Source')} | {item.get('type', 'evidence')} | "
                        f"score {float(item.get('score', 0)):.3f}]**{indirect_tag} {item.get('text', '')}"
                    )
                for ev in extra_reviews:
                    indirect_tag = " *(Indirect sibling-year)*" if ev.get("indirect") else " *(Exact year)*"
                    st.markdown(
                        f"- **[{ev.get('source', 'Source')} | review]**{indirect_tag} {ev.get('text', '')}"
                    )
            with tab_parts:
                if not parts:
                    st.caption("No aftermarket parts linked.")
                for part in parts:
                    facts = part.get("facts") or {}
                    fit = "Verified fitment" if facts.get("verified_fitment") else "Fitment unverified"
                    price_str = f" · ${float(facts['price']):,.2f}" if facts.get("price") is not None else ""
                    part_link = f" ([Open]({part['url']}))" if part.get("url") else ""
                    st.markdown(
                        f"- **{part.get('source', 'Retailer')}** ({fit}{price_str}): "
                        f"{part.get('text', '')}{part_link}"
                    )
            with tab_recalls:
                if not recalls_list:
                    st.caption("No recalls recorded in the current NHTSA snapshot/response.")
                for recall in recalls_list:
                    facts = recall.get("facts") or {}
                    st.markdown(
                        f"- **{facts.get('campaign_number', 'NHTSA')}** "
                        f"({facts.get('component', 'Component')}): {recall.get('text', '')}"
                    )
            with tab_paths:
                if not graph_paths:
                    st.caption(
                        f"Graph paths are not used in Configuration {result.get('configuration', '')}."
                    )
                for path in graph_paths:
                    chain = " → ".join(path.get("relations") or [])
                    indirect = " *(Indirect sibling-year lineage)*" if path.get("indirect") else ""
                    st.markdown(
                        f"- **{path.get('label', 'Path')}**{indirect} `{chain}` — {path.get('text', '')}"
                    )
            with tab_links:
                if not nav_plan:
                    st.caption("Cross-retailer navigation plan unavailable.")
                for site_key, plan in nav_plan.items():
                    if isinstance(plan, dict) and plan.get("url"):
                        st.markdown(
                            f"- **{site_key.title()}**: `{plan.get('query', '')}` — "
                            f"[{plan.get('target', 'Open search')}]({plan['url']})"
                        )


def _execute_query(
    query: str,
    configuration: str,
    use_model: bool,
    live_nhtsa: bool,
    candidate_limit: int,
) -> None:
    st.session_state["last_query"] = query
    st.session_state["messages"].append({"role": "user", "content": query, "result": None})
    try:
        result = recommend(
            query,
            live_nhtsa=live_nhtsa,
            use_model=use_model,
            configuration=configuration,
            candidate_limit=candidate_limit,
        )
        st.session_state["pending_questions"] = list(result.get("clarification_questions") or [])[:2]
        st.session_state["messages"].append(
            {
                "role": "assistant",
                "content": result.get("message") or "",
                "result": result,
            }
        )
    except (CatalogError, VerificationError, ValueError) as error:
        st.session_state["pending_questions"] = []
        st.session_state["messages"].append(
            {
                "role": "assistant",
                "content": f"Request could not complete: {error}",
                "result": None,
            }
        )


def run_app() -> None:
    """Render the interactive ClickLess Automotive Streamlit chat interface."""
    if st is None:
        raise RuntimeError(
            "Streamlit UI dependencies are not installed. Install with: pip install -e '.[ui]'"
        )

    st.set_page_config(page_title="ClickLess Automotive", layout="wide")
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("last_query", "")
    st.session_state.setdefault("pending_questions", [])

    with st.sidebar:
        st.header("Configuration")
        selected_config = st.selectbox(
            "Pipeline configuration",
            options=list(CONFIGURATION_LABELS),
            index=2,
            format_func=lambda key: CONFIGURATION_LABELS[key],
        )
        use_model = st.checkbox("Use Qwen model", value=True)
        live_nhtsa = st.checkbox("Check live NHTSA recalls", value=False)
        candidate_limit = st.slider("Max candidates", min_value=1, max_value=10, value=3)

        st.divider()
        st.subheader("System status")
        try:
            catalog = load_catalog()
            st.caption(
                f"Catalog: {len(catalog['vehicles'])} variants · "
                f"{len(catalog['listings']):,} listings · "
                f"{len(catalog['reviews']):,} reviews · "
                f"{len(catalog['recalls'])} recalls"
            )
        except CatalogError as error:
            st.error(str(error))

        try:
            qwen_model = QwenClient.from_env().model
            st.caption(f"Qwen configured: `{qwen_model}`")
        except QwenError:
            st.caption(f"Qwen fallback: rule-based ({DEFAULT_QWEN_MODEL} unconfigured)")

        backend = os.getenv("CLICKLESS_GRAPH_BACKEND", "memory")
        st.caption(f"Graph backend: `{backend}`")

        if st.button("Reset conversation", use_container_width=True):
            st.session_state["messages"] = []
            st.session_state["last_query"] = ""
            st.session_state["pending_questions"] = []
            st.rerun()

    st.title("ClickLess Automotive")
    st.caption(
        "Conversational used-car assistant backed by GraphRAG citations, NHTSA safety records, and verified evidence."
    )

    if not st.session_state["messages"]:
        st.markdown("**Try an example request:**")
        cols = st.columns(len(STARTER_PROMPTS))
        for col, (label_text, prompt_text) in zip(cols, STARTER_PROMPTS):
            with col:
                if st.button(label_text, use_container_width=True):
                    _execute_query(
                        prompt_text,
                        selected_config,
                        use_model,
                        live_nhtsa,
                        candidate_limit,
                    )
                    st.rerun()

    messages = st.session_state["messages"]
    for idx, msg in enumerate(messages):
        with st.chat_message(msg["role"]):
            if msg["role"] == "user":
                st.markdown(msg["content"])
                continue

            result = msg.get("result")
            if not result:
                st.error(msg["content"])
                continue

            questions = (result.get("clarification_questions") or [])[:2]
            is_latest = idx == len(messages) - 1
            if questions:
                with st.container(border=True):
                    st.markdown("**Clarify your request**")
                    for q in questions:
                        st.markdown(f"- {q}")
                    if is_latest:
                        quick_opts = clarification_quick_options(questions)
                        if quick_opts:
                            chip_cols = st.columns(len(quick_opts))
                            for opt_idx, (chip_label, addition) in enumerate(quick_opts):
                                with chip_cols[opt_idx]:
                                    if st.button(
                                        chip_label,
                                        key=f"clarify_{idx}_{opt_idx}",
                                        use_container_width=True,
                                    ):
                                        merged = merge_follow_up(
                                            st.session_state["last_query"],
                                            addition,
                                        )
                                        _execute_query(
                                            merged,
                                            selected_config,
                                            use_model,
                                            live_nhtsa,
                                            candidate_limit,
                                        )
                                        st.rerun()

            st.caption(_parsed_summary_text(result))
            if result.get("model_notice"):
                st.caption(result["model_notice"])

            candidates = result.get("candidates") or []
            if not candidates:
                st.info(result.get("message") or "No vehicles match the current request.")
            else:
                if result.get("match_mode") == "closest_alternatives":
                    st.warning(
                        result.get("message")
                        or "No exact matches were found. Showing closest alternatives."
                    )
                for rank_idx, candidate in enumerate(candidates, start=1):
                    _render_candidate_card(candidate, rank_idx, result)

    user_prompt = st.chat_input("Describe the vehicle you need, or answer a clarification question…")
    if user_prompt:
        pending = st.session_state.get("pending_questions") or []
        effective_query = (
            merge_follow_up(st.session_state["last_query"], user_prompt)
            if is_follow_up_reply(user_prompt, pending)
            else user_prompt.strip()
        )
        _execute_query(
            effective_query,
            selected_config,
            use_model,
            live_nhtsa,
            candidate_limit,
        )
        st.rerun()


if __name__ == "__main__":
    run_app()
