from __future__ import annotations

import re
import time
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, Tuple

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field


app = FastAPI()

START_TS = time.time()

# In-memory stores keyed by scope then context_id
CONTEXTS: Dict[str, Dict[str, Dict[str, Any]]] = {
    "category": {},
    "merchant": {},
    "customer": {},
    "trigger": {},
}

CONVERSATIONS: Dict[str, Dict[str, Any]] = {}
SENT_SUPPRESSIONS: set[str] = set()
SUPPRESSED_MERCHANTS: set[str] = set()

AUTO_REPLY_MARKERS = [
    "thank you for contacting",
    "we will respond shortly",
    "away from keyboard",
    "auto-reply",
    "auto reply",
    "out of office",
    "will get back to you",
]

HOSTILE_MARKERS = [
    "stop messaging",
    "unsubscribe",
    "spam",
    "useless",
    "don't message",
    "do not message",
]

COMMIT_MARKERS = [
    "yes",
    "ok",
    "okay",
    "go ahead",
    "proceed",
    "send",
    "do it",
    "sounds good",
]

OUT_OF_SCOPE_MARKERS = [
    "gst",
    "tax",
    "itr",
    "accounting",
    "loan",
    "legal",
]


class ContextPush(BaseModel):
    scope: str
    context_id: str
    version: int = Field(ge=0)
    payload: Dict[str, Any]
    delivered_at: str


class TickRequest(BaseModel):
    now: str
    available_triggers: list[str]


class ReplyRequest(BaseModel):
    conversation_id: str
    merchant_id: str
    customer_id: str | None = None
    from_role: str
    message: str
    received_at: str
    turn_number: int


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _parse_iso(ts: str) -> datetime | None:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError:
        return None


def _get_context(scope: str, context_id: str) -> Dict[str, Any] | None:
    entry = CONTEXTS.get(scope, {}).get(context_id)
    if not entry:
        return None
    return entry.get("payload")


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _contains_any(text: str, markers: Iterable[str]) -> bool:
    return any(marker in text for marker in markers)


def _is_auto_reply(message: str) -> bool:
    text = _normalize_text(message)
    return _contains_any(text, AUTO_REPLY_MARKERS)


def _is_hostile(message: str) -> bool:
    text = _normalize_text(message)
    return _contains_any(text, HOSTILE_MARKERS)


def _is_commitment(message: str) -> bool:
    text = _normalize_text(message)
    if "not" in text or "don't" in text or "do not" in text:
        return False
    return _contains_any(text, COMMIT_MARKERS)


def _is_out_of_scope(message: str) -> bool:
    text = _normalize_text(message)
    return _contains_any(text, OUT_OF_SCOPE_MARKERS)


def _merchant_salutation(merchant: Dict[str, Any]) -> str:
    identity = merchant.get("identity", {})
    category_slug = merchant.get("category_slug", "")
    first_name = identity.get("owner_first_name")
    if category_slug == "dentists" and first_name:
        return f"Dr. {first_name}"
    if first_name:
        return first_name
    return identity.get("name", "there")


def _category_style(category_slug: str) -> Dict[str, str]:
    if category_slug == "dentists":
        return {"prefix": "Clinical update", "cta": "Want a 2-min summary?"}
    if category_slug == "salons":
        return {"prefix": "Quick idea", "cta": "Want a ready-to-post draft?"}
    if category_slug == "restaurants":
        return {"prefix": "Operator note", "cta": "Want a draft for today?"}
    if category_slug == "gyms":
        return {"prefix": "Coach note", "cta": "Want a simple plan to act on?"}
    if category_slug == "pharmacies":
        return {"prefix": "Pharmacy update", "cta": "Want a precise checklist?"}
    return {"prefix": "Quick update", "cta": "Want details?"}


def _category_hint(category_slug: str) -> str:
    if category_slug == "dentists":
        return "patient recall and trust"
    if category_slug == "salons":
        return "slots and stylist demand"
    if category_slug == "restaurants":
        return "dine-in and delivery flow"
    if category_slug == "gyms":
        return "member retention"
    if category_slug == "pharmacies":
        return "stock and refills"
    return "core demand"


def _category_terms(category_slug: str) -> Dict[str, str]:
    if category_slug == "dentists":
        return {"noun": "clinic", "audience": "patients", "focus": "recall and hygiene"}
    if category_slug == "salons":
        return {"noun": "salon", "audience": "clients", "focus": "slots and stylist demand"}
    if category_slug == "restaurants":
        return {"noun": "restaurant", "audience": "guests", "focus": "delivery and dine-in"}
    if category_slug == "gyms":
        return {"noun": "gym", "audience": "members", "focus": "trials and retention"}
    if category_slug == "pharmacies":
        return {"noun": "pharmacy", "audience": "customers", "focus": "refills and stock"}
    return {"noun": "business", "audience": "customers", "focus": "core demand"}


def _category_voice_word(category: Dict[str, Any]) -> str | None:
    voice = category.get("voice", {})
    vocab = voice.get("vocab_allowed", [])
    if vocab:
        return vocab[0]
    return None


def _category_seasonal_note(category: Dict[str, Any]) -> str | None:
    beats = category.get("seasonal_beats", [])
    if beats:
        note = beats[0].get("note")
        if note:
            return note
    return None


def _category_trend_signal(category: Dict[str, Any]) -> str | None:
    trends = category.get("trend_signals", [])
    if trends:
        query = trends[0].get("query")
        delta = trends[0].get("delta_yoy")
        if query and isinstance(delta, (int, float)):
            return f"{query} +{delta:.0%} YoY"
        if query:
            return query
    return None


def _language_tail(language_pref: str | None) -> str:
    if not language_pref:
        return ""
    pref = language_pref.lower()
    if "hi" in pref:
        return " Aap ka preferred time batayein."
    if "mr" in pref:
        return " Kripya preferred time batayein."
    if "kn" in pref:
        return " Dayavittu preferred time heli."
    if "te" in pref:
        return " Meeku saraina time cheppandi."
    return ""


def _pick_voice_term(category: Dict[str, Any]) -> str | None:
    voice = category.get("voice", {})
    vocab = voice.get("vocab_allowed", [])
    if vocab:
        return vocab[0]
    return None


def _signal_snippet(signals: set[str]) -> str:
    if not signals:
        return ""
    for signal in signals:
        if signal.startswith("stale_posts"):
            days = signal.split(":", 1)[1] if ":" in signal else ""
            return f"Last post {days} ago."
        if signal.startswith("renewal_due_soon"):
            days = signal.split(":", 1)[1] if ":" in signal else ""
            return f"Renewal due in {days}."
        if signal == "ctr_below_peer_median":
            return "CTR below peer median."
        if signal == "unverified_gbp":
            return "GBP unverified."
        if signal == "no_active_offers":
            return "No active offers right now."
        if signal == "perf_dip_severe":
            return "Sharp dip vs recent baseline."
        if signal.startswith("dormant_with_vera"):
            return "No replies in the last 2 weeks."
        if signal == "high_risk_adult_cohort":
            return "High-risk adult cohort in roster."
    return ""


def _metric_label(metric: str, category_slug: str) -> str:
    if metric == "calls":
        if category_slug == "dentists":
            return "patient calls"
        if category_slug == "salons":
            return "booking calls"
        if category_slug == "restaurants":
            return "order calls"
        if category_slug == "gyms":
            return "trial calls"
        if category_slug == "pharmacies":
            return "pharmacy calls"
    if metric == "views":
        return "profile views"
    if metric == "ctr":
        return "CTR"
    return metric


def _active_offer_title(merchant: Dict[str, Any]) -> str | None:
    offers = merchant.get("offers", [])
    for offer in offers:
        if offer.get("status") == "active":
            return offer.get("title")
    return None


def _category_offer_suggestion(category: Dict[str, Any]) -> str | None:
    offers = category.get("offer_catalog", [])
    for offer in offers:
        title = offer.get("title")
        if title:
            return title
    return None


def _find_digest_item(category: Dict[str, Any], item_id: str) -> Dict[str, Any] | None:
    for item in category.get("digest", []):
        if item.get("id") == item_id:
            return item
    return None


def _rank_trigger(
    trigger: Dict[str, Any],
    merchant: Dict[str, Any],
    now: datetime,
) -> Tuple[int, str]:
    urgency = int(trigger.get("urgency", 0))
    scope = trigger.get("scope", "merchant")
    kind = trigger.get("kind", "")
    expires_at = _parse_iso(trigger.get("expires_at", ""))
    if expires_at and now > expires_at:
        return (-1, kind)

    score = urgency * 100
    if scope == "customer":
        score += 20

    if kind in {"active_planning_intent", "recall_due", "perf_dip", "chronic_refill_due"}:
        score += 15

    if expires_at:
        hours_left = (expires_at - now).total_seconds() / 3600
        if hours_left <= 24:
            score += 10

    signals = set(merchant.get("signals", []))
    signal_note = _signal_snippet(signals)
    if "engaged_in_last_48h" in signals or "high_engagement" in signals:
        score += 5

    if kind == "perf_dip" and any("perf_dip" in signal for signal in signals):
        score += 5
    if kind == "renewal_due" and any("renewal_due" in signal for signal in signals):
        score += 5
    if kind == "gbp_unverified" and "unverified_gbp" in signals:
        score += 5
    if kind == "dormant_with_vera" and any("dormant" in signal for signal in signals):
        score += 5

    return (score, kind)


def _compose_message(
    category: Dict[str, Any],
    merchant: Dict[str, Any],
    trigger: Dict[str, Any],
    customer: Dict[str, Any] | None,
) -> Dict[str, Any]:
    scope = trigger.get("scope", "merchant")
    kind = trigger.get("kind", "")
    payload = trigger.get("payload", {})
    salutation = _merchant_salutation(merchant)
    merchant_name = merchant.get("identity", {}).get("name", "your business")
    identity = merchant.get("identity", {})
    locality = identity.get("locality")
    city = identity.get("city")
    location = ", ".join([p for p in [locality, city] if p])
    offer_title = _active_offer_title(merchant)
    category_slug = merchant.get("category_slug", "")
    style = _category_style(category_slug)
    hint = _category_hint(category_slug)
    terms = _category_terms(category_slug)
    send_as = "merchant_on_behalf" if scope == "customer" else "vera"
    suppression_key = trigger.get("suppression_key", trigger.get("id", ""))
    signals = set(merchant.get("signals", []))
    signal_note = _signal_snippet(signals)
    language_pref = None
    if customer:
        language_pref = customer.get("identity", {}).get("language_pref")
    else:
        langs = identity.get("languages") or []
        if isinstance(langs, list) and langs:
            language_pref = "/".join(langs)
    lang_tail = _language_tail(language_pref)

    if kind == "research_digest":
        item_id = payload.get("top_item_id")
        digest_item = _find_digest_item(category, item_id) if item_id else None
        title = digest_item.get("title") if digest_item else "A new research digest item"
        source = digest_item.get("source") if digest_item else ""
        trial_n = digest_item.get("trial_n") if digest_item else None
        patient_segment = digest_item.get("patient_segment") if digest_item else None
        trial_line = f"{trial_n}-patient trial" if trial_n else "Study"
        segment_line = f" for {patient_segment.replace('_', ' ')}" if patient_segment else ""
        location_line = f" ({location})" if location else ""
        audience_line = f" for your {terms['audience']}" if terms.get("audience") else ""
        voice_term = _pick_voice_term(category)
        term_line = f" (" + voice_term + ")" if voice_term else ""
        if category_slug == "dentists":
            tag_line = "Recall interval impact"
        elif category_slug == "restaurants":
            tag_line = "Operator takeaway"
        else:
            tag_line = "Key takeaway"
        body = (
            f"{salutation}, {style['prefix']}{location_line}: {title}. "
            f"{trial_line}{segment_line}{audience_line}{term_line}. {tag_line}. "
            f"{style['cta'].replace('?', '')} "
            "and I will pull the abstract."
        )
        if source:
            body = f"{body} — {source}"
        cta = "open_ended"
        rationale = f"Trigger {kind} with digest item {item_id} for {merchant_name}"
        template_name = f"vera_{kind}_v1"
        template_params = [salutation, title, source]
    elif kind == "regulation_change":
        item_id = payload.get("top_item_id")
        digest_item = _find_digest_item(category, item_id) if item_id else None
        title = digest_item.get("title") if digest_item else "Regulation update"
        deadline = payload.get("deadline_iso")
        deadline_line = f" Deadline: {deadline}." if deadline else ""
        location_line = f" ({location})" if location else ""
        body = (
            f"{salutation}, {style['prefix']}{location_line}: {title}.{deadline_line} "
            f"{style['cta']}"
        )
        cta = "open_ended"
        rationale = f"Trigger {kind} deadline {deadline}"
        template_name = f"vera_{kind}_v1"
        template_params = [salutation, title, deadline or ""]
    elif kind == "recall_due" and customer:
        customer_name = customer.get("identity", {}).get("name", "there")
        slots = payload.get("available_slots", [])
        slot_labels = [slot.get("label") for slot in slots if slot.get("label")]
        slot_line = " or ".join(slot_labels[:2])
        due_date = payload.get("due_date")
        offer_line = f" {offer_title}." if offer_title else ""
        if slot_line:
            body = (
                f"Hi {customer_name}, {merchant_name} here. Your recall is due"
                f"{f' by {due_date}' if due_date else ''}. "
                f"Slots ready: {slot_line}. Reply 1 or 2 to confirm.{offer_line}"
            )
        else:
            body = (
                f"Hi {customer_name}, {merchant_name} here. Your recall is due"
                f"{f' by {due_date}' if due_date else ''}. "
                f"Want me to suggest a slot?{offer_line}"
            )
        body = f"{body}{lang_tail}"
        cta = "binary"
        rationale = f"Trigger {kind} for customer {customer.get('customer_id')}"
        template_name = "merchant_recall_reminder_v1"
        template_params = [customer_name, merchant_name, slot_line, offer_title or ""]
    elif kind in {"perf_dip", "perf_spike"}:
        metric = payload.get("metric") or "performance"
        metric_label = _metric_label(metric, category_slug)
        delta_pct = payload.get("delta_pct")
        window = payload.get("window") or "recent window"
        if isinstance(delta_pct, (int, float)):
            delta_line = f"{delta_pct:+.0%}"
        else:
            delta_line = str(delta_pct) if delta_pct is not None else ""
        perf = merchant.get("performance", {})
        views = perf.get("views")
        calls = perf.get("calls")
        ctr = perf.get("ctr")
        perf_line = f"Views {views}, calls {calls}, ctr {ctr}." if views is not None else ""
        peer_stats = category.get("peer_stats", {})
        peer_ctr = peer_stats.get("avg_ctr")
        peer_calls = peer_stats.get("avg_calls_30d")
        peer_line = ""
        if ctr is not None and peer_ctr is not None:
            peer_line = f" Peer avg ctr {peer_ctr}."
        peer_calls_line = ""
        if calls is not None and peer_calls is not None:
            peer_calls_line = f" Peer avg calls {peer_calls}/30d."
        offer_hint = ""
        if offer_title:
            offer_hint = f" Active offer: {offer_title}."
        else:
            suggested_offer = _category_offer_suggestion(category)
            if suggested_offer:
                offer_hint = f" Suggested offer: {suggested_offer}."
        focus_line = terms.get("focus")
        focus_text = f"Focus: {focus_line}. " if focus_line else ""
        signal_line = f" {signal_note}" if signal_note else ""
        agg = merchant.get("customer_aggregate", {})
        lapsed_180 = agg.get("lapsed_180d_plus")
        lapsed_90 = agg.get("lapsed_90d_plus")
        retention_6mo = agg.get("retention_6mo_pct")
        lapsed_line = ""
        if lapsed_180:
            lapsed_line = f" {lapsed_180} lapsed 180d+ in roster."
        elif lapsed_90:
            lapsed_line = f" {lapsed_90} lapsed 90d+ in roster."
        retention_line = ""
        if retention_6mo is not None:
            retention_line = f" 6mo retention {retention_6mo:.0%}."
        high_risk = agg.get("high_risk_adult_count")
        cohort_line = f" High-risk adults: {high_risk}." if high_risk else ""
        seasonal_note = payload.get("season_note") or _category_seasonal_note(category)
        seasonal_line = f" Seasonal note: {seasonal_note}." if seasonal_note else ""
        trend_note = _category_trend_signal(category)
        trend_line = f" Trend: {trend_note}." if trend_note else ""
        voice_word = _category_voice_word(category)
        voice_line = f" ({voice_word})" if voice_word else ""
        baseline = payload.get("vs_baseline")
        baseline_line = f" vs baseline {baseline}." if baseline is not None else ""
        location_line = f" ({location})" if location else ""
        if category_slug == "dentists":
            category_tail = (
                "Chair utilization dipped; recall and hygiene reminders are the fastest fix. "
                "Want a 2-line recall push + whitening add-on draft?"
            )
        elif category_slug == "salons":
            category_tail = (
                "Stylist slots look softer; a midweek slot-fill + add-on upgrade helps. "
                "Want a slot-fill draft for this week?"
            )
        elif category_slug == "restaurants":
            category_tail = (
                "Covers may be leaking; a dine-in vs delivery split often stabilizes orders. "
                "Want a tonight-only offer draft?"
            )
        elif category_slug == "gyms":
            category_tail = (
                "Trial-to-member conversion can drift; a 3-day nudge helps. "
                "Want a reactivation message for this week?"
            )
        elif category_slug == "pharmacies":
            category_tail = (
                "Refill intent may be slipping; a refill reminder plus stock callout helps. "
                "Want a repeat-customer refill note?"
            )
        else:
            category_tail = style["cta"]
        body = (
            f"{salutation}, {metric_label} moved {delta_line} over {window}{location_line}{voice_line}. "
            f"{perf_line}{offer_hint}{peer_line}{peer_calls_line}{baseline_line}{signal_line}"
            f"{lapsed_line}{retention_line}{cohort_line}{seasonal_line}{trend_line} {focus_text}{category_tail}"
        )
        cta = "open_ended"
        rationale = f"Trigger {kind} on {metric} delta {delta_line}"
        template_name = f"vera_{kind}_v1"
        template_params = [salutation, metric_label, delta_line, window]
    elif kind == "festival_upcoming":
        festival = payload.get("festival")
        days_until = payload.get("days_until")
        date = payload.get("date")
        location_line = f" in {location}" if location else ""
        body = (
            f"{salutation}, {festival} is coming in {days_until} days ({date}){location_line}. "
            f"{style['cta']}"
        )
        cta = "open_ended"
        rationale = f"Trigger {kind} for {festival}"
        template_name = f"vera_{kind}_v1"
        template_params = [salutation, festival, str(days_until), date or ""]
    elif kind == "review_theme_emerged":
        theme = payload.get("theme") or "a recurring theme"
        occurrences = payload.get("occurrences_30d") or "some"
        common_quote = payload.get("common_quote") or "(no quote captured)"
        theme_line = theme.replace("_", " ")
        if category_slug == "restaurants" and "delivery" in theme_line:
            action_hint = "I can draft a delivery-time response + prep-time fix."
        elif category_slug == "salons" and "wait" in theme_line:
            action_hint = "I can draft a stylist-slot response + wait-time fix."
        elif category_slug == "dentists" and "wait" in theme_line:
            action_hint = "I can draft a chair-time response + schedule tweak."
        elif category_slug == "gyms" and "crowd" in theme_line:
            action_hint = "I can draft a peak-hour response + slot plan."
        elif category_slug == "pharmacies" and "wait" in theme_line:
            action_hint = "I can draft a counter-wait response + queue fix."
        else:
            action_hint = "I can draft a response template."
        agg = merchant.get("customer_aggregate", {})
        delivery_orders = agg.get("delivery_orders_30d")
        dine_in_orders = agg.get("dine_in_orders_30d")
        volume_line = ""
        if delivery_orders or dine_in_orders:
            volume_line = f" Delivery {delivery_orders}/30d, dine-in {dine_in_orders}/30d." if delivery_orders and dine_in_orders else ""
        offer_hint = f" Active offer: {offer_title}." if offer_title else ""
        voice_word = _category_voice_word(category)
        voice_line = f" ({voice_word})" if voice_word else ""
        body = (
            f"{salutation}, {occurrences} recent reviews mention {theme_line}{voice_line}. "
            f"Example: '{common_quote}'.{volume_line}{offer_hint} {action_hint} Want me to post it?"
        )
        cta = "open_ended"
        rationale = f"Trigger {kind} theme {theme_line}"
        template_name = f"vera_{kind}_v1"
        template_params = [salutation, theme_line, str(occurrences), common_quote]
    elif kind == "competitor_opened":
        competitor = payload.get("competitor_name") or "a nearby competitor"
        distance = payload.get("distance_km")
        their_offer = payload.get("their_offer") or "an introductory offer"
        distance_line = f" {distance} km away" if distance is not None else " nearby"
        body = (
            f"{salutation}, new competitor {competitor} opened{distance_line}. "
            f"They are promoting: {their_offer}. Want a counter-offer draft?"
        )
        cta = "open_ended"
        rationale = f"Trigger {kind} for competitor {competitor}"
        template_name = f"vera_{kind}_v1"
        template_params = [salutation, competitor, str(distance or ""), their_offer]
    elif kind == "active_planning_intent":
        intent_topic = payload.get("intent_topic") or "your plan"
        last_msg = payload.get("merchant_last_message")
        if last_msg:
            body = (
                f"{salutation}, got your note about {intent_topic}. "
                f"'{last_msg}' — want me to draft a simple package outline?"
            )
        else:
            body = (
                f"{salutation}, want me to draft a simple package outline for {intent_topic}?"
            )
        cta = "open_ended"
        rationale = f"Trigger {kind} intent {intent_topic}"
        template_name = f"vera_{kind}_v1"
        template_params = [salutation, intent_topic]
    elif kind == "supply_alert":
        molecule = payload.get("molecule")
        batches = ", ".join(payload.get("affected_batches", [])[:3])
        manufacturer = payload.get("manufacturer")
        maker_line = f" by {manufacturer}" if manufacturer else ""
        agg = merchant.get("customer_aggregate", {})
        chronic_rx = agg.get("chronic_rx_count")
        chronic_line = f" Chronic patients on file: {chronic_rx}." if chronic_rx else ""
        voice_word = _category_voice_word(category)
        voice_line = f" ({voice_word})" if voice_word else ""
        location_line = f" in {location}" if location else ""
        if category_slug == "pharmacies":
            body = (
                f"{salutation}, pharmacy supply alert for {molecule}{maker_line}{location_line}{voice_line}. "
                f"Affected batches: {batches}.{chronic_line} Want a recall notice + counter checklist?"
            )
        else:
            body = (
                f"{salutation}, supply alert for {molecule}{maker_line}{location_line}{voice_line}. "
                f"Affected batches: {batches}. Want a quick customer notice draft?"
            )
        cta = "open_ended"
        rationale = f"Trigger {kind} molecule {molecule}"
        template_name = f"vera_{kind}_v1"
        template_params = [salutation, molecule, batches]
    elif kind == "chronic_refill_due" and customer:
        customer_name = customer.get("identity", {}).get("name", "there")
        molecules = ", ".join(payload.get("molecule_list", [])[:3])
        runs_out = payload.get("stock_runs_out_iso")
        body = (
            f"Hi {customer_name}, {merchant_name} here. Your refill for {molecules} "
            f"runs out around {runs_out}. Want us to arrange a refill?"
        )
        body = f"{body}{lang_tail}"
        cta = "binary"
        rationale = f"Trigger {kind} for customer {customer.get('customer_id')}"
        template_name = "pharmacy_refill_reminder_v1"
        template_params = [customer_name, molecules, runs_out or ""]
    elif kind == "gbp_unverified":
        uplift = payload.get("estimated_uplift_pct")
        path = payload.get("verification_path")
        uplift_line = f"Potential uplift ~{uplift:.0%}." if isinstance(uplift, (int, float)) else ""
        location_line = f" in {location}" if location else ""
        peer_calls = category.get("peer_stats", {}).get("avg_calls_30d")
        calls = merchant.get("performance", {}).get("calls")
        peer_calls_line = ""
        if calls is not None and peer_calls is not None:
            peer_calls_line = f" Your calls {calls}/30d vs peer {peer_calls}."
        offer_hint = ""
        if offer_title:
            offer_hint = f" Active offer: {offer_title}."
        else:
            suggested_offer = _category_offer_suggestion(category)
            if suggested_offer:
                offer_hint = f" Suggested offer: {suggested_offer}."
        seasonal_note = _category_seasonal_note(category)
        seasonal_line = f" Seasonal: {seasonal_note}." if seasonal_note else ""
        trend_note = _category_trend_signal(category)
        trend_line = f" Trend: {trend_note}." if trend_note else ""
        voice_word = _category_voice_word(category)
        voice_line = f" ({voice_word})" if voice_word else ""
        if category_slug == "restaurants":
            benefit_line = "Verified profiles get more map clicks for dining searches and reservation calls."
        elif category_slug == "salons":
            benefit_line = "Verified profiles lift appointment enquiries and stylist visibility."
        elif category_slug == "dentists":
            benefit_line = "Verified profiles build trust for first-time patients and recall callers."
        elif category_slug == "pharmacies":
            benefit_line = "Verified profiles improve call-through for urgent needs and refills."
        elif category_slug == "gyms":
            benefit_line = "Verified profiles improve trial walk-ins and class enquiries."
        else:
            benefit_line = "Verified profiles improve discovery."
        body = (
            f"{salutation}, your Google Business Profile is still unverified for the {terms['noun']}{location_line}{voice_line}. "
            f"{uplift_line}{peer_calls_line}{offer_hint}{seasonal_line}{trend_line} "
            f"{benefit_line} Verification path: {path}. "
            "Want me to start it today?"
        )
        cta = "binary"
        rationale = f"Trigger {kind} verification_path {path}"
        template_name = f"vera_{kind}_v1"
        template_params = [salutation, path, uplift_line]
    elif kind == "cde_opportunity":
        credits = payload.get("credits")
        fee = payload.get("fee")
        body = (
            f"{salutation}, CDE opportunity: {credits} credits, fee {fee}. "
            "Want the registration link and a reminder?"
        )
        cta = "open_ended"
        rationale = f"Trigger {kind} credits {credits}"
        template_name = f"vera_{kind}_v1"
        template_params = [salutation, str(credits), str(fee)]
    elif kind == "customer_lapsed_hard" and customer:
        customer_name = customer.get("identity", {}).get("name", "there")
        days = payload.get("days_since_last_visit")
        body = (
            f"Hi {customer_name}, {merchant_name} here. We have not seen you in {days} days. "
            "Want a quick check-in or a preferred slot?"
        )
        body = f"{body}{lang_tail}"
        cta = "open_ended"
        rationale = f"Trigger {kind} days_since_last_visit {days}"
        template_name = "merchant_winback_v1"
        template_params = [customer_name, str(days)]
    elif kind == "trial_followup" and customer:
        customer_name = customer.get("identity", {}).get("name", "there")
        trial_date = payload.get("trial_date")
        options = payload.get("next_session_options", [])
        option_label = options[0].get("label") if options else None
        option_line = f" Next option: {option_label}." if option_label else ""
        body = (
            f"Hi {customer_name}, thanks for the trial on {trial_date}.{option_line} "
            "Want to lock it in?"
        )
        body = f"{body}{lang_tail}"
        cta = "binary"
        rationale = f"Trigger {kind} trial_date {trial_date}"
        template_name = "merchant_trial_followup_v1"
        template_params = [customer_name, trial_date or "", option_label or ""]
    elif kind == "category_seasonal":
        trends = payload.get("trends", [])
        trend_line = ", ".join(trends[:3])
        body = (
            f"{salutation}, seasonal demand shifts: {trend_line}. "
            "Want a quick shelf/update plan?"
        )
        cta = "open_ended"
        rationale = f"Trigger {kind} trends {trend_line}"
        template_name = f"vera_{kind}_v1"
        template_params = [salutation, trend_line]
    elif kind == "milestone_reached":
        value_now = payload.get("value_now")
        milestone_value = payload.get("milestone_value")
        body = (
            f"{salutation}, you're at {value_now} reviews, close to {milestone_value}. "
            "Want a 2-line review ask you can send today?"
        )
        cta = "binary"
        rationale = f"Trigger {kind} milestone {milestone_value}"
        template_name = f"vera_{kind}_v1"
        template_params = [salutation, str(value_now), str(milestone_value)]
    elif kind == "renewal_due":
        days_remaining = payload.get("days_remaining")
        plan = payload.get("plan")
        amount = payload.get("renewal_amount")
        body = (
            f"{salutation}, your {plan} plan renews in {days_remaining} days. "
            f"Renewal amount: ₹{amount}. Want me to send the renewal link?"
        )
        cta = "binary"
        rationale = f"Trigger {kind} days_remaining {days_remaining}"
        template_name = f"vera_{kind}_v1"
        template_params = [salutation, plan or "", str(days_remaining), str(amount)]
    elif kind == "dormant_with_vera":
        days = payload.get("days_since_last_merchant_message")
        last_topic = payload.get("last_topic")
        topic_line = f" Last topic: {last_topic}." if last_topic else ""
        body = (
            f"{salutation}, it's been {days} days since we last connected.{topic_line} "
            "Want a quick 2-minute check and next-step suggestion?"
        )
        cta = "binary"
        rationale = f"Trigger {kind} days_since_last_message {days}"
        template_name = f"vera_{kind}_v1"
        template_params = [salutation, str(days), last_topic or ""]
    elif kind == "winback_eligible":
        days = payload.get("days_since_expiry")
        perf_dip = payload.get("perf_dip_pct")
        lapsed = payload.get("lapsed_customers_added_since_expiry")
        perf_line = f"perf down {perf_dip:+.0%}" if isinstance(perf_dip, (int, float)) else ""
        lapsed_line = f"{lapsed} lapsed added" if lapsed is not None else ""
        detail_line = ", ".join([t for t in [perf_line, lapsed_line] if t])
        agg = merchant.get("customer_aggregate", {})
        lapsed_90 = agg.get("lapsed_90d_plus")
        lapsed_180 = agg.get("lapsed_180d_plus")
        retention_3mo = agg.get("retention_3mo_pct")
        retention_line = f" 3mo retention {retention_3mo:.0%}." if retention_3mo is not None else ""
        roster_line = ""
        if lapsed_90:
            roster_line = f" {lapsed_90} lapsed 90d+ in roster."
        elif lapsed_180:
            roster_line = f" {lapsed_180} lapsed 180d+ in roster."
        trend_note = _category_trend_signal(category)
        trend_line = f" Trend: {trend_note}." if trend_note else ""
        voice_word = _category_voice_word(category)
        voice_line = f" ({voice_word})" if voice_word else ""
        if category_slug == "salons":
            winback_target = "lapsed clients"
            tactic_line = "Add a midweek slot-fill + add-on upgrade."
        elif category_slug == "gyms":
            winback_target = "lapsed members"
            tactic_line = "Offer a 3-day trial return or buddy pass."
        elif category_slug == "restaurants":
            winback_target = "inactive diners"
            tactic_line = "Offer a weekday special or delivery-freebie."
        elif category_slug == "pharmacies":
            winback_target = "repeat customers"
            tactic_line = "Offer a refill reminder + loyalty incentive."
        else:
            winback_target = "lapsed customers"
            tactic_line = "Offer a short-window return incentive."
        offer_hint = f" Active offer: {offer_title}." if offer_title else ""
        body = (
            f"{salutation}, winback window at {days} days post-expiry{voice_line}. {detail_line}. "
            f"{tactic_line}{offer_hint}{roster_line}{retention_line}{trend_line} "
            f"Want a winback draft to re-activate {winback_target}?"
        )
        cta = "open_ended"
        rationale = f"Trigger {kind} days_since_expiry {days}"
        template_name = f"vera_{kind}_v1"
        template_params = [salutation, str(days), detail_line]
    else:
        body = f"{salutation}, quick update for {merchant_name}. Focus: {hint}. {style['cta']}"
        cta = "open_ended"
        rationale = f"Trigger {kind} generic handling"
        template_name = "vera_generic_v1"
        template_params = [salutation, merchant_name]

    return {
        "body": body,
        "cta": cta,
        "send_as": send_as,
        "suppression_key": suppression_key,
        "rationale": rationale,
        "template_name": template_name,
        "template_params": template_params,
    }


@app.get("/v1/healthz")
def healthz():
    uptime = int(time.time() - START_TS)
    return {
        "status": "ok",
        "uptime_seconds": uptime,
        "contexts_loaded": {
            "category": len(CONTEXTS["category"]),
            "merchant": len(CONTEXTS["merchant"]),
            "customer": len(CONTEXTS["customer"]),
            "trigger": len(CONTEXTS["trigger"]),
        },
    }


@app.get("/v1/metadata")
def metadata():
    return {
        "team_name": "Team Candidate",
        "team_members": ["Your Name"],
        "model": "deterministic-rules",
        "approach": "deterministic compose() with grounded templates",
        "contact_email": "you@example.com",
        "version": "0.1.0",
        "submitted_at": _now_iso(),
    }


@app.post("/v1/context")
def push_context(req: ContextPush):
    if req.scope not in CONTEXTS:
        return JSONResponse(
            status_code=400,
            content={"accepted": False, "reason": "invalid_scope", "details": req.scope},
        )

    scope_store = CONTEXTS[req.scope]
    existing = scope_store.get(req.context_id)
    if existing:
        current_version = existing["version"]
        if req.version < current_version:
            return JSONResponse(
                status_code=409,
                content={
                    "accepted": False,
                    "reason": "stale_version",
                    "current_version": current_version,
                },
            )
        if req.version == current_version:
            return {
                "accepted": True,
                "ack_id": f"ack_{req.context_id}_v{req.version}",
                "stored_at": existing["stored_at"],
            }

    scope_store[req.context_id] = {
        "version": req.version,
        "payload": req.payload,
        "delivered_at": req.delivered_at,
        "stored_at": _now_iso(),
    }

    return {
        "accepted": True,
        "ack_id": f"ack_{req.context_id}_v{req.version}",
        "stored_at": scope_store[req.context_id]["stored_at"],
    }


@app.post("/v1/tick")
def tick(req: TickRequest):
    now = _parse_iso(req.now) or datetime.now(timezone.utc)
    candidates: list[Tuple[int, str, Dict[str, Any]]] = []
    for trigger_id in req.available_triggers:
        trigger = _get_context("trigger", trigger_id)
        if not trigger:
            continue
        merchant_id = trigger.get("merchant_id")
        merchant = _get_context("merchant", merchant_id) if merchant_id else None
        if not merchant:
            continue
        if trigger.get("scope") == "customer":
            customer_id = trigger.get("customer_id")
            if customer_id and not _get_context("customer", customer_id):
                continue
        score, _ = _rank_trigger(trigger, merchant, now)
        if score < 0:
            continue
        candidates.append((score, trigger_id, trigger))

    if not candidates:
        return {"actions": []}

    candidates.sort(key=lambda item: (-item[0], item[1]))
    _, trigger_id, trigger = candidates[0]
    suppression_key = trigger.get("suppression_key", trigger_id)
    if suppression_key in SENT_SUPPRESSIONS:
        return {"actions": []}

    merchant_id = trigger.get("merchant_id")
    customer_id = trigger.get("customer_id")
    merchant = _get_context("merchant", merchant_id) if merchant_id else None
    if not merchant:
        return {"actions": []}
    if merchant_id in SUPPRESSED_MERCHANTS:
        return {"actions": []}

    category_slug = merchant.get("category_slug")
    category = _get_context("category", category_slug) if category_slug else None
    customer = _get_context("customer", customer_id) if customer_id else None
    if not category:
        return {"actions": []}

    try:
        action = _compose_message(category, merchant, trigger, customer)
    except Exception:
        return {"actions": []}
    conversation_id = f"conv_{trigger_id}"
    CONVERSATIONS[conversation_id] = {
        "trigger_id": trigger_id,
        "merchant_id": merchant_id,
        "customer_id": customer_id,
        "last_action": action,
    }
    SENT_SUPPRESSIONS.add(suppression_key)

    return {
        "actions": [
            {
                "conversation_id": conversation_id,
                "merchant_id": merchant_id,
                "customer_id": customer_id,
                "send_as": action["send_as"],
                "trigger_id": trigger_id,
                "template_name": action.get("template_name"),
                "template_params": action.get("template_params", []),
                "body": action["body"],
                "cta": action["cta"],
                "suppression_key": action["suppression_key"],
                "rationale": action["rationale"],
            }
        ]
    }


@app.post("/v1/reply")
def reply(req: ReplyRequest):
    message = req.message or ""
    convo = CONVERSATIONS.setdefault(req.conversation_id, {"auto_reply_count": 0})
    if _is_auto_reply(message):
        convo["auto_reply_count"] = int(convo.get("auto_reply_count", 0)) + 1
        if convo["auto_reply_count"] >= 3:
            return {
                "action": "end",
                "rationale": "Auto-reply repeated; ending to avoid loops",
            }
        return {
            "action": "wait",
            "wait_seconds": 1800,
            "rationale": "Auto-reply detected; backing off",
        }

    if _is_hostile(message):
        if req.merchant_id:
            SUPPRESSED_MERCHANTS.add(req.merchant_id)
        return {
            "action": "end",
            "rationale": "Merchant requested to stop messaging",
        }

    if _is_out_of_scope(message):
        return {
            "action": "send",
            "body": "That is outside what I can help with directly. Coming back to this thread — should I proceed with the draft?",
            "cta": "binary",
            "rationale": "Out-of-scope request; redirecting to original task",
        }

    last_action = convo.get("last_action", {})
    if _is_commitment(message):
        follow_up = "Done. I will send it now."
        if last_action.get("cta") == "binary":
            follow_up = "Confirmed. I will proceed now."
        return {
            "action": "send",
            "body": follow_up,
            "cta": "open_ended",
            "rationale": "Merchant committed; proceeding with next step",
        }

    return {
        "action": "send",
        "body": "Got it. One quick question so I can proceed: do you want this to go out today?",
        "cta": "binary",
        "rationale": "Clarify intent before proceeding",
    }
