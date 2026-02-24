"""
phone_service/number_checker.py — NEXUS SPAM SHIELD v5.0
Real unknown number intelligence engine with heuristic analysis.
"""
import re, logging
from datetime import datetime
from phone_service.db import get_number, log_activity

logger = logging.getLogger("NexusSpamShield.NumberChecker")

SUSPICIOUS_PREFIXES = {
    "900","976","809","284","876","268","473","664","649",
    "767","784","868","869","758","246","441","345",
    "1900","0900","0976"
}
TOLL_FREE_PREFIXES = {"800","888","877","866","855","844","833"}
KNOWN_SPAM_PATTERNS = [
    r"^(\d)\1{6,}", r"^(\d{2})\1{3,}",
    r"^1234567", r"^9876543", r"^0{3,}",
]
RECOMMENDATIONS = {
    "UNKNOWN":    "safe to answer",
    "LOW_TRUST":  "monitor activity",
    "SUSPICIOUS": "caution advised",
    "HIGH_RISK":  "likely spam — do not engage",
}


def normalise(raw):
    cleaned = re.sub(r"[\s\-\(\)\+\.]", "", str(raw).strip())
    return cleaned if re.match(r"^\d{7,15}$", cleaned) else None


def _score_to_risk_level(score):
    if score <= 20: return "LOW"
    if score <= 50: return "MEDIUM"
    if score <= 80: return "HIGH"
    return "CRITICAL"

def _score_to_trust_class(score):
    if score <= 20: return "UNKNOWN"
    if score <= 50: return "LOW_TRUST"
    if score <= 80: return "SUSPICIOUS"
    return "HIGH_RISK"


def _analyse_unknown(phone):
    score = 0; signals = []
    n = len(phone)
    if n < 7:
        score += 30; signals.append("abnormally short number")
    elif n > 13:
        score += 25; signals.append("abnormally long number")
    elif n not in (10, 11, 12):
        score += 10; signals.append(f"non-standard length ({n} digits)")

    for plen in (4, 3):
        pfx = phone[:plen]
        if pfx in SUSPICIOUS_PREFIXES:
            score += 35; signals.append(f"premium-rate/high-risk prefix ({pfx})"); break

    for plen in (3, 4):
        pfx = phone[:plen]
        if pfx in TOLL_FREE_PREFIXES:
            score += 10; signals.append(f"toll-free prefix ({pfx})"); break

    for pat in KNOWN_SPAM_PATTERNS:
        if re.search(pat, phone):
            score += 20; signals.append("synthetic digit pattern detected"); break

    if len(set(phone)) == 1:
        score += 40; signals.append("single-digit number")

    counts   = {d: phone.count(d) for d in set(phone)}
    max_freq = max(counts.values()) / len(phone)
    if max_freq > 0.6:
        score += 15; signals.append(f"low digit diversity ({int(max_freq*100)}% dominant)")

    score += 8; signals.append("absent from threat database")
    score += 5; signals.append("identity unverified")
    score = min(score, 100)

    risk_level  = _score_to_risk_level(score)
    trust_class = _score_to_trust_class(score)
    rec         = RECOMMENDATIONS[trust_class]
    sig_text    = "; ".join(signals)
    pfx_info    = f"prefix '{phone[:3]}'"

    summaries = {
        "UNKNOWN": (
            f"Number {phone} is absent from the threat intelligence database. "
            f"Structural analysis of {pfx_info} returned risk score {score}/100. "
            f"Signals: {sig_text}. No known threat indicators. Recommendation: {rec.upper()}."
        ),
        "LOW_TRUST": (
            f"Number {phone} is unlisted. Heuristic flags: {sig_text}. "
            f"Risk score {score}/100 — elevated above baseline. "
            f"Limited verifiability. Recommendation: {rec.upper()}."
        ),
        "SUSPICIOUS": (
            f"Number {phone} triggered multiple heuristic flags: {sig_text}. "
            f"Risk score {score}/100. Structure consistent with automated dialing or fraud. "
            f"Recommendation: {rec.upper()}."
        ),
        "HIGH_RISK": (
            f"THREAT ALERT — {phone} exhibits high-risk structural patterns. "
            f"Risk score {score}/100. Critical signals: {sig_text}. "
            f"Profile consistent with spam, robocall, or toll-fraud. "
            f"Recommendation: {rec.upper()}."
        ),
    }

    return {
        "risk_score": score, "risk_level": risk_level,
        "trust_classification": trust_class, "recommendation": rec,
        "intelligence_summary": summaries[trust_class],
        "heuristic_signals": signals,
    }


def check_number(raw_phone):
    phone = normalise(raw_phone)
    if not phone:
        return {"valid": False, "error": "Invalid format. Use 7-15 digits."}

    record = get_number(phone)
    log_activity("check", phone, record["label"] if record else "unknown")

    if record:
        stored_score = record.get("risk_score") or 0
        trust_class  = _score_to_trust_class(stored_score)
        rec = RECOMMENDATIONS.get(trust_class, "monitor activity")
        if record["label"] == "spam":
            summary = (f"Number {phone} is confirmed SPAM with {record['report_count']} "
                       f"community reports. Risk score: {stored_score}/100. Do not engage.")
            rec = "likely spam — do not engage"
        else:
            summary = (f"Number {phone} is in the database (label: {record['label']}). "
                       f"Risk score: {stored_score}/100. No active spam classification.")
        return {
            "valid": True, "phone": phone, "found": True,
            "label": record["label"], "is_spam": record["label"] == "spam",
            "name": record["name"], "report_count": record["report_count"],
            "risk_level": record["risk_level"], "risk_score": stored_score,
            "trust_classification": trust_class, "recommendation": rec,
            "intelligence_summary": summary, "heuristic_signals": [],
            "analysis_count": record.get("analysis_count", 0),
            "first_seen": record.get("first_seen"), "last_checked": record.get("last_checked"),
            "last_updated": record["last_updated"],
        }

    intel = _analyse_unknown(phone)
    return {
        "valid": True, "phone": phone, "found": False,
        "label": "unknown", "is_spam": False, "name": None, "report_count": 0,
        "risk_level": intel["risk_level"], "risk_score": intel["risk_score"],
        "trust_classification": intel["trust_classification"],
        "recommendation": intel["recommendation"],
        "intelligence_summary": intel["intelligence_summary"],
        "heuristic_signals": intel["heuristic_signals"],
        "analysis_count": 0, "first_seen": None, "last_checked": None, "last_updated": None,
    }