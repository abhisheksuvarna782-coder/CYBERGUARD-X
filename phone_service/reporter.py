"""
phone_service/reporter.py — NEXUS SPAM SHIELD v5.0
"""
import logging
from datetime import datetime
from phone_service.db import upsert_report, log_activity
from phone_service.number_checker import normalise, _score_to_trust_class, RECOMMENDATIONS

logger = logging.getLogger("NexusSpamShield.Reporter")

def report_number(raw_phone, name=None):
    phone = normalise(raw_phone)
    if not phone:
        return {"valid": False, "error": "Invalid format. Use 7-15 digits."}

    record    = upsert_report(phone, name)
    timestamp = datetime.utcnow().isoformat() + "Z"
    log_activity("report", phone, "spam")

    cnt        = record["report_count"]
    risk_score = record.get("risk_score", 0) or 0
    risk_level = record["risk_level"]
    trust_cls  = _score_to_trust_class(risk_score)
    rec        = RECOMMENDATIONS.get(trust_cls, "likely spam — do not engage")

    summary = (
        f"Number {phone} reported as SPAM. Total community reports: {cnt}. "
        f"Risk score recalculated to {risk_score}/100 — {risk_level} threat level. "
        f"Database updated at {timestamp}. Recommendation: {rec.upper()}."
    )
    logger.info("Reported phone=%s reports=%d risk=%d level=%s", phone, cnt, risk_score, risk_level)

    return {
        "valid": True, "phone": phone, "label": "spam", "is_spam": True,
        "name": record["name"], "report_count": cnt,
        "risk_level": risk_level, "risk_score": risk_score,
        "trust_classification": trust_cls, "recommendation": rec,
        "intelligence_summary": summary,
        "last_updated": record["last_updated"], "reported_at": timestamp,
        "message": f"Reported. Reports: {cnt}. Risk: {risk_score}/100.",
    }