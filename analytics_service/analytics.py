"""
analytics_service/analytics.py — NEXUS SPAM SHIELD v5.0
Central analytics aggregator.
"""
import logging
from datetime import datetime, timedelta
from phone_service.db import get_stats, get_recent_activity

logger = logging.getLogger("NexusSpamShield.Analytics")

_session = {
    "messages_scanned": 0, "spam_detected": 0,
    "safe_detected": 0, "total_risk_score": 0,
    "started_at": datetime.utcnow().isoformat() + "Z",
}

def record_scan(is_spam: bool, risk_score: int = 0):
    _session["messages_scanned"] += 1
    if is_spam: _session["spam_detected"] += 1
    else:       _session["safe_detected"]  += 1
    _session["total_risk_score"] += risk_score

def get_session_stats() -> dict:
    total = _session["messages_scanned"]
    avg_risk = round(_session["total_risk_score"] / total, 1) if total else 0
    return {
        "messages_scanned": total,
        "spam_detected":    _session["spam_detected"],
        "safe_detected":    _session["safe_detected"],
        "spam_rate":        round(_session["spam_detected"] / total * 100, 1) if total else 0,
        "avg_risk_score":   avg_risk,
        "session_started":  _session["started_at"],
    }

def get_full_analytics() -> dict:
    phone_stats     = get_stats()
    recent_activity = get_recent_activity(20)
    session_stats   = get_session_stats()
    return {
        "phone_stats":     phone_stats,
        "recent_activity": recent_activity,
        "session_stats":   session_stats,
        "timestamp":       datetime.utcnow().isoformat() + "Z",
    }