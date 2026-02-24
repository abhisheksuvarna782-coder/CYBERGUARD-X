"""
server.py — NEXUS SPAM SHIELD v5.0
Flask backend: message analysis + phone intelligence + analytics.
"""
import logging, os
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from datetime import datetime

from ml_service.predictor import SpamPredictor
from ml_service.explainer import SpamExplainer
from phone_service.db import init_db, get_stats, get_recent_activity, delete_number
from phone_service.number_checker import check_number
from phone_service.reporter import report_number
from analytics_service.analytics import record_scan, get_full_analytics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/app.log", mode="a"),
    ],
)
logger = logging.getLogger("NexusSpamShield")

app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app)

init_db()
predictor = SpamPredictor()
explainer = SpamExplainer(predictor.vectorizer, predictor.model)

def _validate_msg(msg):
    if not msg or not isinstance(msg, str): return False, "Message required."
    if len(msg.strip()) < 3:               return False, "Message too short."
    if len(msg) > 10000:                   return False, "Message too long (max 10,000 chars)."
    return True, ""

@app.route("/")
def index():
    return send_from_directory("templates", "index.html")

@app.route("/api/status")
def api_status():
    stats = get_stats()
    model_info = predictor.meta if predictor.meta else {}
    return jsonify({
        "status": "online", "version": "5.0.0",
        "model":  model_info.get("model_name", "Unknown"),
        "model_f1": model_info.get("f1_score", "N/A"),
        "model_accuracy": model_info.get("accuracy", "N/A"),
        "threshold": predictor.threshold,
        "vocab_size": model_info.get("vocab_size", "N/A"),
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "model_loaded": predictor.is_ready(),
        "phone_db": stats,
    })

@app.route("/api/predict", methods=["POST"])
@app.route("/api/predict-message", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}
    msg  = data.get("message", "")
    ok, err = _validate_msg(msg)
    if not ok:
        return jsonify({"error": err}), 422
    try:
        res  = predictor.predict(msg)
        expl = explainer.explain(msg, res["label"], res.get("risk_score", 0))
        record_scan(res["label"] == "SPAM", res.get("risk_score", 0))
        logger.info("MSG label=%s spam=%.1f%% risk=%d len=%d",
                    res["label"], res["spam_prob"]*100, res.get("risk_score",0), len(msg))
        return jsonify({
            "label":       res["label"],
            "is_spam":     res["label"] == "SPAM",
            "confidence":  {"spam": round(res["spam_prob"]*100, 2), "ham": round(res["ham_prob"]*100, 2)},
            "risk_score":  res.get("risk_score", 0),
            "explanation": expl,
            "char_count":  len(msg),
            "timestamp":   datetime.utcnow().isoformat() + "Z",
        })
    except Exception as e:
        logger.error("Predict error: %s", e, exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route("/api/check-number", methods=["POST"])
def api_check_number():
    data  = request.get_json(silent=True) or {}
    phone = data.get("phone", "").strip()
    if not phone:
        return jsonify({"error": "Phone number required."}), 422
    result = check_number(phone)
    if not result["valid"]:
        return jsonify({"error": result["error"]}), 422
    return jsonify(result)

@app.route("/api/report-number", methods=["POST"])
def api_report_number():
    data  = request.get_json(silent=True) or {}
    phone = data.get("phone", "").strip()
    name  = data.get("name",  "").strip() or None
    if not phone:
        return jsonify({"error": "Phone number required."}), 422
    result = report_number(phone, name)
    if not result["valid"]:
        return jsonify({"error": result["error"]}), 422
    return jsonify(result)

@app.route("/api/analytics")
def api_analytics():
    return jsonify(get_full_analytics())

@app.route("/api/remove-number", methods=["POST"])
def api_remove_number():
    data  = request.get_json(silent=True) or {}
    phone = data.get("phone", "").strip()
    if not phone:
        return jsonify({"error": "Phone number required."}), 422
    result = delete_number(phone)
    logger.info("Removed phone=%s from DB", phone)
    return jsonify(result)

import os

if __name__ == "__main__":
    logger.info("NEXUS SPAM SHIELD v5.0 -> Railway deployment")

    port = int(os.environ.get("PORT", 5000))  # Railway provides PORT
    app.run(debug=False, host="0.0.0.0", port=port)