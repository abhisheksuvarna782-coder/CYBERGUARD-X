"""
ml_service/explainer.py — NEXUS SPAM SHIELD v5.0
AI explainability: top features, token highlighting, threat summary.
"""
import numpy as np, re, logging

logger = logging.getLogger("NexusSpamShield.Explainer")

SPAM_SIGNAL_WORDS = {
    "free","win","winner","won","cash","prize","claim","offer",
    "urgent","congratulations","click","call","now","limited","deal",
    "guaranteed","money","credit","loan","discount","selected",
    "promotion","bonus","reward","gift","exclusive","subscribe",
    "verify","account","password","bank","invoice","confirm",
    "nigeria","transfer","million","billion","investment","profit",
    "pharmacy","pills","text","txt","sms","mobile","ringtone",
    "xxx","adult","dating","singles","bitcoin","crypto","forex",
    "otp","kyc","scheme","survey","lottery","earn","salary",
    "hiring","apply","register","hurry","expires","suspended",
}

THREAT_LEVELS = {
    (0,  25):  ("LOW",      "No significant threat indicators."),
    (25, 50):  ("MODERATE", "Some suspicious patterns detected."),
    (50, 75):  ("HIGH",     "Multiple threat indicators found."),
    (75, 101): ("CRITICAL", "Strong spam indicators. High risk."),
}


def _threat_level(risk_score: int):
    for (lo, hi), (level, desc) in THREAT_LEVELS.items():
        if lo <= risk_score < hi:
            return level, desc
    return "CRITICAL", "Extreme threat level."


class SpamExplainer:
    def __init__(self, vectorizer, model):
        self.vectorizer = vectorizer
        self.model = model
        if vectorizer is not None:
            try:
                self.feature_names = list(vectorizer.get_feature_names_out())
            except AttributeError:
                self.feature_names = list(vectorizer.get_feature_names())
        else:
            self.feature_names = []

    def explain(self, message: str, label: str, risk_score: int = 0) -> dict:
        if not self.feature_names:
            return {"top_words":[],"highlighted_tokens":[],"summary":"Explanation unavailable.","signal_words_found":[],"threat_level":"UNKNOWN","threat_desc":""}
        try:
            vector = self.vectorizer.transform([message])
            scores = np.asarray(vector.todense()).flatten()
            nz_idx = np.where(scores > 0)[0]
            scored = sorted(
                [(self.feature_names[i], float(scores[i])) for i in nz_idx],
                key=lambda x: x[1], reverse=True)
            top_words = [{"word": w, "score": round(s, 4)} for w, s in scored[:12]]
            tokens = set(re.findall(r"\b\w+\b", message.lower()))
            signal_words_found = sorted(tokens & SPAM_SIGNAL_WORDS)
            highlighted_tokens = self._highlight(message, signal_words_found, scored)
            threat_level, threat_desc = _threat_level(risk_score)
            summary = self._summary(label, top_words, signal_words_found, risk_score)
            return {
                "top_words": top_words,
                "highlighted_tokens": highlighted_tokens,
                "summary": summary,
                "signal_words_found": signal_words_found,
                "threat_level": threat_level,
                "threat_desc": threat_desc,
            }
        except Exception as e:
            logger.warning("Explainer error: %s", e)
            return {"top_words":[],"highlighted_tokens":[],"summary":"Explanation unavailable.","signal_words_found":[],"threat_level":"UNKNOWN","threat_desc":""}

    def _highlight(self, message, signal_words, scored):
        top_feat = {w for w, _ in scored[:15]}
        tokens   = re.findall(r"\S+|\s+", message)
        result   = []
        for token in tokens:
            clean = re.sub(r"[^\w]", "", token).lower()
            if clean in signal_words:
                result.append({"word": token, "type": "spam"})
            elif clean in top_feat:
                result.append({"word": token, "type": "flagged"})
            else:
                result.append({"word": token, "type": "safe"})
        return result

    def _summary(self, label, top_words, signal_words, risk_score):
        word_list   = ", ".join([w["word"] for w in top_words[:5]]) or "none"
        signal_list = ", ".join(signal_words[:4]) or None
        if label == "SPAM":
            s = f"THREAT DETECTED — Risk score: {risk_score}/100. Key indicators: {word_list}."
            if signal_list:
                s += f" High-risk keywords: {signal_list}."
            s += " Pattern profile matches known spam templates."
        else:
            s = f"SIGNAL CLEAR — Risk score: {risk_score}/100. Dominant features: {word_list}."
            if signal_list:
                s += f" Cautionary words noted ({signal_list}) but below spam threshold."
            else:
                s += " No high-risk patterns detected."
        return s