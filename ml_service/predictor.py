"""
ml_service/predictor.py — NEXUS SPAM SHIELD v5.0
Loads trained model artifacts and performs inference with calibrated threshold.
"""
import pickle, re, logging, os

logger = logging.getLogger("NexusSpamShield.Predictor")

BASE_DIR        = os.path.dirname(os.path.dirname(__file__))
MODEL_PATH      = os.path.join(BASE_DIR, "model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "vectorizer.pkl")
THRESHOLD_PATH  = os.path.join(BASE_DIR, "threshold.pkl")
META_PATH       = os.path.join(BASE_DIR, "model_meta.pkl")

DEFAULT_THRESHOLD = 0.65

SPAM_SIGNALS = {
    'free','win','winner','won','cash','prize','claim','offer','urgent',
    'congratulations','click','limited','deal','guaranteed','money','credit',
    'loan','discount','selected','promotion','bonus','reward','gift',
    'exclusive','subscribe','verify','account','password','bank','invoice',
    'confirm','nigeria','transfer','million','billion','investment','profit',
    'pharmacy','pills','ringtone','xxx','adult','dating','singles',
    'lottery','rupees','recharge','iphone','samsung','earn','salary',
    'hiring','apply','register','order','buy','shop','hurry','expires',
    'suspended','deactivated','compromised','alert','security','phishing',
    'bitcoin','crypto','forex','otp','kyc','scheme','survey',
}


def _classify(spam_prob: float, message: str, threshold: float) -> str:
    words   = message.lower().split()
    wordset = set(words)
    signals = len(wordset & SPAM_SIGNALS)
    if len(words) <= 6 and signals == 0:
        return "NOT SPAM"
    eff = threshold
    if signals >= 3:   eff = max(0.38, threshold - 0.20)
    elif signals >= 2: eff = max(0.48, threshold - 0.12)
    elif signals >= 1: eff = max(0.55, threshold - 0.05)
    return "SPAM" if spam_prob >= eff else "NOT SPAM"


class SpamPredictor:
    def __init__(self):
        self.model = self.vectorizer = None
        self.threshold = DEFAULT_THRESHOLD
        self.meta = {}
        self._load()

    def _load(self):
        try:
            with open(MODEL_PATH, "rb")      as f: self.model     = pickle.load(f)
            with open(VECTORIZER_PATH, "rb") as f: self.vectorizer = pickle.load(f)
            logger.info("Model and vectorizer loaded.")
        except FileNotFoundError as e:
            logger.error("Model files not found: %s", e)
            return
        try:
            with open(THRESHOLD_PATH, "rb") as f: self.threshold = pickle.load(f)
        except Exception:
            pass
        try:
            with open(META_PATH, "rb") as f: self.meta = pickle.load(f)
        except Exception:
            pass
        logger.info("Threshold=%.2f  Model=%s", self.threshold, self.meta.get("model_name","unknown"))

    def is_ready(self) -> bool:
        return self.model is not None and self.vectorizer is not None

    def predict(self, message: str) -> dict:
        if not self.is_ready():
            raise RuntimeError("Model not loaded. Run python train_model.py first.")
        clean = re.sub(r"[^\w\s\'\-]", " ", message.lower().strip())
        clean = re.sub(r"\s+", " ", clean).strip()
        vec   = self.vectorizer.transform([clean])
        proba = self.model.predict_proba(vec)[0]
        spam_prob = float(proba[1])
        ham_prob  = float(proba[0])
        label = _classify(spam_prob, message, self.threshold)
        words     = message.lower().split()
        signals   = len(set(words) & SPAM_SIGNALS)
        risk_score = min(100, int(spam_prob * 100 * (1 + signals * 0.05)))
        return {
            "label":      label,
            "spam_prob":  spam_prob,
            "ham_prob":   ham_prob,
            "vector":     vec,
            "threshold":  self.threshold,
            "risk_score": risk_score,
            "meta":       self.meta,
        }