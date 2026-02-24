"""
phone_service/db.py — NEXUS SPAM SHIELD v5.0
Extended schema: risk_score, trust_classification, analysis_count, first_seen, last_checked.
"""
import sqlite3, os, logging
from datetime import datetime

logger  = logging.getLogger("NexusSpamShield.PhoneDB")
BASE    = os.path.dirname(os.path.dirname(__file__))
DB_PATH = os.path.join(BASE, "phone_numbers.db")

def _conn():
    c = sqlite3.connect(DB_PATH)
    c.row_factory = sqlite3.Row
    return c

def init_db():
    c = _conn()
    c.executescript("""
        CREATE TABLE IF NOT EXISTS phone_numbers (
            id                   INTEGER PRIMARY KEY AUTOINCREMENT,
            phone_number         TEXT    NOT NULL UNIQUE,
            label                TEXT    NOT NULL DEFAULT 'unknown',
            name                 TEXT,
            report_count         INTEGER NOT NULL DEFAULT 0,
            risk_level           TEXT    NOT NULL DEFAULT 'LOW',
            risk_score           INTEGER NOT NULL DEFAULT 0,
            trust_classification TEXT    NOT NULL DEFAULT 'UNKNOWN',
            analysis_count       INTEGER NOT NULL DEFAULT 0,
            first_seen           TEXT,
            last_checked         TEXT,
            last_updated         TEXT    NOT NULL
        );
        CREATE TABLE IF NOT EXISTS activity_log (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            action     TEXT NOT NULL,
            phone      TEXT NOT NULL,
            result     TEXT,
            created_at TEXT NOT NULL
        );
    """)
    # Safe migration for existing databases
    migrations = [
        "ALTER TABLE phone_numbers ADD COLUMN risk_score INTEGER NOT NULL DEFAULT 0",
        "ALTER TABLE phone_numbers ADD COLUMN trust_classification TEXT NOT NULL DEFAULT \'UNKNOWN\'",
        "ALTER TABLE phone_numbers ADD COLUMN analysis_count INTEGER NOT NULL DEFAULT 0",
        "ALTER TABLE phone_numbers ADD COLUMN first_seen TEXT",
        "ALTER TABLE phone_numbers ADD COLUMN last_checked TEXT",
    ]
    for sql in migrations:
        try: c.execute(sql); c.commit()
        except Exception: pass
    c.commit(); c.close()
    logger.info("Phone DB ready at %s", DB_PATH)

def get_number(phone):
    c   = _conn()
    row = c.execute("SELECT * FROM phone_numbers WHERE phone_number=?", (phone,)).fetchone()
    if row:
        now = datetime.utcnow().isoformat() + "Z"
        cnt = (row["analysis_count"] or 0) + 1
        c.execute("UPDATE phone_numbers SET analysis_count=?, last_checked=? WHERE phone_number=?",
                  (cnt, now, phone))
        c.commit()
    c.close()
    return dict(row) if row else None

def upsert_report(phone, name=None):
    now = datetime.utcnow().isoformat() + "Z"
    c   = _conn()
    ex  = c.execute("SELECT * FROM phone_numbers WHERE phone_number=?", (phone,)).fetchone()
    if ex:
        cnt   = ex["report_count"] + 1
        risk  = _risk_level(cnt)
        score = _risk_score(cnt)
        trust = _trust_class(score)
        c.execute("""UPDATE phone_numbers
                     SET report_count=?,risk_level=?,risk_score=?,trust_classification=?,
                         label=\'spam\',last_updated=?,last_checked=?,name=COALESCE(?,name)
                     WHERE phone_number=?""",
                  (cnt, risk, score, trust, now, now, name, phone))
    else:
        score = _risk_score(1); trust = _trust_class(score)
        c.execute("""INSERT INTO phone_numbers
                     (phone_number,label,name,report_count,risk_level,risk_score,
                      trust_classification,analysis_count,first_seen,last_checked,last_updated)
                     VALUES (?,?,?,1,?,?,?,1,?,?,?)""",
                  (phone,"spam",name,_risk_level(1),score,trust,now,now,now))
    c.commit()
    row = c.execute("SELECT * FROM phone_numbers WHERE phone_number=?", (phone,)).fetchone()
    c.close()
    return dict(row)

def log_activity(action, phone, result):
    c = _conn()
    c.execute("INSERT INTO activity_log(action,phone,result,created_at) VALUES(?,?,?,?)",
              (action,phone,result,datetime.utcnow().isoformat()+"Z"))
    c.commit(); c.close()

def get_recent_activity(limit=20):
    c    = _conn()
    rows = c.execute("SELECT * FROM activity_log ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    c.close()
    return [dict(r) for r in rows]

def get_stats():
    c = _conn()
    s = {
        "total_numbers":  c.execute("SELECT COUNT(*) FROM phone_numbers").fetchone()[0],
        "spam_numbers":   c.execute("SELECT COUNT(*) FROM phone_numbers WHERE label=\'spam\'").fetchone()[0],
        "safe_numbers":   c.execute("SELECT COUNT(*) FROM phone_numbers WHERE label=\'safe\'").fetchone()[0],
        "total_checks":   c.execute("SELECT COUNT(*) FROM activity_log WHERE action=\'check\'").fetchone()[0],
        "total_reports":  c.execute("SELECT COUNT(*) FROM activity_log WHERE action=\'report\'").fetchone()[0],
        "avg_risk_score": c.execute("SELECT AVG(risk_score) FROM phone_numbers WHERE risk_score>0").fetchone()[0] or 0,
    }
    s["avg_risk_score"] = round(float(s["avg_risk_score"]), 1)
    c.close()
    return s


def delete_number(phone):
    """Remove a phone number and all its activity from the database."""
    c = _conn()
    row = c.execute("SELECT * FROM phone_numbers WHERE phone_number=?", (phone,)).fetchone()
    if not row:
        c.close()
        return {"success": False, "message": f"Number {phone} not found in database."}
    c.execute("DELETE FROM phone_numbers WHERE phone_number=?", (phone,))
    c.execute("DELETE FROM activity_log WHERE phone=?", (phone,))
    c.commit()
    c.close()
    logger.info("Deleted phone_number=%s from DB", phone)
    return {"success": True, "message": f"Number {phone} removed successfully.", "phone": phone}

def _risk_level(n):
    if n >= 10: return "CRITICAL"
    if n >= 5:  return "HIGH"
    if n >= 2:  return "MEDIUM"
    return "LOW"

def _risk_score(n):
    if n == 1:  return 30
    if n == 2:  return 45
    if n == 3:  return 58
    if n == 4:  return 68
    if n == 5:  return 75
    if n <= 7:  return 83
    if n <= 9:  return 91
    return 97

def _trust_class(score):
    if score <= 20: return "UNKNOWN"
    if score <= 50: return "LOW_TRUST"
    if score <= 80: return "SUSPICIOUS"
    return "HIGH_RISK"