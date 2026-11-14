"""VERITIQ Advanced (Pro) features: opt-in persistent memory, recommendation/impact tracking,
optional LLM-reworded explanations, export and delete. Standard use never touches the database.
Stdlib only. Backend-enforced: every persistent call checks the Advanced settings."""
import sqlite3, csv, io, json, os, urllib.request
from datetime import datetime

DB = os.environ.get("VERITIQ_DB", "veritiq_pro.db")
STATUSES = ["New", "Under Review", "Accepted", "Planned", "In Progress", "Implemented", "Rejected", "Deferred"]
DEFAULTS = {"pro_enabled": "0", "consent": "0", "impact_tracker": "1", "llm_enabled": "0",
            "llm_endpoint": "http://localhost:11434/v1", "llm_model": "llama3.1"}
_SESSION_KEY = ""   # API key lives in memory only, never in the database or on disk

class ProDisabled(Exception): pass

def _c():
    con = sqlite3.connect(DB); con.row_factory = sqlite3.Row; return con

def init_db():
    with _c() as c:
        c.executescript("""
        CREATE TABLE IF NOT EXISTS settings(k TEXT PRIMARY KEY, v TEXT);
        CREATE TABLE IF NOT EXISTS runs(id INTEGER PRIMARY KEY, ts TEXT, n_charges INT, n_flagged INT, est_total REAL);
        CREATE TABLE IF NOT EXISTS recs(id INTEGER PRIMARY KEY, run_id INT, identified TEXT, tx_date TEXT, vendor TEXT,
          amount REAL, type TEXT, reason TEXT, score REAL, est_impact REAL, status TEXT DEFAULT 'New',
          accepted_on TEXT, implemented_on TEXT, actual_savings REAL, notes TEXT DEFAULT '');
        CREATE INDEX IF NOT EXISTS ix_recs_status ON recs(status);""")

def get(k):
    init_db()
    with _c() as c:
        r = c.execute("SELECT v FROM settings WHERE k=?", (k,)).fetchone()
    return r["v"] if r else DEFAULTS.get(k, "")

def set_(k, v):
    init_db()
    with _c() as c: c.execute("INSERT OR REPLACE INTO settings VALUES(?,?)", (k, str(v)))

def is_enabled(): return get("pro_enabled") == "1" and get("consent") == "1"

def require_pro():
    if not is_enabled():
        raise ProDisabled("Advanced features are off, or storage consent has not been given (Advanced tab).")

def set_api_key(k):
    global _SESSION_KEY; _SESSION_KEY = k or ""

def save_run(flags, n_charges):
    require_pro(); now = datetime.now().isoformat(timespec="seconds")
    with _c() as c:
        cur = c.execute("INSERT INTO runs(ts,n_charges,n_flagged,est_total) VALUES(?,?,?,?)",
                        (now, n_charges, len(flags), float(flags.est_impact.sum()) if len(flags) else 0.0))
        rid = cur.lastrowid
        for _, r in flags.iterrows():
            c.execute("INSERT INTO recs(run_id,identified,tx_date,vendor,amount,type,reason,score,est_impact) VALUES(?,?,?,?,?,?,?,?,?)",
                      (rid, now, str(r.date)[:10], r.vendor, float(r.amount), r.type, r.reason, float(r.score), float(r.est_impact)))
    return rid

def list_recs(status=None):
    require_pro()
    q, a = "SELECT * FROM recs", ()
    if status and status != "All": q, a = q + " WHERE status=?", (status,)
    with _c() as c: return [dict(r) for r in c.execute(q + " ORDER BY id DESC", a)]

def update_rec(rec_id, status, actual=None, notes=""):
    require_pro()
    if status not in STATUSES: raise ValueError(f"Unknown status: {status}")
    today = datetime.now().date().isoformat()
    with _c() as c:
        r = c.execute("SELECT * FROM recs WHERE id=?", (rec_id,)).fetchone()
        if not r: raise ValueError(f"No recommendation with id {rec_id}")
        acc = r["accepted_on"] or (today if status in ("Accepted", "Planned", "In Progress", "Implemented") else None)
        imp = r["implemented_on"] or (today if status == "Implemented" else None)
        c.execute("UPDATE recs SET status=?, accepted_on=?, implemented_on=?, actual_savings=COALESCE(?,actual_savings), notes=? WHERE id=?",
                  (status, acc, imp, actual, notes or r["notes"], rec_id))

def impact_summary():
    """Accepting a recommendation is NOT counted as savings. 'Reported actual' is what the user entered after
    implementation; VERITIQ does not verify it against ledger data."""
    require_pro()
    with _c() as c:
        rows = [dict(r) for r in c.execute("SELECT * FROM recs")]
    acc = [r for r in rows if r["status"] in ("Accepted", "Planned", "In Progress", "Implemented")]
    imp = [r for r in rows if r["status"] == "Implemented"]
    rep = [r for r in imp if r["actual_savings"] is not None]
    proj_rep = sum(r["est_impact"] for r in rep); act = sum(r["actual_savings"] for r in rep)
    return {"identified_total": round(sum(r["est_impact"] for r in rows), 2), "n_recs": len(rows),
            "accepted_est": round(sum(r["est_impact"] for r in acc), 2), "n_implemented": len(imp),
            "pending_est": round(sum(r["est_impact"] for r in acc if r["status"] != "Implemented"), 2),
            "projected_for_reported": round(proj_rep, 2), "reported_actual": round(act, 2),
            "difference": round(act - proj_rep, 2), "n_with_outcome": len(rep)}

def history():
    require_pro()
    with _c() as c: return [dict(r) for r in c.execute("SELECT * FROM runs ORDER BY id DESC")]

def export_csv():
    require_pro(); recs = list_recs()
    buf = io.StringIO()
    if recs:
        w = csv.DictWriter(buf, fieldnames=list(recs[0].keys())); w.writeheader(); w.writerows(recs)
    return buf.getvalue()

def clear_workspace():
    require_pro()
    with _c() as c: c.execute("DELETE FROM recs"); c.execute("DELETE FROM runs")

# ---- Optional LLM rewording (OpenAI-compatible endpoint: OpenAI, Groq, Ollama /v1) ----
def _chat(messages):
    if get("llm_enabled") != "1": raise RuntimeError("LLM explanations are turned off.")
    req = urllib.request.Request(get("llm_endpoint").rstrip("/") + "/chat/completions",
        data=json.dumps({"model": get("llm_model"), "messages": messages, "max_tokens": 120}).encode(),
        headers={"Content-Type": "application/json", **({"Authorization": f"Bearer {_SESSION_KEY}"} if _SESSION_KEY else {})})
    with urllib.request.urlopen(req, timeout=20) as r:
        return json.load(r)["choices"][0]["message"]["content"].strip()

def llm_test():
    try: _chat([{"role": "user", "content": "Reply with the word OK."}]); return "Connected."
    except Exception as e: return f"Connection failed: {e}"

def llm_reword(vendor, amount, reason):
    """Sends ONLY vendor, amount and the rule-based reason to the configured endpoint. Falls back to the original reason on any error."""
    try:
        return _chat([{"role": "system", "content": "Rewrite in one plain sentence for a small-business owner. Use only the facts given. Do not add numbers."},
                      {"role": "user", "content": f"{vendor}, ${amount:,.2f}: {reason}"}])
    except Exception:
        return reason
