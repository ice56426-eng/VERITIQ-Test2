"""VERITIQ demo app (Gradio). Run: python app.py
Standard use stores nothing. Advanced tab turns on opt-in Pro features (see pro.py)."""
import tempfile, gradio as gr, pandas as pd
import pro
from veritiq import make_data, train_final_model, flag_transactions, cards_html

MODEL = train_final_model()
DISCLAIMER = "Prototype, not financial advice. Model trained on synthetic data; flags are leads to review, not verdicts."
COLS = ["id", "tx_date", "vendor", "amount", "type", "est_impact", "status", "actual_savings", "reason"]

SHOW = ["date", "vendor", "amount", "score", "type", "est_impact", "reason"]

def run(file, use_demo, sens):
    try:
        raw = make_data()[["date", "vendor", "amount"]] if (use_demo or file is None) else pd.read_csv(file.name)
        flags = flag_transactions(raw, MODEL, threshold=float(sens))
    except Exception as e:   # readable message instead of a crash
        return pd.DataFrame(), "", f"Could not check this file: {e}"
    note = ""
    if pro.is_enabled():
        if pro.get("llm_enabled") == "1":   # reword top 5 only; sends vendor, amount, reason to your endpoint
            for i in flags.index[:5]:
                flags.loc[i, "reason"] = pro.llm_reword(flags.loc[i, "vendor"], flags.loc[i, "amount"], flags.loc[i, "reason"])
        pro.save_run(flags, len(raw)); note = " Saved to your Pro workspace."
    warn = " ".join(flags.attrs.get("warnings", []))
    out = flags[SHOW].copy(); out["score"] = out.score.round(2); out["date"] = out.date.astype(str).str[:10]
    return out, cards_html(flags), f"{len(raw)} charges checked, {len(flags)} flagged.{note} {warn} " + DISCLAIMER

def tracker(status):
    try:
        recs = pro.list_recs(status); s = pro.impact_summary()
    except pro.ProDisabled as e:
        return pd.DataFrame(columns=COLS), str(e)
    df = pd.DataFrame(recs, columns=COLS) if recs else pd.DataFrame(columns=COLS)
    md = (f"**Identified (estimated):** ${s['identified_total']:,.2f} across {s['n_recs']} flags  \n"
          f"**Accepted/planned/in progress/implemented (estimated, NOT realized):** ${s['accepted_est']:,.2f}  \n"
          f"**Implemented:** {s['n_implemented']}  ·  **Pending estimate:** ${s['pending_est']:,.2f}  \n"
          f"**Projected vs reported actual** (for {s['n_with_outcome']} items with an outcome): "
          f"${s['projected_for_reported']:,.2f} vs ${s['reported_actual']:,.2f} (difference ${s['difference']:,.2f}).  \n"
          "_Estimates are not guaranteed. 'Reported actual' is what you typed in; VERITIQ does not verify it._")
    return df, md

def update(rec_id, status, actual, notes):
    try:
        pro.update_rec(int(rec_id), status, None if actual in (None, "") else float(actual), notes or "")
        return "Updated."
    except Exception as e:
        return f"Could not update: {e}"

def history():
    try: return pd.DataFrame(pro.history())
    except pro.ProDisabled as e: return pd.DataFrame({"message": [str(e)]})

def save_settings(enabled, consent, llm_on, endpoint, model, key):
    pro.set_("pro_enabled", "1" if enabled else "0"); pro.set_("consent", "1" if consent else "0")
    pro.set_("llm_enabled", "1" if llm_on else "0"); pro.set_("llm_endpoint", endpoint); pro.set_("llm_model", model)
    pro.set_api_key(key)
    if enabled and not consent: return "Saved, but persistent storage stays OFF until you tick the consent box."
    return "Settings saved. Pro features are " + ("ON." if pro.is_enabled() else "OFF (nothing is stored).")

def export():
    try:
        f = tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False); f.write(pro.export_csv()); f.close(); return f.name
    except pro.ProDisabled: return None

def clear(confirm):
    if not confirm: return "Tick the confirmation box first."
    try: pro.clear_workspace(); return "Workspace cleared (settings kept)."
    except pro.ProDisabled as e: return str(e)

with gr.Blocks(title="VERITIQ", theme=gr.themes.Base(primary_hue="purple")) as demo:
    gr.Image("assets/logo.png", height=200, show_label=False, interactive=False)
    with gr.Tab("Check charges"):
        gr.Markdown("Upload a CSV with `date, vendor, amount`, or use synthetic demo data. **Standard mode: nothing is saved.**")
        f = gr.File(label="Transactions CSV", file_types=[".csv"]); d = gr.Checkbox(label="Use synthetic demo data", value=True)
        sens = gr.Slider(0.1, 0.8, value=0.2, step=0.05, label="Sensitivity threshold: lower = catches more but more false alarms (0.2 matches our test)")
        b = gr.Button("Check for suspicious charges", variant="primary")
        m = gr.Markdown(); cards = gr.HTML(); t = gr.Dataframe(label="All flagged charges"); b.click(run, [f, d, sens], [t, cards, m])
    with gr.Tab("Tracker (Pro)"):
        flt = gr.Dropdown(["All"] + pro.STATUSES, value="All", label="Filter by status"); rb = gr.Button("Refresh")
        tt = gr.Dataframe(label="Recommendations"); sm = gr.Markdown(); rb.click(tracker, flt, [tt, sm])
        gr.Markdown("### Update a recommendation")
        rid = gr.Number(label="ID", precision=0); st = gr.Dropdown(pro.STATUSES, value="Accepted", label="Status")
        act = gr.Number(label="Actual savings you measured (optional, set after implementing)"); nt = gr.Textbox(label="Notes")
        ub = gr.Button("Update"); um = gr.Markdown(); ub.click(update, [rid, st, act, nt], um)
    with gr.Tab("History (Pro)"):
        hb = gr.Button("Load past runs"); ht = gr.Dataframe(); hb.click(history, None, ht)
    with gr.Tab("Advanced"):
        gr.Markdown("## Advanced (Pro) features\nOff by default. Turning them on stores your results in a local file (`veritiq_pro.db`) on this computer.")
        e = gr.Checkbox(label="Enable Pro features (persistent memory, history, tracker)", value=pro.get("pro_enabled") == "1")
        c = gr.Checkbox(label="I consent to saving flagged charges and my decisions locally", value=pro.get("consent") == "1")
        gr.Markdown("### Optional: AI-reworded explanations\nSends only vendor, amount and the rule-based reason of the top 5 flags to the endpoint you enter. "
                    "Works with OpenAI-compatible endpoints (OpenAI, Groq, local Ollama). Your key is kept in memory only and never saved. If it fails, the original explanation is used.")
        lo = gr.Checkbox(label="Enable AI-reworded explanations", value=pro.get("llm_enabled") == "1")
        ep = gr.Textbox(label="Endpoint", value=pro.get("llm_endpoint")); md = gr.Textbox(label="Model", value=pro.get("llm_model"))
        ky = gr.Textbox(label="API key (not needed for local Ollama)", type="password")
        sv = gr.Button("Save settings", variant="primary"); tc = gr.Button("Test AI connection"); sm2 = gr.Markdown()
        sv.click(save_settings, [e, c, lo, ep, md, ky], sm2); tc.click(lambda: pro.llm_test(), None, sm2)
        gr.Markdown("### Your data")
        eb = gr.Button("Export history (CSV)"); ef = gr.File(); eb.click(export, None, ef)
        cf = gr.Checkbox(label="Yes, permanently delete all saved runs and recommendations"); cb = gr.Button("Clear workspace"); cm = gr.Markdown(); cb.click(clear, cf, cm)
        gr.Markdown("### Not implemented\nGoogle Drive / Sheets / Gmail, scheduled reports and alerts, Claude/Gemini-native adapters. These are roadmap items, not working features.")

if __name__ == "__main__":
    demo.launch()
