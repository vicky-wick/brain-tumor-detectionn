import os
import json
import traceback
import warnings

warnings.filterwarnings("ignore")

import streamlit as st
import torch
from transformers import AutoImageProcessor, SwinForImageClassification
from PIL import Image

try:
    from groq import Groq
    _GROQ_AVAILABLE = True
except ImportError:
    _GROQ_AVAILABLE = False

# ─────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────
st.set_page_config(
    page_title="NeuroScan AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────
#  CUSTOM CSS  — forced light/dark override
# ─────────────────────────────────────────
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700;900&family=IBM+Plex+Mono:wght@400;700&display=swap');

/* ── Global reset ── */
html, body { background: #f5f5f0 !important; }

.stApp {
    background: #f5f5f0 !important;
    font-family: 'Space Grotesk', sans-serif !important;
}

/* Kill streamlit padding */
.block-container { padding-top: 2rem !important; }

/* ── Force text black everywhere ── */
.stApp, .stApp * {
    color: #111111 !important;
    font-family: 'Space Grotesk', sans-serif !important;
}

/* ── Hide streamlit chrome ── */
#MainMenu, footer, header {visibility: hidden !important;}

/* ─── HERO HEADER ─── */
.ns-hero {
    background: #5d3fd3;
    border: 6px solid #111;
    box-shadow: 12px 12px 0 #111;
    padding: 36px 24px;
    text-align: center;
    margin-bottom: 40px;
}
.ns-hero h1 {
    color: #ffffff !important;
    font-size: clamp(2.4rem, 6vw, 4.2rem) !important;
    font-weight: 900 !important;
    text-transform: uppercase !important;
    letter-spacing: -3px !important;
    line-height: 1 !important;
    margin: 0 !important;
}
.ns-hero p {
    color: #d4c8ff !important;
    font-size: 1.1rem !important;
    margin: 8px 0 0 !important;
}

/* ─── CARD ─── */
.ns-card {
    background: #ffffff;
    border: 4px solid #111;
    box-shadow: 8px 8px 0 #111;
    padding: 28px;
    margin-bottom: 28px;
}
.ns-card h3 {
    font-size: 1.1rem !important;
    font-weight: 900 !important;
    text-transform: uppercase !important;
    letter-spacing: 2px !important;
    border-bottom: 3px solid #111;
    padding-bottom: 10px;
    margin: 0 0 16px !important;
}

/* ─── FILE UPLOADER FIX ─── */
[data-testid="stFileUploadDropzone"] {
    background-color: #ffffff !important;
    border: 3px dashed #5d3fd3 !important;
    color: #111 !important;
}
[data-testid="stFileUploadDropzone"] * { color: #111 !important; }
[data-testid="stFileUploaderFileName"] { color: #111 !important; }

/* Upload label from st.file_uploader */
[data-testid="stWidgetLabel"] { color: #111 !important; }

/* ─── BUTTONS ─── */
.stButton > button {
    background: #111 !important;
    color: #fff !important;
    border: 4px solid #111 !important;
    border-radius: 0 !important;
    font-weight: 900 !important;
    font-size: 1rem !important;
    text-transform: uppercase !important;
    letter-spacing: 1px !important;
    padding: 14px 28px !important;
    box-shadow: 5px 5px 0 #5d3fd3 !important;
    transition: transform .1s, box-shadow .1s !important;
    width: 100% !important;
}
.stButton > button:hover {
    background: #5d3fd3 !important;
    transform: translate(3px, 3px) !important;
    box-shadow: 2px 2px 0 #5d3fd3 !important;
}

/* ─── RESULT BANNER ─── */
.ns-result {
    border: 6px solid #111;
    padding: 36px;
    text-align: center;
    box-shadow: 12px 12px 0 #111;
    margin-bottom: 28px;
}
.ns-result.pos { background: #ff4742; }
.ns-result.neg { background: #00e676; }
.ns-result h1 {
    color: #111 !important;
    font-size: clamp(1.8rem, 5vw, 3rem) !important;
    font-weight: 900 !important;
    text-transform: uppercase !important;
    margin: 0 0 8px !important;
}
.ns-result h2 {
    color: #111 !important;
    font-size: 1.5rem !important;
    font-weight: 700 !important;
    margin: 0 !important;
}

/* ─── PROBABILITY BARS ─── */
.ns-stat {
    display: flex;
    justify-content: space-between;
    align-items: center;
    background: #fff;
    border: 2px solid #111;
    padding: 10px 18px;
    margin: 6px 0;
    font-family: 'IBM Plex Mono', monospace !important;
    font-weight: 700;
}
.ns-stat span { color: #111 !important; }
.ns-stat .bar-fill {
    height: 6px;
    background: #5d3fd3;
    margin-top: 4px;
    border: 1px solid #111;
}

/* ─── DIAGNOSIS BOX ─── */
.ns-diag {
    background: #ffeb3b;
    border: 5px solid #111;
    box-shadow: 10px 10px 0 #111;
    padding: 32px;
    margin-bottom: 28px;
    font-size: 1.05rem;
    line-height: 1.7;
}
.ns-diag h3 {
    color: #111 !important;
    font-weight: 900 !important;
    text-transform: uppercase !important;
    letter-spacing: 2px !important;
    border-bottom: 3px solid #111;
    padding-bottom: 8px;
    margin-bottom: 16px !important;
}
.ns-diag p, .ns-diag li { color: #111 !important; }

/* ─── DOWNLOAD BTN ─── */
.stDownloadButton > button {
    background: #fff !important;
    color: #111 !important;
    border: 4px solid #111 !important;
    border-radius: 0 !important;
    font-weight: 900 !important;
    box-shadow: 5px 5px 0 #111 !important;
}
.stDownloadButton > button:hover {
    background: #f5f5f0 !important;
    transform: translate(3px, 3px) !important;
    box-shadow: 2px 2px 0 #111 !important;
}

/* ─── FOOTER ─── */
.ns-footer {
    background: #111;
    border: 4px solid #5d3fd3;
    box-shadow: 8px 8px 0 #5d3fd3;
    padding: 18px;
    text-align: center;
    margin-top: 60px;
}
.ns-footer span {
    color: #ffffff !important;
    font-weight: 900 !important;
    letter-spacing: 3px !important;
    text-transform: uppercase !important;
    font-size: 1rem !important;
}

/* ─── SPINNER ─── */
[data-testid="stSpinner"] { color: #111 !important; }
</style>
""",
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────
#  MODEL LOADER
# ─────────────────────────────────────────
@st.cache_resource(show_spinner="Initializing Neural Engine...")
def load_model():
    path = os.getcwd()
    processor = AutoImageProcessor.from_pretrained(path, local_files_only=True)
    model = SwinForImageClassification.from_pretrained(path, local_files_only=True)
    model.eval()
    with open(os.path.join(path, "config.json")) as f:
        cfg = json.load(f)
    return processor, model, cfg.get("id2label", {})


# ─────────────────────────────────────────
#  GROQ ADVICE
# ─────────────────────────────────────────
def ai_diagnosis(tumor_type: str) -> str:
    if not _GROQ_AVAILABLE:
        return "⚠️ Groq library not installed. Run `pip install groq`."
    try:
        groq_client = Groq(api_key="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")
        prompt = (
            f"MRI result: **{tumor_type.replace('_', ' ').title()}**. "
            "Provide: 1) What this condition is, 2) Typical symptoms, "
            "3) Common medical management/medications. "
            "Be professional, under 200 words, and end with a disclaimer."
        )
        resp = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are an expert neurologist providing clinical summaries."},
                {"role": "user", "content": prompt},
            ],
            model="llama-3.3-70b-versatile",
        )
        return resp.choices[0].message.content
    except Exception as exc:
        return f"AI insight unavailable: {exc}"


# ─────────────────────────────────────────
#  HERO
# ─────────────────────────────────────────
st.markdown(
    '<div class="ns-hero"><h1>NeuroScan AI</h1>'
    '<p>Advanced Brain Tumor Classification System</p></div>',
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────
#  LOAD MODEL
# ─────────────────────────────────────────
try:
    processor, model, id2label = load_model()
except Exception:
    st.error("⛔ Could not load model. Make sure config.json and pytorch_model.bin are present.")
    st.code(traceback.format_exc())
    st.stop()

# ─────────────────────────────────────────
#  UPLOAD SECTION
# ─────────────────────────────────────────
st.markdown('<div class="ns-card"><h3>📁 Upload MRI Scan</h3>', unsafe_allow_html=True)
uploaded = st.file_uploader(
    label="Upload MRI image (JPG/PNG)",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed",
)
st.markdown('</div>', unsafe_allow_html=True)

if uploaded is None:
    # Prompt to upload — no empty cards
    st.info("↑ Upload an axial T1-weighted brain MRI image to begin classification.")
    st.markdown(
        '<div class="ns-footer"><span>Made by Devanshu Mahato</span></div>',
        unsafe_allow_html=True,
    )
    st.stop()

# ─────────────────────────────────────────
#  IMAGE PREVIEW + RUN BUTTON
# ─────────────────────────────────────────
img = Image.open(uploaded).convert("RGB")

col_img, col_ctrl = st.columns([1, 1], gap="large")

with col_img:
    st.markdown('<div class="ns-card"><h3>🖼 Uploaded Scan</h3>', unsafe_allow_html=True)
    st.image(img, use_column_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col_ctrl:
    st.markdown('<div class="ns-card"><h3>⚙️ Run Classifier</h3>', unsafe_allow_html=True)
    st.write("Current Engine: **NeuroScan v1.0** (96.47% accuracy)")
    st.write("Classes: Glioma · Meningioma · No Tumor · Pituitary")
    run = st.button("RUN DIAGNOSTIC ▶")
    st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────
#  INFERENCE
# ─────────────────────────────────────────
if run:
    with st.spinner("Analyzing neural patterns…"):
        inputs = processor(images=img, return_tensors="pt")
        with torch.no_grad():
            logits = model(**inputs).logits
        idx = logits.argmax(-1).item()
        label = id2label.get(str(idx), "unknown")
        probs = torch.nn.functional.softmax(logits, dim=-1)[0]
        conf = probs[idx].item()
        all_probs = {id2label.get(str(i), f"Class {i}"): p.item() for i, p in enumerate(probs)}
        advice = ai_diagnosis(label)

    st.session_state.update(
        label=label, conf=conf, all_probs=all_probs, advice=advice
    )

# ─────────────────────────────────────────
#  SHOW RESULTS (if available in session)
# ─────────────────────────────────────────
if "label" in st.session_state:
    label = st.session_state["label"]
    conf = st.session_state["conf"]
    all_probs = st.session_state["all_probs"]
    advice = st.session_state["advice"]

    is_tumor = label != "no_tumor"
    banner_cls = "pos" if is_tumor else "neg"

    # ── Result Banner ──
    st.markdown(
        f'<div class="ns-result {banner_cls}">'
        f'<h1>{label.replace("_", " ").upper()}</h1>'
        f'<h2>CONFIDENCE: {conf:.1%}</h2>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ── Probability Matrix ──
    st.markdown('<div class="ns-card"><h3>📊 Probability Matrix</h3>', unsafe_allow_html=True)
    for lbl, prob in sorted(all_probs.items(), key=lambda x: -x[1]):
        pct = int(prob * 100)
        st.markdown(
            f'<div class="ns-stat">'
            f'<span>{lbl.replace("_", " ").title()}</span>'
            f'<span>{prob:.2%}</span>'
            f'</div>'
            f'<div style="height:6px;background:#5d3fd3;width:{pct}%;border:1px solid #111;margin-bottom:2px;"></div>',
            unsafe_allow_html=True,
        )
    st.markdown('</div>', unsafe_allow_html=True)

    # ── AI Physician Insight ──
    st.markdown(
        f'<div class="ns-diag"><h3>🩺 Clinical Insight (AI)</h3>{advice}</div>',
        unsafe_allow_html=True,
    )

    # ── Download Report ──
    report = (
        "══════════════════════════════════════\n"
        "         NEUROSCAN AI REPORT\n"
        "     Developed by Devanshu Mahato\n"
        "══════════════════════════════════════\n\n"
        f"DIAGNOSIS   : {label.replace('_', ' ').upper()}\n"
        f"CONFIDENCE  : {conf:.2%}\n\n"
        "── PROBABILITY MATRIX ──\n"
    )
    for lbl, prob in sorted(all_probs.items(), key=lambda x: -x[1]):
        report += f"  {lbl:<25} {prob:.2%}\n"
    report += f"\n── CLINICAL INSIGHT ──\n{advice}\n\n"
    report += "DISCLAIMER: AI-generated for research only. Consult a licensed neurologist.\n"

    st.download_button(
        label="📥  DOWNLOAD MEDICAL REPORT",
        data=report,
        file_name=f"NeuroScan_{label}.txt",
        mime="text/plain",
    )

# ─────────────────────────────────────────
#  FOOTER
# ─────────────────────────────────────────
st.markdown(
    '<div class="ns-footer"><span>Made by Devanshu Mahato</span></div>',
    unsafe_allow_html=True,
)
