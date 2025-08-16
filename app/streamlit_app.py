# app/streamlit_app.py
import os
import io
import shutil
from pathlib import Path
from typing import Tuple, List

import streamlit as st
import requests
import pandas as pd
import altair as alt

# ---------- Optional libs (graceful fallbacks) ----------
try:
    import fitz  # PyMuPDF (PDF text extraction)
except Exception:
    fitz = None

try:
    import pdfplumber  # PDF table extraction
except Exception:
    pdfplumber = None

try:
    from PIL import Image, ImageOps, ImageFilter  # used only for OCR cleanup
except Exception:
    Image = None
    ImageOps = None
    ImageFilter = None

try:
    import pytesseract  # OCR
except Exception:
    pytesseract = None

try:
    from moviepy.editor import VideoFileClip  # video->audio / frames
except Exception:
    VideoFileClip = None

# for writing PNG from numpy frames without PIL
try:
    import imageio.v2 as imageio
except Exception:
    imageio = None

try:
    import openai  # optional: Whisper transcription
except Exception:
    openai = None


# ---------- Configure external tool paths (Tesseract) ----------
# Ensure pytesseract knows where the tesseract binary is (macOS/Homebrew path included)
try:
    if pytesseract is not None:
        TESS_CMD = (
            os.getenv("TESSERACT_CMD")
            or shutil.which("tesseract")
            or "/opt/homebrew/bin/tesseract"  # common on Apple Silicon
        )
        try:
            from pytesseract import pytesseract as _pt
            _pt.tesseract_cmd = TESS_CMD
        except Exception:
            pytesseract.pytesseract.tesseract_cmd = TESS_CMD  # type: ignore
except Exception:
    pass  # keep OCR optional; don't crash if binding fails


# ---------- App config / state ----------
API_BASE = os.getenv("API_BASE", "http://127.0.0.1:8000")

# Sticky Pinecone namespace across reruns/tabs
if "pc_namespace" not in st.session_state:
    st.session_state.pc_namespace = "snowflake"


def _inject_css():
    st.markdown(
        """
        <style>
        /* CSS Variables for consistent theming */
        :root {
          --accent: #7C3AED;
          --bg: #0F1117;
          --bg-2: #161A23;
          --stroke: #232734;
          --text: #E6E6E6;
          --muted: #A8B0BF;
          --appbar-h: 64px;
          --maxw: 1200px;
          --success: #00D26A;
          --error: #F44336;
        }

        /* Base styles */
        html, body {
          background: var(--bg);
          color: var(--text);
          font-size: 16px;
          line-height: 1.5;
          font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
          -webkit-font-smoothing: antialiased;
          -moz-osx-font-smoothing: grayscale;
        }

        /* Prevent layout shift */
        .main {
          overflow-x: hidden;
        }

        /* Main container with space for fixed header */
        .block-container {
          max-width: var(--maxw);
          padding-top: calc(var(--appbar-h) + 2rem) !important;
          padding-bottom: 3rem;
          margin: 0 auto;
        }

        /* Hide Streamlit's default header */
        header[data-testid="stHeader"],
        div[data-testid="stToolbar"],
        #MainMenu,
        footer {
          display: none !important;
        }

        /* Fixed app bar */
        .appbar {
          position: fixed;
          top: 0;
          left: 0;
          right: 0;
          height: var(--appbar-h);
          z-index: 999;
          background: var(--bg);
          border-bottom: 1px solid var(--stroke);
          backdrop-filter: blur(10px);
          -webkit-backdrop-filter: blur(10px);
        }

        .appbar-inner {
          max-width: var(--maxw);
          height: 100%;
          margin: 0 auto;
          padding: 0 2rem;
          display: flex;
          align-items: center;
          gap: 1.5rem;
        }

        .brand {
          font-weight: 700;
          font-size: 1.125rem;
          color: var(--text);
          display: flex;
          align-items: center;
          white-space: nowrap;
        }

        .brand .emoji {
          margin-right: 0.5rem;
          font-size: 1.25rem;
        }

        .brand-sub {
          color: var(--muted);
          font-size: 0.875rem;
          font-weight: 400;
          margin-left: 1rem;
        }

        .right-chips {
          margin-left: auto;
          display: flex;
          flex-wrap: wrap;
          gap: 0.5rem;
        }

        .pill-kpi {
          display: inline-flex;
          align-items: center;
          gap: 0.5rem;
          border: 1px solid var(--stroke);
          border-radius: 999px;
          padding: 0.375rem 0.75rem;
          font-size: 0.8125rem;
          background: var(--bg-2);
          color: var(--text);
          transition: all 0.2s ease;
        }

        .pill-kpi:hover {
          border-color: var(--accent);
          transform: translateY(-1px);
        }

        .pill-kpi strong {
          color: var(--accent);
          font-weight: 600;
        }

        /* Tabs styling */
        .stTabs [data-baseweb="tab-list"] {
          gap: 0.5rem;
          background: transparent;
          border-bottom: 1px solid var(--stroke);
          padding-bottom: 0;
        }

        .stTabs [data-baseweb="tab"] {
          background: var(--bg-2);
          border: 1px solid var(--stroke);
          border-radius: 10px 10px 0 0;
          padding: 0.5rem 1rem;
          color: var(--text);
          font-weight: 500;
          transition: all 0.2s ease;
        }

        .stTabs [data-baseweb="tab"]:hover {
          background: rgba(124, 58, 237, 0.1);
        }

        .stTabs [aria-selected="true"] {
          background: var(--accent);
          border-color: var(--accent);
          color: white;
        }

        /* Cards */
        .section-card {
          background: var(--bg-2);
          border: 1px solid var(--stroke);
          border-radius: 12px;
          padding: 1.25rem;
          margin: 1rem 0;
          transition: all 0.2s ease;
        }

        .section-card:hover {
          border-color: rgba(124, 58, 237, 0.3);
        }

        .answer-card {
          background: linear-gradient(135deg, rgba(124, 58, 237, 0.1) 0%, rgba(124, 58, 237, 0.05) 100%);
          border: 1px solid rgba(124, 58, 237, 0.4);
          box-shadow: 0 4px 12px rgba(124, 58, 237, 0.1);
        }

        /* Buttons */
        div.stButton > button {
          background: var(--accent);
          color: white;
          border: none;
          border-radius: 8px;
          padding: 0.625rem 1.25rem;
          font-weight: 600;
          font-size: 0.9375rem;
          transition: all 0.2s ease;
          box-shadow: 0 2px 8px rgba(124, 58, 237, 0.25);
        }

        div.stButton > button:hover {
          background: #6B2FD6;
          transform: translateY(-2px);
          box-shadow: 0 4px 12px rgba(124, 58, 237, 0.35);
        }

        div.stButton > button:active {
          transform: translateY(0);
        }

        /* Input fields */
        input, textarea, select {
          font-size: 1rem !important;
          background: var(--bg-2) !important;
          border: 1px solid var(--stroke) !important;
          color: var(--text) !important;
          border-radius: 8px !important;
          transition: all 0.2s ease !important;
        }

        input:focus, textarea:focus, select:focus {
          border-color: var(--accent) !important;
          box-shadow: 0 0 0 2px rgba(124, 58, 237, 0.1) !important;
        }

        /* Sliders */
        .stSlider > div > div {
          background: var(--stroke);
        }

        .stSlider > div > div > div {
          background: var(--accent);
        }

        /* Checkboxes */
        .stCheckbox > label {
          color: var(--text) !important;
        }

        /* Expanders */
        .streamlit-expanderHeader {
          background: var(--bg-2);
          border: 1px solid var(--stroke);
          border-radius: 8px;
          color: var(--text) !important;
        }

        .streamlit-expanderHeader:hover {
          border-color: var(--accent);
        }

        /* Images */
        .stImage > img {
          border-radius: 8px;
          box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
        }

        /* Dataframes */
        .dataframe {
          border: 1px solid var(--stroke) !important;
          border-radius: 8px !important;
          overflow: hidden;
        }

        /* Success/Error/Warning messages */
        .stAlert {
          border-radius: 8px;
          border: 1px solid;
        }

        div[data-baseweb="notification"] {
          border-radius: 8px;
        }

        /* Code blocks */
        .stCodeBlock {
          border-radius: 8px;
          border: 1px solid var(--stroke);
        }

        /* Footer */
        .footer {
          color: var(--muted);
          font-size: 0.875rem;
          text-align: center;
          margin-top: 3rem;
          padding-top: 2rem;
          border-top: 1px solid var(--stroke);
        }

        /* Responsive design */
        @media (max-width: 768px) {
          .appbar-inner {
            padding: 0 1rem;
          }
          
          .brand-sub {
            display: none;
          }
          
          .right-chips {
            gap: 0.25rem;
          }
          
          .pill-kpi {
            font-size: 0.75rem;
            padding: 0.25rem 0.5rem;
          }
        }

        /* Loading spinner */
        .stSpinner > div {
          border-color: var(--accent) !important;
        }

        /* Animations */
        @keyframes fadeIn {
          from { opacity: 0; transform: translateY(10px); }
          to { opacity: 1; transform: translateY(0); }
        }

        .section-card {
          animation: fadeIn 0.3s ease;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _header():
    try:
        cfg = requests.get(f"{API_BASE}/config", timeout=2).json()
        llm = cfg.get("llm_backend", "—")
        emb = cfg.get("embeddings_backend", "—")
        has = "✅" if cfg.get("has_key") else "❌"
    except Exception:
        llm, emb, has = "—", "—", "❌"

    st.markdown(
        f"""
        <div class="appbar">
          <div class="appbar-inner">
            <div class="brand">
              <span class="emoji">🤖</span>
              GenAI Doc Assistant
              <span class="brand-sub">Ask your docs · Summaries · Tables · Charts</span>
            </div>
            <div class="right-chips">
              <span class="pill-kpi">API: <strong>{API_BASE}</strong></span>
              <span class="pill-kpi">LLM: <strong>{llm}</strong></span>
              <span class="pill-kpi">Emb: <strong>{emb}</strong></span>
              <span class="pill-kpi">OpenAI key: <strong>{has}</strong></span>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.title("GenAI Doc Assistant")


# ---------- Utilities ----------
def _sanitize_name(name: str) -> str:
    return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in name)


def _fetch_image_bytes_from_url(url: str, max_mb: int = 15) -> bytes:
    """Download an image from a DIRECT image URL and validate content-type/size."""
    headers = {"User-Agent": "Mozilla/5.0 (compatible; GenAI-Doc-Assistant; +local)"}
    r = requests.get(url, headers=headers, timeout=15, allow_redirects=True)
    r.raise_for_status()
    ctype = (r.headers.get("content-type") or "").lower()
    if not ctype.startswith("image/"):
        raise ValueError(
            f"URL is not a direct image (content-type: {ctype}). "
            "Paste a URL that ends with .png / .jpg / .jpeg / .gif / .webp."
        )
    if len(r.content) > max_mb * 1024 * 1024:
        raise ValueError(f"Image larger than {max_mb} MB.")
    return r.content


def _vision_summary(img_bytes: bytes, prompt: str) -> str:
    """Call the backend /describe_image for a vision summary."""
    files = {"file": ("image.png", img_bytes, "image/png")}
    data = {"prompt": prompt}
    r = requests.post(f"{API_BASE}/describe_image", data=data, files=files, timeout=120)
    r.raise_for_status()
    out = r.json()
    return str(out.get("answer", "")).strip()


# ---- PDF text (PyMuPDF) ----
def _extract_pdf_text(file_bytes: bytes) -> str:
    if not fitz:
        return ""
    out: List[str] = []
    try:
        with fitz.open(stream=io.BytesIO(file_bytes), filetype="pdf") as doc:
            for p in doc:
                out.append(p.get_text() or "")
    except Exception:
        return ""
    return "\n".join(out).strip()


# ---- OCR scanned PDF (render page -> OCR) ----
def _ocr_pdf_text(file_bytes: bytes) -> str:
    if not (fitz and Image and pytesseract):
        return ""
    parts: List[str] = []
    try:
        with fitz.open(stream=io.BytesIO(file_bytes), filetype="pdf") as doc:
            for page in doc:
                # increased from 200 to 300 DPI for better OCR accuracy
                pm = page.get_pixmap(dpi=300)
                img = Image.frombytes("RGB", [pm.width, pm.height], pm.samples)
                parts.append(pytesseract.image_to_string(img))
    except Exception:
        return ""
    return "\n".join(parts).strip()


# ---- PDF structured (text + tables) ----
def _extract_pdf_structured(file_bytes: bytes) -> Tuple[str, List[pd.DataFrame]]:
    """Extract plain text + tables. Robust to ragged rows & duplicate headers."""
    text = _extract_pdf_text(file_bytes)
    tables: List[pd.DataFrame] = []
    if not pdfplumber:
        return text, tables

    try:
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for page in pdf.pages:
                raw = page.extract_tables() or []
                for tbl in raw:
                    if not tbl or len(tbl) < 2:
                        continue
                    w = max(len(r) for r in tbl)
                    rows = [(r + [""] * (w - len(r))) for r in tbl]
                    header, body = rows[0], rows[1:]

                    cols, seen = [], {}
                    for h in header:
                        h = (h or "").strip() or "col"
                        seen[h] = seen.get(h, 0) + 1
                        cols.append(f"{h}_{seen[h]}" if seen[h] > 1 else h)

                    df = pd.DataFrame(body, columns=cols)

                    # Conservative numeric coercion per column
                    for i in range(df.shape[1]):
                        s = df.iloc[:, i].astype("string")
                        s_clean = (
                            s.str.replace(",", "", regex=False)
                             .str.replace("%", "", regex=False)
                             .str.strip()
                        )
                        num = pd.to_numeric(s_clean, errors="coerce")
                        if num.notna().sum() >= max(1, int(0.6 * len(num))):
                            df.iloc[:, i] = num
                        else:
                            df.iloc[:, i] = s

                    df = df.dropna(how="all", axis=1).dropna(how="all", axis=0)
                    if not df.empty:
                        tables.append(df)
    except Exception:
        pass

    return text.strip(), tables


# ---- Image OCR ----
def _extract_image_text(file_bytes: bytes) -> str:
    if not (Image and pytesseract and ImageOps and ImageFilter):
        return ""
    import tempfile
    try:
        img = Image.open(io.BytesIO(file_bytes))
        # Normalize + denoise for OCR
        img = img.convert("L")
        img = ImageOps.autocontrast(img)
        img = img.filter(ImageFilter.SHARPEN)

        with tempfile.NamedTemporaryFile(suffix=".png", delete=True) as tmp:
            img.save(tmp.name, format="PNG")
            norm = Image.open(tmp.name)
            # PSM 6 = assume a block of text (good default for screenshots)
            return pytesseract.image_to_string(norm, config="--psm 6")
    except Exception:
        try:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=True) as tmp:
                tmp.write(file_bytes); tmp.flush()
                norm = Image.open(tmp.name).convert("L")
                norm = ImageOps.autocontrast(norm)
                return pytesseract.image_to_string(norm, config="--psm 6")
        except Exception:
            return ""


# ---- Audio transcription via OpenAI Whisper (optional) ----
def _transcribe_audio_via_openai(wav_path: Path) -> str:
    if openai is None or not os.getenv("OPENAI_API_KEY"):
        return ""
    try:
        # OpenAI Python SDK v1+
        from openai import OpenAI  # type: ignore
        client = OpenAI()
        with open(wav_path, "rb") as f:
            tr = client.audio.transcriptions.create(model="whisper-1", file=f)
        return getattr(tr, "text", "").strip() or str(tr)
    except Exception:
        try:
            # Fallback for older client versions
            with open(wav_path, "rb") as f:
                tr = openai.audio.transcriptions.create(model="whisper-1", file=f)  # type: ignore
            return getattr(tr, "text", "").strip() or str(tr)
        except Exception:
            return ""


# ---- Video text extraction (audio transcription) ----
def _extract_video_text(file_bytes: bytes, filename: str) -> str:
    """Extract audio from video and transcribe it."""
    if not VideoFileClip:
        return ""
    
    # Create temp directory if needed
    tmp = Path(".streamlit_tmp")
    tmp.mkdir(exist_ok=True)
    
    # Save video temporarily
    video_path = tmp / _sanitize_name(filename)
    audio_path = tmp / f"{_sanitize_name(Path(filename).stem)}.wav"
    
    try:
        # Write video bytes to temp file
        video_path.write_bytes(file_bytes)
        
        # Extract audio from video
        try:
            clip = VideoFileClip(str(video_path))
            if clip.audio is None:
                clip.close()
                return ""
            
            # Write audio to WAV file
            clip.audio.write_audiofile(
                str(audio_path),
                logger=None,  # suppress MoviePy output
                verbose=False,
                ffmpeg_params=["-ac", "1", "-ar", "16000"]  # mono, 16kHz for Whisper
            )
            clip.close()
        except Exception as e:
            try:
                clip.close()
            except Exception:
                pass
            return ""
        
        # Transcribe the audio
        transcript = _transcribe_audio_via_openai(audio_path)
        
        # If OpenAI transcription fails or is unavailable, return empty
        if not transcript:
            return ""
            
        return transcript
        
    except Exception as e:
        return ""
    finally:
        # Clean up temp files
        try:
            video_path.unlink(missing_ok=True)
            audio_path.unlink(missing_ok=True)
        except Exception:
            pass


# ---- Video midframe extraction (for visual summary) ----
def _video_midframe_png(file_bytes: bytes, filename: str, at_ratio: float = 0.5) -> bytes:
    """Extract a mid-frame from a video and return as PNG bytes (uses MoviePy + imageio)."""
    if not VideoFileClip or not imageio:
        return b""
    tmp = Path(".streamlit_tmp"); tmp.mkdir(exist_ok=True)
    vp = tmp / _sanitize_name(filename)
    vp.write_bytes(file_bytes)
    try:
        clip = VideoFileClip(str(vp))
        t = (clip.duration or 0) * at_ratio if clip.duration else 0
        frame = clip.get_frame(t)  # numpy array (H, W, 3)
        clip.close()
    except Exception:
        try:
            clip.close()
        except Exception:
            pass
        vp.unlink(missing_ok=True)
        return b""
    # encode to PNG
    buf = io.BytesIO()
    try:
        imageio.imwrite(buf, frame, format="png")
        png_bytes = buf.getvalue()
    except Exception:
        png_bytes = b""
    finally:
        try:
            vp.unlink(missing_ok=True)
        except Exception:
            pass
    return png_bytes


# ---- MAIN EXTRACTION ORCHESTRATOR ----
def _extract_text(upload, do_ocr: bool = False) -> Tuple[str, str, List[pd.DataFrame]]:
    """Returns (text, info, tables)."""
    if upload is None:
        return "", "No file uploaded.", []
    suffix = (Path(upload.name).suffix or "").lower()
    data = upload.read()

    if suffix == ".pdf":
        text2, tables = _extract_pdf_structured(data)
        if do_ocr and not text2.strip():
            ocr_text = _ocr_pdf_text(data)
            if ocr_text.strip():
                text2 = ocr_text
        if not text2.strip():
            text2 = _extract_pdf_text(data)
        return text2, "PDF extracted", tables

    if suffix in {".png", ".jpg", ".jpeg"}:
        return _extract_image_text(data), "Image OCR", []

    if suffix in {".txt", ".md"}:
        try:
            return data.decode("utf-8", errors="ignore"), "Text loaded", []
        except Exception:
            return data.decode("latin-1", errors="ignore"), "Text loaded (latin1)", []

    if suffix in {".mp4", ".mov", ".m4v"}:
        return _extract_video_text(data, upload.name), "Video transcribed", []

    try:
        return data.decode("utf-8", errors="ignore"), "Raw bytes decoded", []
    except Exception:
        return "", "Unknown file type", []


# ---- Display helpers ----
def _show_table_and_chart(df: pd.DataFrame, title: str = ""):
    if df is None or df.empty:
        return
    if title:
        st.markdown(f"**{title}**")
    st.dataframe(df, use_container_width=True)

    # simple chart if we have numeric + another column
    num_cols = df.select_dtypes(include="number").columns
    if len(num_cols) >= 1 and len(df.columns) >= 2:
        cats = [c for c in df.columns if c not in num_cols]
        if cats:
            x_col = cats[0]
        else:
            df = df.reset_index(names="index")
            x_col = "index"
        y_col = num_cols[0]
        chart = (
            alt.Chart(df)
            .mark_bar()
            .encode(x=alt.X(x_col, sort=None), y=y_col, tooltip=list(df.columns))
            .properties(height=280)
        )
        st.altair_chart(chart, use_container_width=True)


# ---------- Page Configuration ----------
st.set_page_config(page_title="GenAI Doc Assistant", page_icon="🤖", layout="wide")
_inject_css()
_header()

# Prevent initial layout shift
st.markdown('<div style="height: calc(var(--appbar-h) + 1rem);"></div>', unsafe_allow_html=True)

# ---------- Main Tabs ----------
tab_ask, tab_upload, tab_photo = st.tabs(
    ["🤖 Ask", "📥 Upload (PDF/Image/Text/Video)", "🖼️ Photo Summary"]
)

# =============== ASK TAB ===================
with tab_ask:
    st.caption("Ask the knowledge base using FAISS or Pinecone.")

    c1, c2, c3 = st.columns([2, 1, 1])

    with c1:
        q = st.text_input("Your question", placeholder="e.g., How do I rotate Snowflake keys?")

    with c2:
        use_pine = st.checkbox("Use Pinecone", value=True)  # default ON

    with c3:
        topk = st.slider("Top-K", 1, 10, 8)

    ns = st.text_input(
        "Pinecone namespace (optional)",
        value=st.session_state.pc_namespace,
        key="pc_namespace_ask",
    )
    if ns.strip():
        st.session_state.pc_namespace = ns.strip()

    if st.button("Ask", type="primary"):
        if not q.strip():
            st.warning("Please enter a question.")
            st.stop()
        try:
            if use_pine:
                payload = {"question": q, "top_k": topk}
                if ns.strip():
                    payload["namespace"] = ns.strip()
                url = f"{API_BASE}/ask_pinecone"
            else:
                payload = {"question": q, "k": topk}
                url = f"{API_BASE}/ask"

            with st.spinner("Thinking…"):
                r = requests.post(url, json=payload, timeout=120)
                r.raise_for_status()
                data = r.json()

            st.markdown("#### Answer")
            st.markdown('<div class="section-card answer-card">', unsafe_allow_html=True)
            st.write(data.get("answer", "(no answer)"))
            st.markdown("</div>", unsafe_allow_html=True)

            with st.expander("Sources"):
                for s in (data.get("sources") or []):
                    st.code(s)
        except Exception as e:
            st.error(f"Error: {e}")

    st.divider()
    if st.button("Rebuild FAISS KB from data/raw (ingest)"):
        with st.spinner("Ingesting…"):
            r = requests.post(f"{API_BASE}/ingest", timeout=120)
            if r.ok:
                st.success(r.json())
            else:
                st.error(r.text)

# =============== UPLOAD TAB ===================
with tab_upload:
    st.caption("Upload a document to summarize and (optionally) index to Pinecone or FAISS.")
    do_ocr = st.checkbox("Try OCR for scanned PDFs (slower)", value=False)
    upload = st.file_uploader(
        "Upload a file",
        type=["pdf", "png", "jpg", "jpeg", "txt", "md", "mp4", "mov", "m4v"],
        accept_multiple_files=False,
    )

    if upload is not None:
        # keep a copy of the raw bytes for vision/video fallbacks
        try:
            raw_bytes = upload.getvalue()
        except Exception:
            raw_bytes = None

        ext = (Path(upload.name).suffix or "").lower()
        text, info, tables = _extract_text(upload, do_ocr=do_ocr)

        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown(f"**Detected:** {info}  •  **Chars:** {len(text)}")
        st.markdown("</div>", unsafe_allow_html=True)

        # Helpful note if short
        too_short = len((text or "").strip()) < 20
        if too_short:
            st.info(
                "Not enough text for the summarizer (needs ≥ 20 chars). "
                "For scanned PDFs, enable **'Try OCR for scanned PDFs'** and re-upload."
            )

        # Vision fallback for IMAGES when OCR yields little/none
        if ext in {".png", ".jpg", ".jpeg"} and too_short and raw_bytes:
            if st.button("Use vision to describe this image"):
                try:
                    ans = _vision_summary(
                        raw_bytes,
                        "Describe the image. If there are charts or tables, summarize the key numbers accurately."
                    )
                    st.markdown("#### Vision Description")
                    st.markdown('<div class="section-card answer-card">', unsafe_allow_html=True)
                    st.write(ans or "(no answer)")
                    st.markdown("</div>", unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Vision describe failed: {e}")

        # Vision fallback for VIDEOS when no/short transcript
        if ext in {".mp4", ".mov", ".m4v"} and too_short and raw_bytes:
            st.caption("No/short audio transcript detected. You can still get a visual description from a keyframe.")
            if st.button("Visual summary of a keyframe"):
                if not VideoFileClip or not imageio:
                    st.error("Video/encoding libraries not available. Install moviepy + imageio + imageio-ffmpeg.")
                else:
                    frame_png = _video_midframe_png(raw_bytes, upload.name, at_ratio=0.5)
                    if not frame_png:
                        st.error("Couldn't extract a keyframe (check ffmpeg install).")
                    else:
                        try:
                            ans = _vision_summary(
                                frame_png,
                                "Describe this video frame. If text or metrics are visible, quote accurate numbers."
                            )
                            st.markdown("#### Visual Keyframe Summary")
                            st.markdown('<div class="section-card answer-card">', unsafe_allow_html=True)
                            st.write(ans or "(no answer)")
                            st.markdown("</div>", unsafe_allow_html=True)
                        except Exception as e:
                            st.error(f"Keyframe summary failed: {e}")

        if tables:
            st.subheader(f"Detected tables: {len(tables)}")
            for i, df in enumerate(tables, 1):
                _show_table_and_chart(df, title=f"Table {i}")

        if text.strip():
            with st.expander("Preview extracted text", expanded=False):
                st.text_area("text", value=text[:20000], height=240)

            # ---- Summarize ----
            st.subheader("Summarize")
            max_words = st.slider("Max words", 50, 600, 250, 25)

            if st.button("Summarize this document"):
                txt = (text or "").strip()
                try:
                    with st.spinner("Summarizing…"):
                        r = requests.post(
                            f"{API_BASE}/summarize",
                            json={"text": txt, "max_words": int(max_words)},
                            timeout=180,
                        )
                        r.raise_for_status()
                        data = r.json()

                    st.markdown("#### Summary")
                    st.markdown('<div class="section-card answer-card">', unsafe_allow_html=True)
                    st.write(data.get("summary", ""))
                    st.markdown("</div>", unsafe_allow_html=True)

                    if data.get("bullets"):
                        st.markdown("**Highlights:**")
                        for b in data["bullets"]:
                            st.markdown(f"- {b}")

                    # Download summary as Markdown
                    md = (
                        f"# Summary\n\n{data.get('summary','')}\n\n## Highlights\n"
                        + "\n".join(f"- {b}" for b in data.get("bullets", []))
                    )
                    st.download_button(
                        "⬇️ Download summary (.md)",
                        data=md.encode("utf-8"),
                        file_name=f"{_sanitize_name(Path(upload.name).stem)}_summary.md",
                        mime="text/markdown",
                    )

                    if data.get("table_csv"):
                        from io import StringIO
                        try:
                            df_csv = pd.read_csv(StringIO(data["table_csv"]))
                            _show_table_and_chart(df_csv, title="Table from summary")
                        except Exception:
                            pass

                except requests.HTTPError as e:
                    body = getattr(e.response, "text", "")
                    st.error(f"Summarize error: {e}\n\n{body}")
                except Exception as e:
                    st.error(f"Summarize error: {e}")

            st.divider()

            # ---- Index to Pinecone ----
            st.subheader("Index to Pinecone")
            ns2 = st.text_input(
                "Namespace (optional)",
                value=st.session_state.pc_namespace,
                key="pc_namespace_upsert",
            )
            # keep the namespace sticky across tabs/reruns
            st.session_state.pc_namespace = ns2.strip() or st.session_state.pc_namespace

            if st.button("Index this file to Pinecone"):
                try:
                    chunks = [text[i:i + 1200] for i in range(0, len(text), 1100)]
                    base_id = _sanitize_name(Path(upload.name).stem)
                    ids = [f"{base_id}-{i}" for i in range(len(chunks))]
                    payload = {"texts": chunks, "ids": ids}
                    if ns2.strip():
                        payload["namespace"] = ns2.strip()
                    with st.spinner("Indexing…"):
                        r = requests.post(f"{API_BASE}/upsert_pinecone", json=payload, timeout=240)
                        r.raise_for_status()
                        st.success("Indexed to Pinecone ✅")
                        st.json(r.json())
                except Exception as e:
                    st.error(f"Pinecone upsert error: {e}")

            # ---- Save to local KB (FAISS) ----
            st.subheader("Save to local KB (FAISS)")
            if st.button("Save text to data/raw and Rebuild KB"):
                try:
                    raw_dir = Path("data/raw"); raw_dir.mkdir(parents=True, exist_ok=True)
                    out_path = raw_dir / f"{_sanitize_name(Path(upload.name).stem)}.txt"
                    out_path.write_text(text)
                    with st.spinner("Ingesting…"):
                        r = requests.post(f"{API_BASE}/ingest", timeout=120)
                        r.raise_for_status()
                        st.success(f"Saved to {out_path} and ingested")
                        st.json(r.json())
                except Exception as e:
                    st.error(f"Save/ingest error: {e}")
        else:
            st.warning("Could not extract any text from this file.")
    else:
        st.info("Upload a file to begin.")

# =============== PHOTO SUMMARY TAB ===================
with tab_photo:
    st.caption("Upload a photo or paste a DIRECT image URL to get a concise, accurate description.")

    src = st.radio("Image source", ["Upload", "From URL"], horizontal=True)
    img_bytes: bytes = b""

    if src == "Upload":
        up = st.file_uploader(
            "Choose a photo",
            type=["png", "jpg", "jpeg", "gif", "webp"],
            accept_multiple_files=False,
        )
        if up is not None:
            img_bytes = up.read()
    else:
        url = st.text_input("Image URL (must be a direct link ending in .png/.jpg/.jpeg/.gif/.webp)")
        if url:
            try:
                img_bytes = _fetch_image_bytes_from_url(url)
            except Exception as e:
                st.error(f"Could not fetch image from URL: {e}")

    # Preview if we have bytes
    if img_bytes:
        try:
            st.image(img_bytes, use_container_width=True)
        except Exception:
            st.warning("Fetched data is not a valid image. Make sure the URL points directly to an image file.")

    prompt = st.text_input(
        "Instruction (optional)",
        value="Describe the image. If there are charts or tables, summarize the key numbers accurately.",
    )

    colA, colB = st.columns([1, 1])
    with colA:
        go = st.button("Summarize photo", type="primary")
    with colB:
        clr = st.button("Clear")

    if clr:
        try:
            st.rerun()
        except Exception:
            st.experimental_rerun()

    if go:
        if not img_bytes:
            st.warning("Please upload or fetch a valid image first.")
        else:
            try:
                files = {"file": ("image.png", img_bytes, "image/png")}
                data = {"prompt": prompt}
                with st.spinner("Analyzing photo…"):
                    r = requests.post(f"{API_BASE}/describe_image", data=data, files=files, timeout=120)
                    r.raise_for_status()
                    out = r.json()
                st.markdown("#### Photo Summary")
                st.markdown('<div class="section-card answer-card">', unsafe_allow_html=True)
                st.write(out.get("answer", "(no answer)"))
                st.markdown("</div>", unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Photo summarize error: {e}")

# ---------- Footer ----------
st.markdown(
    '<div class="footer">Built with Streamlit • Altair • PyMuPDF • pdfplumber • FAISS/Pinecone</div>',
    unsafe_allow_html=True,
)