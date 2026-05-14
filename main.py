import cv2
import os
import numpy as np
import streamlit as st
from PIL import Image
from keras.applications import MobileNetV2
from keras.applications.mobilenet_v2 import preprocess_input, decode_predictions


# ── Model helpers (unchanged) ─────────────────────────────────────────────────

def load_model():
    try:
        model = MobileNetV2(weights="imagenet")
        return model
    except OSError:
        keras_cache_dir = os.path.expanduser("~/.keras/models")
        for file in os.listdir(keras_cache_dir):
            if "mobilenet_v2" in file:
                os.remove(os.path.join(keras_cache_dir, file))
        model = MobileNetV2(weights="imagenet")
        return model


def preprocess_image(image):
    image = np.array(image)
    image = cv2.resize(image, (224, 224))
    image = preprocess_input(image)
    image = np.expand_dims(image, axis=0)
    return image


def classify_image(model, image):
    preds = model.predict(image)
    decoded_preds = decode_predictions(preds, top=3)[0]
    return decoded_preds


def main():
    st.set_page_config(
        page_title="Image Classifier",
        page_icon="🔍",
        layout="centered",
    ) 

    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght@0,9..144,300;0,9..144,600;0,9..144,700;1,9..144,300&family=DM+Mono:wght@400;500&family=Manrope:wght@300;400;500;600&display=swap');

    :root {
        --ink:       #1a1a1a;
        --ink-soft:  #5a5a5a;
        --ink-faint: #9a9a9a;
        --rule:      #e4e0d8;
        --cream:     #faf8f4;
        --paper:     #ffffff;
        --signal:    #c8470a;
        --radius:    10px;
    }

    html, body, [class*="css"] {
        font-family: 'Manrope', sans-serif;
        background: var(--cream) !important;
        color: var(--ink) !important;
    }

    #MainMenu, footer, header { visibility: hidden; }
    .block-container {
        max-width: 680px !important;
        padding: 3.5rem 1.5rem 5rem !important;
    }

    .masthead {
        border-bottom: 1.5px solid var(--ink);
        padding-bottom: 1rem;
        margin-bottom: 2.2rem;
    }
    .masthead-label {
        font-family: 'DM Mono', monospace;
        font-size: 0.65rem;
        letter-spacing: 0.18em;
        text-transform: uppercase;
        color: var(--ink-faint);
        margin-bottom: 0.45rem;
    }
    .masthead h1 {
        font-family: 'Fraunces', serif !important;
        font-size: 2.6rem !important;
        font-weight: 700 !important;
        line-height: 1.05 !important;
        letter-spacing: -0.02em;
        color: var(--ink) !important;
        margin: 0 !important;
    }
    .masthead h1 em {
        font-style: italic;
        font-weight: 300;
        color: var(--signal);
    }
    .masthead-sub {
        font-size: 0.88rem;
        color: var(--ink-soft);
        margin-top: 0.55rem;
        font-weight: 400;
        line-height: 1.55;
    }

    [data-testid="stFileUploader"] > div:first-child {
        background: var(--paper) !important;
        border: 1.5px solid var(--rule) !important;
        border-radius: var(--radius) !important;
        padding: 1.4rem !important;
        transition: border-color 0.2s;
    }
    [data-testid="stFileUploader"] > div:first-child:hover {
        border-color: var(--ink-soft) !important;
    }
    [data-testid="stFileUploader"] label {
        color: var(--ink-soft) !important;
        font-size: 0.75rem !important;
    }

    [data-testid="stImage"] img {
        border-radius: var(--radius) !important;
        border: 1px solid var(--rule) !important;
        width: 100% !important;
    }
    [data-testid="stImage"] figcaption { display: none !important; }

    .stButton > button {
        background: var(--ink) !important;
        color: var(--cream) !important;
        font-family: 'Manrope', sans-serif !important;
        font-weight: 500 !important;
        font-size: 0.75rem !important;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        border: none !important;
        border-radius: var(--radius) !important;
        padding: 0.7rem 2rem !important;
        width: 100%;
        transition: background 0.18s, transform 0.1s !important;
    }
    .stButton > button:hover {
        background: #333 !important;
        transform: translateY(-1px) !important;
    }
    .stButton > button:active { transform: translateY(0) !important; }

    .results-label {
        font-family: 'DM Mono', monospace;
        font-size: 0.62rem;
        letter-spacing: 0.2em;
        text-transform: uppercase;
        color: var(--ink-faint);
        margin: 1.8rem 0 0.9rem;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid var(--rule);
    }

    .pred-row {
        display: flex;
        align-items: center;
        gap: 1rem;
        padding: 0.85rem 0;
        border-bottom: 1px solid var(--rule);
    }
    .pred-row:last-child { border-bottom: none; }
    .pred-num {
        font-family: 'DM Mono', monospace;
        font-size: 0.7rem;
        color: var(--ink-faint);
        width: 1.4rem;
        flex-shrink: 0;
    }
    .pred-label {
        flex: 1;
        font-size: 0.98rem;
        font-weight: 500;
        color: var(--ink);
        text-transform: capitalize;
    }
    .pred-row.top .pred-label { font-weight: 600; color: var(--signal); }
    .pred-bar-wrap {
        width: 100px;
        height: 4px;
        background: var(--rule);
        border-radius: 99px;
        flex-shrink: 0;
    }
    .pred-bar-fill {
        height: 100%;
        border-radius: 99px;
        background: var(--ink);
    }
    .pred-row.top .pred-bar-fill { background: var(--signal); }
    .pred-score {
        font-family: 'DM Mono', monospace;
        font-size: 0.78rem;
        color: var(--ink-soft);
        width: 3rem;
        text-align: right;
        flex-shrink: 0;
    }
    .pred-row.top .pred-score { color: var(--signal); font-weight: 500; }

    [data-testid="stSpinner"] p { color: var(--ink-soft) !important; }
    </style>
    """, unsafe_allow_html=True)

  
    st.markdown("""
    <div class="masthead">
        <div class="masthead-label">Computer Vision · Image Recognition · Top-3 Results</div>
        <h1>Image <em>Classifier</em></h1>
        <p class="masthead-sub">Upload a photo — get three smart predictions under a second.</p>
    </div>
    """, unsafe_allow_html=True)

    # Cache the model loading to speed up subsequent classifications
    @st.cache_resource
    def load_cached_model():
        return load_model()

 # file uploader with preview
    uploaded_file = st.file_uploader(
        "Upload an image (JPEG, PNG)",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, use_container_width=True)

        st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)
        btn = st.button("Classify Image")

        if btn:
            model = load_cached_model()
            with st.spinner("Analyzing..."):
                processed = preprocess_image(image)
                predictions = classify_image(model, processed)

            if predictions:
                st.markdown('<div class="results-label">Predictions</div>', unsafe_allow_html=True)

                for i, (_, label, score) in enumerate(predictions):
                    top_cls   = "top" if i == 0 else ""
                    num       = f"0{i+1}"
                    pct       = score * 100
                    bar_width = int(pct)
                    pretty    = label.replace("_", " ")

                    st.markdown(f"""
                    <div class="pred-row {top_cls}">
                        <span class="pred-num">{num}</span>
                        <span class="pred-label">{pretty}</span>
                        <div class="pred-bar-wrap">
                            <div class="pred-bar-fill" style="width:{bar_width}%"></div>
                        </div>
                        <span class="pred-score">{pct:.1f}%</span>
                    </div>
                    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()