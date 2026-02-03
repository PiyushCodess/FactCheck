import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from src.utils import PredictionPipeline
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from wordcloud import WordCloud


# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="FactCheck - AI Misinformation Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        color: #eee;
    }
    .main-title {
        text-align: center;
        font-size: 2.8em;
        font-weight: bold;
        background: linear-gradient(90deg, #e94560, #0f3460);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding: 20px 0;
    }
    .subtitle {
        text-align: center;
        color: #a8a8b3;
        font-size: 1.1em;
        margin-bottom: 30px;
    }
    .stTextarea textarea {
        background-color: #1e2a3a !important;
        color: #eee !important;
        border: 1px solid #3a4f6b !important;
        border-radius: 10px !important;
        font-size: 1em !important;
    }
    .stTextarea textarea:focus {
        border-color: #e94560 !important;
        box-shadow: 0 0 10px rgba(233,69,96,0.3) !important;
    }
    .result-box-fake {
        background: linear-gradient(135deg, rgba(233,69,96,0.15), rgba(180,30,50,0.25));
        border: 2px solid #e94560;
        border-radius: 16px;
        padding: 30px;
        text-align: center;
        margin: 20px 0;
    }
    .result-box-true {
        background: linear-gradient(135deg, rgba(46,213,115,0.15), rgba(30,150,80,0.25));
        border: 2px solid #2ed573;
        border-radius: 16px;
        padding: 30px;
        text-align: center;
        margin: 20px 0;
    }
    .result-label {
        font-size: 2.5em;
        font-weight: bold;
        margin: 10px 0;
    }
    .confidence-text {
        font-size: 1.2em;
        color: #a8a8b3;
    }
    .feature-card {
        background: rgba(255,255,255,0.05);
        border-radius: 10px;
        padding: 15px;
        margin: 8px 0;
        border: 1px solid rgba(255,255,255,0.1);
    }
    .sidebar-title {
        color: #e94560;
        font-size: 1.3em;
        font-weight: bold;
        margin: 15px 0 8px 0;
    }
    .stButton button {
        background: linear-gradient(135deg, #e94560, #c23152) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 12px 24px !important;
        font-size: 1.1em !important;
        font-weight: bold !important;
        cursor: pointer;
        width: 100%;
        transition: transform 0.2s;
    }
    .stButton button:hover {
        transform: scale(1.02);
        box-shadow: 0 4px 15px rgba(233,69,96,0.4);
    }
    .info-box {
        background: rgba(15,52,96,0.4);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
    .divider {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, #3a4f6b, transparent);
        margin: 20px 0;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# LOAD MODEL (Cached)
# ─────────────────────────────────────────────
@st.cache_resource
def load_pipeline():
    pipeline = PredictionPipeline()

    base = os.path.join(os.path.dirname(__file__), '..', 'models')

    # Try to load models in order of preference (smallest first)
    model_files = [
        'logistic_regression.pkl',
        'random_forest.pkl', 
        'xgboost.pkl',
        'ensemble_model.pkl'
    ]
    
    model_loaded = False
    for model_file in model_files:
        model_path = os.path.join(base, model_file)
        if os.path.exists(model_path):
            try:
                # Check size before loading
                size_mb = os.path.getsize(model_path) / (1024 * 1024)
                if size_mb < 100:  # Only load models under 100MB
                    pipeline.model = joblib.load(model_path)
                    st.sidebar.success(f"✅ Loaded: {model_file.replace('.pkl', '').replace('_', ' ').title()}")
                    model_loaded = True
                    break
            except Exception as e:
                st.sidebar.warning(f"⚠️ Could not load {model_file}: {str(e)}")
                continue
    
    if not model_loaded:
        st.error("❌ No model available. Please train a model first.")
        st.stop()

    # Load TF-IDF vectorizer
    tfidf_path = os.path.join(base, 'tfidf_vectorizer.pkl')
    if os.path.exists(tfidf_path):
        try:
            pipeline.tfidf_vectorizer = joblib.load(tfidf_path)
        except:
            st.warning("⚠️ TF-IDF vectorizer not loaded")

    # Load feature columns
    cols_path = os.path.join(base, 'feature_columns.pkl')
    if os.path.exists(cols_path):
        try:
            pipeline.feature_columns = joblib.load(cols_path)
        except:
            st.warning("⚠️ Feature columns not loaded")

    return pipeline


# ─────────────────────────────────────────────
# HELPER: CONFIDENCE GAUGE
# ─────────────────────────────────────────────
def plot_confidence_gauge(confidence, is_fake):
    fig, ax = plt.subplots(figsize=(6, 3.5))
    fig.patch.set_facecolor('none')
    ax.set_facecolor('none')

    theta = np.linspace(np.pi, 0, 100)
    r = 1.0
    ax.plot(r * np.cos(theta), r * np.sin(theta), color='#3a4f6b', linewidth=22, solid_capstyle='butt')

    conf_angle = np.pi - (confidence / 100) * np.pi
    theta_conf = np.linspace(np.pi, conf_angle, 100)
    color = '#e94560' if is_fake else '#2ed573'
    ax.plot(r * np.cos(theta_conf), r * np.sin(theta_conf), color=color, linewidth=22, solid_capstyle='butt')

    needle_angle = conf_angle
    ax.plot([0, 0.62 * np.cos(needle_angle)], [0, 0.62 * np.sin(needle_angle)], color='white', linewidth=3)
    ax.plot(0, 0, 'o', color='white', markersize=10)

    ax.text(-1.15, -0.18, '0%', color='#a8a8b3', fontsize=10, ha='center')
    ax.text(1.15, -0.18, '100%', color='#a8a8b3', fontsize=10, ha='center')
    ax.text(0, -0.35, f'{confidence:.1f}%', color='white', fontsize=22, ha='center', fontweight='bold')
    ax.text(0, -0.55, 'Confidence', color='#a8a8b3', fontsize=11, ha='center')

    ax.set_xlim(-1.4, 1.4)
    ax.set_ylim(-0.7, 1.15)
    ax.set_aspect('equal')
    ax.axis('off')

    return fig


# ─────────────────────────────────────────────
# HELPER: FEATURE BREAKDOWN CHART
# ─────────────────────────────────────────────
def plot_feature_breakdown(features):
    keys = ['sentiment_compound', 'subjectivity', 'polarity',
            'uppercase_ratio', 'punctuation_ratio', 'lexical_diversity']
    labels = ['Sentiment', 'Subjectivity', 'Polarity',
              'CAPS Ratio', 'Punct. Ratio', 'Lexical Diversity']
    values = [abs(features.get(k, 0)) for k in keys]

    fig, ax = plt.subplots(figsize=(7, 4))
    fig.patch.set_facecolor('none')
    ax.set_facecolor('#1e2a3a')

    bars = ax.barh(labels, values, color='#e94560', edgecolor='none', height=0.5)
    ax.set_xlim(0, max(values) * 1.3 if max(values) > 0 else 1)
    ax.tick_params(colors='white', labelsize=11)
    ax.xaxis.set_tick_params(labelbottom=False)
    for spine in ax.spines.values():
        spine.set_color('#3a4f6b')

    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                f'{val:.3f}', va='center', color='white', fontsize=10)

    ax.set_title('Feature Breakdown', color='white', fontsize=13, fontweight='bold', pad=12)
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-title">🛡️ FactCheck</div>', unsafe_allow_html=True)
    st.markdown('<hr style="border-color:#3a4f6b">', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-title">📌 How It Works</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
        <ol style="color:#ccc; font-size:0.9em; padding-left:18px;">
            <li>Paste a news <b>title</b> and <b>article text</b></li>
            <li>The AI extracts <b>linguistic + sentiment</b> features</li>
            <li>An <b>ensemble of ML models</b> classifies the news</li>
            <li>You get a <b>prediction + confidence score</b></li>
        </ol>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-title">⚠️ Disclaimer</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
        <p style="color:#a8a8b3; font-size:0.85em;">
            This tool is for <b>educational and research purposes</b> only. 
            Always verify news from multiple trusted sources. 
            AI predictions are probabilistic and may not be 100% accurate.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-title">🔧 Model Info</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
        <p style="color:#a8a8b3; font-size:0.85em;">
            <b>Dataset:</b> ISOT Fake News Dataset<br>
            <b>Models:</b> Ensemble (LR + RF + XGB + GB + SVM)<br>
            <b>Features:</b> TF-IDF + Sentiment + Linguistic
        </p>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# MAIN PAGE
# ─────────────────────────────────────────────
st.markdown('<div class="main-title">🛡️ FactCheck — AI Misinformation Detector</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Powered by Ensemble Machine Learning & NLP</div>', unsafe_allow_html=True)
st.markdown('<hr class="divider">', unsafe_allow_html=True)

pipeline = load_pipeline()

col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown("### 📰 News Title")
    title_input = st.text_area(
        label="title",
        placeholder="Enter the news headline here...",
        height=70,
        label_visibility="hidden"
    )

    st.markdown("### 📄 News Article")
    text_input = st.text_area(
        label="article",
        placeholder="Paste the full news article text here...",
        height=250,
        label_visibility="hidden"
    )

    analyze_clicked = st.button("🔍 Analyze News", use_container_width=True)

with col2:
    result_placeholder = st.empty()
    gauge_placeholder = st.empty()
    features_placeholder = st.empty()

    if not analyze_clicked:
        result_placeholder.markdown("""
        <div style="
            border: 2px dashed #3a4f6b;
            border-radius: 16px;
            padding: 60px 30px;
            text-align: center;
            margin-top: 40px;
        ">
            <p style="color:#5a6a7a; font-size:1.1em;">
                👈 Enter a news title and article, then click <b>Analyze News</b> to see results here.
            </p>
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# PREDICTION LOGIC
# ─────────────────────────────────────────────
if analyze_clicked:
    if not title_input.strip() and not text_input.strip():
        result_placeholder.error("⚠️ Please enter at least a title or article text.")
    else:
        with st.spinner("🔄 Analyzing..."):
            result = pipeline.predict(
                title=title_input if title_input else "",
                text=text_input if text_input else ""
            )

        is_fake = result['prediction'] == 0
        confidence = result['confidence']
        features = result['features']

        if is_fake:
            result_placeholder.markdown(f"""
            <div class="result-box-fake">
                <div style="font-size:2.8em;">🚨</div>
                <div class="result-label" style="color:#e94560;">FAKE NEWS</div>
                <div class="confidence-text">Likely misinformation detected</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            result_placeholder.markdown(f"""
            <div class="result-box-true">
                <div style="font-size:2.8em;">✅</div>
                <div class="result-label" style="color:#2ed573;">TRUE NEWS</div>
                <div class="confidence-text">Appears to be legitimate news</div>
            </div>
            """, unsafe_allow_html=True)

        fig_gauge = plot_confidence_gauge(confidence, is_fake)
        gauge_placeholder.pyplot(fig_gauge, use_container_width=True)

        fig_features = plot_feature_breakdown(features)
        features_placeholder.pyplot(fig_features, use_container_width=True)

        # ── Detailed Feature Table (below) ──
        st.markdown('<hr class="divider">', unsafe_allow_html=True)
        st.markdown("### 📊 Detailed Feature Analysis")

        feat_col1, feat_col2, feat_col3 = st.columns(3)
        feat_items = list(features.items())

        third = len(feat_items) // 3
        columns = [feat_col1, feat_col2, feat_col3]
        splits = [feat_items[:third], feat_items[third:2*third], feat_items[2*third:]]

        for col, items in zip(columns, splits):
            for key, val in items:
                col.markdown(f"""
                <div class="feature-card">
                    <div style="color:#a8a8b3; font-size:0.8em; text-transform:uppercase;">{key.replace('_',' ')}</div>
                    <div style="color:white; font-size:1.1em; font-weight:bold;">{val:.4f}</div>
                </div>
                """, unsafe_allow_html=True)