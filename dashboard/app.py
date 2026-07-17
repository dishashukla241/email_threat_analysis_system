import os
import re
import email
from email.policy import default
import joblib
import pandas as pd
import numpy as np
import scipy.sparse as sp
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# ==============================================================================
# PAGE CONFIGURATION
# ==============================================================================
st.set_page_config(
    page_title="Email Threat Analysis & Spam Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================================================
# CSS STYLING & SOC DARK THEME INJECTION
# ==============================================================================
st.markdown(
    """
    <style>
    /* Import Premium Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;700&display=swap');

    /* Apply Fonts */
    html, body, [class*="css"], .stMarkdown {
        font-family: 'Outfit', sans-serif;
    }
    
    code, pre, .mono-text {
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* Main Dashboard container styles */
    .reportview-container {
        background: #090D16;
    }

    /* Metric Card Styling */
    .soc-metric-card {
        background-color: #131926;
        border: 1px solid #1E293B;
        border-radius: 10px;
        padding: 18px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2), 0 2px 4px -1px rgba(0, 0, 0, 0.1);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        text-align: left;
        margin-bottom: 12px;
    }
    .soc-metric-card:hover {
        transform: translateY(-3px);
        border-color: #007BFF;
        box-shadow: 0 10px 15px -3px rgba(0, 123, 255, 0.15), 0 4px 6px -2px rgba(0, 123, 255, 0.05);
    }
    .soc-metric-icon {
        font-size: 1.6rem;
        margin-bottom: 8px;
        display: inline-block;
    }
    .soc-metric-label {
        font-size: 0.8rem;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-weight: 600;
        margin-bottom: 4px;
    }
    .soc-metric-value {
        font-size: 1.8rem;
        font-weight: 800;
        color: #F8FAFC;
        line-height: 1.2;
    }
    .soc-metric-footer {
        font-size: 0.75rem;
        color: #64748B;
        margin-top: 6px;
        font-weight: 500;
    }

    /* Cybersecurity Badges */
    .cyber-badge {
        padding: 6px 14px;
        border-radius: 6px;
        font-weight: 700;
        text-transform: uppercase;
        font-size: 0.8rem;
        display: inline-block;
        letter-spacing: 0.06em;
        border: 1px solid transparent;
        text-align: center;
    }
    .cyber-badge-critical {
        background-color: rgba(239, 68, 68, 0.12);
        color: #FF4D4D;
        border-color: rgba(239, 68, 68, 0.3);
        box-shadow: 0 0 10px rgba(239, 68, 68, 0.1);
    }
    .cyber-badge-high {
        background-color: rgba(249, 115, 22, 0.12);
        color: #FFA500;
        border-color: rgba(249, 115, 22, 0.3);
    }
    .cyber-badge-medium {
        background-color: rgba(234, 179, 8, 0.12);
        color: #FFCC00;
        border-color: rgba(234, 179, 8, 0.3);
    }
    .cyber-badge-low {
        background-color: rgba(34, 197, 94, 0.12);
        color: #34C759;
        border-color: rgba(34, 197, 94, 0.3);
    }

    /* Explanation Alert Box */
    .explain-container {
        background-color: #161C2A;
        border-left: 5px solid #007BFF;
        border-radius: 4px;
        padding: 16px;
        margin-top: 15px;
        margin-bottom: 15px;
    }
    .explain-title {
        font-size: 0.85rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #94A3B8;
        margin-bottom: 6px;
    }
    .explain-text {
        font-size: 0.95rem;
        line-height: 1.5;
        color: #F1F5F9;
        font-family: 'JetBrains Mono', monospace;
    }

    /* Indicator checklist styling */
    .indicator-pill {
        display: flex;
        align-items: center;
        justify-content: space-between;
        background-color: #1E2536;
        border: 1px solid #2D3748;
        padding: 8px 12px;
        border-radius: 6px;
        margin-bottom: 8px;
    }
    .indicator-label {
        font-size: 0.85rem;
        color: #E2E8F0;
        font-weight: 500;
    }
    .indicator-val {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.9rem;
        font-weight: 700;
        color: #F8FAFC;
    }
    
    /* Footer styles */
    .soc-footer {
        border-top: 1px solid #1E293B;
        padding: 20px 0;
        margin-top: 40px;
        color: #64748B;
        font-size: 0.85rem;
    }
    
    /* Center title styling */
    .header-badge {
        background-color: rgba(0, 123, 255, 0.1);
        color: #007BFF;
        border: 1px solid rgba(0, 123, 255, 0.3);
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        display: inline-block;
        margin-bottom: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ==============================================================================
# MODEL AND RESOURCE CACHING
# ==============================================================================
@st.cache_resource
def load_ml_models():
    """Loads trained Random Forest classifier and TF-IDF vectorizer."""
    try:
        rf = joblib.load("models/random_forest.pkl")
        tfidf = joblib.load("models/tfidf_vectorizer.pkl")
        return rf, tfidf
    except Exception as e:
        st.warning(f"⚠️ Could not load ML model binary files: {e}. Live threat prediction will run in simulation mode.")
        return None, None

rf_model, tfidf_vectorizer = load_ml_models()

# ==============================================================================
# FEATURE ENGINEERING DICTIONARIES & HELPER FUNCTIONS
# ==============================================================================
SUSPICIOUS_KEYWORDS = [
    "urgent", "verify", "verification", "click", "click here", "login",
    "password", "account", "bank", "update", "confirm", "winner", "won",
    "lottery", "gift", "free", "claim", "limited", "offer", "bonus", "reward",
    "security", "payment", "invoice", "crypto", "bitcoin"
]
URGENCY_WORDS = [
    "urgent", "immediately", "now", "today", "expire", "expired", "warning",
    "important", "action required"
]
MONEY_WORDS = [
    "bank", "payment", "credit", "debit", "refund", "money", "cash",
    "invoice", "bitcoin", "crypto", "prize"
]
SHORTENED_DOMAINS = [
    "bit.ly", "tinyurl", "goo.gl", "ow.ly", "t.co", "buff.ly", "is.gd"
]
SUSPICIOUS_TLDS = [
    ".ru", ".xyz", ".top", ".click", ".work", ".gq", ".tk"
]

# Reputation lists
TRUSTED_DOMAINS = {
    "gmail.com", "googlemail.com", "outlook.com", "hotmail.com", "live.com",
    "yahoo.com", "ymail.com", "icloud.com", "me.com", "microsoft.com",
    "office.com", "apple.com", "github.com", "gitlab.com", "amazon.com",
    "aws.amazon.com", "paypal.com", "linkedin.com", "oracle.com", "adobe.com",
    "zoom.us", "dropbox.com", "proton.me", "protonmail.com", "aol.com",
    "fastmail.com"
}
DISPOSABLE_DOMAINS = {
    "mailinator.com", "guerrillamail.com", "10minutemail.com", "temp-mail.org",
    "tempmail.com", "throwawaymail.com", "fakeinbox.com", "trashmail.com"
}
TRUSTED_TLDS_REP = (".edu", ".gov", ".gov.in", ".ac.in", ".edu.in")
SUSPICIOUS_TLDS_REP = (".ru", ".tk", ".xyz", ".click", ".work", ".top", ".gq", ".zip")

def clean_text(text):
    if not text or pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    words = text.split()
    stop_words = set(ENGLISH_STOP_WORDS)
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

def get_reputation(domain):
    if not domain or pd.isna(domain):
        return "Medium", 60
    domain = str(domain).strip().lower()
    if domain in TRUSTED_DOMAINS:
        return "High", 100
    if domain in DISPOSABLE_DOMAINS:
        return "Very Low", 10
    for tld in TRUSTED_TLDS_REP:
        if domain.endswith(tld):
            return "High", 90
    for tld in SUSPICIOUS_TLDS_REP:
        if domain.endswith(tld):
            return "Low", 25
    return "Medium", 60

def calculate_threat_metrics(spam_probability, kw_count, url_count, ip_count, short_url_count, tld_count, contains_urgency, contains_money, sender_reputation, sender_reputation_score):
    # Rule-based Security Score
    sec_score = (
        kw_count * 8 +
        url_count * 5 +
        ip_count * 5 +
        short_url_count * 8 +
        tld_count * 8 +
        (15 if contains_urgency else 0) +
        (10 if contains_money else 0)
    )
    sec_score = min(max(sec_score, 0.0), 100.0)
    
    # Final Threat Score (70% ML, 20% Security indicators, 10% Sender reputation)
    threat_score = 0.70 * spam_probability + 0.20 * sec_score + 0.10 * sender_reputation_score
    threat_score = round(min(max(threat_score, 0.0), 100.0), 2)
    
    # Threat Level
    if threat_score < 30:
        threat_lvl = "Low"
    elif threat_score < 55:
        threat_lvl = "Medium"
    elif threat_score < 80:
        threat_lvl = "High"
    else:
        threat_lvl = "Critical"
        
    # Blacklist Recommendation
    is_blacklisted = (
        (threat_lvl == "Critical") |
        ((threat_lvl == "High") & (sender_reputation == "Low")) |
        (spam_probability >= 95.0)
    )
    blacklist_rec = "YES" if is_blacklisted else "NO"
    
    # Rationale reasons list
    reasons = []
    if spam_probability >= 90:
        reasons.append(f"High Spam Probability ({spam_probability:.2f}%)")
    if kw_count > 0:
        reasons.append(f"{int(kw_count)} Suspicious Keyword(s)")
    if url_count > 0:
        reasons.append(f"{int(url_count)} URL(s) Found")
    if ip_count > 0:
        reasons.append("IP Address Detected")
    if short_url_count > 0:
        reasons.append("Shortened URL")
    if tld_count > 0:
        reasons.append("Suspicious TLD")
    if contains_urgency:
        reasons.append("Urgent Language")
    if contains_money:
        reasons.append("Financial Language")
    if sender_reputation in ["Low", "Very Low"]:
        reasons.append(f"{sender_reputation} Sender Reputation")
    if threat_lvl in ["High", "Critical"]:
        reasons.append(f"{threat_lvl} Threat Score")
    if not reasons:
        reasons.append("No significant threat indicators")
        
    threat_reason_str = ", ".join(reasons)
    
    return sec_score, threat_score, threat_lvl, blacklist_rec, threat_reason_str

def extract_live_features(subject_val, body_val, attachment_count, has_attachment):
    # Match the clean_text logic from src/02_clean_text.py
    cleaned_body = str(body_val).lower().strip()
    cleaned_body = re.sub(r"\s+", " ", cleaned_body)
    
    # URL and email counts from raw/cleaned body
    url_pattern = r"https?://\S+|www\.\S+"
    email_pattern = r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"
    
    urls_found = re.findall(url_pattern, body_val)
    url_count = len(urls_found)
    
    email_c = len(re.findall(email_pattern, body_val))
    
    # Counts on cleaned_body
    char_count = len(cleaned_body)
    word_count = len(cleaned_body.split())
    
    # Uppercase features (always 0 in training due to premature lowercasing)
    raw_upper_count = sum(1 for c in cleaned_body if c.isupper())
    upper_ratio = (raw_upper_count / char_count) if char_count > 0 else 0.0
    
    raw_digit_count = sum(1 for c in cleaned_body if c.isdigit())
    excl_count = cleaned_body.count("!")
    quest_count = cleaned_body.count("?")
    spec_count = len(re.findall(r"[^\w\s]", cleaned_body))
    lines_c = len(body_val.splitlines())
    
    words_in_raw = cleaned_body.split()
    avg_w_len = round(sum(len(w) for w in words_in_raw) / len(words_in_raw), 2) if len(words_in_raw) > 0 else 0
    
    has_subj = 0 if not subject_val or str(subject_val).strip() == "No Subject" else 1
    has_links = 1 if url_count > 0 else 0
    
    # Security Features
    found_keywords = []
    for word in SUSPICIOUS_KEYWORDS:
        if word in cleaned_body:
            found_keywords.append(word)
    found_keywords = list(set(found_keywords))
    kw_count = len(found_keywords)
    
    # Regex/patterns counts
    ips_found = re.findall(r"(?:\d{1,3}\.){3}\d{1,3}", body_val)
    ip_count = len(ips_found)
    
    short_url_count = sum(domain in cleaned_body for domain in SHORTENED_DOMAINS)
    tld_count = sum(tld in cleaned_body for tld in SUSPICIOUS_TLDS)
    
    # Urgency & Money
    contains_urgency = any(word in cleaned_body for word in URGENCY_WORDS)
    contains_money = any(word in cleaned_body for word in MONEY_WORDS)
    
    # Combine numerical features array exactly matching training sequence
    X_num = np.array([[
        float(url_count),
        float(email_c),
        float(attachment_count),
        float(char_count),
        float(word_count),
        float(raw_upper_count),
        float(upper_ratio),
        float(raw_digit_count),
        float(excl_count),
        float(quest_count),
        float(spec_count),
        float(lines_c),
        float(avg_w_len),
        float(has_subj),
        float(has_links),
        float(has_attachment),
        float(kw_count),
        float(ip_count),
        float(short_url_count),
        float(tld_count),
        float(1.0 if contains_urgency else 0.0),
        float(1.0 if contains_money else 0.0)
    ]])
    
    return X_num, cleaned_body, url_count, urls_found, email_c, char_count, word_count, raw_upper_count, upper_ratio, raw_digit_count, excl_count, quest_count, spec_count, lines_c, avg_w_len, has_subj, has_links, kw_count, found_keywords, ip_count, ips_found, short_url_count, tld_count, contains_urgency, contains_money

# ==============================================================================
# DATA LOADING UTILITY
# ==============================================================================
@st.cache_data
def load_data():
    """
    Loads and preprocesses the threat engine dashboard dataset.
    Searches multiple paths to handle execution context variations gracefully.
    """
    possible_paths = [
        "data/processed/dashboard_dataset.csv",
        "../data/processed/dashboard_dataset.csv",
        os.path.join(os.path.dirname(__file__), "../data/processed/dashboard_dataset.csv"),
        os.path.join(os.path.dirname(__file__), "data/processed/dashboard_dataset.csv"),
    ]
    
    data_path = None
    for path in possible_paths:
        if os.path.exists(path):
            data_path = path
            break
            
    if data_path is None:
        st.error("❌ Dataset dashboard_dataset.csv not found. Please run the backend pipeline to generate processed data.")
        st.stop()
        
    try:
        df = pd.read_csv(data_path)
    except Exception as e:
        st.error(f"❌ Failed to read CSV dataset: {e}")
        st.stop()
        
    # Handle missing values gracefully
    df["Subject"] = df["Subject"].fillna("(No Subject)")
    df["From"] = df["From"].fillna("unknown@unknown_sender.com")
    df["Sender_Domain"] = df["Sender_Domain"].fillna("unknown_sender.com")
    df["Threat_Reason"] = df["Threat_Reason"].fillna("No specific anomalies detected.")
    df["Threat_Level"] = df["Threat_Level"].fillna("Low")
    df["Sender_Reputation"] = df["Sender_Reputation"].fillna("Medium")
    df["Spam_Prediction"] = df["Spam_Prediction"].fillna("Ham")
    df["Blacklist_Recommendation"] = df["Blacklist_Recommendation"].fillna("NO")
    
    # Fill numerical nulls
    numeric_cols = [
        "Spam_Probability", "Security_Score", "Threat_Score", 
        "Sender_Reputation_Score", "Suspicious_Keyword_Count", 
        "URL_Count", "IP_Address_Count", "Shortened_URL_Count", "Suspicious_TLD_Count"
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
            
    # Handle boolean indicators
    df["Contains_Urgency"] = df["Contains_Urgency"].fillna(False).astype(bool)
    df["Contains_Money"] = df["Contains_Money"].fillna(False).astype(bool)
    
    return df

try:
    df_raw = load_data()
except Exception as e:
    st.error(f"Error during initialization: {e}")
    st.stop()

# ==============================================================================
# SIDEBAR FILTERS (DYNAMISM)
# ==============================================================================
st.sidebar.image(
    "https://img.icons8.com/nolan/96/shield.png", 
    width=65
)
st.sidebar.markdown(
    "<h2 style='margin-top:0px; font-weight:700; color:#F8FAFC;'>SOC CONTROLS</h2>", 
    unsafe_allow_html=True
)
st.sidebar.markdown("Filter system records dynamically:")

# Filter: Threat Level
all_threat_levels = ["Critical", "High", "Medium", "Low"]
selected_threat_levels = st.sidebar.multiselect(
    "Threat Level",
    options=all_threat_levels,
    default=all_threat_levels
)

# Filter: Sender Reputation
all_reputations = ["High", "Medium", "Low"]
selected_reputations = st.sidebar.multiselect(
    "Sender Reputation",
    options=all_reputations,
    default=all_reputations
)

# Filter: Spam/Ham Prediction
prediction_options = ["All", "Spam", "Ham"]
selected_prediction = st.sidebar.selectbox(
    "Spam / Ham Prediction",
    options=prediction_options,
    index=0
)

# Filter: Blacklist Recommendation
blacklist_options = ["All", "YES", "NO"]
selected_blacklist = st.sidebar.selectbox(
    "Blacklist Recommendation",
    options=blacklist_options,
    index=0
)

# Sidebar System Health Banner
st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    <div style="background-color: #131926; border: 1px solid #1E293B; padding: 12px; border-radius: 6px;">
        <div style="color: #64748B; font-size: 0.75rem; text-transform: uppercase; font-weight: 700; letter-spacing: 0.05em;">SYSTEM STATS</div>
        <div style="color: #34C759; font-size: 0.85rem; font-weight: 600; margin-top: 4px; display: flex; align-items: center;">
            <span style="height: 8px; width: 8px; background-color: #34C759; border-radius: 50%; display: inline-block; margin-right: 8px; box-shadow: 0 0 8px #34C759;"></span>
            SOC THREAT ENGINE: ACTIVE
        </div>
        <div style="color: #94A3B8; font-size: 0.75rem; margin-top: 4px;">DATABASE COUNT: 4,198</div>
    </div>
    """,
    unsafe_allow_html=True
)

# Apply filters
df_filtered = df_raw.copy()

if selected_threat_levels:
    df_filtered = df_filtered[df_filtered["Threat_Level"].isin(selected_threat_levels)]
else:
    df_filtered = df_filtered.iloc[0:0]

if selected_reputations:
    df_filtered = df_filtered[df_filtered["Sender_Reputation"].isin(selected_reputations)]
else:
    df_filtered = df_filtered.iloc[0:0]

if selected_prediction != "All":
    df_filtered = df_filtered[df_filtered["Spam_Prediction"] == selected_prediction]

if selected_blacklist != "All":
    df_filtered = df_filtered[df_filtered["Blacklist_Recommendation"] == selected_blacklist]

# ==============================================================================
# HEADER
# ==============================================================================
col_title, col_status = st.columns([4, 1])

with col_title:
    st.markdown("<div class='header-badge'>SOC SECURITY SUITE</div>", unsafe_allow_html=True)
    st.markdown(
        "<h1 style='margin:0px; font-weight:800; font-size: 2.3rem; letter-spacing: -0.02em;'>Email Threat Analysis & Spam Detection System</h1>", 
        unsafe_allow_html=True
    )
    st.markdown(
        "<p style='color: #94A3B8; font-size: 1.05rem; margin-top: 4px; margin-bottom: 25px;'>AI-powered Email Security Dashboard using Machine Learning, NLP and Cybersecurity Analytics</p>", 
        unsafe_allow_html=True
    )

with col_status:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        """
        <div style="background-color: #131926; border: 1px solid #1E293B; border-radius: 8px; padding: 10px 15px; text-align: center; float: right;">
            <div style="color: #64748B; font-size: 0.65rem; text-transform: uppercase; font-weight: 700; letter-spacing: 0.05em;">SECURE STAGE</div>
            <div style="color: #007BFF; font-size: 0.95rem; font-weight: 700; margin-top: 2px;">MONITORING MODE</div>
        </div>
        """,
        unsafe_allow_html=True
    )

# ==============================================================================
# KPI METRICS SECTION
# ==============================================================================
total_count = len(df_filtered)
spam_count = len(df_filtered[df_filtered["Spam_Prediction"] == "Spam"])
ham_count = len(df_filtered[df_filtered["Spam_Prediction"] == "Ham"])

# Safety check for empty means
avg_spam_prob = df_filtered["Spam_Probability"].mean() if total_count > 0 else 0
avg_threat_score = df_filtered["Threat_Score"].mean() if total_count > 0 else 0
blacklist_count = len(df_filtered[df_filtered["Blacklist_Recommendation"] == "YES"])

# ==============================================================================
# MAIN TABS
# ==============================================================================
tab_live, tab_analytics, tab_investigation = st.tabs([
    "LIVE THREAT DETECTOR",
    "THREAT ANALYTICS",
    "EMAIL INVESTIGATION & FORENSICS"
])

plotly_layout_opts = dict(
    paper_bgcolor='rgba(19, 25, 38, 0.5)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(color='#E6EDF2', family='Outfit'),
    margin=dict(l=30, r=30, t=50, b=30),
    height=320,
    legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5)
)

# ----------------- TAB 1: LIVE THREAT DETECTOR -----------------
with tab_live:
    st.markdown("<h3 style='font-size:1.3rem; font-weight:700; color:#F8FAFC;'>Real-Time Email Threat Sandbox Analysis</h3>", unsafe_allow_html=True)
    st.markdown("Load a raw email message or enter metadata manually to calculate threat vectors in real-time.")
    
    # Initialize session state for inputs and attachments if not exists
    if "live_sender" not in st.session_state:
        st.session_state.live_sender = "security-update@paypal-verify.ru"
    if "live_subject" not in st.session_state:
        st.session_state.live_subject = "URGENT: Verify your credentials immediately!"
    if "live_body" not in st.session_state:
        st.session_state.live_body = "Dear Customer, we detected a login attempt. Please click here (http://bit.ly/fake-paypal) to verify your account password. Avoid bank account suspension. Transfer money."
    if "attachment_count" not in st.session_state:
        st.session_state.attachment_count = 0
    if "has_attachment" not in st.session_state:
        st.session_state.has_attachment = 0
    if "attachment_metadata" not in st.session_state:
        st.session_state.attachment_metadata = []
        
    live_col1, live_col2 = st.columns([5, 5])
    
    with live_col1:
        st.markdown("<h4 style='font-size:1rem; font-weight:600;'>1. Email Input Parameters</h4>", unsafe_allow_html=True)
        
        input_method = st.radio(
            "Input Method",
            ["Paste Text Manually", "Upload Email File (.eml / .txt)"],
            horizontal=True,
            key="input_method_radio"
        )
        
        if input_method == "Upload Email File (.eml / .txt)":
            uploaded_file = st.file_uploader("Select Email file to scan:", type=["eml", "txt"], key="live_file_uploader")
            if uploaded_file is not None:
                file_id = f"{uploaded_file.name}_{uploaded_file.size}"
                if st.session_state.get("uploaded_file_id") != file_id:
                    st.session_state.uploaded_file_id = file_id
                    try:
                        file_bytes = uploaded_file.read()
                        if uploaded_file.name.endswith(".eml"):
                            msg = email.message_from_bytes(file_bytes, policy=default)
                            st.session_state.live_sender = msg.get("From", "unknown@unknown.com")
                            st.session_state.live_subject = msg.get("Subject", "(No Subject)")
                            
                            # Parse body
                            body_str = ""
                            if msg.is_multipart():
                                body_parts = []
                                for part in msg.walk():
                                    c_type = part.get_content_type()
                                    c_disp = str(part.get("Content-Disposition"))
                                    if c_type == "text/plain" and "attachment" not in c_disp:
                                        body_parts.append(part.get_payload(decode=True).decode(errors="ignore"))
                                body_str = "\n".join(body_parts)
                            else:
                                body_str = msg.get_payload(decode=True).decode(errors="ignore")
                            st.session_state.live_body = body_str
                            
                            # Parse attachments
                            att_count = 0
                            att_metadata = []
                            for part in msg.walk():
                                filename_attr = part.get_filename()
                                if filename_attr:
                                    att_count += 1
                                    c_type = part.get_content_type()
                                    att_metadata.append(f"{filename_attr} ({c_type})")
                            st.session_state.attachment_count = att_count
                            st.session_state.has_attachment = 1 if att_count > 0 else 0
                            st.session_state.attachment_metadata = att_metadata
                        else:
                            text_content = file_bytes.decode(errors="ignore")
                            st.session_state.live_sender = "unknown@unknown.com"
                            st.session_state.live_subject = "(No Subject)"
                            st.session_state.live_body = text_content
                            st.session_state.attachment_count = 0
                            st.session_state.has_attachment = 0
                            st.session_state.attachment_metadata = []
                        st.success(f"✓ Parsed file: {uploaded_file.name}")
                    except Exception as e:
                        st.error(f"Failed parsing file: {e}")
        else:
            # Reset attachments on manual input toggle
            if st.session_state.get("uploaded_file_id") is not None:
                st.session_state.uploaded_file_id = None
                st.session_state.attachment_count = 0
                st.session_state.has_attachment = 0
                st.session_state.attachment_metadata = []
            
        # Input widgets bound to session state
        sender_val = st.text_input("Sender Address (From):", key="live_sender")
        subject_val = st.text_input("Subject:", key="live_subject")
        body_val = st.text_area("Email Body Content:", key="live_body", height=180)
        
        run_analysis = st.button("⚡ Execute Cyber threat Scan", type="primary")
        
    with live_col2:
        st.markdown("<h4 style='font-size:1rem; font-weight:600;'>2. Real-Time Forensic Scan Report</h4>", unsafe_allow_html=True)
        
        if run_analysis:
            # Extract Sender Domain
            sender_domain = "unknown.com"
            if "@" in sender_val:
                sender_domain = sender_val.split("@")[-1].strip().replace(">", "").replace("<", "")
                
            # Get reputation score (Rule-Based Reputation)
            rep_class, rep_score = get_reputation(sender_domain)
            
            # Retrieve attachments
            att_count = st.session_state.attachment_count
            has_att = st.session_state.has_attachment
            
            # Exact training-pipeline feature engineering alignment (unscaled)
            (X_num, cleaned_body, url_count, urls_found, email_c, char_count, word_count, 
             raw_upper_count, upper_ratio, raw_digit_count, excl_count, quest_count, 
             spec_count, lines_c, avg_w_len, has_subj, has_links, kw_count, found_keywords, 
             ip_count, ips_found, short_url_count, tld_count, contains_urgency, 
             contains_money) = extract_live_features(subject_val, body_val, att_count, has_att)
             
            # Run ML model prediction (handles simulation or live model execution)
            prob_val = 0.0
            pred_class = "Ham"
            is_simulated = False
            
            if rf_model is not None and tfidf_vectorizer is not None:
                # Text processing matching tf-idf pipeline
                processed_text = preprocess_text(body_val)
                X_text = tfidf_vectorizer.transform([processed_text])
                
                # Combine sparse matrices
                X_comb = sp.hstack([
                    X_text,
                    sp.csr_matrix(X_num)
                ])
                
                # Predict
                prob_val = float(rf_model.predict_proba(X_comb)[0, 1] * 100)
                pred_class = "Spam" if rf_model.predict(X_comb)[0] == 1 else "Ham"
            else:
                is_simulated = True
                kw_factor = min(kw_count * 15, 60)
                url_factor = 25 if url_count > 0 else 0
                urgency_factor = 15 if contains_urgency else 0
                prob_val = float(kw_factor + url_factor + urgency_factor)
                prob_val = min(max(prob_val, 2.0), 99.8)
                pred_class = "Spam" if prob_val >= 50.0 else "Ham"
                
            # Compute consistent threat metrics via unified helper
            sec_score, threat_score, threat_lvl, blacklist_rec, threat_reason_str = calculate_threat_metrics(
                prob_val, kw_count, url_count, ip_count, short_url_count, tld_count,
                contains_urgency, contains_money, rep_class, rep_score
            )
            
            # Badge styles
            badge_style = "cyber-badge-low"
            if threat_lvl == "Critical":
                badge_style = "cyber-badge-critical"
            elif threat_lvl == "High":
                badge_style = "cyber-badge-high"
            elif threat_lvl == "Medium":
                badge_style = "cyber-badge-medium"
                
            # Render report using Streamlit-native components
            with st.container(border=True):
                # Header
                col_badge, col_status = st.columns([2, 1])
                with col_badge:
                    st.markdown(f'<span class="cyber-badge {badge_style}">{threat_lvl} VERDICT</span>', unsafe_allow_html=True)
                with col_status:
                    st.markdown('<div style="text-align: right; color: #007BFF; font-family: monospace; font-size: 0.85rem; font-weight:700;">SCAN COMPLETE</div>', unsafe_allow_html=True)
                
                if is_simulated:
                    st.error("⚠️ DEMO / SIMULATION MODE — NOT ML MODEL OUTPUT")
                    st.warning("The saved Random Forest model or TF-IDF vectorizer could not be loaded. Live predictions are running in rule-based fallback mode.")
                
                st.write("")
                st.markdown(f"### {subject_val if subject_val else '(No Subject)'}")
                st.markdown(f"**From:** `{sender_val}` *(Domain: `{sender_domain}`)*")
                
                st.write("---")
                
                # Metrics Grid
                m_col1, m_col2 = st.columns(2)
                with m_col1:
                    st.metric(label="Threat Score", value=f"{threat_score:.2f} / 100")
                    st.metric(label="Spam Probability", value=f"{prob_val:.2f}%")
                    st.metric(label="Verdict Prediction", value=pred_class.upper())
                with m_col2:
                    st.metric(label="Rule-Based Sender Reputation", value=f"{rep_class.upper()} ({rep_score}/100)")
                    st.metric(label="Sender Blacklist Recommendation", value=blacklist_rec)
                    
                st.write("---")
                st.markdown("#### URL & Link Indicator Analysis")
                
                ind_col1, ind_col2 = st.columns(2)
                with ind_col1:
                    # Keywords
                    if kw_count > 0:
                        st.write(f" **Suspicious Keywords Found ({kw_count}):** `{', '.join(found_keywords)}`")
                    else:
                        st.write(" **Suspicious Keywords Found:** None")
                        
                    # URLs
                    if url_count > 0:
                        st.write(f" **URL Count:** {url_count}")
                        with st.expander("View Extracted URLs"):
                            for u in urls_found:
                                st.write(f"- `{u}`")
                    else:
                        st.write(" **URL Count:** 0")
                        
                    # IPs
                    if ip_count > 0:
                        st.write(f" **IP Addresses Detected ({ip_count}):** `{', '.join(ips_found)}`")
                    else:
                        st.write(" **IP Addresses Detected:** 0")
                        
                with ind_col2:
                    st.write(f" **Shortened Links:** {short_url_count}")
                    st.write(f" **Suspicious TLDs:** {tld_count}")
                    st.write(f" **Urgent Language:** {'Yes' if contains_urgency else 'No'}")
                    st.write(f" **Financial Trigger:** {'Yes' if contains_money else 'No'}")
                    
                # Display attachments metadata
                if att_count > 0:
                    st.write("---")
                    st.write(f"📎 **Attachments Found ({att_count}):**")
                    for att in st.session_state.attachment_metadata:
                        st.write(f"- `{att}`")
                        
                st.write("---")
                st.markdown("**Threat Assessment Rationale:**")
                st.info(threat_reason_str)
        else:
            st.info("👈 Enter email details and click 'Execute Cyber threat Scan' to trigger real-time ML forensic analysis.")

# ----------------- TAB 2: THREAT ANALYTICS -----------------
with tab_analytics:
    if df_filtered.empty:
        st.info("No emails to display in charts. Adjust filters.")
    else:
        # KPI METRICS SECTION inside Analytics Tab
        kpi1, kpi2, kpi3, kpi4, kpi5, kpi6 = st.columns(6)
        
        with kpi1:
            st.markdown(
                f"""
                <div class="soc-metric-card">
                    <span class="soc-metric-icon"></span>
                    <div class="soc-metric-label">Total Emails</div>
                    <div class="soc-metric-value">{total_count:,}</div>
                    <div class="soc-metric-footer">Analyzed by pipeline</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        with kpi2:
            st.markdown(
                f"""
                <div class="soc-metric-card">
                    <span class="soc-metric-icon"></span>
                    <div class="soc-metric-label">Spam Emails</div>
                    <div class="soc-metric-value" style="color: #FF4D4D;">{spam_count:,}</div>
                    <div class="soc-metric-footer">Flagged malicious</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        with kpi3:
            st.markdown(
                f"""
                <div class="soc-metric-card">
                    <span class="soc-metric-icon"></span>
                    <div class="soc-metric-label">Ham Emails</div>
                    <div class="soc-metric-value" style="color: #34C759;">{ham_count:,}</div>
                    <div class="soc-metric-footer">Verified clean</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        with kpi4:
            st.markdown(
                f"""
                <div class="soc-metric-card">
                    <span class="soc-metric-icon"></span>
                    <div class="soc-metric-label">Avg Spam Prob</div>
                    <div class="soc-metric-value">{avg_spam_prob:.1f}%</div>
                    <div class="soc-metric-footer">ML model confidence</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        with kpi5:
            threat_color = "#34C759"
            if avg_threat_score >= 80:
                threat_color = "#FF4D4D"
            elif avg_threat_score >= 55:
                threat_color = "#FFA500"
            elif avg_threat_score >= 30:
                threat_color = "#FFCC00"
                
            st.markdown(
                f"""
                <div class="soc-metric-card">
                    <span class="soc-metric-icon"></span>
                    <div class="soc-metric-label">Avg Threat Score</div>
                    <div class="soc-metric-value" style="color: {threat_color};">{avg_threat_score:.1f}</div>
                    <div class="soc-metric-footer">Risk score index</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        with kpi6:
            st.markdown(
                f"""
                <div class="soc-metric-card">
                    <span class="soc-metric-icon"></span>
                    <div class="soc-metric-label">Blacklisted Senders</div>
                    <div class="soc-metric-value" style="color: #FF4D4D;">{blacklist_count:,}</div>
                    <div class="soc-metric-footer">Recommended for block</div>
                </div>
                """,
                unsafe_allow_html=True
            )
            
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Row 1: Distribution Pie Charts
        ch_col1, ch_col2, ch_col3 = st.columns(3)
        
        # Chart 1: Threat Level Distribution
        with ch_col1:
            st.markdown("<h4 style='font-size:1rem; font-weight:600; text-align:center;'>Threat Level Distribution</h4>", unsafe_allow_html=True)
            threat_dist = df_filtered["Threat_Level"].value_counts().reset_index()
            threat_dist.columns = ["Threat Level", "Count"]
            
            level_order = {"Critical": 0, "High": 1, "Medium": 2, "Low": 3}
            threat_dist["order"] = threat_dist["Threat Level"].map(level_order)
            threat_dist = threat_dist.sort_values("order").drop(columns="order")
            
            color_map = {"Critical": "#EF4444", "High": "#F97316", "Medium": "#EAB308", "Low": "#22C55E"}
            colors = [color_map.get(lvl, "#007BFF") for lvl in threat_dist["Threat Level"]]
            
            fig1 = go.Figure(data=[go.Pie(
                labels=threat_dist["Threat Level"],
                values=threat_dist["Count"],
                hole=0.45,
                marker=dict(colors=colors),
                textinfo='percent',
                hoverinfo='label+value'
            )])
            fig1.update_layout(**plotly_layout_opts)
            st.plotly_chart(fig1, use_container_width=True, key="threat_level_dist_chart")

        # Chart 2: Spam vs Ham Prediction
        with ch_col2:
            st.markdown("<h4 style='font-size:1rem; font-weight:600; text-align:center;'>Spam vs Ham Distribution</h4>", unsafe_allow_html=True)
            pred_dist = df_filtered["Spam_Prediction"].value_counts().reset_index()
            pred_dist.columns = ["Verdict", "Count"]
            colors2 = ["#EF4444" if v == "Spam" else "#22C55E" for v in pred_dist["Verdict"]]
            
            fig2 = go.Figure(data=[go.Pie(
                labels=pred_dist["Verdict"],
                values=pred_dist["Count"],
                hole=0.45,
                marker=dict(colors=colors2),
                textinfo='percent',
                hoverinfo='label+value'
            )])
            fig2.update_layout(**plotly_layout_opts)
            st.plotly_chart(fig2, use_container_width=True, key="spam_ham_dist_chart")

        # Chart 3: Sender Reputation Distribution
        with ch_col3:
            st.markdown("<h4 style='font-size:1rem; font-weight:600; text-align:center;'>Sender Reputation</h4>", unsafe_allow_html=True)
            rep_dist = df_filtered["Sender_Reputation"].value_counts().reset_index()
            rep_dist.columns = ["Reputation", "Count"]
            
            rep_order = {"High": 0, "Medium": 1, "Low": 2, "Very Low": 3}
            rep_dist["order"] = rep_dist["Reputation"].map(rep_order)
            rep_dist = rep_dist.sort_values("order").drop(columns="order")
            
            rep_colors = {"Very Low": "#EF4444", "Low": "#F97316", "Medium": "#EAB308", "High": "#007BFF"}
            colors3 = [rep_colors.get(rep, "#94A3B8") for rep in rep_dist["Reputation"]]
            
            fig3 = go.Figure(data=[go.Pie(
                labels=rep_dist["Reputation"],
                values=rep_dist["Count"],
                hole=0.45,
                marker=dict(colors=colors3),
                textinfo='percent',
                hoverinfo='label+value'
            )])
            fig3.update_layout(**plotly_layout_opts)
            st.plotly_chart(fig3, use_container_width=True, key="sender_reputation_chart")
            
        st.markdown("<br><hr style='border-color: #1E293B;'>", unsafe_allow_html=True)
        
        # Row 2: Histograms
        ch_col4, ch_col5 = st.columns(2)
        
        with ch_col4:
            st.markdown("<h4 style='font-size:1rem; font-weight:600; text-align:center;'>Histogram of Threat Scores</h4>", unsafe_allow_html=True)
            fig4 = px.histogram(
                df_filtered,
                x="Threat_Score",
                nbins=30,
                color_discrete_sequence=["#007BFF"],
                labels={"Threat_Score": "Threat Score"}
            )
            fig4.update_layout(**plotly_layout_opts)
            fig4.update_layout(
                xaxis_title="Threat Score", 
                yaxis_title="Email Count", 
                bargap=0.08,
                xaxis=dict(gridcolor='#1E293B', range=[0, 100]),
                yaxis=dict(gridcolor='#1E293B')
            )
            st.plotly_chart(fig4, use_container_width=True, key="threat_score_hist_chart")
            
        with ch_col5:
            st.markdown("<h4 style='font-size:1rem; font-weight:600; text-align:center;'>Histogram of Spam Probability</h4>", unsafe_allow_html=True)
            fig5 = px.histogram(
                df_filtered,
                x="Spam_Probability",
                nbins=30,
                color_discrete_sequence=["#FF4D4D"],
                labels={"Spam_Probability": "Spam Probability (%)"}
            )
            fig5.update_layout(**plotly_layout_opts)
            fig5.update_layout(
                xaxis_title="Spam Probability (%)", 
                yaxis_title="Email Count", 
                bargap=0.08,
                xaxis=dict(gridcolor='#1E293B', range=[0, 100]),
                yaxis=dict(gridcolor='#1E293B')
            )
            st.plotly_chart(fig5, use_container_width=True, key="spam_prob_hist_chart")
            
        st.markdown("<br><hr style='border-color: #1E293B;'>", unsafe_allow_html=True)
        
        # Row 3: Security Indicators and Triggers
        ch_col6, ch_col7 = st.columns([3, 2])
        
        with ch_col6:
            st.markdown("<h4 style='font-size:1rem; font-weight:600; text-align:center;'>Suspicious Security Indicator Sums</h4>", unsafe_allow_html=True)
            indicators = {
                "Suspicious Keywords": df_filtered["Suspicious_Keyword_Count"].sum(),
                "Total Embedded Links": df_filtered["URL_Count"].sum(),
                "IP Addresses Detected": df_filtered["IP_Address_Count"].sum(),
                "Shortened Links": df_filtered["Shortened_URL_Count"].sum(),
                "Suspicious TLDs": df_filtered["Suspicious_TLD_Count"].sum()
            }
            ind_df = pd.DataFrame(list(indicators.items()), columns=["Indicator", "Count"])
            
            fig6 = px.bar(
                ind_df,
                x="Count",
                y="Indicator",
                orientation="h",
                color="Indicator",
                color_discrete_sequence=px.colors.sequential.Blues_r,
                text_auto=True
            )
            fig6.update_layout(**plotly_layout_opts)
            fig6.update_layout(
                xaxis_title="Aggregated Count",
                yaxis_title=None,
                showlegend=False,
                xaxis=dict(gridcolor='#1E293B'),
                yaxis=dict(autorange="reversed")
            )
            st.plotly_chart(fig6, use_container_width=True, key="security_indicators_chart")
            
        with ch_col7:
            st.markdown("<h4 style='font-size:1rem; font-weight:600; text-align:center;'>Urgency & Financial Triggers</h4>", unsafe_allow_html=True)
            urgency_pct = (df_filtered["Contains_Urgency"].sum() / total_count) * 100 if total_count > 0 else 0
            money_pct = (df_filtered["Contains_Money"].sum() / total_count) * 100 if total_count > 0 else 0
            
            trigger_df = pd.DataFrame({
                "Trigger": ["Contains Urgency", "Contains Financial Language"],
                "Percentage": [urgency_pct, money_pct]
            })
            
            fig7 = px.bar(
                trigger_df,
                x="Trigger",
                y="Percentage",
                color="Trigger",
                color_discrete_map={"Contains Urgency": "#F97316", "Contains Financial Language": "#EAB308"},
                text_auto=".1f"
            )
            fig7.update_layout(**plotly_layout_opts)
            fig7.update_layout(
                xaxis_title=None,
                yaxis_title="Percentage of Emails (%)",
                showlegend=False,
                yaxis=dict(gridcolor='#1E293B', range=[0, 100])
            )
            st.plotly_chart(fig7, use_container_width=True, key="urgency_financial_chart")

# ----------------- TAB 3: EMAIL INVESTIGATION & FORENSICS -----------------
with tab_investigation:
    if df_filtered.empty:
        st.info("No emails to display. Adjust filters.")
    else:
        st.markdown("<h2 style='font-size:1.6rem; font-weight:700; color:#F8FAFC;'>📧 EMAIL INVESTIGATION & ANALYTICS</h2>", unsafe_allow_html=True)
        
        panel_col_left, panel_col_right = st.columns([7, 5])
        
        with panel_col_left:
            st.markdown("<h4 style='font-size:1.1rem; font-weight:600; margin-bottom:8px;'>High Risk & Filtered Records</h4>", unsafe_allow_html=True)
            search_query = st.text_input("🔍 Search emails (Sender, Subject, Domain):", value="", placeholder="Type sender name or domain...", key="dataset_search_input")
            
            df_table = df_filtered.copy()
            if search_query:
                df_table = df_table[
                    df_table["From"].str.contains(search_query, case=False, na=False) |
                    df_table["Subject"].str.contains(search_query, case=False, na=False) |
                    df_table["Sender_Domain"].str.contains(search_query, case=False, na=False)
                ]
                
            display_cols = [
                "Sender_Domain", "Subject", "Spam_Prediction", 
                "Spam_Probability", "Threat_Score", "Threat_Level", 
                "Sender_Reputation", "Blacklist_Recommendation"
            ]
            
            st.dataframe(
                df_table[display_cols],
                column_config={
                    "Sender_Domain": "Domain",
                    "Subject": "Subject Line",
                    "Spam_Prediction": "Prediction",
                    "Spam_Probability": st.column_config.ProgressColumn(
                        "Spam Prob",
                        format="%.1f%%",
                        min_value=0.0,
                        max_value=100.0,
                    ),
                    "Threat_Score": st.column_config.ProgressColumn(
                        "Threat Score",
                        format="%.1f",
                        min_value=0.0,
                        max_value=100.0,
                    ),
                    "Threat_Level": "Threat Level",
                    "Sender_Reputation": "Reputation",
                    "Blacklist_Recommendation": "Blacklist?"
                },
                use_container_width=True,
                hide_index=False
            )
            
            csv_data = df_filtered.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Filtered Dataset (CSV)",
                data=csv_data,
                file_name="filtered_threat_analysis_data.csv",
                mime="text/csv",
                key="download_filtered_csv_btn"
            )
            
        with panel_col_right:
            st.markdown("<h4 style='font-size:1.1rem; font-weight:600; margin-bottom:8px;'> Forensics & Detail Panel</h4>", unsafe_allow_html=True)
            
            forensic_options = df_filtered.index.tolist()
            
            if len(forensic_options) == 0:
                st.info("No emails to select. Adjust filters.")
            else:
                def get_forensic_label(idx):
                    row = df_filtered.loc[idx]
                    sub = str(row["Subject"])
                    truncated_sub = sub if len(sub) < 32 else sub[:30] + "..."
                    return f"[{row['Threat_Level']}] {truncated_sub} ({row['Sender_Domain']})"
                    
                selected_email_idx = st.selectbox(
                    "Select email index to load for deep investigation:",
                    options=forensic_options,
                    format_func=get_forensic_label,
                    key="investigation_selectbox"
                )
                
                email_data = df_filtered.loc[selected_email_idx]
                
                t_level = email_data["Threat_Level"]
                badge_class = "cyber-badge-low"
                if t_level == "Critical":
                    badge_class = "cyber-badge-critical"
                elif t_level == "High":
                    badge_class = "cyber-badge-high"
                elif t_level == "Medium":
                    badge_class = "cyber-badge-medium"
                
                with st.container(border=True):
                    # Header: Threat Level Badge & Index
                    col_badge, col_uuid = st.columns([2, 1])
                    with col_badge:
                        st.markdown(f'<span class="cyber-badge {badge_class}">{t_level} RISK LEVEL</span>', unsafe_allow_html=True)
                    with col_uuid:
                        st.markdown(f'<div style="text-align: right; color: #94A3B8; font-family: monospace; font-size: 0.85rem;">Index: {selected_email_idx}</div>', unsafe_allow_html=True)
                    
                    st.write("")
                    st.markdown(f"### {email_data['Subject']}")
                    st.markdown(f"**From:** `{email_data['From']}` *(Domain: `{email_data['Sender_Domain']}`)*")
                    
                    st.write("---")
                    
                    # Grid of Metrics
                    m_col1, m_col2 = st.columns(2)
                    with m_col1:
                        st.metric(label="Threat Score", value=f"{email_data['Threat_Score']:.2f} / 100")
                        st.metric(label="Spam Probability", value=f"{email_data['Spam_Probability']:.2f}%")
                        st.metric(label="Verdict", value=str(email_data['Spam_Prediction']).upper())
                    with m_col2:
                        st.metric(label="Rule-Based Sender Reputation", value=f"{str(email_data['Sender_Reputation']).upper()} ({int(email_data['Sender_Reputation_Score'])}/100)")
                        st.metric(label="Blacklist Recommended?", value=str(email_data['Blacklist_Recommendation']))
                    
                    st.write("---")
                    st.markdown("#### Rule-Based Sender & Link Security Indicators")
                    ind_col1, ind_col2 = st.columns(2)
                    with ind_col1:
                        st.write(f" **Suspicious Keywords:** {int(email_data['Suspicious_Keyword_Count'])}")
                        st.write(f" **URL Count:** {int(email_data['URL_Count'])}")
                        st.write(f" **IP Addresses:** {int(email_data['IP_Address_Count'])}")
                    with ind_col2:
                        st.write(f" **Shortened Links:** {int(email_data['Shortened_URL_Count'])}")
                        st.write(f" **Suspicious TLDs:** {int(email_data['Suspicious_TLD_Count'])}")
                        st.write(f" **Urgent Language:** {'Yes' if email_data['Contains_Urgency'] else 'No'}")
                        st.write(f" **Financial Language:** {'Yes' if email_data['Contains_Money'] else 'No'}")
                        
                    st.write("---")
                    st.markdown("**Threat Engine Rationale & Explanation:**")
                    st.info(email_data['Threat_Reason'])
                    
                    # Expandable original email body if available
                    if "Body" in email_data and email_data["Body"]:
                        with st.expander("View Original Email Content"):
                            st.text(email_data["Body"])
                            
        # TOP 10 HIGHEST THREAT EMAILS
        st.markdown("<br>", unsafe_allow_html=True)
        with st.expander(" TOP 10 HIGHEST THREAT EMAILS"):
            st.markdown("Emails sorted descending by overall Threat Score (ML confidence + Security Anomalies + Reputation):")
            top_10 = df_filtered.sort_values(by="Threat_Score", ascending=False).head(10)
            
            st.dataframe(
                top_10[[
                    "From", "Subject", "Spam_Prediction", "Spam_Probability", 
                    "Threat_Score", "Threat_Level", "Sender_Reputation", "Blacklist_Recommendation"
                ]],
                column_config={
                    "From": "From",
                    "Subject": "Subject Line",
                    "Spam_Prediction": "Verdict",
                    "Spam_Probability": st.column_config.ProgressColumn("Spam Probability", format="%.2f%%", min_value=0.0, max_value=100.0),
                    "Threat_Score": st.column_config.ProgressColumn("Threat Score", format="%.2f", min_value=0.0, max_value=100.0),
                    "Threat_Level": "Threat Level",
                    "Sender_Reputation": "Sender Reputation",
                    "Blacklist_Recommendation": "Recommend Blacklist"
                },
                use_container_width=True,
                hide_index=True
            )

st.markdown("<br><hr style='border-color: #1E293B;'>", unsafe_allow_html=True)

# ==============================================================================
# FOOTER & SYSTEM INFO
# ==============================================================================
st.markdown(
    """
    <div class="soc-footer">
        <div style="display:flex; justify-content: space-between; align-items:center;">
            <div>
                <strong>SYSTEM LOGS & AUDIT INFO</strong><br>
                Model Pipeline: Random Forest Classifier (RF-09)<br>
                Data Source: data/processed/dashboard_dataset.csv
            </div>
            <div style="text-align:right;">
                <strong>MODEL CLASSIFICATION PERFORMANCE METRICS</strong><br>
                Accuracy: 96.89% &nbsp;|&nbsp; Precision: 96.31% &nbsp;|&nbsp; Recall: 94.22% &nbsp;|&nbsp; F1 Score: 95.26% &nbsp;|&nbsp; ROC-AUC: 99.44%
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)
