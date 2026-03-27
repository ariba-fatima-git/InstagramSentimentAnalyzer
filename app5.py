"""
STEP 6 — Streamlit Dashboard
================================
Full interactive dashboard for Instagram Comment Sentiment Analysis.

Run:
    streamlit run dashboard/app.py

Sections:
  1. INPUT    — upload CSV or paste Reel URL (triggers Apify)
  2. ANALYSIS — pie chart, bar chart, word clouds
  3. INSIGHTS — cluster cards, AI summary, download report
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import io
import re
import sys
import os
from pathlib import Path
from collections import Counter
from dotenv import load_dotenv
load_dotenv()
from groq import Groq


# ─────────────────────────────────────────────
# BACKEND API KEY  (never shown to the user)
# ─────────────────────────────────────────────
# Priority 1: Streamlit Cloud → add to .streamlit/secrets.toml as:
#   APIFY_TOKEN = "apify_xxxx..."
# Priority 2: local dev → set environment variable APIFY_TOKEN
def _load_apify_token() -> str | None:
    try:
        return st.secrets["APIFY_TOKEN"]
    except (KeyError, FileNotFoundError):
        pass
    return os.environ.get("APIFY_TOKEN")


def _load_groq_token() -> str | None:
    try:
        return st.secrets["GROQ_API_KEY"]
    except (KeyError, FileNotFoundError):
        pass
    return os.environ.get("GROQ_API_KEY")

# 1. Call the loader to get the key string
groq_key_string = _load_groq_token()

# 2. Check if the key exists before starting the engine
if groq_key_string:
    # This 'client' is what actually sends your data to Groq
    client = Groq(api_key=groq_key_string)
else:
    st.error("Missing Groq API Key! Please add it to your .env file.")
# ── Add parent dir to path so we can import our pipeline modules ──
sys.path.insert(0, str(Path(__file__).parent.parent))


# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Instagram Sentiment Analyser",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    /* Main header */
    .main-title {
        font-size: 2.4rem;
        font-weight: 800;
        background: linear-gradient(135deg, #833ab4, #fd1d1d, #fcb045);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        font-size: 1rem;
        color: #888;
        margin-bottom: 2rem;
    }
    /* Metric cards */
    .metric-card {
        background: #1e1e2e;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        border-left: 4px solid;
        margin-bottom: 0.8rem;
    }
    .metric-card.positive { border-color: #00c851; }
    .metric-card.neutral  { border-color: #ffbb33; }
    .metric-card.negative { border-color: #ff4444; }
    .metric-number {
        font-size: 2.5rem;
        font-weight: 800;
        line-height: 1;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #aaa;
        margin-top: 0.2rem;
    }
    /* Cluster cards */
    .cluster-card {
        background: #1e1e2e;
        border-radius: 12px;
        padding: 1.2rem;
        border: 1px solid #333;
        height: 100%;
    }
    .cluster-title {
        font-size: 1.1rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    .keyword-pill {
        display: inline-block;
        background: #333;
        border-radius: 20px;
        padding: 2px 10px;
        font-size: 0.78rem;
        margin: 2px;
        color: #ccc;
    }
    /* Summary box */
    .summary-box {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border-radius: 12px;
        padding: 1.5rem;
        border: 1px solid #444;
        font-size: 1.05rem;
        line-height: 1.7;
        color: #e0e0e0;
    }
    /* Step indicator */
    .step-badge {
        background: #833ab4;
        color: white;
        border-radius: 50%;
        width: 28px;
        height: 28px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-weight: bold;
        margin-right: 8px;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────

STOP_WORDS = set([
    "the","a","an","is","it","this","that","and","or","to","of","in",
    "for","on","with","so","just","im","ive","i","me","my","you","your",
    "we","our","they","them","but","not","no","do","dont","be","was",
    "are","have","had","has","will","can","got","get","like","really",
    "very","much","more","too","been","than","what","its","also",
])


def clean_text_basic(text: str) -> str:
    text = re.sub(r"http\S+|www\.\S+", "", str(text))
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#(\w+)", r"\1", text)
    return re.sub(r"\s+", " ", text).strip()


def top_words(texts: list, n: int = 15) -> list:
    all_words = []
    for text in texts:
        words = re.sub(r"[^a-zA-Z\s]", "", str(text).lower()).split()
        all_words.extend([w for w in words if w not in STOP_WORDS and len(w) > 2])
    return [word for word, _ in Counter(all_words).most_common(n)]


def count_emojis(text: str) -> int:
    pattern = re.compile(
        "[\U00010000-\U0010ffff\U00002600-\U000027BF\U0001F300-\U0001F9FF]+",
        flags=re.UNICODE
    )
    return len(pattern.findall(str(text)))


@st.cache_data
def run_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the raw dataframe."""
    # Normalise comment column
    for col in ["text", "comment", "body", "content"]:
        if col in df.columns:
            df = df.rename(columns={col: "comment"})
            break
    if "comment" not in df.columns:
        st.error("Could not find comment column. Expected: text, comment, body, or content.")
        st.stop()

    df = df.dropna(subset=["comment"])
    df = df[df["comment"].str.strip() != ""]
    df = df.drop_duplicates(subset="comment")
    df = df[df["comment"].str.split().str.len() >= 2]
    df["comment_clean"] = df["comment"].apply(clean_text_basic)
    df["word_count"]    = df["comment_clean"].str.split().str.len()
    df["emoji_count"]   = df["comment"].apply(count_emojis)

    # ── Language detection (best-effort, silent if langdetect not installed) ──
    try:
        from langdetect import detect, LangDetectException
        def safe_detect(text):
            try:
                return detect(str(text))
            except LangDetectException:
                return "unknown"
        df["language"] = df["comment"].apply(safe_detect)
    except ImportError:
        df["language"] = "unknown"

    return df.reset_index(drop=True)


@st.cache_resource
def load_sentiment_model():
    try:
        from transformers import pipeline
        return pipeline(
            "text-classification",
            model="cardiffnlp/twitter-xlm-roberta-base-sentiment",
            top_k=None, truncation=True, max_length=512,
        )
    except Exception:
        return None


# Handles both old english model labels and new XLM multilingual labels
LABEL_MAP = {
    "LABEL_0":  "Negative",
    "LABEL_1":  "Neutral",
    "LABEL_2":  "Positive",
    "negative": "Negative",
    "neutral":  "Neutral",
    "positive": "Positive",
}

def roberta_sentiment(classifier, texts: list) -> pd.DataFrame:
    results = []
    batch_size = 16
    for start in range(0, len(texts), batch_size):
        batch   = texts[start:start+batch_size]
        outputs = classifier(batch)
        for output in outputs:
            # Default all three to 0 first — avoids silent key-miss bugs
            scores = {"Positive": 0.0, "Neutral": 0.0, "Negative": 0.0}
            for d in output:
                mapped = LABEL_MAP.get(d["label"])
                if mapped:
                    scores[mapped] = d["score"]
            best = max(scores, key=scores.get)
            results.append({
                "sentiment":  best,
                "confidence": round(scores[best], 4),
                "score_pos":  round(scores["Positive"], 4),
                "score_neu":  round(scores["Neutral"],  4),
                "score_neg":  round(scores["Negative"], 4),
            })
    return pd.DataFrame(results)
CULTURAL_POSITIVE_TERMS = {
    "mashallah", "masha'allah", "masha allah", "alhamdulillah", "subhanallah",
    "allahu akbar", "inshallah", "jazakallah", "barakallah", "tabarak allah",
    "ramadan mubarak", "eid mubarak", "mabrook", "mubarak",
    "ماشاء الله", "الحمد لله", "سبحان الله",
}


def patch_cultural_sentiment(df: pd.DataFrame, original_texts: list) -> pd.DataFrame:
    df = df.copy()
    for i, text in enumerate(original_texts):
        if i >= len(df):
            break
            
        text_lower = str(text).lower()
        
        # Check if any cultural term exists in this specific comment
        has_cultural_term = any(term.lower() in text_lower for term in CULTURAL_POSITIVE_TERMS)
        
        # LOGIC CHANGE: 
        # If it has a cultural term AND the AI didn't already mark it as Positive, 
        # or if the AI is even slightly unsure (confidence < 0.85)
        if has_cultural_term and (df.at[i, "sentiment"] != "Positive" or df.at[i, "confidence"] < 0.85):
            df.at[i, "sentiment"]  = "Positive"
            df.at[i, "confidence"] = 0.90  # Set it higher so it shows up clearly
            df.at[i, "score_pos"]  = 0.90
            df.at[i, "score_neu"]  = 0.05
            df.at[i, "score_neg"]  = 0.05
            
    return df

def textblob_sentiment(texts: list) -> pd.DataFrame:
    from textblob import TextBlob
    results = []
    for text in texts:
        pol = TextBlob(str(text)).sentiment.polarity
        if pol > 0.05:
            label = "Positive"
        elif pol < -0.05:
            label = "Negative"
        else:
            label = "Neutral"
        results.append({
            "sentiment":  label,
            "confidence": round(abs(pol), 4),
            "score_pos":  round(max(pol, 0), 4),
            "score_neu":  round(1 - abs(pol), 4),
            "score_neg":  round(max(-pol, 0), 4),
        })
    return pd.DataFrame(results)


@st.cache_data
def run_sentiment(_df: pd.DataFrame) -> pd.DataFrame:
    classifier = load_sentiment_model()
    texts      = _df["comment_clean"].astype(str).tolist()

    if classifier:
        with st.spinner("🤖 Running RoBERTa sentiment model..."):
            sent_df = roberta_sentiment(classifier, texts)
        sent_df = patch_cultural_sentiment(sent_df, texts)
    else:
        st.info("💡 RoBERTa not available — using TextBlob as fallback.")
        try:
            sent_df = textblob_sentiment(texts)
        except ImportError:
            # Manual mock for demo purposes
            import random
            random.seed(42)
            results = []
            for _ in texts:
                label = random.choices(
                    ["Positive","Neutral","Negative"], weights=[0.6,0.25,0.15]
                )[0]
                results.append({
                    "sentiment":  label,
                    "confidence": round(random.uniform(0.6, 0.95), 4),
                    "score_pos":  0.0, "score_neu": 0.0, "score_neg": 0.0,
                })
            sent_df = pd.DataFrame(results)

    return pd.concat([_df.reset_index(drop=True), sent_df], axis=1)


@st.cache_data
def run_clustering(_df: pd.DataFrame, n_clusters: int) -> tuple:
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.feature_extraction.text import TfidfVectorizer

    feature_cols = ["score_pos","score_neu","score_neg","confidence"]
    for col in ["word_count","emoji_count"]:
        if col in _df.columns:
            feature_cols.append(col)

    for col in feature_cols:
        if col not in _df.columns:
            _df[col] = 0

    X      = _df[feature_cols].fillna(0).values
    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X)

    km     = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    _df    = _df.copy()
    _df["cluster_id"] = km.fit_predict(X_sc)

    text_col = "comment_clean" if "comment_clean" in _df.columns else "comment"
    profiles = []

    for cid in sorted(_df["cluster_id"].unique()):
        cdf      = _df[_df["cluster_id"] == cid]
        texts    = cdf[text_col].astype(str).tolist()

        # TF-IDF keywords
        try:
            tfidf    = TfidfVectorizer(max_features=100, stop_words=list(STOP_WORDS),
                                       ngram_range=(1,2), min_df=1)
            mat      = tfidf.fit_transform(texts)
            scores   = np.asarray(mat.sum(axis=0)).flatten()
            vocab    = tfidf.get_feature_names_out()
            top_idx  = scores.argsort()[::-1][:8]
            keywords = [vocab[i] for i in top_idx]
        except Exception:
            keywords = top_words(texts, 8)

        # Auto-label
        counts = cdf["sentiment"].value_counts(normalize=True)
        pos = counts.get("Positive", 0)
        neg = counts.get("Negative", 0)
        neu = counts.get("Neutral",  0)
        avg_emojis = cdf.get("emoji_count", pd.Series([0])).mean()

        if pos >= 0.65:
            label = "🔥 Highly Engaged Fans" if avg_emojis > 1 else "💬 Loyal Commenters"
        elif neg >= 0.50:
            label = "👎 Critical Audience"
        elif neg >= 0.30:
            label = "😐 Mixed / Divided Viewers"
        elif neu >= 0.60:
            label = "👀 Passive Viewers"
        else:
            label = "🤔 Curious Explorers"

        _df.loc[_df["cluster_id"] == cid, "cluster_label"] = label
        profiles.append({
            "cluster_id":    int(cid),
            "label":         label,
            "size":          len(cdf),
            "pct_of_total":  round(len(cdf)/len(_df)*100, 1),
            "top_keywords":  keywords,
            "sentiment_dist": cdf["sentiment"].value_counts().to_dict(),
        })

    return _df, profiles


def _build_summary_context(df: pd.DataFrame, profiles: list) -> str:
    """Build a structured data context string to feed into the Groq prompt."""
    counts  = df["sentiment"].value_counts()
    total   = len(df)
    pos_pct = round(counts.get("Positive", 0) / total * 100, 1)
    neg_pct = round(counts.get("Negative", 0) / total * 100, 1)
    neu_pct = round(counts.get("Neutral",  0) / total * 100, 1)
    avg_words = round(df["comment_clean"].str.split().str.len().mean(), 1)
    pos_words = top_words(df[df["sentiment"] == "Positive"]["comment_clean"].tolist(), 6)
    neg_words = top_words(df[df["sentiment"] == "Negative"]["comment_clean"].tolist(), 6)

    lines = [
        f"Total comments analysed: {total}",
        f"Sentiment split — Positive: {pos_pct}%, Neutral: {neu_pct}%, Negative: {neg_pct}%",
        f"Average words per comment: {avg_words}",
        f"Top positive keywords: {', '.join(pos_words) if pos_words else 'N/A'}",
        f"Top negative keywords: {', '.join(neg_words) if neg_words else 'N/A'}",
        "",
        "Audience segments:",
    ]
    for p in profiles:
        lines.append(
            f"  - {p['label']}: {p['size']} comments ({p['pct_of_total']}%), "
            f"keywords: {', '.join(p['top_keywords'][:5])}, "
            f"sentiment: {p['sentiment_dist']}"
        )
    return "\n".join(lines)


def generate_summary(df: pd.DataFrame, profiles: list) -> str:
    """
    Generate an AI-powered audience insight summary using Groq (llama-3.3-70b-versatile).
    Falls back to a template-based summary if the GROQ_API_KEY is not configured.
    """
    groq_key = _load_groq_token()

    # ── Groq AI path ──────────────────────────────────────────────────────────
    if groq_key:
        try:
            import requests as _req
            context = _build_summary_context(df, profiles)
            prompt = (
                "You are a social-media analyst. Based on the Instagram comment "
                "analysis data below, write a concise 4–6 sentence audience insight "
                "summary. Highlight the overall sentiment tone, what positive viewers "
                "appreciate, what critics mention, and which audience segment dominates. "
                "Use plain prose — no bullet points, no headers.\n\n"
                f"{context}"
            )
            resp = _req.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {groq_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "llama-3.3-70b-versatile",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 300,
                    "temperature": 0.6,
                },
                timeout=20,
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            st.warning(f"⚠️ Groq API error — falling back to template summary. ({e})")

    # ── Template fallback (no API key or Groq call failed) ───────────────────
    counts  = df["sentiment"].value_counts()
    total   = len(df)
    pos_pct = round(counts.get("Positive", 0) / total * 100, 1)
    neg_pct = round(counts.get("Negative", 0) / total * 100, 1)
    neu_pct = round(counts.get("Neutral",  0) / total * 100, 1)
    tone = ("overwhelmingly positive" if pos_pct >= 70 else
            "mostly positive"          if pos_pct >= 55 else
            "notably critical"         if neg_pct >= 40 else
            "largely neutral"          if neu_pct >= 50 else "mixed")
    pos_words = top_words(df[df["sentiment"] == "Positive"]["comment_clean"].tolist(), 5)
    neg_words = top_words(df[df["sentiment"] == "Negative"]["comment_clean"].tolist(), 5)
    avg_words = df["comment_clean"].str.split().str.len().mean()
    top_cluster = max(profiles, key=lambda c: c["size"]) if profiles else None

    summary = (
        f"Analysis of **{total} comments** reveals a **{tone}** audience response. "
        f"**{pos_pct}%** Positive · **{neg_pct}%** Negative · **{neu_pct}%** Neutral. "
        f"Viewers average **{avg_words:.0f} words** per comment, indicating "
        f"{'high' if avg_words > 8 else 'brief'} engagement. "
    )
    if pos_words:
        summary += f"Positive comments revolve around: *{', '.join(pos_words)}*. "
    if neg_words:
        summary += f"Critics focus on: *{', '.join(neg_words)}*. "
    if top_cluster:
        summary += (
            f"The largest audience segment ({top_cluster['pct_of_total']}%) "
            f"is **{top_cluster['label']}**, discussing: "
            f"*{', '.join(top_cluster['top_keywords'][:4])}*."
        )
    return summary


def make_wordcloud_fig(words: list, color: str):
    """Generate a simple frequency bar chart as word cloud substitute."""
    import plotly.graph_objects as go
    if not words:
        return None
    sizes = list(range(len(words), 0, -1))
    fig = go.Figure(go.Bar(
        x=words, y=sizes,
        marker_color=color,
        text=words,
        textposition="auto",
    ))
    fig.update_layout(
        showlegend=False,
        margin=dict(l=10, r=10, t=10, b=10),
        height=250,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showticklabels=False),
        yaxis=dict(showticklabels=False),
    )
    return fig


def build_pdf_report(df: pd.DataFrame, profiles: list, summary: str) -> bytes:
    """Build a simple CSV + summary as download bundle."""
    buf = io.StringIO()
    buf.write("INSTAGRAM SENTIMENT ANALYSIS REPORT\n")
    buf.write("="*50 + "\n\n")
    buf.write("SUMMARY\n")
    buf.write("-"*30 + "\n")
    # Strip markdown bold markers for plain text
    plain_summary = re.sub(r"\*\*(.+?)\*\*", r"\1", summary)
    plain_summary = re.sub(r"\*(.+?)\*", r"\1", plain_summary)
    buf.write(plain_summary + "\n\n")
    buf.write("SENTIMENT DISTRIBUTION\n")
    buf.write("-"*30 + "\n")
    counts = df["sentiment"].value_counts()
    total  = len(df)
    for label in ["Positive","Neutral","Negative"]:
        n   = counts.get(label, 0)
        pct = n/total*100
        buf.write(f"{label}: {n} ({pct:.1f}%)\n")
    buf.write("\nCLUSTER PROFILES\n")
    buf.write("-"*30 + "\n")
    for p in profiles:
        buf.write(f"\n{p['label']} ({p['size']} comments, {p['pct_of_total']}%)\n")
        buf.write(f"  Keywords: {', '.join(p['top_keywords'][:6])}\n")
        buf.write(f"  Sentiment: {p['sentiment_dist']}\n")
    return buf.getvalue().encode("utf-8")


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚙️ Settings")
    n_clusters = st.slider("Audience Clusters (k)", 2, 6, 4)
    st.divider()
    st.markdown("### 📖 Pipeline Steps")
    st.markdown("""
    1. **Collect** — CSV, URL or manual input
    2. **Clean** — deduplicate, filter, detect language
    3. **Sentiment** — XLM-RoBERTa (100+ languages)
    4. **Cluster** — KMeans + TF-IDF
    5. **Insights** — summary + report
    """)
    st.divider()
    st.markdown("**Model:** `cardiffnlp/twitter-xlm-roberta-base-sentiment` 🌍")
    st.caption("Supports English, Urdu, Arabic, Hindi, French, Spanish and 95+ more languages.")


# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────

st.markdown('<div class="main-title">📊 Instagram Sentiment Analyser</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Audience intelligence from Instagram Reel comments</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────
# SHARED ANALYSIS RENDERER
# Called by each tab with its own data + unique prefix
# ─────────────────────────────────────────────

def render_analysis(df_raw: pd.DataFrame, prefix: str, source_label: str = ""):
    import plotly.express as px
    import plotly.graph_objects as go

    df_clean = run_cleaning(df_raw)
    df_sent  = run_sentiment(df_clean)
    df_final, profiles = run_clustering(df_sent, n_clusters)

    # ── SECTION 2 — ANALYSIS ──────────────────
    st.markdown("---")
    
    st.markdown("### <span class='step-badge'>1</span> Data Input", unsafe_allow_html=True)
    
    st.caption(f"📌 Source: {source_label}")

    counts = df_final["sentiment"].value_counts()
    total  = len(df_final)
    lang_counts = df_final["language"].value_counts() if "language" in df_final.columns else None

    # ── Metric row ──
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Comments", f"{total:,}")
    with col2:
        n = counts.get("Positive", 0)
        st.metric("✅ Positive", f"{n} ({n/total*100:.0f}%)")
    with col3:
        n = counts.get("Neutral", 0)
        st.metric("➖ Neutral", f"{n} ({n/total*100:.0f}%)")
    with col4:
        n = counts.get("Negative", 0)
        st.metric("❌ Negative", f"{n} ({n/total*100:.0f}%)")

    # ── Language breakdown badge row ──
    if lang_counts is not None and not lang_counts.index.tolist() == ["unknown"]:
        st.markdown("#### 🌍 Languages Detected")
        lang_cols = st.columns(min(len(lang_counts), 6))
        lang_names = {
            "en":"English","ur":"Urdu","ar":"Arabic","hi":"Hindi",
            "fr":"French","es":"Spanish","de":"German","tr":"Turkish",
            "pt":"Portuguese","it":"Italian","zh-cn":"Chinese","ja":"Japanese",
            "ko":"Korean","ru":"Russian","nl":"Dutch","pl":"Polish",
        }
        for i, (lang, cnt) in enumerate(lang_counts.head(6).items()):
            with lang_cols[i]:
                name = lang_names.get(lang, lang.upper())
                pct  = round(cnt / total * 100, 1)
                st.markdown(f"""
                <div style="background:#1e1e2e;border-radius:10px;padding:0.8rem;
                            text-align:center;border:1px solid #333;">
                    <div style="font-size:1.4rem">{name}</div>
                    <div style="font-size:1.5rem;font-weight:800;color:#a78bfa">{cnt}</div>
                    <div style="font-size:0.75rem;color:#888">{pct}% of comments</div>
                </div>
                """, unsafe_allow_html=True)

        # Language pie chart
        lang_df = lang_counts.reset_index()
        lang_df.columns = ["Language", "Count"]
        lang_df["Language"] = lang_df["Language"].apply(lambda x: lang_names.get(x, x.upper()))
        fig_lang = px.pie(
            lang_df, names="Language", values="Count",
            hole=0.4, title="Comment Language Distribution",
        )
        fig_lang.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(t=40, b=20), height=320,
        )
        st.plotly_chart(fig_lang, use_container_width=True, key=f"{prefix}_lang_pie")

    # ── Sentiment charts ──
    st.markdown("#### Sentiment Breakdown")
    pie_data = df_final["sentiment"].value_counts().reset_index()
    pie_data.columns = ["Sentiment", "Count"]

    col_a, col_b = st.columns(2)
    with col_a:
        fig_pie = px.pie(
            pie_data, names="Sentiment", values="Count", color="Sentiment",
            color_discrete_map={"Positive":"#00c851","Neutral":"#ffbb33","Negative":"#ff4444"},
            hole=0.45,
        )
        fig_pie.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", y=-0.1), margin=dict(t=20, b=20),
        )
        st.plotly_chart(fig_pie, use_container_width=True, key=f"{prefix}_pie")

    with col_b:
        fig_bar = px.bar(
            pie_data, x="Sentiment", y="Count", color="Sentiment",
            color_discrete_map={"Positive":"#00c851","Neutral":"#ffbb33","Negative":"#ff4444"},
            text="Count",
        )
        fig_bar.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            showlegend=False, margin=dict(t=20, b=20), yaxis=dict(gridcolor="#333"),
        )
        fig_bar.update_traces(textposition="outside")
        st.plotly_chart(fig_bar, use_container_width=True, key=f"{prefix}_bar")

    # ── Keyword themes ──
    st.markdown("#### 🔑 Keyword Themes by Sentiment")
    wc1, wc2, wc3 = st.columns(3)
    for wcol, label, color in zip(
        [wc1, wc2, wc3], ["Positive", "Neutral", "Negative"],
        ["#00c851", "#ffbb33", "#ff4444"]
    ):
        with wcol:
            st.markdown(f"**{label}**")
            subset = df_final[df_final["sentiment"] == label]["comment_clean"].tolist()
            words  = top_words(subset, 10)
            fig_wc = make_wordcloud_fig(words, color)
            if fig_wc:
                st.plotly_chart(fig_wc, use_container_width=True, key=f"{prefix}_wc_{label}")

    # ── Data table ──
    with st.expander("🔍 View Processed Data"):
        display_cols = ["comment", "language", "sentiment", "confidence",
                        "score_pos", "score_neg", "cluster_label"]
        display_cols = [c for c in display_cols if c in df_final.columns]
        st.dataframe(df_final[display_cols], use_container_width=True)

    # ── SECTION 3 — INSIGHTS ─────────────────
    st.markdown("---")
    st.markdown("### <span class='step-badge'>3</span> Audience Insights", unsafe_allow_html=True)

    st.markdown("#### 👥 Audience Segments")
    seg_cols = st.columns(min(len(profiles), 4))
    for i, profile in enumerate(profiles):
        with seg_cols[i % len(seg_cols)]:
            sent = profile["sentiment_dist"]
            dom_color = (
                "#00c851" if sent.get("Positive",0) >= max(sent.get("Neutral",0), sent.get("Negative",0))
                else "#ff4444" if sent.get("Negative",0) >= sent.get("Neutral",0)
                else "#ffbb33"
            )
            st.markdown(f"""
            <div class="cluster-card">
                <div class="cluster-title" style="color:{dom_color}">{profile['label']}</div>
                <div style="font-size:1.8rem;font-weight:800;margin:0.3rem 0">{profile['size']}</div>
                <div style="color:#aaa;font-size:0.8rem;margin-bottom:0.8rem">
                    comments · {profile['pct_of_total']}% of total
                </div>
                <div style="margin-bottom:0.5rem;font-size:0.8rem;color:#888">Top Keywords:</div>
                <div>{''.join(f'<span class="keyword-pill">{k}</span>' for k in profile['top_keywords'][:6])}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("#### 📝 AI-Generated Summary")
    summary = generate_summary(df_final, profiles)
    st.markdown(f'<div class="summary-box">{summary}</div>', unsafe_allow_html=True)

    st.markdown("#### 📈 Model Confidence Distribution")
    fig_conf = px.histogram(
        df_final, x="confidence", color="sentiment", nbins=20, barmode="overlay",
        color_discrete_map={"Positive":"#00c851","Neutral":"#ffbb33","Negative":"#ff4444"},
        opacity=0.75,
    )
    fig_conf.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(gridcolor="#333"), margin=dict(t=20, b=20), height=300,
    )
    st.plotly_chart(fig_conf, use_container_width=True, key=f"{prefix}_conf")

    st.markdown("---")
    st.markdown("### ⬇️ Download Report")
    dl1, dl2 = st.columns(2)
    with dl1:
        st.download_button(
            "📥 Download Full CSV",
            data=df_final.to_csv(index=False).encode("utf-8"),
            file_name="sentiment_results.csv", mime="text/csv",
            key=f"{prefix}_dl_csv",
        )
    with dl2:
        st.download_button(
            "📄 Download Text Report",
            data=build_pdf_report(df_final, profiles, summary),
            file_name="sentiment_report.txt", mime="text/plain",
            key=f"{prefix}_dl_txt",
        )


# ─────────────────────────────────────────────
# SECTION 1 — INPUT  (3 fully independent tabs)
# ─────────────────────────────────────────────

st.markdown("---")
if not any(k in st.session_state for k in ["csv_ready", "url_ready", "manual_ready"]):
   st.markdown("### <span class='step-badge'>1</span> Data Input", unsafe_allow_html=True)

tab_upload, tab_url, tab_manual = st.tabs([
    "📁 Upload CSV",
    "🔗 Paste Reel URL",
    "✏️ Type / Paste Comments",
])

# ── TAB 1: Upload CSV ──────────────────────────────────────────
with tab_upload:
    uploaded = st.file_uploader(
        "Upload your Apify-exported CSV",
        type=["csv"],
        help="Must have a column named: comment, text, body, or content",
        key="csv_uploader",
    )
    if uploaded:
        df_csv = pd.read_csv(uploaded)
        st.success(f"✅ Loaded {len(df_csv)} rows")
        st.dataframe(df_csv.head(5), use_container_width=True)
        if st.button("▶ Analyse CSV", key="btn_csv"):
            st.session_state["csv_ready"] = df_csv

    if "csv_ready" in st.session_state:
        render_analysis(st.session_state["csv_ready"],
                        prefix="csv", source_label="Uploaded CSV")


# ── TAB 2: Paste Reel URL ──────────────────────────────────────
with tab_url:
    reel_url = st.text_input("Instagram Reel URL",
                             placeholder="https://www.instagram.com/reel/...",
                             key="reel_url")

    # Load API token from backend — user never needs to enter it
    apify_token = _load_apify_token()
    if not apify_token:
        st.error(
            "⚠️ Apify API token is not configured. "
            "Please ask the administrator to set the `APIFY_TOKEN` secret."
        )

    if st.button("🚀 Scrape & Analyse", key="btn_scrape") and reel_url and apify_token:
        import requests, time

        try:
            run_resp = requests.post(
                "https://api.apify.com/v2/acts/apify~instagram-comment-scraper/runs",
                headers={"Authorization": f"Bearer {apify_token}"},
                json={"directUrls": [reel_url], "resultsLimit": 200},
                timeout=30,
            )
        except Exception as e:
            st.error(f"Network error: {e}")
            st.stop()

        if run_resp.status_code != 201:
            st.error(f"Apify error {run_resp.status_code}: {run_resp.text[:200]}")
            st.stop()

        run_id     = run_resp.json()["data"]["id"]
        dataset_id = run_resp.json()["data"]["defaultDatasetId"]
        status_ph  = st.empty()
        prog_bar   = st.progress(0)
        max_wait, poll_interval, elapsed = 120, 5, 0

        while elapsed < max_wait:
            sr     = requests.get(
                f"https://api.apify.com/v2/actor-runs/{run_id}",
                headers={"Authorization": f"Bearer {apify_token}"}, timeout=15,
            )
            status = sr.json()["data"]["status"]
            prog_bar.progress(min(elapsed / max_wait, 0.95))
            status_ph.info(f"⏳ Scraping... ({elapsed}s) — {status}")
            if status == "SUCCEEDED":
                prog_bar.progress(1.0)
                status_ph.success("✅ Done! Fetching comments...")
                break
            elif status in ("FAILED", "ABORTED", "TIMED-OUT"):
                status_ph.error(f"❌ Run {status}.")
                st.stop()
            time.sleep(poll_interval)
            elapsed += poll_interval
        else:
            st.error("⏱ Timed out. Upload CSV manually instead.")
            st.stop()

        items = requests.get(
            f"https://api.apify.com/v2/datasets/{dataset_id}/items?format=json&clean=true",
            headers={"Authorization": f"Bearer {apify_token}"}, timeout=30,
        ).json()

        if not items:
            st.error("No comments found. Post may be private or have no comments.")
            st.stop()

        st.session_state["url_ready"] = pd.DataFrame(items)
        st.success(f"✅ Scraped {len(items)} comments — running analysis!")

    if "url_ready" in st.session_state:
        render_analysis(st.session_state["url_ready"],
                        prefix="url", source_label="Live Scraped Reel")


# ── TAB 3: Type / Paste Comments ──────────────────────────────
with tab_manual:
    st.markdown("Type or paste comments below — **one comment per line**. Supports any language.")

    user_input = st.text_area(
        "Enter comments",
        placeholder="Love this content!\nبہت اچھا ویڈیو تھا 🔥\nNot impressed at all.\nC'est vraiment incroyable!",
        height=220,
        key="manual_input",
    )

    if st.button("🔍 Analyse", key="btn_manual"):
        lines = [l.strip() for l in user_input.strip().splitlines() if l.strip()]
        if not lines:
            st.warning("Please enter at least one comment.")
        elif len(lines) == 1:
            try:
                from textblob import TextBlob
                pol = TextBlob(lines[0]).sentiment.polarity
            except Exception:
                pol = 0.0
            label = "Positive 😊" if pol > 0.05 else "Negative 😞" if pol < -0.05 else "Neutral 😐"
            color = "#00c851"    if pol > 0.05 else "#ff4444"     if pol < -0.05 else "#ffbb33"
            st.markdown(f"""
            <div style="background:#1e1e2e;border-radius:12px;padding:1.5rem;
                        border-left:5px solid {color};margin-top:1rem;">
                <div style="font-size:0.8rem;color:#aaa;margin-bottom:0.5rem">YOUR COMMENT</div>
                <div style="font-size:1rem;color:#e0e0e0;margin-bottom:1.2rem">"{lines[0]}"</div>
                <div style="font-size:2.2rem;font-weight:800;color:{color}">{label}</div>
                <div style="font-size:0.85rem;color:#888;margin-top:0.4rem">
                    Confidence: {round(abs(pol), 2)}
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.session_state["manual_ready"] = pd.DataFrame({"text": lines})
            st.success(f"✅ {len(lines)} comments loaded — running analysis!")

    if "manual_ready" in st.session_state:
        render_analysis(st.session_state["manual_ready"],
                        prefix="manual", source_label="Manually Entered Comments")

if not any(k in st.session_state for k in ["csv_ready", "url_ready", "manual_ready"]):
    st.info("👆 Choose a tab above — upload a CSV, paste a Reel URL, or type comments directly.")
else:
    with st.expander("🔄 Analyse a different source", expanded=False):
        st.markdown("Load new data by switching tabs above, or clear current results below.")
        if st.button("🗑️ Clear all results"):
            for k in ["csv_ready", "url_ready", "manual_ready"]:
                st.session_state.pop(k, None)
            st.rerun()