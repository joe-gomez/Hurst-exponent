# 📊 Streamlit App: Hurst Exponent of a Story's Sentiment Trajectory

# Install required packages before running:
pip install streamlit nltk matplotlib numpy hurst

import streamlit as st
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import numpy as np
from hurst import compute_Hc

nltk.download('punkt')

st.title("📖 Hurst Exponent Analyzer for Story Sentiment")

uploaded_file = st.file_uploader("Upload a .txt file of your story", type=["txt"])

if uploaded_file is not None:
    text = uploaded_file.read().decode('utf-8')
    sentences = nltk.sent_tokenize(text)
    st.write(f"Number of sentences detected: {len(sentences)}")

    sia = SentimentIntensityAnalyzer()
    sentiments = [sia.polarity_scores(s)['compound'] for s in sentences]

    st.subheader("Sentiment Trajectory")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(sentiments, color='purple', alpha=0.7)
    ax.set_xlabel('Sentence index')
    ax.set_ylabel('Sentiment score (-1 to +1)')
    st.pyplot(fig)

    # --- Compute Hurst exponent using hurst package ---
    H, c, data = compute_Hc(sentiments)

    st.subheader("📊 Hurst Exponent Result")
    st.write(f"Hurst exponent: **{H:.3f}**")

    if H > 0.5:
        st.info("Persistent trend: emotional rises tend to follow rises.")
    elif H < 0.5:
        st.info("Anti-persistent: quick reversals in emotional tone.")
    else:
        st.info("Memoryless: similar to white noise.")

    st.subheader("Cumulative Sentiment Arc")
    cum_sentiment = np.cumsum(sentiments)
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    ax2.plot(cum_sentiment, color='teal')
    ax2.set_xlabel('Sentence index')
    ax2.set_ylabel('Cumulative sentiment')
    st.pyplot(fig2)

    st.success("✅ Analysis complete! Try another text to compare.")
else:
    st.write("👆 Upload a text file to get started.")
