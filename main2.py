# 📓 Hurst Exponent of a Story's Sentiment Trajectory
# This notebook loads a text, computes sentiment per sentence,
# then computes and plots the Hurst exponent to analyze emotional persistence.

# ✅ Install required packages first (uncomment and run if needed)
# !pip install nltk hurst matplotlib

import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
from hurst import compute_Hc

# Download punkt if not already installed
nltk.download('punkt')

# --- 1️⃣ Load your story ---
filename = 'my_story.txt'  # Change to your file name
with open(filename, encoding='utf-8') as f:
    text = f.read()

# --- 2️⃣ Split text into sentences ---
sentences = nltk.sent_tokenize(text)
print(f"Loaded {len(sentences)} sentences.")

# --- 3️⃣ Compute sentiment per sentence ---
sia = SentimentIntensityAnalyzer()
sentiments = [sia.polarity_scores(s)['compound'] for s in sentences]

# Plot the raw sentiment trajectory
plt.figure(figsize=(12,4))
plt.plot(sentiments, color='purple', alpha=0.7)
plt.title('Sentiment Trajectory (compound score per sentence)')
plt.xlabel('Sentence index')
plt.ylabel('Sentiment score (-1 to +1)')
plt.show()

# --- 4️⃣ Compute the Hurst exponent ---
H, c, data = compute_Hc(sentiments)

print("\n📊 Hurst exponent:", round(H, 3))
if H > 0.5:
    print("Interpretation: persistent trend; emotional rises tend to follow rises.")
elif H < 0.5:
    print("Interpretation: anti-persistent; quick reversals in emotional tone.")
else:
    print("Interpretation: memoryless; like white noise.")

# --- 5️⃣ Optional: plot cumulative sum (emotional arc) ---
import numpy as np
cum_sentiment = np.cumsum(sentiments)

plt.figure(figsize=(12,4))
plt.plot(cum_sentiment, color='teal')
plt.title('Cumulative Sentiment Arc')
plt.xlabel('Sentence index')
plt.ylabel('Cumulative sentiment')
plt.show()

# ✅ Done! You can now interpret your story’s emotional rhythm quantitatively.
# Try comparing different texts or genres to see if your writing style has higher or lower Hurst exponent.
