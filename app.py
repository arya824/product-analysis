import streamlit as st
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Download VADER lexicon if not already present
nltk.download('vader_lexicon')

# Initialize analyzer
sia = SentimentIntensityAnalyzer()

# Streamlit app
st.title("📝 Sentiment Analysis Web App")
st.write("Analyze the sentiment of your text using **NLTK VADER**.")

# Input box
user_input = st.text_area("Enter your text here:")

if st.button("Analyze Sentiment"):
    if user_input.strip() != "":
        # Get sentiment scores
        sentiment = sia.polarity_scores(user_input)

        # Show results
        st.write("### Sentiment Scores:")
        st.json(sentiment)

        # Decide overall sentiment
        if sentiment['compound'] >= 0.05:
            st.success("Overall Sentiment: 😊 Positive")
        elif sentiment['compound'] <= -0.05:
            st.error("Overall Sentiment: 😞 Negative")
        else:
            st.warning("Overall Sentiment: 😐 Neutral")
    else:
        st.warning("Please enter some text to analyze!")
