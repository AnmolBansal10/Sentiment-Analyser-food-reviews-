import streamlit as st
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# --- Page Configuration ---
st.set_page_config(
    page_title="Sentiment Spice",
    page_icon="🌶️",
    layout="centered"
)

# --- Model Loading (Cached for performance) ---
@st.cache_resource
def load_models():
    """Load and cache the NLP models to avoid reloading on each run."""
    # RoBERTa Model
    roberta_model_name = "cardiffnlp/twitter-roberta-base-sentiment"
    tokenizer = AutoTokenizer.from_pretrained(roberta_model_name)
    model = AutoModelForSequenceClassification.from_pretrained(roberta_model_name)
    
    # VADER Model
    # Try to download vader_lexicon, handle error if it fails
    try:
        nltk.data.find('sentiment/vader_lexicon.zip')
    except nltk.downloader.DownloadError:
        st.info("Downloading VADER lexicon for the first time...")
        nltk.download('vader_lexicon')
        
    sia = SentimentIntensityAnalyzer()
    
    return tokenizer, model, sia

tokenizer, model, sia = load_models()

# --- Custom Styling ---
def local_css():
    """Injects custom CSS for theming."""
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&family=Tiro+Devanagari+Hindi:wght@400&display=swap');
        
        /* Gradient Background */
        .stApp {
            background-image: linear-gradient(to top right, #f8b400, #f87217, #e11d48);
            background-attachment: fixed;
            background-size: cover;
        }
        
        /* Title Font */
        .stTitle, .stHeader {
            font-family: 'Tiro Devanagari Hindi', serif;
        }
        
        /* Custom Card Styling */
        .result-card {
            background-color: rgba(255, 255, 255, 0.1);
            border: 1px solid rgba(255, 255, 255, 0.2);
            padding: 20px;
            border-radius: 15px;
            margin-bottom: 20px;
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
        }
        
        /* Progress Bar Styling */
        .progress-container {
            display: flex;
            align-items: center;
            margin-bottom: 8px;
        }
        .progress-label {
            width: 70px; /* Fixed width for labels */
            font-weight: 600;
        }
        .progress-bar-bg {
            flex-grow: 1;
            background-color: rgba(0, 0, 0, 0.2);
            border-radius: 10px;
            height: 10px;
            margin: 0 10px;
        }
        .progress-bar {
            height: 10px;
            border-radius: 10px;
        }
        .pos-bar { background-color: #22c55e; } /* Green */
        .neu-bar { background-color: #facc15; } /* Yellow */
        .neg-bar { background-color: #ef4444; } /* Red */

    </style>
    """, unsafe_allow_html=True)

# --- Analysis Functions ---
def is_food_related(text):
    """Checks if the text contains food-related keywords."""
    food_keywords = [
        'food', 'eat', 'taste', 'tasted', 'delicious', 'restaurant', 'cafe', 'dish', 'recipe',
        'ingredients', 'cook', 'chef', 'menu', 'order', 'delivery', 'beverage', 'drink',
        'meal', 'breakfast', 'lunch', 'dinner', 'snack', 'flavor', 'aroma', 'tasty',
        'sweet', 'sour', 'salty', 'bitter', 'spicy', 'bland', 'fresh', 'stale',
        'hungry', 'dining', 'cuisine', 'serve', 'waiter', 'waitress', 'kitchen',
        # A few common dishes to catch more cases
        'biryani', 'pizza', 'burger', 'curry', 'sushi', 'pasta', 'salad', 'soup', 'dessert'
    ]
    text_lower = text.lower()
    return any(keyword in text_lower for keyword in food_keywords)

def analyze_roberta(text):
    """Performs sentiment analysis using the RoBERTa model."""
    encoded_text = tokenizer(text, return_tensors='pt')
    output = model(**encoded_text)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    return {'neg': scores[0], 'neu': scores[1], 'pos': scores[2]}

def get_sentiment_details(score_dict, model_type='roberta'):
    """Determines the overall sentiment and associated emoji."""
    if model_type == 'vader':
        score = score_dict['compound']
        if score >= 0.05:
            return "Positive", "😍"
        elif score <= -0.05:
            return "Negative", "😠"
        else:
            return "Neutral", "🤔"
    else: # roberta or final
        if score_dict['pos'] > score_dict['neg'] and score_dict['pos'] > score_dict['neu']:
            return "Positive", "😍"
        elif score_dict['neg'] > score_dict['pos'] and score_dict['neg'] > score_dict['neu']:
            return "Negative", "😠"
        else:
            return "Neutral", "🤔"
            
# --- UI Rendering Functions ---
def styled_header(text, size=5, text_align="center"):
    """Creates a styled header with custom font."""
    st.markdown(f'<h{size} style="text-align: {text_align}; font-family: \'Tiro Devanagari Hindi\', serif;">{text}</h{size}>', unsafe_allow_html=True)

def display_progress_bar(label, value, color_class):
    """Displays a custom styled progress bar."""
    st.markdown(f"""
        <div class="progress-container">
            <span class="progress-label">{label}</span>
            <div class="progress-bar-bg">
                <div class="progress-bar {color_class}" style="width: {value * 100:.2f}%;"></div>
            </div>
            <span>{value:.2f}</span>
        </div>
    """, unsafe_allow_html=True)

# --- Main Application ---
local_css()

st.markdown('<h1 class="stTitle" style="text-align: center;">Sentiment Spice</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; opacity: 0.9;">Analyse Your Indian Food Reviews</p>', unsafe_allow_html=True)

review_text = st.text_area(
    "Enter your food review here:",
    height=150,
    placeholder="e.g., The biryani was absolutely aromatic and delicious, a must-try!"
)

if st.button("Analyze Sentiment"):
    if review_text.strip():
        # Check if the input is food-related before analyzing
        if not is_food_related(review_text):
            st.warning("🤔 This is a food review analyzer. Please enter text related to food or dining experiences for the best results.")
        else:
            with st.spinner("Simmering the results..."):
                # --- VADER Analysis ---
                vader_scores = sia.polarity_scores(review_text)
                vader_sentiment, vader_emoji = get_sentiment_details(vader_scores, 'vader')

                # --- RoBERTa Analysis ---
                roberta_scores = analyze_roberta(review_text)
                roberta_sentiment, roberta_emoji = get_sentiment_details(roberta_scores)

                # --- Blended Result ---
                vader_compound = vader_scores['compound']
                roberta_compound = roberta_scores['pos'] - roberta_scores['neg']
                # Weighted average: VADER 40%, RoBERTa 60%
                final_score = (vader_compound * 0.4) + (roberta_compound * 0.6)
                
                final_sentiment, final_emoji = "Neutral", "🤔"
                if final_score >= 0.05:
                    final_sentiment, final_emoji = "Positive", "😍"
                elif final_score <= -0.05:
                    final_sentiment, final_emoji = "Negative", "😠"

            # --- Display Results ---
            col1, col2 = st.columns(2)

            with col1:
                st.markdown('<div class="result-card">', unsafe_allow_html=True)
                styled_header("VADER Analysis", 3)
                st.markdown(f"<p style='text-align:center; font-size: 50px;'>{vader_emoji}</p>", unsafe_allow_html=True)
                st.markdown(f"<p style='text-align:center; font-size: 24px; font-weight: 600;'>{vader_sentiment}</p>", unsafe_allow_html=True)
                st.markdown("---")
                display_progress_bar("Positive", vader_scores['pos'], "pos-bar")
                display_progress_bar("Neutral", vader_scores['neu'], "neu-bar")
                display_progress_bar("Negative", vader_scores['neg'], "neg-bar")
                st.markdown("---")
                st.metric(label="Compound Score", value=f"{vader_scores['compound']:.2f}")
                st.markdown('</div>', unsafe_allow_html=True)

            with col2:
                st.markdown('<div class="result-card">', unsafe_allow_html=True)
                styled_header("RoBERTa Analysis", 3)
                st.markdown(f"<p style='text-align:center; font-size: 50px;'>{roberta_emoji}</p>", unsafe_allow_html=True)
                st.markdown(f"<p style='text-align:center; font-size: 24px; font-weight: 600;'>{roberta_sentiment}</p>", unsafe_allow_html=True)
                st.markdown("---")
                display_progress_bar("Positive", roberta_scores['pos'], "pos-bar")
                display_progress_bar("Neutral", roberta_scores['neu'], "neu-bar")
                display_progress_bar("Negative", roberta_scores['neg'], "neg-bar")
                st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div class="result-card">', unsafe_allow_html=True)
            styled_header("Final Blended Result", 2)
            st.markdown("<p style='text-align:center; font-size: 14px; opacity: 0.8;'>(VADER 40% + RoBERTa 60%)</p>", unsafe_allow_html=True)
            st.markdown(f"<p style='text-align:center; font-size: 60px;'>{final_emoji}</p>", unsafe_allow_html=True)
            st.markdown(f"<p style='text-align:center; font-size: 32px; font-weight: 600;'>{final_sentiment}</p>", unsafe_allow_html=True)
            st.metric(label="Weighted Score", value=f"{final_score:.2f}")
            st.markdown('</div>', unsafe_allow_html=True)

    else:
        st.warning("🌶️ Please enter a review to analyze.")

st.markdown("<br><br><p style='text-align:center; font-size: 12px; opacity: 0.7;'>Powered by VADER & RoBERTa</p>", unsafe_allow_html=True)

