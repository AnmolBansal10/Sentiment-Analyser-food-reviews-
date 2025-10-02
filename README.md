# Sentiment Spice - Food Review Analyzer 🌶️

A powerful sentiment analysis tool specifically designed for analyzing food reviews using advanced NLP models. This project combines VADER and RoBERTa models to provide accurate sentiment analysis for food-related text.

## Features

- **Dual Model Analysis**: Uses both VADER and RoBERTa models for comprehensive sentiment analysis
- **Food-Focused**: Optimized for food and dining experience reviews
- **Interactive Web Interface**: Beautiful Streamlit-based web application
- **Real-time Analysis**: Instant sentiment scoring with visual progress bars
- **Blended Results**: Combines both models for more accurate predictions

## Models Used

1. **VADER (Valence Aware Dictionary and sEntiment Reasoner)**: Lexicon and rule-based sentiment analysis
2. **RoBERTa**: Pre-trained transformer model fine-tuned on Twitter sentiment data

## Installation

1. Clone the repository:
```bash
git clone https://github.com/AnmolBansal10/Sentiment-Analyser-food-reviews-.git
cd Sentiment-Analyser-food-reviews-
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download NLTK data (if not already downloaded):
```python
import nltk
nltk.download('vader_lexicon')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger')
```

## Usage

### Streamlit Web App
Run the main application:
```bash
streamlit run app1.py
```

### HTML Interface
Open `sentiment_ananlyser.html` in your web browser for a static demonstration.

### Jupyter Notebook
Open `sentimental_ananlysis_try.ipynb` to explore the analysis process and model comparison.

## Project Structure

```
├── app1.py                          # Main Streamlit application
├── sentiment_ananlyser.html         # HTML interface
├── sentimental_ananlysis_try.ipynb  # Jupyter notebook with analysis
├── roberta_sentiment_model/         # Pre-trained RoBERTa model files
│   ├── config.json
│   ├── pytorch_model.bin
│   ├── tokenizer.json
│   └── ...
├── requirements.txt                 # Python dependencies
└── README.md                       # This file
```

## How It Works

1. **Text Input**: Users enter their food review text
2. **Food Detection**: The system checks if the text is food-related
3. **Dual Analysis**: Both VADER and RoBERTa models analyze the text
4. **Blended Results**: Results are combined using weighted averaging (VADER 40% + RoBERTa 60%)
5. **Visual Output**: Results are displayed with progress bars and sentiment indicators

## Example Usage

```python
# Example review
review = "The biryani was absolutely aromatic and delicious, a must-try!"

# The system will analyze this and provide:
# - VADER sentiment scores (positive, neutral, negative, compound)
# - RoBERTa sentiment scores (positive, neutral, negative)
# - Final blended result with confidence scores
```

## Technologies Used

- **Python 3.8+**
- **Streamlit** - Web application framework
- **Transformers** - Hugging Face transformers library
- **NLTK** - Natural Language Toolkit
- **PyTorch** - Deep learning framework
- **Pandas** - Data manipulation
- **Matplotlib/Seaborn** - Data visualization

## Contributing

Feel free to submit issues and enhancement requests!

## License

This project is open source and available under the [MIT License](LICENSE).

## Acknowledgments

- [Hugging Face](https://huggingface.co/) for the pre-trained models
- [Cardiff NLP](https://cardiffnlp.github.io/) for the Twitter RoBERTa model
- [NLTK](https://www.nltk.org/) for VADER sentiment analysis
