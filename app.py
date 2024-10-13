import streamlit as st
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pandas as pd
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import time
import io
import json

# Download NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

st.set_page_config(page_title="Advanced Sentiment Analysis Dashboard", page_icon="🧠", layout="wide")

# Sentiment Analyzer class
class SentimentAnalyzer:
    def __init__(self, model_name):
        self.llm = Ollama(model=model_name)
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a sentiment analysis expert. Analyze the sentiment of the given text and respond with ONLY ONE WORD: 'positive', 'negative', or 'neutral'."),
            ("human", "Analyze the sentiment of the following text: {text}")
        ])
        self.output_parser = StrOutputParser()
        self.chain = self.prompt | self.llm | self.output_parser

    def analyze_sentiment(self, text):
        try:
            result = self.chain.invoke({"text": text})
            st.write("Raw model output:", result)  # Debug output
            
            # Extract sentiment from the output
            sentiment = self.extract_sentiment(result)
            
            st.write("Extracted sentiment:", sentiment)  # Debug output
            
            # Ensure the sentiment is one of the expected values
            if sentiment not in ['positive', 'negative', 'neutral']:
                raise ValueError(f"Unexpected sentiment: {sentiment}")
            
            return {
                'classification': sentiment,
                'confidence': 1.0,  # Default confidence
                'summary': "Summary not available"  # Default summary
            }
        except Exception as e:
            st.error(f"An error occurred during sentiment analysis: {str(e)}")
            st.error(f"Raw output: {result}")  # Debug output
            return None

    def extract_sentiment(self, text):
        # List of possible sentiment words
        sentiments = ['positive', 'negative', 'neutral']
        
        # Convert text to lowercase for case-insensitive matching
        text = text.lower()
        
        # Check for each sentiment word in the text
        for sentiment in sentiments:
            if sentiment in text:
                return sentiment
        
        # If no sentiment is found, return 'unknown'
        return 'unknown'

# Caching the SentimentAnalyzer instance
@st.cache_resource
def get_sentiment_analyzer(model_name):
    return SentimentAnalyzer(model_name)

# Function to create a word cloud
def create_word_cloud(text):
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text)
    filtered_text = [word.lower() for word in word_tokens if word.isalnum() and word.lower() not in stop_words]
    
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(filtered_text))
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    return fig

# Function to create a sentiment gauge chart
def create_sentiment_gauge(confidence, sentiment):
    colors = {'positive': 'green', 'neutral': 'gray', 'negative': 'red'}
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = confidence,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': f"Sentiment: {sentiment.capitalize()}"},
        gauge = {
            'axis': {'range': [0, 1]},
            'bar': {'color': colors[sentiment.lower()]},
            'steps': [
                {'range': [0, 0.33], 'color': "lightgray"},
                {'range': [0.33, 0.66], 'color': "gray"},
                {'range': [0.66, 1], 'color': "darkgray"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': confidence
            }
        }
    ))
    return fig

# Function to create a sample CSV
def create_sample_csv():
    sample_data = [
        {"text": "I love this product! It's amazing and works perfectly.", "sentiment": "positive"},
        {"text": "The service was terrible and the staff was rude.", "sentiment": "negative"},
        {"text": "The movie was okay, nothing special but not bad either.", "sentiment": "neutral"},
        {"text": "This restaurant has the best pizza I've ever tasted!", "sentiment": "positive"},
        {"text": "I'm feeling quite indifferent about the whole situation.", "sentiment": "neutral"}
    ]
    df = pd.DataFrame(sample_data)
    csv = df.to_csv(index=False)
    return csv

# Main function
def main():
    st.title("🧠 Advanced Sentiment Analysis Dashboard")
    st.write("Analyze the sentiment of your text using various LLM models!")

    # Sidebar for model selection and options
    st.sidebar.title("Settings")
    model_name = st.sidebar.selectbox("Select LLM Model", [
        "tinyllama", "llama2", "phi3", "mistral", "codellama", "orca-mini", 
        "vicuna", "stable-beluga", "neural-chat", "starling-lm", "starcoder", 
        "wizardcoder", "falcon", "openhermes", "nous-hermes", "sqlcoder", 
        "replit-code", "deepseek-coder", "phind-codellama", "magicoder", 
        "santacoder", "wizardlm", "openorca", "dolphin", "yarn-mistral", 
        "solar", "yi", "qwen", "meditron", "zephyr", "xwin", "openchat"
    ])
    
    analyzer = get_sentiment_analyzer(model_name)

    # Text input area
    text_input = st.text_area("Enter your text here:", height=150)

    # Batch analysis option
    use_batch = st.checkbox("Enable batch analysis")
    
    if use_batch:
        col1, col2 = st.columns(2)
        with col1:
            uploaded_file = st.file_uploader("Upload a CSV file with texts to analyze", type="csv")
        with col2:
            st.write("Don't have a CSV file? Download a sample:")
            sample_csv = create_sample_csv()
            st.download_button(
                label="Download Sample CSV",
                data=sample_csv,
                file_name="sample_sentiment_analysis.csv",
                mime="text/csv"
            )
        
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            if 'text' not in df.columns:
                st.error("The CSV file must contain a 'text' column.")
            else:
                texts = df['text'].tolist()
    else:
        texts = [text_input] if text_input else []

    if st.button("Analyze Sentiment"):
        if texts:
            results = []
            progress_bar = st.progress(0)
            for i, text in enumerate(texts):
                st.write(f"Analyzing text {i+1}: {text[:100]}...")  # Debug output
                with st.spinner(f"Analyzing text {i+1}/{len(texts)}..."):
                    try:
                        result = analyzer.analyze_sentiment(text)
                        if result:
                            results.append(result)
                            st.success(f"Successfully analyzed text {i+1}")
                        else:
                            st.warning(f"Failed to analyze text {i+1}. Skipping...")
                    except Exception as e:
                        st.error(f"Error analyzing text {i+1}: {str(e)}")
                        st.error(f"Text causing error: {text[:100]}...")  # Show part of the problematic text
                progress_bar.progress((i + 1) / len(texts))
            
            if results:
                st.subheader("Results:")
                for i, result in enumerate(results):
                    with st.expander(f"Analysis {i+1}"):
                        st.write(f"**Text:** {texts[i][:100]}...")
                        st.write(f"**Sentiment:** {result['classification']}")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.plotly_chart(create_sentiment_gauge(result['confidence'], result['classification']), use_container_width=True)
                        with col2:
                            st.pyplot(create_word_cloud(texts[i]))
                
                # Overall statistics
                sentiments = [r['classification'].lower() for r in results]
                sentiment_counts = pd.Series(sentiments).value_counts()
                st.subheader("Overall Sentiment Distribution")
                st.bar_chart(sentiment_counts)
            else:
                st.error("No valid results were obtained. Please try again or check your input.")
                st.error("If the problem persists, try a different model or check your Ollama installation.")
        else:
            st.warning("Please enter some text to analyze or upload a CSV file for batch analysis.")

    # Display model information
    st.sidebar.subheader("Model Information")
    st.sidebar.write(f"Current model: {model_name}")
    st.sidebar.write("Model capabilities may vary. Ensure you have the selected model installed in Ollama.")
    st.sidebar.write("Note: Not all models are optimized for sentiment analysis. Results may vary.")

    # Add a footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("Created by Anoop Maurya with ❤️ using Streamlit")

if __name__ == "__main__":
    main()
