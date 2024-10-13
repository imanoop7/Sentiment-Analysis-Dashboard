# Advanced Sentiment Analysis Dashboard

## Overview
This project is an Advanced Sentiment Analysis Dashboard built using Streamlit and LangChain. It leverages various Large Language Models (LLMs) through Ollama to perform sentiment analysis on user-provided text or batch inputs via CSV files.

For a detailed guide on how this project was built, check out the article: [Beginner's Guide to Building an Advanced Sentiment Analysis Dashboard with Streamlit and Ollama](https://medium.com/@mauryaanoop3/beginners-guide-to-building-an-advanced-sentiment-analysis-dashboard-with-streamlit-and-ollama-ba09023a91fa)

## Features
- Single text and batch sentiment analysis
- Multiple LLM model selection (including TinyLlama, Llama2, Phi3, Mistral, and many more)
- Interactive visualizations:
  - Sentiment gauge charts
  - Word clouds
  - Overall sentiment distribution bar charts
- Batch analysis with CSV file upload
- Sample CSV download for batch analysis testing
- Detailed error handling and debugging output

## Installation

### Prerequisites
- Python 3.7+
- Ollama (with desired models installed)

### Setup
1. Clone the repository:
   ```
   git clone https://github.com/imanoop7/Sentiment-Analysis-Dashboard
   cd sentiment-analysis-dashboard
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

3. Ensure you have Ollama installed and the desired models downloaded.

## Usage

1. Run the Streamlit app:
   ```
   streamlit run app.py
   ```

2. Open your web browser and navigate to the URL provided by Streamlit (usually `http://localhost:8501`).

3. Use the dashboard:
   - Select an LLM model from the sidebar
   - Enter text in the input area or upload a CSV file for batch analysis
   - Click "Analyze Sentiment" to process the input
   - View the results, including sentiment classification, visualizations, and overall statistics

## Project Structure

- `app.py`: Main Streamlit application file
- `requirements.txt`: List of Python dependencies
- `README.md`: Project documentation (this file)

## Dependencies

- streamlit
- langchain
- langchain-community
- ollama
- plotly
- wordcloud
- matplotlib
- pandas
- nltk

## Notes

- The availability and performance of models may vary depending on your Ollama installation.
- Not all models are optimized for sentiment analysis tasks. Results may vary between different models.

## Contributing

Contributions to improve the dashboard are welcome. Please feel free to submit issues or pull requests.

## License

This project is open-source and available under the [MIT License](LICENSE).

## Acknowledgments

Created by Anoop Maurya using Streamlit and LangChain.

## Learn More

For a step-by-step guide on how this dashboard was created, check out the accompanying article:
[Beginner's Guide to Building an Advanced Sentiment Analysis Dashboard with Streamlit and Ollama](https://medium.com/@mauryaanoop3/beginners-guide-to-building-an-advanced-sentiment-analysis-dashboard-with-streamlit-and-ollama-ba09023a91fa)
