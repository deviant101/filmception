# 🎬 Filmception

**An AI-powered Multilingual Movie Summary Translator and Genre Classifier**

![Python Badge](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit Badge](https://img.shields.io/badge/Streamlit-1.10.0%2B-FF4B4B)
![scikit-learn Badge](https://img.shields.io/badge/scikit--learn-1.0.0%2B-F7931E)
![NLTK Badge](https://img.shields.io/badge/NLTK-3.7%2B-green)

## 📋 Overview

Filmception is an interactive AI system that processes movie summaries to provide two main functions:

1. **Multilingual Translation and Audio Conversion**: Convert movie summaries into audio in multiple languages (Arabic, Urdu, Korean, and English).
2. **Genre Prediction**: Predict movie genres based on the summary using a machine learning model trained on the CMU Movie Summary dataset.

The system provides a user-friendly Streamlit interface where users can input movie summaries, receive genre predictions, listen to translated audio, and browse a library of pre-generated audio summaries.

## ✨ Features

- **Text Translation**: Translate movie summaries to Arabic, Urdu, Korean, and English
- **Audio Conversion**: Convert translated text to speech in respective languages
- **Genre Prediction**: Predict movie genres from summaries using machine learning
- **Audio Library**: Browse 50+ pre-generated movie summary audios in multiple languages
- **Interactive UI**: Clean, intuitive Streamlit-based user interface

## 🧰 Technology Stack

- **Python**: Core programming language
- **Streamlit**: Web interface development
- **NLTK**: Natural language processing for text cleaning and preprocessing
- **scikit-learn**: Machine learning for genre prediction
- **googletrans**: Text translation API integration
- **gTTS (Google Text-to-Speech)**: Audio conversion
- **pandas & numpy**: Data manipulation 
- **matplotlib & seaborn**: Visualization

## 📊 Dataset

This project uses the [CMU Movie Summary Dataset](https://www.kaggle.com/datasets/msafi04/movies-genre-dataset-cmu-movie-summary), which includes:

- 42,306 movie plot summaries extracted from Wikipedia
- Movie metadata including genres, release dates, runtime, etc.
- Character metadata

## 🛠️ Installation

### Prerequisites

- Python 3.8 or higher
- Git

### Setup Instructions

1. Clone the repository:
   ```bash
   git clone https://github.com/deviant101/filmception.git
   cd filmception
   ```

2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the CMU Movie Summary Dataset and place it in the `MovieSummaries` directory.

4. Generate sample audio files (optional, but recommended):
   ```bash
   python generate_sample_audios.py
   ```

5. Run the Streamlit application:
   ```bash
   streamlit run main.py
   ```
   This will start the Streamlit server and automatically open the application in your default web browser.

## 🎮 Usage

### Translating and Converting Summaries to Audio

1. Navigate to the "Movie Summary Audio Conversion" section
2. Enter a movie plot summary in the text area
3. Click on your preferred language (Arabic, Urdu, Korean, or English)
4. The system will translate the text and generate an audio file
5. Listen to the audio or download it for later use

### Predicting Movie Genres

1. Navigate to the "Genre Prediction" section
2. Enter a movie plot summary in the text area
3. Click "Predict Genre"
4. View the predicted genres and a brief analysis

### Browsing Audio Library

1. Navigate to the "Audio Library" section
2. Browse through the collection of pre-generated movie summary audios
3. Use the search function to find specific summaries
4. Click on any summary to expand and listen to its translations
5. Download audio files if needed

## 🧠 Model Details

The genre prediction model uses:

- **Text Vectorization**: TF-IDF (Term Frequency-Inverse Document Frequency)
- **Algorithm**: Multi-output Logistic Regression
- **Classification Type**: Multi-label (each movie can belong to multiple genres)
- **Training Method**: Supervised learning on cleaned and preprocessed movie summaries
- **Evaluation Metrics**: Accuracy, Precision, Recall, F1-score

## 📁 Project Structure

```
filmception/
├── audio_files/              # Generated audio files
├── models/                   # Trained machine learning models
├── MovieSummaries/           # Dataset directory
│   ├── plot_summaries.txt    # Movie summaries
│   ├── movie.metadata.tsv    # Movie metadata
│   └── preprocessed_data.csv # Cleaned and preprocessed data
├── generate_sample_audios.py # Script to generate sample audio files
├── main.py                   # Main application file
├── requirements.txt          # Project dependencies
├── streamlit_app.py          # Streamlit interface
├── technical_report.md       # Detailed technical documentation
└── README.md                 # Project documentation
```

## 📚 Future Improvements

- Expand language support to include more languages
- Implement more advanced NLP techniques for better genre prediction
- Add user accounts to save favorite translations and predictions
- Improve audio quality and pronunciation for non-English languages
- Add visual analysis of genre predictions with confidence scores

## 📖 Documentation

The project includes two main documentation files:

1. **README.md** (this file): Provides an overview, installation instructions, and general usage guidelines.

2. **[Technical Report](technical_report.md)**: Contains detailed information about the system architecture, implementation details, algorithms used, and performance analysis. This comprehensive report is intended for developers and researchers interested in the technical aspects of the project.

## 📄 License

This project is licensed under the [MIT License](LICENSE) - see the LICENSE file for details.

## 🙏 Acknowledgements

- CMU for providing the Movie Summary Dataset
- Google Cloud for translation and text-to-speech APIs
- The Streamlit team for their excellent framework