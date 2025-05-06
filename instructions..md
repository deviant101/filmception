# Filmception

**An AI-powered Multilingual Movie Summary Translator and Genre Classifier**

---

## Overview

This project is designed to develop a comprehensive system for processing movie summaries, predicting movie genres, and converting movie summaries into audio formats. The goal is to create a user-interactive menu-based system where:

- A user can input a movie summary.
- The system will offer the option to convert the summary into audio in multiple languages (Arabic, Urdu, and Korean).
- The system will also provide an option to predict the movie's genre(s) based on the summary using a machine learning model.

The system will allow users to select language preferences for audio conversion, as well as the option to classify the movie into one or more genres based on the machine learning model trained on the CMU Movie Summary dataset.

**Dataset**: [CMU Movie Summary Dataset](https://www.kaggle.com/datasets/msafi04/movies-genre-dataset-cmu-movie-summary)

---

## Project Components

### 1. Data Preprocessing and Cleaning

#### Summary Extraction and Cleaning
- Preprocess the movie summaries extracted from the CMU Movie Summary Dataset. The preprocessing will involve:
  - Removing special characters, stopwords, and redundant spaces.
  - Lowercasing the text to standardize it.
  - Tokenization to break down the summaries into individual words or phrases.
  - Stemming/Lemmatization to reduce words to their base or root form.
  - Removing non-relevant words like numbers and punctuations that don’t add value to the genre classification.

#### Metadata Extraction
- From the `movie.metadata.tsv` file, extract the genre information for each movie. The genres are provided as multi-labels, indicating that each movie can belong to multiple genres (e.g., "Action", "Adventure", "Comedy").
- The final output from this preprocessing should be a new cleaned file containing the Movie ID, summary, and genres (one or more). This file will be used for training the genre prediction model.

#### Train-Test Split
- Perform a train-test split on the dataset. The split should be carefully chosen to prevent data leakage and ensure robust model evaluation.

---

### 2. Text Translation and Audio Conversion

#### Text Translation
- Once the movie summaries are cleaned and prepared, translate each summary into Arabic, Urdu, and Korean. This will test the capability to work with multilingual data.
- Use existing translation tools or APIs such as Google Translate, MarianMT (Hugging Face), or DeepL to achieve this.

#### Audio Conversion
- After translation, the translated summaries will be converted into audio using a Text-to-Speech (TTS) engine (such as gTTS, pyttsx3, or Amazon Polly).
- The audio will be available for playback, and the user will be able to choose which language they want to listen to. This allows the user to hear the movie summaries in multiple languages based on their preference.

> **Note**: At least 50 movie summaries from the dataset should be translated, converted to audio, and saved.

---

### 3. Movie Genre Prediction Model

#### Model Development
- Build a machine learning model to predict the genres of movies based on their summaries. This will be a multi-label classification problem, where a movie can belong to more than one genre.
- Choose the model architecture:
  - Simpler model: Logistic Regression.

#### Feature Extraction
- You will need to extract features from the movie summaries. This could involve using TF-IDF, word embeddings (like Word2Vec, GloVe), or even directly using pre-trained transformers for text representatio

#### Evaluation
- Evaluate the model’s performance using the following metrics:
  - **Accuracy**: The overall accuracy of the model.
  - **Precision, Recall, F1-Score**: Assess the performance of the model in identifying specific genres.
  - **Confusion Matrix**: Visualize the classification results for the training and test sets.

---

### 4. Streamlit Based Graphical User Interface (Menu-Based System)

The core of this project is an interactive, menu-driven system. The workflow will be as follows:

1. **Input**: The user inputs a movie summary.
2. **Options**:
   - **Option 1: Convert Summary to Audio**
     - The user selects the language for the audio (e.g., Arabic, Urdu, Korean).
     - The system generates the audio and plays it for the user.
   - **Option 2: Predict Genre**
     - The user selects the option to classify the genre of the movie based on the summary.
     - The system outputs the predicted genre(s) of the movie.

The system should be designed in a way that allows users to explore both features sequentially.

---