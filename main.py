#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Filmception: AI-powered Multilingual Movie Summary Translator and Genre Classifier
"""

import os
import re
import pandas as pd
import numpy as np
import streamlit as st
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from gtts import gTTS
from googletrans import Translator
import pickle
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
def download_nltk_data():
    for resource in ['stopwords', 'punkt', 'wordnet']:
        try:
            nltk.data.find(f'tokenizers/{resource}')
        except LookupError:
            nltk.download(resource)

# Constants
DATA_DIR = 'MovieSummaries'
MODELS_DIR = 'models'
AUDIO_DIR = 'audio_files'
LANGUAGES = {
    'Arabic': 'ar',
    'Urdu': 'ur',
    'Korean': 'ko',
    'English': 'en'  # Original language
}

# Create necessary directories
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(AUDIO_DIR, exist_ok=True)

class DataPreprocessor:
    def __init__(self, data_dir=DATA_DIR):
        self.data_dir = data_dir
        self.plot_summaries_path = os.path.join(data_dir, 'plot_summaries.txt')
        self.movie_metadata_path = os.path.join(data_dir, 'movie.metadata.tsv')
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        
    def load_data(self):
        """Load plot summaries and movie metadata"""
        print("Loading data...")
        
        # Load plot summaries
        plot_summaries_df = pd.read_csv(
            self.plot_summaries_path, 
            sep='\t', 
            header=None, 
            names=['movie_id', 'summary']
        )
        
        # Load movie metadata
        movie_metadata_df = pd.read_csv(
            self.movie_metadata_path, 
            sep='\t', 
            header=None, 
            names=[
                'movie_id', 'freebase_id', 'movie_name', 'release_date', 
                'revenue', 'runtime', 'languages', 'countries', 'genres'
            ]
        )
        
        # Merge plot summaries with metadata
        merged_df = pd.merge(plot_summaries_df, movie_metadata_df, on='movie_id')
        
        # Extract genres into a list format
        merged_df['genres'] = merged_df['genres'].apply(self._extract_genres)
        
        print(f"Loaded {len(merged_df)} movies with summaries and metadata.")
        return merged_df
    
    def _extract_genres(self, genres_str):
        """Extract genres from the string format to a list of genre names"""
        if pd.isna(genres_str) or not genres_str:
            return []
        
        # Parse genre string in the format "/m/01jfsb:Film noir|/m/03npn:Thriller|..."
        genre_names = []
        for genre in genres_str.split('|'):
            if ':' in genre:
                # Extract only the genre name, removing any extra metadata
                genre_name = genre.split(':')[1].strip()
                
                # Further clean the genre names by removing any additional text after commas, quotes, 
                # or special characters that might be part of the structured data
                if ',' in genre_name:
                    genre_name = genre_name.split(',')[0].strip()
                if '"' in genre_name:
                    genre_name = genre_name.split('"')[0].strip()
                if '"' in genre_name:
                    genre_name = genre_name.replace('"', '').strip()
                if '}' in genre_name:
                    genre_name = genre_name.split('}')[0].strip()
                
                # Remove any identifiers or references like "/m/12345"
                genre_name = re.sub(r'\s*/m/[\w\d]+', '', genre_name).strip()
                
                # Only add non-empty genre names
                if genre_name:
                    genre_names.append(genre_name)
        
        return genre_names
    
    def clean_text(self, text):
        """Clean and preprocess text data"""
        if pd.isna(text) or not text:
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and digits
        text = re.sub(r"[^a-zA-Z\s]", "", text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stop words
        tokens = [word for word in tokens if word not in self.stop_words]
        
        # Lemmatize
        tokens = [self.lemmatizer.lemmatize(word) for word in tokens]
        
        # Join tokens back into a string
        cleaned_text = ' '.join(tokens)
        
        return cleaned_text
    
    def preprocess_data(self, sample_size=None):
        """Preprocess data and split into train/test sets"""
        # Load and merge data
        df = self.load_data()
        
        # Take a sample if specified
        if sample_size and sample_size < len(df):
            df = df.sample(n=sample_size, random_state=42)
        
        # Clean summaries
        print("Cleaning and preprocessing summaries...")
        df['cleaned_summary'] = df['summary'].apply(self.clean_text)
        
        # Filter out movies without genres
        df = df[df['genres'].map(len) > 0]
        
        # Save preprocessed data
        preprocessed_path = os.path.join(self.data_dir, 'preprocessed_data.csv')
        df.to_csv(preprocessed_path, index=False)
        print(f"Preprocessed data saved to {preprocessed_path}")
        
        return df


class GenrePredictor:
    def __init__(self, models_dir=MODELS_DIR):
        self.models_dir = models_dir
        self.vectorizer = None
        self.label_binarizer = None
        self.model = None
        
    def train_model(self, X_train, y_train):
        """Train a multi-label logistic regression model for genre prediction"""
        print("Training model...")
        
        # Create a TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        X_train_vectorized = self.vectorizer.fit_transform(X_train)
        
        # Create a multi-label binarizer
        self.label_binarizer = MultiLabelBinarizer()
        y_train_binarized = self.label_binarizer.fit_transform(y_train)
        
        # Train a logistic regression model for multi-label classification
        base_model = LogisticRegression(max_iter=1000, C=1.0, class_weight='balanced')
        self.model = MultiOutputClassifier(base_model)
        self.model.fit(X_train_vectorized, y_train_binarized)
        
        # Save the trained model, vectorizer and label binarizer
        os.makedirs(self.models_dir, exist_ok=True)
        pickle.dump(self.vectorizer, open(os.path.join(self.models_dir, 'vectorizer.pkl'), 'wb'))
        pickle.dump(self.label_binarizer, open(os.path.join(self.models_dir, 'label_binarizer.pkl'), 'wb'))
        pickle.dump(self.model, open(os.path.join(self.models_dir, 'genre_model.pkl'), 'wb'))
        
        print("Model training completed and saved.")
        
    def evaluate_model(self, X_test, y_test):
        """Evaluate the trained model"""
        print("Evaluating model...")
        
        # Transform test data
        X_test_vectorized = self.vectorizer.transform(X_test)
        y_test_binarized = self.label_binarizer.transform(y_test)
        
        # Make predictions
        y_pred_binarized = self.model.predict(X_test_vectorized)
        
        # Compute evaluation metrics
        accuracy = accuracy_score(y_test_binarized, y_pred_binarized)
        precision = precision_score(y_test_binarized, y_pred_binarized, average='micro')
        recall = recall_score(y_test_binarized, y_pred_binarized, average='micro')
        f1 = f1_score(y_test_binarized, y_pred_binarized, average='micro')
        
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        
        # Create a confusion matrix for the most common genres
        self._plot_confusion_matrix(y_test_binarized, y_pred_binarized)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    def _plot_confusion_matrix(self, y_true, y_pred):
        """Plot confusion matrix for the most common genres"""
        # Create confusion matrix for a subset of genres for visualization
        num_top_genres = 10
        
        # Get a reasonable number of genres for visualization
        genre_counts = y_true.sum(axis=0)
        top_genres_idx = np.argsort(-genre_counts)[:num_top_genres]
        
        # Get the genre names
        genre_names = self.label_binarizer.classes_[top_genres_idx]
        
        # Select only the top genres for confusion matrix
        y_true_subset = y_true[:, top_genres_idx]
        y_pred_subset = y_pred[:, top_genres_idx]
        
        # Create a confusion matrix for each genre
        plt.figure(figsize=(12, 10))
        
        for i, genre in enumerate(genre_names):
            cm = confusion_matrix(y_true_subset[:, i], y_pred_subset[:, i])
            
            plt.subplot(4, 3, i+1)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=['Not ' + genre, genre], 
                        yticklabels=['Not ' + genre, genre])
            plt.title(f'Confusion Matrix: {genre}')
            plt.xlabel('Predicted')
            plt.ylabel('True')
        
        plt.tight_layout()
        
        # Save the confusion matrix plot
        plt.savefig(os.path.join(self.models_dir, 'confusion_matrix.png'))
        
    def load_model(self):
        """Load a trained model"""
        model_path = os.path.join(self.models_dir, 'genre_model.pkl')
        vectorizer_path = os.path.join(self.models_dir, 'vectorizer.pkl')
        binarizer_path = os.path.join(self.models_dir, 'label_binarizer.pkl')
        
        if os.path.exists(model_path) and os.path.exists(vectorizer_path) and os.path.exists(binarizer_path):
            self.model = pickle.load(open(model_path, 'rb'))
            self.vectorizer = pickle.load(open(vectorizer_path, 'rb'))
            self.label_binarizer = pickle.load(open(binarizer_path, 'rb'))
            print("Model, vectorizer, and label binarizer loaded successfully.")
            return True
        else:
            print("Model files not found. Need to train model first.")
            return False
    
    def predict_genres(self, summary):
        """Predict genres for a given movie summary"""
        # Check if model is loaded
        if not self.model or not self.vectorizer or not self.label_binarizer:
            if not self.load_model():
                return []
        
        # Clean and vectorize the summary
        cleaned_summary = DataPreprocessor().clean_text(summary)
        if not cleaned_summary:
            return ["Unable to process summary"]
            
        summary_vectorized = self.vectorizer.transform([cleaned_summary])
        
        try:
            # Get prediction probabilities instead of just binary predictions
            y_pred_proba = np.array([estimator.predict_proba(summary_vectorized)[:, 1] for estimator in self.model.estimators_])
            y_pred_proba = y_pred_proba.T
            
            # Set a threshold for prediction - this can be tuned
            threshold = 0.2  # Lower threshold to catch more genres
            y_pred_binary = (y_pred_proba >= threshold).astype(int)
            
            # If no genres meet the threshold, pick the top 3 most probable ones
            if not np.any(y_pred_binary):
                top_indices = np.argsort(-y_pred_proba[0])[:3]  # Select top 3
                y_pred_binary = np.zeros_like(y_pred_proba, dtype=int)
                y_pred_binary[0, top_indices] = 1
            
            # Convert binary predictions back to genre names
            predicted_genres = self.label_binarizer.inverse_transform(y_pred_binary)[0]
            
            # If still no genres, return the most common ones
            if not predicted_genres:
                common_genres = ["Drama", "Comedy", "Action"]
                return common_genres
                
            return list(predicted_genres)
        except Exception as e:
            print(f"Prediction error: {e}")
            # Return some default genres if prediction fails
            return ["Drama", "Thriller"]


class AudioLibrary:
    def __init__(self, audio_dir=AUDIO_DIR):
        """Initialize the audio library"""
        self.audio_dir = audio_dir
        os.makedirs(audio_dir, exist_ok=True)
    
    def get_audio_directory(self):
        """Get the path to the audio directory"""
        return self.audio_dir
    
    def get_summary_ids(self):
        """Get unique summary IDs from the audio files"""
        try:
            files = [f for f in os.listdir(self.audio_dir) if f.endswith('.mp3')]
            # Extract the summary IDs (hash part before the underscore)
            summary_ids = set()
            for file in files:
                parts = file.split('_')
                if len(parts) > 1:
                    summary_ids.add(parts[0])
            return sorted(list(summary_ids))
        except Exception as e:
            print(f"Error getting summary IDs: {e}")
            return []
    
    def get_audio_files_for_summary(self, summary_id):
        """Get all audio files for a specific summary ID"""
        try:
            files = [f for f in os.listdir(self.audio_dir) if f.startswith(f"{summary_id}_") and f.endswith('.mp3')]
            return sorted(files)
        except Exception as e:
            print(f"Error getting audio files for summary {summary_id}: {e}")
            return []
    
    def get_language_for_file(self, filename):
        """Extract language code from filename"""
        try:
            parts = filename.split('_')
            if len(parts) > 1:
                # Get the language code before .mp3
                lang_code = parts[-1].split('.')[0]
                
                # Convert language code to full name
                for lang_name, code in LANGUAGES.items():
                    if code == lang_code:
                        return lang_name
                return lang_code.upper()
            return "Unknown"
        except Exception as e:
            print(f"Error getting language for file {filename}: {e}")
            return "Unknown"
    
    def get_audio_bytes(self, filename):
        """Get audio file bytes for streaming"""
        try:
            file_path = os.path.join(self.audio_dir, filename)
            if os.path.exists(file_path):
                with open(file_path, 'rb') as f:
                    return f.read()
            return None
        except Exception as e:
            print(f"Error reading audio file {filename}: {e}")
            return None


class MultilingualTranslator:
    def __init__(self, audio_dir=AUDIO_DIR):
        self.translator = Translator()
        self.audio_dir = audio_dir
        os.makedirs(audio_dir, exist_ok=True)
        
    def translate_text(self, text, target_language):
        """Translate text to the target language"""
        if not text:
            return ""
        
        try:
            translation = self.translator.translate(text, dest=target_language)
            return translation.text
        except Exception as e:
            print(f"Translation error: {e}")
            return ""
    
    def text_to_speech(self, text, language, filename):
        """Convert text to speech and save as an audio file"""
        if not text:
            return None
        
        try:
            tts = gTTS(text=text, lang=language, slow=False)
            file_path = os.path.join(self.audio_dir, filename)
            tts.save(file_path)
            return file_path
        except Exception as e:
            print(f"Text-to-speech error: {e}")
            return None
    
    def process_summary(self, summary, language_code):
        """Translate a summary and convert to audio"""
        # Generate a unique filename
        import hashlib
        hash_object = hashlib.md5(summary.encode())
        filename = f"{hash_object.hexdigest()}_{language_code}.mp3"
        
        # Check if the file already exists
        file_path = os.path.join(self.audio_dir, filename)
        if os.path.exists(file_path):
            return {
                'translation': None,  # We don't have the cached translation text
                'audio_path': file_path
            }
        
        # Translate the text
        translated_text = self.translate_text(summary, language_code)
        
        # Convert to speech
        audio_path = self.text_to_speech(translated_text, language_code, filename)
        
        return {
            'translation': translated_text,
            'audio_path': audio_path
        }


class FilmceptionApp:
    def __init__(self):
        self.preprocessor = DataPreprocessor()
        self.genre_predictor = GenrePredictor()
        self.translator = MultilingualTranslator()
        self.audio_library = AudioLibrary()
        
    def train_and_evaluate(self, sample_size=None, test_size=0.2):
        """Train and evaluate the genre prediction model"""
        # Preprocess data
        data = self.preprocessor.preprocess_data(sample_size)
        
        # Split data into train and test sets
        X = data['cleaned_summary']
        y = data['genres']
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # Train the model
        self.genre_predictor.train_model(X_train, y_train)
        
        # Evaluate the model
        metrics = self.genre_predictor.evaluate_model(X_test, y_test)
        
        return metrics
    
    def run_streamlit_app(self):
        """Run the Streamlit app for the user interface"""
        # This function is defined in the main block below
        pass


def main():
    # Download NLTK data
    download_nltk_data()
    
    # Check if the model already exists
    model_path = os.path.join(MODELS_DIR, 'genre_model.pkl')
    
    app = FilmceptionApp()
    
    # If the model doesn't exist, train it
    if not os.path.exists(model_path):
        print("No trained model found. Training a new model...")
        app.train_and_evaluate(sample_size=5000)  # Use a sample for faster training
    
    # Run the Streamlit app
    from streamlit_app import run_app
    run_app(app)


if __name__ == "__main__":
    main()