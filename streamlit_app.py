#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Streamlit application for the Filmception project
"""

import os
import streamlit as st
import time
from PIL import Image
import base64
from io import BytesIO
import re
import pandas as pd

# Constants
LANGUAGES = {
    'Arabic': 'ar',
    'Urdu': 'ur',
    'Korean': 'ko',
    'English': 'en'  # Original language
}

def run_app(app):
    """
    Run the Streamlit application
    
    Parameters:
    app -- The FilmceptionApp instance
    """
    # Configure the page
    st.set_page_config(
        page_title="Filmception",
        page_icon="🎬",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Add custom CSS
    st.markdown("""
        <style>
            .main-header {
                font-size: 2.5rem;
                color: #FF4B4B;
                text-align: center;
                margin-bottom: 2rem;
            }
            .sub-header {
                font-size: 1.8rem;
                color: #4B8BFF;
                margin-top: 1.5rem;
                margin-bottom: 1rem;
            }
            .info-text {
                font-size: 1.1rem;
            }
            .footer {
                text-align: center;
                color: #888888;
                margin-top: 3rem;
            }
            .language-button {
                margin: 0.5rem;
                padding: 0.5rem 1rem;
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 5px;
                text-align: center;
                text-decoration: none;
                display: inline-block;
                font-size: 16px;
                cursor: pointer;
            }
            .audio-card {
                background-color: #f8f9fa;
                border-radius: 10px;
                padding: 15px;
                margin-bottom: 15px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            }
            .audio-card:hover {
                box-shadow: 0 8px 12px rgba(0,0,0,0.15);
                transform: translateY(-2px);
                transition: all 0.3s ease;
            }
        </style>
    """, unsafe_allow_html=True)

    # App title and description
    st.markdown('<h1 class="main-header">🎬 Filmception</h1>', unsafe_allow_html=True)
    st.markdown('<p class="info-text">An AI-powered Multilingual Movie Summary Translator and Genre Classifier</p>', 
                unsafe_allow_html=True)

    # Sidebar with app options
    st.sidebar.title("Options")
    app_mode = st.sidebar.radio(
        "Choose the functionality",
        ["Movie Summary Audio Conversion", "Genre Prediction", "Audio Library", "About Project"]
    )

    # Main content based on selected mode
    if app_mode == "Movie Summary Audio Conversion":
        show_summary_translation(app)
    elif app_mode == "Genre Prediction":
        show_genre_prediction(app)
    elif app_mode == "Audio Library":
        show_audio_library(app)
    else:
        show_about_project()

    # Footer
    st.markdown(
        '<div class="footer">Filmception - AI Class Project</div>',
        unsafe_allow_html=True
    )

def show_summary_translation(app):
    """Display the summary translation and audio conversion interface"""
    
    st.markdown('<h2 class="sub-header">Movie Summary Translation & Audio Conversion</h2>', 
                unsafe_allow_html=True)
    
    st.write("Enter a movie summary, select the language, and convert it to audio.")
    
    # Text area for movie summary input
    summary = st.text_area("Enter movie summary:", 
                          height=200, 
                          max_chars=5000,
                          placeholder="Enter a movie plot summary here...")
    
    # Only proceed if there is text input
    if summary:
        st.write("Select language for translation and audio conversion:")
        
        # Create columns for language selection
        cols = st.columns(len(LANGUAGES))
        
        for i, (language_name, language_code) in enumerate(LANGUAGES.items()):
            with cols[i]:
                if st.button(f"{language_name}", key=f"lang_{language_name}"):
                    process_language(app, summary, language_name, language_code)
    else:
        st.info("Please enter a movie summary to continue.")

def process_language(app, summary, language_name, language_code):
    """Process the summary for a selected language"""
    
    # Show a progress bar for processing
    progress_text = f"Converting to {language_name}..."
    my_bar = st.progress(0, text=progress_text)
    
    # Process in stages for better UX
    my_bar.progress(25, text=f"{progress_text} (Translating)")
    
    # Translate and convert to audio
    result = app.translator.process_summary(summary, language_code)
    
    my_bar.progress(75, text=f"{progress_text} (Generating audio)")
    
    if result['audio_path'] and os.path.exists(result['audio_path']):
        # Display the translation if available
        if result['translation']:
            st.subheader(f"Translation ({language_name}):")
            st.write(result['translation'])
        
        # Display audio player
        st.subheader(f"Audio ({language_name}):")
        
        # Get the audio file as bytes
        audio_file = open(result['audio_path'], 'rb')
        audio_bytes = audio_file.read()
        audio_file.close()
        
        # Display the audio player
        st.audio(audio_bytes, format='audio/mp3')
        
        # Provide download link
        st.download_button(
            label=f"Download {language_name} Audio",
            data=audio_bytes,
            file_name=os.path.basename(result['audio_path']),
            mime="audio/mp3"
        )
    else:
        st.error(f"Failed to generate {language_name} audio. Please try again.")
    
    my_bar.progress(100, text="Completed!")

def show_genre_prediction(app):
    """Display the genre prediction interface"""
    
    st.markdown('<h2 class="sub-header">Movie Genre Prediction</h2>', 
                unsafe_allow_html=True)
    
    st.write("Enter a movie summary to predict its genre(s).")
    
    # Check if the model is loaded
    try:
        app.genre_predictor.load_model()
        
        # Text area for movie summary input
        summary = st.text_area("Enter movie summary:", 
                              height=200, 
                              max_chars=5000,
                              placeholder="Enter a movie plot summary here...")
        
        # Only predict if there is text input
        if summary and st.button("Predict Genre"):
            with st.spinner("Analyzing movie summary..."):
                # Predict genres
                predicted_genres = app.genre_predictor.predict_genres(summary)
                
                # Clean up any remaining metadata from genre names
                clean_genres = []
                for genre in predicted_genres:
                    # Remove any Freebase IDs like "/m/12345"
                    clean_genre = re.sub(r'\s*/m/[\w\d]+', '', genre)
                    
                    # Remove any special characters and extra metadata
                    clean_genre = re.sub(r'[{}\"\'\[\]]', '', clean_genre)
                    
                    # Split on comma and take first part (main genre name)
                    if ',' in clean_genre:
                        clean_genre = clean_genre.split(',')[0].strip()
                    
                    clean_genres.append(clean_genre.strip())
                
                # Display results
                st.subheader("Predicted Genres:")
                
                if clean_genres:
                    # Create a horizontal list of genres with styling
                    genre_html = '<div style="display: flex; flex-wrap: wrap; gap: 8px; margin-top: 10px;">'
                    for genre in clean_genres:
                        genre_html += f'<span style="background-color: #4B8BFF; color: white; padding: 5px 10px; border-radius: 15px;">{genre}</span>'
                    genre_html += '</div>'
                    
                    st.markdown(genre_html, unsafe_allow_html=True)
                    
                    # Display a brief analysis
                    st.subheader("Genre Analysis:")
                    
                    if len(clean_genres) > 1:
                        st.write(f"This movie summary suggests elements of {', '.join(clean_genres[:-1])} and {clean_genres[-1]}.") 
                    else:
                        st.write(f"This movie summary suggests elements of {clean_genres[0]}.") 
                else:
                    st.info("Could not determine genres from this summary. Try adding more details.")
        
    except Exception as e:
        st.error(f"Error loading the genre prediction model: {str(e)}")
        st.info("You may need to train the model first by running the main.py script.")

def show_audio_library(app):
    """Display the audio library interface"""
    
    st.markdown('<h2 class="sub-header">Audio Library</h2>', 
                unsafe_allow_html=True)
    
    st.write("Browse and listen to movie summaries in multiple languages.")
    
    # Check if the audio library exists
    audio_dir = app.audio_library.get_audio_directory()
    if os.path.exists(audio_dir):
        # Get all unique summary IDs
        summary_ids = app.audio_library.get_summary_ids()
        
        if summary_ids:
            # Display summary count
            st.info(f"Library contains {len(summary_ids)} movie summaries in multiple languages.")
            
            # Add search functionality
            search_text = st.text_input("🔍 Search for specific summary IDs:", "")
            
            # Filter summary IDs based on search
            if search_text:
                filtered_ids = [sid for sid in summary_ids if search_text.lower() in sid.lower()]
                if not filtered_ids:
                    st.warning("No matches found. Showing all summaries.")
                    filtered_ids = summary_ids
            else:
                filtered_ids = summary_ids
            
            # Create pagination
            items_per_page = 5
            total_pages = (len(filtered_ids) + items_per_page - 1) // items_per_page
            
            col1, col2, col3 = st.columns([1, 3, 1])
            with col2:
                page = st.number_input("Page", min_value=1, max_value=max(1, total_pages), value=1, step=1)
            
            # Get current page items
            start_idx = (page - 1) * items_per_page
            end_idx = min(start_idx + items_per_page, len(filtered_ids))
            current_page_ids = filtered_ids[start_idx:end_idx]
            
            # Display page information
            st.write(f"Showing summaries {start_idx + 1}-{end_idx} of {len(filtered_ids)}")
            
            # Display each summary with its audio files
            for summary_id in current_page_ids:
                with st.expander(f"Summary ID: {summary_id}", expanded=False):
                    audio_files = app.audio_library.get_audio_files_for_summary(summary_id)
                    
                    if audio_files:
                        # Group by language
                        for audio_file in audio_files:
                            language = app.audio_library.get_language_for_file(audio_file)
                            
                            # Display language and audio player
                            st.subheader(f"{language}")
                            
                            # Get audio bytes
                            audio_bytes = app.audio_library.get_audio_bytes(audio_file)
                            
                            if audio_bytes:
                                # Create columns for audio player and download button
                                col1, col2 = st.columns([3, 1])
                                
                                with col1:
                                    st.audio(audio_bytes, format='audio/mp3')
                                
                                with col2:
                                    st.download_button(
                                        label="Download",
                                        data=audio_bytes,
                                        file_name=audio_file,
                                        mime="audio/mp3"
                                    )
                            else:
                                st.error(f"Could not load audio file: {audio_file}")
                    else:
                        st.warning("No audio files found for this summary.")
                
                st.markdown("---")  # Add separator between summaries
            
            # Add pagination navigation
            col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
            with col1:
                if page > 1:
                    if st.button("◀️ Previous"):
                        st.session_state.page = page - 1
                        st.experimental_rerun()
            with col4:
                if page < total_pages:
                    if st.button("Next ▶️"):
                        st.session_state.page = page + 1
                        st.experimental_rerun()
        else:
            st.warning("No audio files found in the library.")
    else:
        st.error("Audio library directory does not exist.")
        
        # Suggest generating audio files
        st.info("You can generate audio files in the 'Movie Summary Audio Conversion' section or run the generate_sample_audios.py script to create sample audio files.")

def show_about_project():
    """Display information about the project"""
    
    st.markdown('<h2 class="sub-header">About Filmception</h2>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    ### Project Overview
    
    **Filmception** is an AI-powered system that processes movie summaries to provide two main functionalities:
    
    1. **Multilingual Translation and Audio Conversion**: Convert movie summaries into audio in multiple languages (Arabic, Urdu, and Korean).
    
    2. **Genre Prediction**: Predict movie genres based on the summary using a machine learning model trained on the CMU Movie Summary dataset.
    
    ### Dataset
    
    This project uses the [CMU Movie Summary Dataset](https://www.kaggle.com/datasets/msafi04/movies-genre-dataset-cmu-movie-summary), which contains:
    
    - 42,306 movie plot summaries extracted from Wikipedia
    - Movie metadata including genres, release dates, and more
    - Character metadata
    
    ### Technical Components
    
    - **Data Preprocessing**: Cleaning and normalizing movie summaries through tokenization, lemmatization, and removal of stopwords and special characters.
    
    - **Multilingual Support**: Translation of movie summaries into Arabic, Urdu, and Korean using the Google Translate API.
    
    - **Audio Conversion**: Text-to-speech functionality using gTTS (Google Text-to-Speech) to convert translated summaries into audio.
    
    - **Machine Learning**: A logistic regression model for multi-label genre classification based on TF-IDF features extracted from movie summaries.
    
    - **Interactive Interface**: A user-friendly Streamlit interface for interacting with the system's functionalities.
    
    ### How to Use
    
    1. Navigate to the "Movie Summary Audio Conversion" tab to translate and convert summaries to audio.
    2. Navigate to the "Genre Prediction" tab to predict the genre of a movie based on its summary.
    3. Navigate to the "Audio Library" tab to browse and manage generated audio files.
    
    """)

if __name__ == "__main__":
    st.error("This file should be imported and run from main.py, not executed directly.")