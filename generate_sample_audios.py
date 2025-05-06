#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script to translate 50 movie summaries from the dataset to multiple languages and convert them to audio
"""

import os
import pandas as pd
import time
from tqdm import tqdm
import random
import nltk

# Import necessary classes from main.py
from main import MultilingualTranslator, DataPreprocessor, LANGUAGES

# Ensure NLTK data is downloaded
def download_nltk_data():
    for resource in ['stopwords', 'punkt', 'wordnet']:
        try:
            nltk.data.find(f'tokenizers/{resource}')
        except LookupError:
            nltk.download(resource)

def main():
    """Generate audio files for 50 randomly selected movie summaries in multiple languages"""
    print("Starting the audio generation process...")
    
    # Initialize the translator
    translator = MultilingualTranslator()
    
    # Load the preprocessed data or original data if needed
    preprocessed_path = os.path.join('MovieSummaries', 'preprocessed_data.csv')
    
    # Check if preprocessed data exists, if not create it
    if not os.path.exists(preprocessed_path):
        print("Preprocessed data not found. Processing the dataset first...")
        preprocessor = DataPreprocessor()
        data = preprocessor.preprocess_data(sample_size=5000)  # Use a sample for faster processing
    else:
        print(f"Loading preprocessed data from {preprocessed_path}...")
        data = pd.read_csv(preprocessed_path)
    
    # Select 50 random summaries
    num_samples = min(50, len(data))
    print(f"Selecting {num_samples} random movie summaries...")
    
    # Use summaries that are not too short or too long for better results
    filtered_data = data[(data['summary'].str.len() > 200) & (data['summary'].str.len() < 2000)]
    
    if len(filtered_data) < num_samples:
        print("Warning: Not enough summaries meeting length criteria. Using all available summaries.")
        filtered_data = data
    
    selected_data = filtered_data.sample(n=num_samples, random_state=42)
    
    # Track progress
    total_languages = len(LANGUAGES)
    total_operations = num_samples * total_languages
    
    print(f"Translating and converting {num_samples} summaries to {total_languages} languages...")
    print(f"Total operations to perform: {total_operations}")
    
    # Track which files were successfully processed
    successful_files = []
    
    # Process each summary for each language
    for idx, row in tqdm(selected_data.iterrows(), total=len(selected_data), desc="Processing Summaries"):
        summary = row['summary']
        movie_id = row['movie_id']
        
        # Truncate very long summaries to avoid translation API limitations
        if len(summary) > 1000:
            summary = summary[:1000] + "..."
        
        print(f"\nProcessing summary for movie ID: {movie_id}")
        
        for language_name, language_code in LANGUAGES.items():
            try:
                print(f"  Translating to {language_name} ({language_code})...")
                
                # Process the summary (translate and convert to audio)
                result = translator.process_summary(summary, language_code)
                
                if result['audio_path']:
                    print(f"  ✓ Successfully saved audio to {result['audio_path']}")
                    successful_files.append(result['audio_path'])
                else:
                    print(f"  ✗ Failed to generate audio for {language_name}")
                
                # Sleep briefly to avoid hitting API rate limits
                time.sleep(1)
                
            except Exception as e:
                print(f"  ✗ Error processing {language_name} for movie {movie_id}: {str(e)}")
    
    print("\nAudio generation process completed.")
    print(f"Successfully generated {len(successful_files)} audio files out of {total_operations} attempted.")
    print(f"Audio files are saved in the '{translator.audio_dir}' directory.")

if __name__ == "__main__":
    # Download NLTK data
    download_nltk_data()
    
    # Run the main function
    main()