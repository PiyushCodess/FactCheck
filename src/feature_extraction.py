import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import re
from textblob import TextBlob

try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon')


class FeatureExtractor:
    def __init__(self):
        self.tfidf_vectorizer = None
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        
    def extract_tfidf_features(self, texts, max_features=5000, ngram_range=(1, 2)):
        """
        Extract TF-IDF features from texts
        """
        if self.tfidf_vectorizer is None:
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=max_features,
                ngram_range=ngram_range,
                min_df=2,
                max_df=0.95
            )
            tfidf_features = self.tfidf_vectorizer.fit_transform(texts)
        else:
            tfidf_features = self.tfidf_vectorizer.transform(texts)
        
        return tfidf_features
    
    def extract_sentiment_features(self, text):
        """
        Extract sentiment scores using VADER
        """
        if pd.isna(text) or text == "":
            return {
                'sentiment_compound': 0,
                'sentiment_pos': 0,
                'sentiment_neu': 0,
                'sentiment_neg': 0
            }
        
        scores = self.sentiment_analyzer.polarity_scores(text)
        return {
            'sentiment_compound': scores['compound'],
            'sentiment_pos': scores['pos'],
            'sentiment_neu': scores['neu'],
            'sentiment_neg': scores['neg']
        }
    
    def extract_linguistic_features(self, text, title=""):
        """
        Extract linguistic and statistical features
        """
        features = {}
        
        if pd.isna(text):
            text = ""
        if pd.isna(title):
            title = ""
        
        features['text_length'] = len(text)
        features['word_count'] = len(text.split())
        features['avg_word_length'] = np.mean([len(word) for word in text.split()]) if text else 0
        features['title_length'] = len(title)
        
        features['uppercase_ratio'] = sum(1 for c in text if c.isupper()) / len(text) if len(text) > 0 else 0
        features['title_has_exclamation'] = 1 if '!' in title else 0
        features['title_has_question'] = 1 if '?' in title else 0
        
        features['exclamation_count'] = text.count('!')
        features['question_count'] = text.count('?')
        features['punctuation_ratio'] = sum(1 for c in text if c in '!?.,;:') / len(text) if len(text) > 0 else 0
        
        sentences = text.split('.')
        features['sentence_count'] = len([s for s in sentences if s.strip()])
        features['avg_sentence_length'] = features['word_count'] / features['sentence_count'] if features['sentence_count'] > 0 else 0
        
        unique_words = len(set(text.split()))
        features['lexical_diversity'] = unique_words / features['word_count'] if features['word_count'] > 0 else 0
        
        try:
            blob = TextBlob(text)
            features['subjectivity'] = blob.sentiment.subjectivity
            features['polarity'] = blob.sentiment.polarity
        except:
            features['subjectivity'] = 0
            features['polarity'] = 0
        
        return features
    
    def extract_all_features(self, df):
        """
        Extract all features from dataframe
        """
        print("Extracting sentiment features...")
        sentiment_features = df['cleaned_text'].apply(self.extract_sentiment_features)
        sentiment_df = pd.DataFrame(sentiment_features.tolist())
        
        print("Extracting linguistic features...")
        linguistic_features = df.apply(
            lambda row: self.extract_linguistic_features(row['text'], row['title']), 
            axis=1
        )
        linguistic_df = pd.DataFrame(linguistic_features.tolist())
        
        feature_df = pd.concat([sentiment_df, linguistic_df], axis=1)
        
        return feature_df


if __name__ == "__main__":
    extractor = FeatureExtractor()
    
    sample_text = "This is BREAKING NEWS!!! The president made shocking statements. What will happen next?"
    sample_title = "SHOCKING: President's Statement!"
    
    print("Sentiment features:")
    print(extractor.extract_sentiment_features(sample_text))
    
    print("\nLinguistic features:")
    print(extractor.extract_linguistic_features(sample_text, sample_title))