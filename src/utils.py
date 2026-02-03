from xml.sax.handler import all_features
import pandas as pd
import numpy as np
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob
import joblib

for item in ['punkt', 'stopwords', 'wordnet', 'vader_lexicon', 'punkt_tab']:
    try:
        nltk.data.find(item)
    except LookupError:
        nltk.download(item, quiet=True)


class PredictionPipeline:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        self.tfidf_vectorizer = None
        self.model = None
        self.feature_columns = None
    
    def load_model(self, model_path, tfidf_path=None):
        """
        Load trained model
        """
        self.model = joblib.load(model_path)
        if tfidf_path:
            self.tfidf_vectorizer = joblib.load(tfidf_path)
        print(f"Model loaded from {model_path}")
    
    def clean_text(self, text):
        """
        Clean raw text
        """
        if pd.isna(text):
            return ""
        text = text.lower()
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        text = re.sub(r'\S+@\S+', '', text)
        text = re.sub(r'@\w+|#\w+', '', text)
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = re.sub(r'\d+', '', text)
        text = ' '.join(text.split())
        return text
    
    def remove_stopwords(self, text):
        tokens = word_tokenize(text)
        return ' '.join([w for w in tokens if w not in self.stop_words])
    
    def lemmatize(self, text):
        tokens = word_tokenize(text)
        return ' '.join([self.lemmatizer.lemmatize(w) for w in tokens])
    
    def preprocess(self, text):
        """
        Full preprocessing pipeline
        """
        text = self.clean_text(text)
        text = self.remove_stopwords(text)
        text = self.lemmatize(text)
        return text
    
    def extract_sentiment_features(self, text):
        scores = self.sentiment_analyzer.polarity_scores(text)
        return {
            'sentiment_compound': scores['compound'],
            'sentiment_pos': scores['pos'],
            'sentiment_neu': scores['neu'],
            'sentiment_neg': scores['neg']
        }
    
    def extract_linguistic_features(self, text, title=""):
        features = {}
        if pd.isna(text): text = ""
        if pd.isna(title): title = ""
        
        features['text_length'] = len(text)
        features['word_count'] = len(text.split())
        features['avg_word_length'] = np.mean([len(w) for w in text.split()]) if text else 0
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
    
    def extract_features(self, title, text):
        """
        Extract all features from raw text
        """
        combined = title + ' ' + text
        cleaned = self.preprocess(combined)
        
        sentiment = self.extract_sentiment_features(cleaned)
        
        linguistic = self.extract_linguistic_features(text, title)
        
        all_features = {**sentiment, **linguistic}
        features_df = pd.DataFrame([all_features])
        
        return features_df, cleaned
    
    def align_features(self, features_df):
        """
        Align feature dataframe to exactly match training columns.
        Adds missing columns as 0, removes extras, and reorders.
        """
        if self.feature_columns is None:
            return features_df

        for col in self.feature_columns:
            if col not in features_df.columns:
                features_df[col] = 0.0

        features_df = features_df[self.feature_columns].copy()

        features_df = features_df.astype(float)

        return features_df

    def predict(self, title, text):
        """
        Full prediction pipeline
        """
        features_df, cleaned_text = self.extract_features(title, text)

        if self.tfidf_vectorizer is not None:
            tfidf = self.tfidf_vectorizer.transform([cleaned_text])
            tfidf_df = pd.DataFrame(
                tfidf.toarray(),
                columns=[f'tfidf_{i}' for i in range(tfidf.shape[1])]
            )
            features_df = pd.concat([features_df.reset_index(drop=True), 
                                     tfidf_df.reset_index(drop=True)], axis=1)

        features_df = self.align_features(features_df)

        prediction = None
        confidence = None

        if hasattr(self.model, 'predict_proba') and hasattr(self.model, 'models'):
            proba = self.model.predict_proba(features_df)
            prediction = int((proba[0] >= 0.5))
            confidence = float(max(proba[0], 1 - proba[0])) * 100
        elif hasattr(self.model, 'predict_proba'):
            proba = self.model.predict_proba(features_df)[0]
            prediction = int(self.model.predict(features_df)[0])
            confidence = float(max(proba)) * 100
        else:
            prediction = int(self.model.predict(features_df)[0])
            confidence = None

        sentiment = self.extract_sentiment_features(cleaned_text)
        linguistic = self.extract_linguistic_features(text, title)
        all_features = {**sentiment, **linguistic}

        return {
            'prediction': prediction,
            'label': 'FAKE NEWS' if prediction == 0 else 'TRUE NEWS',
            'confidence': confidence,
            'features': all_features
        }


if __name__ == "__main__":
    pipeline = PredictionPipeline()
    print("Prediction pipeline loaded successfully!")