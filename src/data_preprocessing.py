import pandas as pd
import numpy as np
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import spacy

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

class TextPreprocessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        
    def clean_text(self, text):
        """
        Clean and preprocess text
        """
        if pd.isna(text):
            return ""
        
        text = text.lower()
        
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        text = re.sub(r'\S+@\S+', '', text)
        
        text = re.sub(r'@\w+|#\w+', '', text)
        
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        text = re.sub(r'\d+', '', text)
        
        text = ' '.join(text.split())
        
        return text
    
    def remove_stopwords(self, text):
        """
        Remove stopwords from text
        """
        tokens = word_tokenize(text)
        filtered_tokens = [word for word in tokens if word not in self.stop_words]
        return ' '.join(filtered_tokens)
    
    def lemmatize_text(self, text):
        """
        Lemmatize text
        """
        tokens = word_tokenize(text)
        lemmatized = [self.lemmatizer.lemmatize(word) for word in tokens]
        return ' '.join(lemmatized)
    
    def preprocess(self, text, remove_stopwords=True, lemmatize=True):
        """
        Complete preprocessing pipeline
        """
        text = self.clean_text(text)
        
        if remove_stopwords:
            text = self.remove_stopwords(text)
        
        if lemmatize:
            text = self.lemmatize_text(text)
        
        return text


def load_and_prepare_data(fake_path, true_path):
    """
    Load and prepare the ISOT dataset
    """
    fake_df = pd.read_csv(fake_path)
    true_df = pd.read_csv(true_path)
    
    fake_df['label'] = 0
    true_df['label'] = 1
    
    df = pd.concat([fake_df, true_df], ignore_index=True)
    
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    df['title'].fillna('', inplace=True)
    df['text'].fillna('', inplace=True)
    df['subject'].fillna('unknown', inplace=True)
    
    df['combined_text'] = df['title'] + ' ' + df['text']
    
    return df


if __name__ == "__main__":
    preprocessor = TextPreprocessor()
    
    sample_text = "Breaking News!!! Visit https://example.com for more info. Email us at test@example.com #FakeNews"
    
    print("Original text:")
    print(sample_text)
    print("\nCleaned text:")
    print(preprocessor.preprocess(sample_text))