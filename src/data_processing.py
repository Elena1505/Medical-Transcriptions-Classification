import pandas as pd 
from pandas import DataFrame
from typing import List
import os 
import re 
from sklearn.feature_extraction.text import TfidfVectorizer

# NLTK
import nltk 
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')


def filter_data(df:DataFrame):
    df_filtred = df[['transcription', 'medical_specialty']]
    return df_filtred


def process_null_values(df:DataFrame):
    df_cleaned = df.dropna(subset=['transcription', 'medical_specialty'])
    df_cleaned.reset_index(drop=True, inplace=True)
    return df_cleaned


def tokenize_sentences(sentence:str): 
    word_tokens = word_tokenize(sentence)
    return word_tokens


def stop_word(lst_words:List[str]):
    stop = list(set(stopwords.words('english'))) + ['[', ']', ',', '.', ':', '?', '(', ')', '!']
    filtered_lst_words = [w for w in lst_words if not w in stop]
    filtered_lst_words = [w for w in filtered_lst_words if len(w) > 2]
    return filtered_lst_words


def lowercase(lst_words:List[str]):
    lowercase_words = [w.lower() for w in lst_words]
    return lowercase_words


def lemmatize(lst_words:List[str]):
    lemmatizer = WordNetLemmatizer()
    lemmatized_words = [lemmatizer.lemmatize(w) for w in lst_words]
    return lemmatized_words


def cleaned_sentences(df:DataFrame, path:str):
    lst_words = []
    for i in range(len(df)):
        description = df.loc[i, 'transcription']
        tokenized = tokenize_sentences(description)
        stopped = stop_word(tokenized)
        lowercased = lowercase(stopped)
        lemmatized = lemmatize(lowercased)
        lst_words.append(lemmatized)
    df['cleaned_transcription'] = lst_words
    with open(os.path.join(path, 'mtsamples.csv'), 'w') as f:
        f.write(df.to_csv(index=False))
    return df


def bag_of_word(df:DataFrame):
    vectorizer = TfidfVectorizer(analyzer='word', stop_words='english', max_df=0.7, min_df=1)
    df['cleaned_transcription'] = df['cleaned_transcription'].apply(lambda x: ' '.join(x) if isinstance(x, list) else x)
    tfidf = vectorizer.fit_transform(df['cleaned_transcription'])
    features_name = sorted(vectorizer.get_feature_names_out())
    return features_name