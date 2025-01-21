import pandas as pd 
from pandas import DataFrame
import nltk 
nltk.download('punkt_tab')
from nltk.tokenize import word_tokenize


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


def cleaned_sentences(df:DataFrame):
    lst_words = []
    for i in range(len(df)):
        description = df.loc[i, 'transcription']
        bow_result = tokenize_sentences(description)
        lst_words.append(bow_result)
    df['lst_words'] = lst_words
    return df