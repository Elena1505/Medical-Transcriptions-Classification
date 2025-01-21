import pandas as pd 
from pandas import DataFrame
from typing import List

# NLTK
import nltk 
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('punkt_tab')
nltk.download('stopwords')


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


def cleaned_sentences(df:DataFrame):
    lst_words = []
    for i in range(len(df)):
        description = df.loc[i, 'transcription']
        tokenized = tokenize_sentences(description)
        stopped = stop_word(tokenized)
        lowercased = lowercase(stopped)

        lst_words.append(lowercased)
    df['lst_words'] = lst_words
    return df