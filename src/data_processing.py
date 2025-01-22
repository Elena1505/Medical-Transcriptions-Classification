import numpy as np 
import matplotlib.pyplot as plt

from pandas import DataFrame
from typing import List
import os 
import re 
import gc 

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans 
from sklearn import metrics

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


def remove_repeated_letter_words(lst_words:List[str]):
    pattern = r'\b([a-zA-Z])\1+\b'
    cleaned_text = [re.sub(pattern, '', text) for text in lst_words]
    return cleaned_text


def cleaned_sentences(df:DataFrame, path:str):
    lst_words = []
    for i in range(len(df)):
        description = df.loc[i, 'transcription']
        tokenized = tokenize_sentences(description)
        stopped = stop_word(tokenized)
        lowercased = lowercase(stopped)
        lemmatized = lemmatize(lowercased)
        removed_repeated_letter = remove_repeated_letter_words(lemmatized)
        lst_words.append(removed_repeated_letter)
    df['cleaned_transcription'] = lst_words
    with open(os.path.join(path, 'mtsamples.csv'), 'w') as f:
        f.write(df.to_csv(index=False))
    return df


def bag_of_word(df:DataFrame):
    vectorizer = TfidfVectorizer(analyzer='word', stop_words='english', max_df=0.7, min_df=1)
    df['cleaned_transcription'] = df['cleaned_transcription'].apply(lambda x: ' '.join(x) if isinstance(x, list) else x)
    tfidf = vectorizer.fit_transform(df['cleaned_transcription'])
    return tfidf



def ARI_fct(tfidf, df:DataFrame) :
    categ = list(set(df['medical_specialty']))
    categ_num = [(1-categ.index(df.iloc[i]['medical_specialty'])) for i in range(len(df))]

    num_labels = len(categ)
    tsne = TSNE(n_components = 2, perplexity = 53, n_iter = 2000, init = 'random', learning_rate = 200, random_state=42)
    X_tsne = tsne.fit_transform(tfidf)
    
    cls = KMeans(n_clusters = num_labels, n_init=100, random_state=42)
    cls.fit(X_tsne)
    ARI = np.round(metrics.adjusted_rand_score(categ_num, cls.labels_),4)
    return ARI, X_tsne, cls.labels_


def TSNE_visu_fct(df:DataFrame, X_tsne, labels, ARI, figure_path:str) :
    fig = plt.figure(figsize=(15,6))

    categ = list(set(df['medical_specialty']))
    categ_num = [(1-categ.index(df.iloc[i]['medical_specialty'])) for i in range(len(df))]
    
    ax = fig.add_subplot(121)
    scatter = ax.scatter(X_tsne[:,0],X_tsne[:,1], c = categ_num, cmap='Set1')
    ax.legend(handles = scatter.legend_elements()[0], labels = categ, loc = "best", title = "Categorie")
    plt.title('Représentation des articles par catégories réelles')
    
    ax = fig.add_subplot(122)
    scatter = ax.scatter(X_tsne[:,0],X_tsne[:,1], c = labels, cmap='Set1')
    ax.legend(handles = scatter.legend_elements()[0], labels = set(labels), loc = "best", title = "Clusters")
    plt.title('Représentation des articles par clusters')

    plt.savefig(os.path.join(figure_path, 'TSNE.png'))
    plt.close()
    print("ARI : ", ARI)
