import pandas as pd 
from pandas import DataFrame
import seaborn as sns 
import matplotlib.pyplot as plt 
import os 


def describe(raw_path: str, figure_path:str): 
    df = pd.read_csv(raw_path)
    with open(os.path.join(figure_path, 'data.md'), 'w') as f:
        f.write(df.to_markdown(index=False))    
    return df


def data_categories_count(df: DataFrame, figure_path:str):
    medical_specialties = df.groupby('medical_specialty').size().reset_index(name="count").sort_values(by='count', ascending=False)
    with open(os.path.join(figure_path, 'medical_specialty_count.md'), 'w') as f:
        f.write(medical_specialties.to_markdown(index=False))     


def plot_data(df:DataFrame, figure_path:str):
    count_data = df['medical_specialty'].value_counts()
    plt.figure(figsize=(12, 8))
    sns.countplot(y='medical_specialty', data=df, order = count_data.nlargest(20).index, palette='rainbow')
    plt.tight_layout()
    plt.savefig(os.path.join(figure_path, 'medical_specialty_repartition.png'))
    plt.close()

