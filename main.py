from src.data_exploration import read_data, data_categories_count, plot_data
from src.data_processing import filter_data, process_null_values, cleaned_sentences, bag_of_word
import os 

def main(): 

    raw_data_path = '/home/lelou1505/DS/medical_transcription/data/raw/mtsamples.csv'
    fig_path = '/home/lelou1505/DS/medical_transcription/reports'
    processed_data_path = '/home/lelou1505/DS/medical_transcription/data/processed'

    # Data exploring 
    first_df = read_data(raw_path=raw_data_path, figure_path=fig_path)
    data_categories_count(df=first_df, figure_path=fig_path)
    plot_data(df=first_df, figure_path=fig_path)

    # Data processing 
    filtred_df = filter_data(df=first_df)
    cleaned_df = process_null_values(df=filtred_df)
    cleaned_df = cleaned_sentences(df=cleaned_df, path=processed_data_path)

    print(bag_of_word(df=cleaned_df))


if __name__ == "__main__":
    main()