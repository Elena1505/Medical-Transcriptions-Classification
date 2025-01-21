from src.data_exploration import describe, data_categories_count, plot_data
from src.data_processing import filter_data, process_null_values

def main(): 

    raw_data_path = '/home/lelou1505/DS/medical_transcription/data/raw/mtsamples.csv'
    fig_path = '/home/lelou1505/DS/medical_transcription/reports'

    # Data exploring 
    first_df = describe(raw_path=raw_data_path, figure_path=fig_path)
    data_categories_count(df=first_df, figure_path=fig_path)
    plot_data(df=first_df, figure_path=fig_path)

    # Data processing 
    filtred_df = filter_data(df=first_df)
    cleaned_df = process_null_values(df=filtred_df)

    print(cleaned_df.iloc[200]['transcription'])


if __name__ == "__main__":
    main()