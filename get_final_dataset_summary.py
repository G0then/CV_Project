import pandas as pd

# Load form CSVs file
df = pd.read_csv("flickr_dataset_summary.csv")
df_words_to_remove = pd.read_csv("distinct_data.csv", sep=';')

# Extract the words after "_" for True values in df_words_to_remove
words_to_remove = df_words_to_remove[~df_words_to_remove['Usable']]['Words'].apply(lambda x: x.lower())

# Remove rows containing the words after "_"
df = df[~df['Folder Name'].apply(lambda x: any(word in x for word in words_to_remove))]

# Create a list of specific File Folders to remove
files_folder_to_remove = ["classic_castle", "lonely_forest", "dry_ice", "awesome_night", "great_night", "holy_night"]

# Filter rows based on the specific column
df = df[~df['Folder Name'].str.contains('|'.join(files_folder_to_remove), case=False)]

csv_path = "flickr_final_dataset_summary.csv"
df.to_csv(csv_path, index=False)