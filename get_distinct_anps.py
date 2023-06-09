import pandas as pd

# Load the dataset from CSV file
dataset_path = "flickr_dataset_summary.csv"
df = pd.read_csv(dataset_path)

prefixes = []
words = []

for line in df['Folder Name']:
    if "_" in line:
        prefix, word = line.split('_')
        prefixes.append(prefix)
        words.append(word)

distinct_prefixes = list(set(prefixes))
distinct_words = list(set(words))

# Optionally, sort the lists in ascending order
distinct_prefixes.sort()
distinct_words.sort()

# Create a DataFrame from the lists
data = pd.DataFrame({'Words': distinct_words})

# Save the DataFrame to an Excel file
data.to_csv('distinct_data.csv', index=False)

print("Prefixes:")
print(distinct_prefixes)

print("\nWords:")
print(distinct_words)