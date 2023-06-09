import os
import glob
import pandas as pd

# Set the path to your dataset and text file
dataset_path = 'F:/Toolkit/ComputacaoVisual/Images_with_CC/bi_concepts1553/'
txt_file = 'F:/Toolkit/ComputacaoVisual/3244ANPs.txt'

# Load the ANP data from the text file
anp_data = []
with open(txt_file, 'r') as file:
    lines = file.readlines()
    for line in lines:
        if line.startswith('\t'):
            line = line.replace('[', '').replace(']', '')
            row = line.strip().split()
            anp_data.append(row)

# Create a mapping between ANP names and sentiment scores
anp_sentiment_mapping = {}
for row in anp_data:
    if len(row) == 5:
        anp = row[0]
        sentiment = float(row[2])
        anp_sentiment_mapping[anp] = sentiment

# Convert mapping to a list of key-value pairs
mapping_list = list(anp_sentiment_mapping.items())

def count_images_in_folder(folder_path, file_extensions=[".jpg", ".jpeg", ".png", ".gif"]):
    image_count = 0
    for extension in file_extensions:
        search_pattern = os.path.join(folder_path, "*" + extension)
        image_count += len(glob.glob(search_pattern))

    print("Number of images= ", image_count)
    return image_count

data = []
for (anp,sentiment_value) in mapping_list:
    print("A verificar o ANP: ", anp)
    anp_folder_path = dataset_path + anp + '/'
    image_count = 0

    # Check if the folder exists (because some folder doesnt exist)
    if not os.path.exists(anp_folder_path):
        print(f"Folder does not exist: {anp_folder_path}")
        continue

    # Iterate over image files
    image_count = count_images_in_folder(anp_folder_path)
    data.append([anp, sentiment_value, image_count])

df = pd.DataFrame(data, columns=["Folder Name", "Sentiment", "Image Count"])
print(df)

csv_path = "flickr_dataset_summary.csv"
df.to_csv(csv_path, index=False)
