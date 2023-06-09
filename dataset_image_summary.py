import pandas as pd

# Load form CSVs file
df = pd.read_csv("flickr_final_dataset_summary.csv")

# Group by Sentiment and calculate the sum of Folder Name and Image Count
grouped_df = df.groupby(pd.cut(df["Sentiment"], bins=[-float("inf"), -0.5, 0.5, float("inf")])).agg({
    "Folder Name": "count",
    "Image Count": "sum"
})

# Rename the sentiment group labels
grouped_df.index = grouped_df.index.map({
    pd.Interval(-float("inf"), -0.5): "Negative",
    pd.Interval(-0.5, 0.5): "Neutral",
    pd.Interval(0.5, float("inf")): "Positive"
})

# Calculate the sums
sums = grouped_df.sum()

# Print the resulting dataframe
print(grouped_df)
print("\nAll Data:\n", sums)
