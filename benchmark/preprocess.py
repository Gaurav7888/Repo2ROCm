import pandas as pd

# Load the dataset
data = pd.read_excel("LatestPaperBenchmark_Dataset.xlsx")

# 1. Remove the '#' column
if '#' in data.columns:
    data.drop(columns=['#'], inplace=True)

# 2. Rename columns:
# 'Paper Title' -> 'paper_title'
# 'Open-Source Repository / Code Link' -> 'github_link'
data.rename(columns={
    'Paper Title': 'paper_title',
    'Open-Source Repository / Code Link': 'github_link'
}, inplace=True)

# Save the preprocessed data to a new CSV file
data.to_csv("preprocessed_dataset.csv", index=False)

# Inspect the first few rows
print(data.head())
