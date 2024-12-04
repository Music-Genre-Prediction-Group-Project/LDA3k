import pandas as pd
from sklearn.model_selection import train_test_split

# Load the dataset
df = pd.read_csv('dataset/genly3k_transformed.csv')

# Split the dataset
train_df, test_df = train_test_split(df, test_size=0.2, random_state=2520)

# Save to CSV files
train_df.to_csv('dataset/genly3k_train.csv', index=False)
test_df.to_csv('dataset/genly3k_test.csv', index=False)
