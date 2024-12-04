import csv
import pandas as pd
import os

# Define file paths
input_csv_path = 'dataset/genly_dataset_3k.csv'
word_ids_path = 'dataset/word-ids-index-1.txt'
output_csv_path = 'dataset/transformed_genly_dataset_3k.csv'

# Ensure the output directory exists
os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)

# Step 1: Read and Parse the Original Dataset
data = []
with open(input_csv_path, 'r', newline='', encoding='utf-8') as file:
    reader = csv.reader(file)
    for row in reader:
        track_id = row[0]
        genre = row[1]
        word_data = row[2:]  # Remaining columns
        data.append((track_id, genre, word_data))

# Step 2: Process Word Data
processed_data = []
for track_id, genre, word_data in data:
    word_freq = {i: 0 for i in range(1, 5001)}  # Initialize all word frequencies to 0
    for item in word_data:
        if item:  # Check if item is not empty
            word_id, freq = item.split(':')
            word_freq[int(word_id)] = int(freq)
    processed_data.append((track_id, genre, word_freq))

# Step 3: Create a DataFrame
# Create column names
columns = ['track_id', 'genre'] + [f'word_{i}' for i in range(1, 5001)]

# Populate the DataFrame
rows = []
for track_id, genre, word_freq in processed_data:
    row = {'track_id': track_id, 'genre': genre}
    row.update({f'word_{i}': word_freq[i] for i in range(1, 5001)})
    rows.append(row)

df = pd.DataFrame(rows, columns=columns)

# Step 4: Map Word IDs to Actual Words
# Read the word-ids-index-1.txt file
with open(word_ids_path, 'r', encoding='utf-8') as file:
    content = file.read()

# Split the content by commas to get the list of words
words = content.split(',')

# Create a mapping from word_id to word
word_id_to_word = {i + 1: word for i, word in enumerate(words) if word}

# Rename the columns in the DataFrame
new_columns = ['track_id', 'genre'] + [word_id_to_word.get(i, f'word_{i}') for i in range(1, 5001)]
df.columns = new_columns

# Step 5: Save the Transformed Dataset
df.to_csv(output_csv_path, index=False, encoding='utf-8')

print(f"Transformed dataset saved to {output_csv_path}")
