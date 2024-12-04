import pandas as pd
from gensim import corpora
from gensim.models import LdaModel
import os

num_topics = 2048

# Define file paths
input_csv_path = 'dataset/genly3k_train.csv'
lda_model_path = f'models/model{num_topics}/lda{num_topics}.model'
word_id_map = 'dataset/word-ids-index-1.txt'

# Ensure the output directory exists
os.makedirs(os.path.dirname(lda_model_path), exist_ok=True)

# Step 1: Load the word map
with open(word_id_map, 'r') as f:
    words = f.read().strip().split(',')
    word_id_to_word = {str(i + 1): word for i, word in enumerate(words)}

# Step 2: Load the Transformed Dataset
print("Loading the dataset...")
df = pd.read_csv(input_csv_path)
print("Preparing data for LDA...")
texts = []
for _, row in df.iterrows():
    lyric_data = row.iloc[2]
    document = []
    if pd.notna(lyric_data):
        word_entries = lyric_data.split(',')
        for entry in word_entries:
            word_id, count = entry.split(':')
            word = word_id_to_word.get(word_id)
            if word:
                document.extend([word] * int(count))
    texts.append(document)

dictionary = corpora.Dictionary(texts)
print(f"Dictionary created with {len(dictionary)} unique tokens.")

# Convert documents into a document-term matrix
corpus = [dictionary.doc2bow(text) for text in texts]
print("Corpus created.")

# Step 3: Train the LDA Model
print("Training the LDA model...")
# Set parameters
passes = 10       # Number of passes through the corpus during training

# Train the LDA model
lda_model = LdaModel(corpus=corpus,
                     id2word=dictionary,
                     num_topics=num_topics,
                     passes=passes,
                     random_state=2520)
print("LDA model trained.")

# Examine the Topics
print("Top words per topic:")
for idx, topic in lda_model.print_topics(-1):
    print(f"Topic {idx}: {topic}")

# Save the LDA Model
lda_model.save(lda_model_path)
print(f"LDA model saved to {lda_model_path}")
