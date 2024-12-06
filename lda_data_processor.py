# Description: Processes a dataset with lyric data to a dataset with topic data

from gensim.models import LdaModel
from gensim.corpora import Dictionary
import pandas as pd
import os

# File paths
dataset_type = "full" # train or test
dataset_path = f"dataset/genly3k_{dataset_type}.csv"
num_topics = 128
lda_model_path = f"models/model{num_topics}_full/lda{num_topics}.model"
id2word_path = f"models/model{num_topics}_full/lda{num_topics}.model.id2word"
output_path = f"dataset/topics_{dataset_type}/genly3k_{dataset_type}{num_topics}.csv"

os.makedirs(os.path.dirname(output_path), exist_ok=True)

# Load the LDA model
print("Loading the LDA model...")
lda_model = LdaModel.load(lda_model_path)

# Load the id2word dictionary
print("Loading the id2word dictionary...")
id2word = Dictionary.load(id2word_path)

# Function to parse lyrics string and convert it into a BoW representation
def parse_lyrics(lyrics_string):
    bow = []
    for pair in lyrics_string.split(','):
        word_id, count = map(int, pair.split(':'))
        if word_id in id2word:
          bow.append((word_id, count))
        else:
          # Skip the word
          pass
    return bow

# Function to infer topics from input
def infer_topics(input_string):
    bow = parse_lyrics(input_string)
    topics = lda_model.get_document_topics(bow, minimum_probability=0)
    # Convert topics to a list of probabilities
    topic_probs = [prob for _, prob in topics]
    return topic_probs

print("Loading the dataset...")
column_names = ["track_id", "genre", "lyric_data"]
df = pd.read_csv(dataset_path, header=None, names=column_names)

print("Inferring the topics...")
df["topic_distribution"] = df["lyric_data"].apply(lambda x: infer_topics(x))

print("Formatting and saving the new dataset...")
topics_df = pd.DataFrame(df["topic_distribution"].tolist(), index=df.index)
topics_df.columns = [f"topic_{i}" for i in range(len(topics_df.columns))]
df = pd.concat([df[["track_id", "genre"]], topics_df], axis=1)
df.to_csv(output_path, index=False)

print("Done!")
