from gensim.models import LdaModel
from gensim.corpora import Dictionary
import numpy as np

lda_model_path = "models/lda_model.model"
id2word_path = "models/lda_model.model.id2word"

print("Loading the LDA model...")
lda_model = LdaModel.load(lda_model_path)

print("Loading the id2word dictionary...")
id2word = Dictionary.load(id2word_path)

# parse input string and convert it into a bow representation
def parse_input(input_string):
    """Parse a string like '2:20,4:1' into a bag-of-words format."""
    bow = []
    for pair in input_string.split(','):
        word_id, count = map(int, pair.split(':'))
        bow.append((word_id, count))
    return bow

# infer topics from input
def infer_topics(input_string):
    """Infer topic distribution for a given input."""
    bow = parse_input(input_string)
    topics = lda_model.get_document_topics(bow, minimum_probability=0.001)
    return topics

# print words associated with a topic
def print_topic_words(topic_id, top_n=10):
    """Print the top words in a topic."""
    words = lda_model.show_topic(topic_id, top_n)
    print(f"Topic {topic_id} Top Words:")
    for word, prob in words:
        print(f"  {word}: {prob:.4f}")
    print()

input_string = "2:20,4:1,5:1,72:2,93:7,133:4,210:1,319:1,629:1,701:13,1047:1,1089:13,2284:1,2796:13"
print(f"Input: {input_string}")
topic_distribution = infer_topics(input_string)

print("\nTopic Distribution:")
for topic_id, confidence in topic_distribution:
    print(f"Topic {topic_id}: {confidence:.4f}")

print("\nWords in Relevant Topics:")
for topic_id, _ in topic_distribution:
    print_topic_words(topic_id, top_n=10)