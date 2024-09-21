import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from scipy.spatial import distance

# Load the Universal Sentence Encoder model
module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
model = hub.load(module_url)
print("module %s loaded" % module_url)

# Function to embed the sentences
def embed(input):
    return model(input)

# Function to find the sentence with maximum similarity score
def get_max_similarity_score(sentences, query):
    query_vec = embed([query])
    max_score = -1
    max_sentence = None
    
    for sent in sentences:
        sent_vec = embed([sent])
        similarity_score = 1 - distance.cosine(query_vec[0], sent_vec[0])
        
        if similarity_score > max_score:
            max_score = similarity_score
            max_sentence = sent
            
    return max_sentence, max_score

# Test the function
test_sentence = "I liked the movie very much"
sentences = ["The movie is good.",
             "We are learning NLP through GeeksforGeeks",
             "The baby learned to walk in the 5th month itself"]

max_sentence, max_score = get_max_similarity_score(sentences, test_sentence)
print(f'Most similar sentence: "{max_sentence}" with similarity score: {max_score}')
