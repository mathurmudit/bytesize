from distutils import text_file
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import os, sys
import sklearn
from sklearn.metrics.pairwise import cosine_similarity

# get cosine similarity matrix
def cos_sim(input_vectors):
    similarity = cosine_similarity(input_vectors)
    return similarity

# get topN similar sentences
def get_top_similar(sentence, sentence_list, similarity_matrix, topN):
    # find the index of sentence in list
    index = sentence_list.index(sentence)
    # get the corresponding row in similarity matrix
    similarity_row = np.array(similarity_matrix[index, :])
    # get the indices of top similar
    indices = similarity_row.argsort()[-topN:][::-1]
    return [sentence_list[i] for i in indices]


module_url = "https://tfhub.dev/google/universal-sentence-encoder/2" #@param ["https://tfhub.dev/google/universal-sentence-encoder/2", "https://tfhub.dev/google/universal-sentence-encoder-large/3"]

# Import the Universal Sentence Encoder's TF Hub module
embed = hub.Module(module_url)

# Reduce logging output.
tf.logging.set_verbosity(tf.logging.ERROR)

# variable that holds the txt file
text_file = open('ECON175 Essay #2.txt', 'r')

sentences_list = text_file.split('.')


with tf.Session() as session:

  session.run([tf.global_variables_initializer(), tf.tables_initializer()])
  sentences_embeddings = session.run(embed(sentences_list))

similarity_matrix = cos_sim(np.array(sentences_embeddings))

sentence = "How does oil affect Venezuela's economy?"
top_similar = get_top_similar(sentence, sentences_list, similarity_matrix, 5)

# printing the list using loop 
for x in range(len(top_similar)): 
    print(top_similar[x])
