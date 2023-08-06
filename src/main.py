# Import the necessary libraries:
import os  # For reading files and managing paths
import numpy as np  # For performing mathematical operations
from scipy.sparse import lil_matrix  # For handling sparse matrices
from sklearn.decomposition import TruncatedSVD  # For Singular Value Decomposition (SVD)
from sklearn.metrics.pairwise import cosine_similarity  # For calculating cosine similarity between vectors

# Define the path to the corpus folder and obtain the list of text files
corpus_folder = "./corpus"
file_names = [f for f in os.listdir(corpus_folder) if f.endswith(".txt")]

# Initialize an empty list to store the words from the corpus
corpus = []

# Read each text file in the corpus folder and append the words to the corpus list
for file_name in file_names:
    file_path = os.path.join(corpus_folder, file_name)
    with open(file_path, "r") as corpusFile:
        for linea in corpusFile:
            word_line = linea.strip().split()
            corpus.extend(word_line)

# Function to create a co-occurrence matrix from the corpus with a given window size
def create_co_occurrence_matrix(corpus, window_size=4):
    vocab = set(corpus)  # Create a set of unique words in the corpus
    word2id = {word: i for i, word in enumerate(vocab)}  # Create a word-to-index dictionary for the words
    id2word = {i: word for i, word in enumerate(vocab)}  # Create an index-to-word dictionary for the words
    matrix = lil_matrix((len(vocab), len(vocab)))  # Initialize an empty sparse matrix of size len(vocab) x len(vocab)

    # Iterate through the corpus to fill the co-occurrence matrix
    for i in range(len(corpus)):
        for j in range(max(0, i - window_size), min(len(corpus), i + window_size)):
            if i != j:
                matrix[word2id[corpus[i]], word2id[corpus[j]]] += 1

    return matrix, word2id, id2word

# Function to perform SVD on the co-occurrence matrix and reduce the dimensionality
def perform_svd(matrix, n_components=300):
    n_components = min(n_components, matrix.shape[1] - 1)
    svd = TruncatedSVD(n_components=n_components)
    return svd.fit_transform(matrix)

# Function to create word embeddings from the corpus using the co-occurrence matrix and SVD
def create_word_embeddings(corpus):
    matrix, word2id, id2word = create_co_occurrence_matrix(corpus)  # Create the co-occurrence matrix
    word_embeddings = perform_svd(matrix)  # Perform SVD on the matrix
    return word_embeddings, word2id, id2word

# Create the word embeddings from the given corpus
embeddings, word2id, id2word = create_word_embeddings(corpus)

# Function to calculate the cosine similarity between two word vectors
def get_word_similarity(embeddings, word2id, word1, word2):
    word1_vector = embeddings[word2id[word1]]  # Get the vector representation of word1
    word2_vector = embeddings[word2id[word2]]  # Get the vector representation of word2

    # Compute the cosine similarity between the two vectors
    similarity = cosine_similarity(word1_vector.reshape(1, -1), word2_vector.reshape(1, -1))

    return similarity[0][0]

# Example usage: Calculate the similarity between the word embeddings for 'sun' and 'sky'
similarity = get_word_similarity(embeddings, word2id, 'sun', 'sky')
print(f"The distance between the two words is: {similarity}")
