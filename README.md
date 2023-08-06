# GloVe Word Embeddings in Python

This repository contains a simple implementation of the GloVe technique for generating word embeddings in Python. GloVe (Global Vectors for Word Representation) is a popular unsupervised learning method that leverages co-occurrence information from large-scale corpora to produce dense word vectors.

## How to Install

Before using the package, you need to install the required dependencies. To do this, make sure you have a requirements.txt file in your project's root directory. Then, follow these steps:

1. Ensure you have Python installed on your system. This project is compatible with Python 3.x.

2. Install the required dependencies from the requirements.txt file:

```bash
pip install -r requirements.txt
```

That's it! You've now installed the necessary dependencies for this project.

## Usage

To use the GloVe implementation in your project, follow these steps:

1. Modify the last lines in the code and add your words

```bash
similarity = get_word_similarity(embeddings, word2id, 'sun', 'sky')
print(f"The distance between the two words is: {similarity}")
```

2. Run the code
```bash
python3 ./src/main.py
```