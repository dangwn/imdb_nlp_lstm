'''
Author: Dan Gawne
Date: 2021-01-22
'''

import requests
import json
import numpy as np
from nlp_utils import tokenize, lemmatize

#--------------------------------------------------------------------------------------------------
# Word Vectorization
#--------------------------------------------------------------------------------------------------
# Google's Dictionary of the top 88_000ish most common words
r = requests.get('https://storage.googleapis.com/tensorflow/tf-keras-datasets/imdb_word_index.json')
word_dict = json.loads(r.content)

def word_vectorize(word:str, vector_size = 10_000):
    '''
    Returns a one-hot encoded vector of an input word
    If the word is not in the top <vector_size> most used, words, a zero vector is returned
    =============================================
    Inputs:
      - word       : The string you want to vectorize
      - vector_size: The size of the vector
        - default: 10,000
    Returns:
      - A numpy array vector
    '''
    vector = [0]*vector_size
    
    #See if original form is in dictionary, if not check if lemmatized version is
    ind = word_dict.get(word,-1)
    if ind == -1:
        ind = word_dict.get(lemmatize([word])[0],vector_size + 1)

    if ind <= vector_size:
        vector[ind-1] = 1
    return np.array(vector)

def word_vectorize_doc(doc:str, vector_size = 10_000, final_doc_len = 100):
    '''
    Returns an array of word vectors for a given document
    =============================================
    Inputs:
      - doc          : The document to vectorize
      - vector_size  : The size of the vectors for each word
      - final_doc_len: The length of the array of vectors
    Returns:
      - A numpy array of vectors
    '''
    tokenized_doc = tokenize(doc)
    
    # Tokenize document
    if len(tokenized_doc) > final_doc_len:
        tokenized_doc = tokenized_doc[:final_doc_len]

    # Create an ordered sequence of vectors
    vectors = []
    for word in tokenized_doc:
        vectors.append(word_vectorize(word, vector_size = vector_size))
    while(len(vectors) < final_doc_len):
        vectors.append(np.zeros(vector_size))

    return np.array(vectors)

#--------------------------------------------------------------------------------------------------
# Character Vectorization
#--------------------------------------------------------------------------------------------------
alphabet = 'abcdefghijklmnopqrstuvwxyz0123456789!"%^&*()_=-[]}{:;<>/?`\' '
char_dict = {ch:ind for ind,ch in enumerate(list(set(alphabet[:])))}

def char_vectorize(char : str):
    assert len(char) == 1

    vector = [0] * len(char_dict)
    ind = char_dict.get(char, -1)
    if ind != -1:
        vector[ind] = 1
    return np.array(vector)

def char_vectorize_doc(doc:str, final_doc_len = 200):

    all_chars = doc
    if len(doc) > final_doc_len:
        all_chars = doc[:final_doc_len]

    vectors = [char_vectorize(c) for c in all_chars]
    if len(vectors) < final_doc_len:
        for _ in range(final_doc_len - len(vectors)):
            vectors.append(np.zeros(len(char_dict)))
    return np.array(vectors)



