'''
Author: Dan Gawne
Date: 2021-01-22
'''

import nltk

import re

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk import word_tokenize

__all__ = ["process_text"]

nltk.download("wordnet")
nltk.download("stopwords")
nltk.download("words")
nltk.download("punkt")

#Objects for the functions
stop_words = list(stopwords.words("English"))
word_dict = set(word.lower() for word in nltk.corpus.words.words())
lemmatizer = WordNetLemmatizer()

def case_normalize(text : str):
    '''
    Takes in a text string and returns a string with all characters in lower case. 
    '''
    return text.lower()

def remove_punctuation(text : str):
    '''
    Takes in a text string and returns a string with all the punctuation removed.
    '''
    new_str = sample_doc = re.sub(r"[^a-zA-Z0-9]"," ", text).strip()
    return new_str

def tokenize(text : str):
    '''
    Takes in a text string and returns a list where each item corresponds to a token.
    '''
    return word_tokenize(text)

def remove_stopwords(tokenized_text : list):
    '''
    Takes in a list of text, and returns another list where the stop words have been removed.
    '''
    processed_text = [word for word in tokenized_text if word not in stop_words]
    return processed_text

def remove_unknown_words(tokenized_text : list):
    '''
    Takes in a list of text, and returns another list where unknown words have been removed.
    '''
    processed_text = [word for word in tokenized_text if word not in word_dict]
    return processed_text

def lemmatize(tokenized_text : list):
    '''
    Takes in a list of text, and returns another list where each word has been lemmatized.
    '''
    lemmatized_text = [lemmatizer.lemmatize(word) for word in tokenized_text]#
    return lemmatized_text

def process_text(text : str, spacer = " "):
    '''
    Takes in a raw text document and performs the following steps in order:
    - punctuation removal
    - case normalization
    - tokenization
    - remove stopwords
    - lemmatization
 
    Then returns a string containing the processed text
    '''
    txt_no_punc = remove_punctuation(text)
    txt_normalized = case_normalize(txt_no_punc)
    token_txt = tokenize(txt_normalized)
    no_stop_txt = remove_stopwords(token_txt)
    lemmatized_txt = lemmatize(no_stop_txt)
    
    return spacer.join(lemmatized_txt)
