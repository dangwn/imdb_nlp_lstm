'''
Author: Dan Gawne
Date: 2021-01-22
'''

import pandas as pd
import data_utils
import yaml
import os
import numpy as np

with open('file_paths.yml','r') as f:
    file_paths = yaml.safe_load(f)
data_dir = file_paths['data_dir'][0]

review_data = pd.read_csv(os.path.join(data_dir, 'reviews.csv'))

def create_data(final_doc_len = 200):
    features = []
    labels = []

    for i in range(len(review_data)):
        features.append(data_utils.char_vectorize_doc(
            review_data.loc[i,'review'], final_doc_len = final_doc_len
        ))
        labels.append(review_data.loc[i,'sentiment'])
        
    X = np.array(features)
    y = np.array(labels)

    return X, y

if __name__ == '__main__':
    X,y = create_data(final_doc_len = 140)
    print(X[0], y[0])


