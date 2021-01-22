import pandas as pd
import data_utils
import yaml
import os
import numpy as np

with open('file_paths.yml','r') as f:
    file_paths = yaml.safe_load(f)
data_dir = file_paths['data_dir'][0]

review_data = pd.read_csv(os.path.join(data_dir, 'reviews.csv'))

def create_data(vector_size = 10_000, final_doc_len = 100):
    features = []
    labels = []

    for i in range(len(review_data)):
        features.append(data_utils.vectorize_doc(
            review_data.loc[i,'review'], vector_size = vector_size, final_doc_len = final_doc_len
        ))
        labels.append(review_data.loc[i,'sentiment'])
        
    X = np.array(features)
    y = np.array(labels)

    return X, y

if __name__ == '__main__':
    X,y = create_data(vector_size=1_000, final_doc_len = 50)
    print(X.shape, y.shape)


