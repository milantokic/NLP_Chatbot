#Import the libraries used for counting the tf-idf
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import BallTree
from sklearn.neighbors import KDTree
import pickle
from sklearn.neighbors import NearestNeighbors

#import the dataset using the pandas
dataset = pd.read_csv('insurance_qna_dataset.csv', sep='\t').iloc[:, 1:]
dataset_group_by_question = dataset.groupby('Question', as_index=False).agg(lambda x: np.unique(x).tolist())
#dataset_groupBy_question = dataset_delete_nanValues.groupby('Question', as_index=False).agg({'Answer': lambda d: ', '.join(set(d))})
dataset_question = pd.Series(dataset_group_by_question['Question'])

#Using the TdidfVectorizer to calculate the tf-idf parameter
tfidf = TfidfVectorizer()
count_tfidf = tfidf.fit_transform(dataset_question)
count_tfidf_features = tfidf.get_feature_names()

#Ask an insurance related question
new_question = [input("Please enter your insurance related question: ")]
count_tfidf_new_question = tfidf.transform(new_question)

count_tfidf_todense = count_tfidf.todense()
count_tfidf_new_question_todense = count_tfidf_new_question.todense()

k_neighbors = 20
tree = KDTree(count_tfidf_todense, metric='manhattan')
indices = tree.query(count_tfidf_new_question_todense[:1], k=k_neighbors, return_distance=False)
print(indices)

print(f"The {k_neighbors} most similar questions according to KNN KD Tree manhattan (tf-idf): ")
for question in range(0, k_neighbors):
    q = indices[0][question]
    print(dataset_question[q])

print('\n\n')

k_neighbors = 20
tree = KDTree(count_tfidf_todense, metric='euclidean')
indices = tree.query(count_tfidf_new_question_todense[:1], k=k_neighbors, return_distance=False)
print(indices)

print(f"The {k_neighbors} most similar questions according to KNN KD Tree euclidean (tf-idf): ")
for question in range(0, k_neighbors):
    q = indices[0][question]
    print(dataset_question[q])



