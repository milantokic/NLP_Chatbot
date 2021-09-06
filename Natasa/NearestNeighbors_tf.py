#Import the libraries used for counting the tf-idf
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import BallTree
import pickle
from sklearn.neighbors import NearestNeighbors

#https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html
#import the dataset using the pandas
dataset = pd.read_csv('insurance_qna_dataset.csv', sep='\t').iloc[:, 1:]
dataset_group_by_question = dataset.groupby('Question', as_index=False).agg(lambda x: np.unique(x).tolist())
#dataset_groupBy_question = dataset_delete_nanValues.groupby('Question', as_index=False).agg({'Answer': lambda d: ', '.join(set(d))})
dataset_question = pd.Series(dataset_group_by_question['Question'])
#dataset_question_test = dataset_group_by_question.iloc[:, 0]


#Using the TdidfVectorizer to calculate the tf-idf parameter
tf = CountVectorizer()
count_tf = tf.fit_transform(dataset_question)
count_tf_features = tf.get_feature_names()

#Ask an insurance related question
new_question = [input("Please enter your insurance related question: ")]
count_tf_new_question = tf.transform(new_question)

count_tf_todense = count_tf.todense()
count_tf_new_question_todense = count_tf_new_question.todense()

k_neighbors = 3
knn = NearestNeighbors(n_neighbors=k_neighbors, algorithm='kd_tree')
knn_fitted = knn.fit(count_tf_todense)
indices = knn.kneighbors(count_tf_new_question_todense, return_distance=False)
print(indices)
#indx = indices.flatten()
#questions = dataset_question.iloc[indx, :]

for question in range(0, k_neighbors):
    q = indices[0][question]
    print(dataset_question[q])





