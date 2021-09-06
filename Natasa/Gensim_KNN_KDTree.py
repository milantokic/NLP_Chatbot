import pandas as pd
import numpy as np

import gensim.downloader as api
import textPreprocessing as tp
import distances as dis

from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KDTree

wv = api.load('word2vec-google-news-300')

### import dataset using pandas ###
dataset = pd.read_csv("C:\\Users\\Natasa\\Documents\\Machine Learning\\insurance_qna_dataset.csv", sep='\t').iloc[:, 1:]
dataset_group_by_question = dataset.groupby('Question', as_index=False).agg(lambda x: np.unique(x).tolist())
dataset_question = dataset_group_by_question.iloc[:, 0]
question_list = [question for question in dataset_question]

top_similar_question = 20
new_question = input("Please enter an insurance question: ")

#######################################################################################
############################# Preprocess the corpus basic #############################
#######################################################################################

corpus = [tp.preprocess(question) for question in question_list]
### Remove docs that don't include any words in W2V's vocab ###
corpus, question_list = tp.filter_docs(corpus, question_list, lambda doc: tp.has_vector_representation(wv, doc))
### Filter out any empty docs with stemming ###
corpus, question_list = tp.filter_docs(corpus, question_list, lambda doc: (len(doc) != 0))

### create X ndarray with stemmed words ###
x = []
for question in corpus:  # append the vector for each document
    x.append(tp.document_vector_sum(wv, question))
X = np.array(x)  # list to array

new_question_basic = tp.preprocess_stem(new_question)
new_question_basic = [tp.document_vector_sum(wv, new_question_basic)]

print(f"This are the most {top_similar_question} similar question according to cosine similarity: \n")
model = NearestNeighbors(n_neighbors=top_similar_question, algorithm='kd_tree').fit(X)
similar_q = model.kneighbors(new_question_basic)
print(dataset_question[similar_q[1].flatten()])