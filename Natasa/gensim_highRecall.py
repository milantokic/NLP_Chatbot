import pandas as pd
import numpy as np

import gensim.downloader as api
import textPreprocessing as tp
import distances as dis
from sklearn.neighbors import NearestNeighbors

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
new_question_basic = tp.document_vector_sum(wv, new_question_basic)

print(f"This are the most {top_similar_question} similar question according to cosine similarity: \n")
number_of_question = len(corpus)
dis.cosin_dist(number_of_question, X, new_question_basic, top_similar_question, dataset_question)

print(f"This are the most {top_similar_question} similar question according to KD tree: \n")
model = NearestNeighbors(n_neighbors=top_similar_question, algorithm='kd_tree').fit(X)
similar_q = model.kneighbors([new_question_basic])
print(dataset_question[similar_q[1].flatten()])

###############################################################################################
############################# Preprocess the corpus with stemming #############################
###############################################################################################

corpus_stem = [tp.preprocess_stem(question) for question in question_list]
### Remove docs that don't include any words in W2V's vocab ###
corpus_stem, question_list = tp.filter_docs(corpus_stem, question_list, lambda doc: tp.has_vector_representation(wv, doc))
### Filter out any empty docs with stemming ###
corpus_stem, question_list = tp.filter_docs(corpus_stem, question_list, lambda doc: (len(doc) != 0))

### create X ndarray with stemmed words ###
x = []
for question in corpus_stem:  # append the vector for each document
    x.append(tp.document_vector_sum(wv, question))
X_stem = np.array(x)  # list to array

new_question_stem = tp.preprocess_stem(new_question)
new_question_stem = tp.document_vector_sum(wv, new_question_stem)

print(f"This are the most {top_similar_question} similar question according to cosine similarity with stemmed words: \n")
number_of_question_stem = len(corpus_stem)
dis.cosin_dist(number_of_question_stem, X_stem, new_question_stem, top_similar_question, dataset_question)

print(f"This are the most {top_similar_question} similar question according to KD tree with stemming: \n")
model = NearestNeighbors(n_neighbors=top_similar_question, algorithm='kd_tree').fit(X_stem)
similar_q_stem = model.kneighbors([new_question_stem])
print(dataset_question[similar_q_stem[1].flatten()])

####################################################################################
##################### Preprocess the corpus with lemmatization #####################
####################################################################################

corpus_lemma = [tp.preprocess_lemma(question) for question in question_list]
# Remove docs that don't include any words in W2V's vocab
corpus_lemma, question_list = tp.filter_docs(corpus_lemma, question_list, lambda doc: tp.has_vector_representation(wv, doc))
# Filter out any empty docs with lemmatization
corpus_lemma, question_list = tp.filter_docs(corpus_lemma, question_list, lambda doc: (len(doc) != 0))

# create X ndarray with lemmatized words
x = []
for question in corpus_lemma:  # append the vector for each question
    x.append(tp.document_vector_sum(wv, question))
X_lemma = np.array(x)  # list to array

new_question_lemma = tp.preprocess_lemma(new_question)
new_question_lemma = tp.document_vector_sum(wv, new_question_lemma)

print(f"This are the {top_similar_question} most similar question according to cosine similarity with lemmatized words: \n")
number_of_question_lemma = len(corpus_lemma)
dis.cosin_dist(number_of_question_lemma, X_lemma, new_question_lemma, top_similar_question, dataset_question)

print(f"This are the most {top_similar_question} similar question according to KD tree with lemmatization: \n")
model = NearestNeighbors(n_neighbors=top_similar_question, algorithm='kd_tree').fit(X_lemma)
similar_q_lemma = model.kneighbors([new_question_lemma])
print(dataset_question[similar_q_lemma[1].flatten()])

