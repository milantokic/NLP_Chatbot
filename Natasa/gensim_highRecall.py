import pandas as pd
import numpy as np

import gensim.downloader as api
import textPreprocessing as tp
import test_algorithmsForHighRecall as thr

wv = api.load('word2vec-google-news-300')

### import dataset using pandas ###
dataset = pd.read_csv("C:\\Users\\Natasa\\Documents\\Machine Learning\\insurance_qna_dataset.csv", sep='\t').iloc[:, 1:]
dataset_group_by_question = dataset.groupby('Question', as_index=False).agg(lambda x: np.unique(x).tolist())
dataset_question = dataset_group_by_question.iloc[:, 0]
question_list = [question for question in dataset_question]

test_questions = {'0': 'What Is The Best Life Insurance To Buy? ',
                  '1': 'What Is The Comprehensive Life Insurance To Buy? ',
                  '2': 'What Is The Comprehensive Life Insurance To Purchase? ',
                  '3': 'What Is The Most Outstanding Guarantee To Purchase? ',

                  '4': 'How Much Does A Whole Life Insurance Policy Typically Cost? ',
                  '5': 'How Much Does A Whole Life Insurance Policy Usually Cost? ',
                  '6': 'What Is The Typically Price Of Whole Life Insurance Policy? ',
                  '7': 'At What Price Could Life Insurance Be Bought? ',

                  '8': 'Who Should Get Critical Illness Insurance? ',
                  '9': 'Who Should Buy Critical Illness Insurance? ',
                  '10': 'Critical Illness Insurance Should Be Got By Who? ',
                  '11': 'Which People Should Buy Censorious Illness Insurance? ',

                  '12': 'Where To Buy Good Life Insurance? ',
                  '13': 'Where To Get Quality Life Insurance? ',
                  '14': 'Where Is The Best Place To Buy Quality Life Insurance? ',
                  '15': 'Which Is The Best Insurance Company to Get Satisfying Life Insurance? ',

                  '16': 'How Can I Find Who My Car Insurance Is With? ',
                  '17': 'How To Find Who My Car Insurance Is With? ',
                  '18': 'How To Find Out Which Is My Vehicle Insurance Company? ',
                  '19': 'At Which Company My Vehicle Is Insured? ',

                  '20': 'What Does Home Insurance Cost? ',
                  '21': 'What Does Property Insurance Cost? ',
                  '22': 'What Is The Price Of Home Insurance? ',
                  '23': 'What Does Insuring My House Cost? ',

                  '24': 'What Is The Purpose Of Life Insurance? ',
                  '25': 'What Is The Motive Of Life Insurance? ',
                  '26': 'What Is The Idea Of Life Indemnity? ',
                  '27': 'What Is The Idea Of Buying Life Indemnity? ',

                  '28': 'Are New Cars Cheaper To Insure Than Older Cars? ',
                  '29': 'Are New Vehicles Cheaper To Insure Than Older? ',
                  '30': 'Is It Cheaper To Insure, Old Or New Cars? ',
                  '31': 'Is It More Expensive To Insure New Vehicles Or Old? ',

                  '32': 'What Is The Best Disability Insurance Company? ',
                  '33': 'What Is The Best Invalidity Insurance Company? ',
                  '34': 'What Is The Recommended Invalidity Insurance Company? ',
                  '35': 'At What Company Should I Buy Disability Insurance? ',

                  '36': 'How Much Money Does A Life Insurance Salesman Make? ',
                  '37': 'How Much Cash Does A Life Protections Sales Representative Make? ',
                  '38': 'How Much Cash Does A Life Assurances Deals Agent Make? ',
                  '39': 'What Is The Amount Of Cash That A Life Protections Sales Make? '
                  }

the_top_similar = 30

#######################################################################################
############################# Preprocess the corpus basic #############################
#######################################################################################

corpus = [tp.preprocess(question) for question in question_list]
### Remove docs that don't include any words in W2V's vocab ###
corpus, question_list = tp.filter_docs(corpus, question_list, lambda doc: tp.has_vector_representation(wv, doc))
### Filter out any empty docs with stemming ###
corpus, question_list = tp.filter_docs(corpus, question_list, lambda doc: (len(doc) != 0))

### create Basic X ndarray with words using sum ###
x = []
for question in corpus:  # append the vector for each document
    x.append(tp.document_vector_sum(wv, question))
X_sum = np.array(x)  # list to array

### create Basic X ndarray with words using mean ###
x = []
for question in corpus:  # append the vector for each document
    x.append(tp.document_vector_mean(wv, question))
X_mean = np.array(x)  # list to array

thr.t_algorithm_gensim(operation = 'sum', preprocessing = 'default', test_questions = test_questions, wv = wv, X = X_sum, the_top_similar = the_top_similar)
thr.t_algorithm_gensim(operation = 'mean', preprocessing = 'default', test_questions = test_questions, wv = wv, X = X_mean, the_top_similar = the_top_similar)

'''
new_question = input("Please enter an insurance question: ")

new_question_basic = tp.preprocess(new_question)
new_question_basic = tp.document_vector_sum(wv, new_question_basic)

print(f"This are the most {top_similar_question} similar question according to cosine similarity: \n")
number_of_question = len(corpus)
dis.cosin_dist(number_of_question, X, new_question_basic, top_similar_question, dataset_question)

print(f"This are the most {top_similar_question} similar question according to KD tree: \n")
model = NearestNeighbors(n_neighbors=top_similar_question, algorithm='kd_tree').fit(X)
similar_q = model.kneighbors([new_question_basic])
print(dataset_question[similar_q[1].flatten()])
'''

###############################################################################################
############################# Preprocess the corpus with stemming #############################
###############################################################################################

corpus_stem = [tp.preprocess_stem(question) for question in question_list]
### Remove docs that don't include any words in W2V's vocab ###
corpus_stem, question_list = tp.filter_docs(corpus_stem, question_list, lambda doc: tp.has_vector_representation(wv, doc))
### Filter out any empty docs with stemming ###
corpus_stem, question_list = tp.filter_docs(corpus_stem, question_list, lambda doc: (len(doc) != 0))

### create X ndarray with stemmed words using sum ###
x = []
for question in corpus_stem:  # append the vector for each document
    x.append(tp.document_vector_sum(wv, question))
X_stem_sum = np.array(x)  # list to array

### create X ndarray with stemmed words using mean ###
x = []
for question in corpus_stem:  # append the vector for each document
    x.append(tp.document_vector_mean(wv, question))
X_stem_mean = np.array(x)  # list to array

thr.t_algorithm_gensim(operation = 'sum', preprocessing = 'stem', test_questions = test_questions, wv = wv, X = X_stem_sum, the_top_similar = the_top_similar)
thr.t_algorithm_gensim(operation = 'mean', preprocessing = 'stem', test_questions = test_questions, wv = wv, X = X_stem_mean, the_top_similar = the_top_similar)


'''
new_question_stem = tp.preprocess_stem(new_question)
new_question_stem = tp.document_vector_sum(wv, new_question_stem)

print(f"This are the most {top_similar_question} similar question according to cosine similarity with stemmed words: \n")
number_of_question_stem = len(corpus_stem)
dis.cosin_dist(number_of_question_stem, X_stem, new_question_stem, top_similar_question, dataset_question)

print(f"This are the most {top_similar_question} similar question according to KD tree with stemming: \n")
model = NearestNeighbors(n_neighbors=top_similar_question, algorithm='kd_tree').fit(X_stem)
similar_q_stem = model.kneighbors([new_question_stem])
print(dataset_question[similar_q_stem[1].flatten()])
'''

####################################################################################
##################### Preprocess the corpus with lemmatization #####################
####################################################################################

corpus_lemma = [tp.preprocess_lemma(question) for question in question_list]
# Remove docs that don't include any words in W2V's vocab
corpus_lemma, question_list = tp.filter_docs(corpus_lemma, question_list, lambda doc: tp.has_vector_representation(wv, doc))
# Filter out any empty docs with lemmatization
corpus_lemma, question_list = tp.filter_docs(corpus_lemma, question_list, lambda doc: (len(doc) != 0))

# create X ndarray with lemmatized words using sum
x = []
for question in corpus_lemma:  # append the vector for each question
    x.append(tp.document_vector_sum(wv, question))
X_lemma_sum = np.array(x)  # list to array

# create X ndarray with lemmatized words using mean
x = []
for question in corpus_lemma:  # append the vector for each question
    x.append(tp.document_vector_mean(wv, question))
X_lemma_mean = np.array(x)  # list to array

thr.t_algorithm_gensim(operation = 'sum', preprocessing = 'lemma', test_questions = test_questions, wv = wv, X = X_lemma_sum, the_top_similar = the_top_similar)
thr.t_algorithm_gensim(operation = 'mean', preprocessing = 'lemma', test_questions = test_questions, wv = wv, X = X_lemma_mean, the_top_similar = the_top_similar)


'''
new_question_lemma = tp.preprocess_lemma(new_question)
new_question_lemma = tp.document_vector_sum(wv, new_question_lemma)

print(f"This are the {top_similar_question} most similar question according to cosine similarity with lemmatized words: \n")
number_of_question_lemma = len(corpus_lemma)
dis.cosin_dist(number_of_question_lemma, X_lemma, new_question_lemma, top_similar_question, dataset_question)

print(f"This are the most {top_similar_question} similar question according to KD tree with lemmatization: \n")
model = NearestNeighbors(n_neighbors=top_similar_question, algorithm='kd_tree').fit(X_lemma)
similar_q_lemma = model.kneighbors([new_question_lemma])
print(dataset_question[similar_q_lemma[1].flatten()])
'''
