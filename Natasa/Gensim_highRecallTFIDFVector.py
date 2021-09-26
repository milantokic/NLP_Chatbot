#Izracunati teszinsku sumu i tezinski prosek koristeci idf komponentu, weighted average and weighted sum


#Import the libraries used for counting the tf
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
import gensim.downloader as api
import textPreprocessing as tp
import test_algorithmsForHighRecall as thr
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer

#import dataset using pandas
dataset = pd.read_csv('insurance_qna_dataset.csv', sep='\t').iloc[:, 1:]
dataset_group_by_question = dataset.groupby('Question', as_index=False).agg(lambda x: np.unique(x).tolist())
dataset_question = dataset_group_by_question.iloc[:, 0]
#dataset_question = dataset_group_by_question.iloc[10:15,0]
number_of_question = len(dataset_question)

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

wv = api.load('word2vec-google-news-300')
question_list = [question for question in dataset_question]

corpus = [tp.preprocess(question) for question in question_list]
### Remove docs that don't include any words in W2V's vocab ###
corpus, question_list = tp.filter_docs(corpus, question_list, lambda doc: tp.has_vector_representation(wv, doc))
### Filter out any empty docs with stemming ###
corpus, question_list = tp.filter_docs(corpus, question_list, lambda doc: (len(doc) != 0))

#Basic TfIdf Vectorizer
tfidf = TfidfVectorizer(use_idf=True)
count_tfidf = tfidf.fit_transform(dataset_question)

idf_values = tp.create_idf_list(tfidf)

### create Basic X ndarray with words using sum ###
x = []
for question in corpus:  # append the vector for each document
    x.append(tp.document_vector_weighted_sum(wv, question, idf_values))
X_sum = np.array(x) # list to array

### create Basic X ndarray with words using sum ###
x = []
for question in corpus:  # append the vector for each document
    x.append(tp.document_vector_weighted_mean(wv, question, idf_values))
X_mean = np.array(x)  # list to array

thr.t_algorithm_gensim(operation = 'weightedSum', preprocessing = 'default', test_questions = test_questions, wv = wv, X = X_sum, the_top_similar = the_top_similar, idf_words = idf_values)
thr.t_algorithm_gensim(operation = 'weightedMean', preprocessing = 'default', test_questions = test_questions, wv = wv, X = X_mean, the_top_similar = the_top_similar, idf_words = idf_values)

################################################################
###################Preprocess the corpus with stemming##########
################################################################

corpus_stem = [tp.preprocess_stem(question) for question in question_list]
### Remove docs that don't include any words in W2V's vocab ###
corpus_stem, question_list = tp.filter_docs(corpus_stem, question_list, lambda doc: tp.has_vector_representation(wv, doc))
### Filter out any empty docs with stemming ###
corpus_stem, question_list = tp.filter_docs(corpus_stem, question_list, lambda doc: (len(doc) != 0))

# SnawBall  Tf CountVectorizer
stemmer = SnowballStemmer(language='english')
analyzer = TfidfVectorizer().build_analyzer()
def stemmed_words(text):
    return (stemmer.stem(w) for w in analyzer(text))

stem_tfidf = TfidfVectorizer(analyzer=stemmed_words,use_idf=True)
count_stem_tfidf = stem_tfidf.fit_transform(dataset_question)

idf_values_stem = tp.create_idf_list(stem_tfidf)

### create Basic X ndarray with words using sum ###
x = []
for question in corpus_stem:  # append the vector for each document
    x.append(tp.document_vector_weighted_sum(wv, question, idf_values_stem))
X_stem_sum = np.array(x) # list to array

### create Basic X ndarray with words using mean ###
x = []
for question in corpus_stem:  # append the vector for each document
    x.append(tp.document_vector_weighted_mean(wv, question, idf_values_stem))
X_stem_mean = np.array(x)  # list to array

thr.t_algorithm_gensim(operation = 'weightedSum', preprocessing = 'stem', test_questions = test_questions, wv = wv, X = X_stem_sum, the_top_similar = the_top_similar, idf_words = idf_values_stem)
thr.t_algorithm_gensim(operation = 'weightedMean', preprocessing = 'stem', test_questions = test_questions, wv = wv, X = X_stem_mean, the_top_similar = the_top_similar, idf_words = idf_values_stem)

################################################################
##############Preprocess the corpus with lemmatization##########
################################################################

corpus_lemma = [tp.preprocess_lemma(question) for question in question_list]
### Remove docs that don't include any words in W2V's vocab ###
corpus_lemma, question_list = tp.filter_docs(corpus_lemma, question_list, lambda doc: tp.has_vector_representation(wv, doc))
### Filter out any empty docs with stemming ###
corpus_lemma, question_list = tp.filter_docs(corpus_lemma, question_list, lambda doc: (len(doc) != 0))

# LemmaTokenizer TfIdf CountVectorizer
lemmatizer = WordNetLemmatizer()
analyzer = TfidfVectorizer().build_analyzer()
def lemmatizer_words(text):
    return (lemmatizer.lemmatize(w) for w in analyzer(text))

lemma_tfidf = TfidfVectorizer(analyzer=lemmatizer_words)
count_lemma_tfidf = lemma_tfidf.fit_transform(dataset_question)

idf_values_lemma = tp.create_idf_list(lemma_tfidf)

### create Basic X ndarray with words using sum ###
x = []
for question in corpus_lemma:  # append the vector for each document
    x.append(tp.document_vector_weighted_sum(wv, question, idf_values_lemma))
X_lemma_sum = np.array(x) # list to array

### create Basic X ndarray with words using mean ###
x = []
for question in corpus_lemma:  # append the vector for each document
    x.append(tp.document_vector_weighted_mean(wv, question, idf_values_lemma))
X_lemma_mean = np.array(x)  # list to array

thr.t_algorithm_gensim(operation = 'weightedSum', preprocessing = 'lemma', test_questions = test_questions, wv = wv, X = X_lemma_sum, the_top_similar = the_top_similar, idf_words = idf_values_lemma)
thr.t_algorithm_gensim(operation = 'weightedMean', preprocessing = 'lemma', test_questions = test_questions, wv = wv, X = X_lemma_mean, the_top_similar = the_top_similar, idf_words = idf_values_lemma)






